#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn
from torchvision import models

try:
    import lpips
except Exception:  # pragma: no cover
    lpips = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _pil_to_float_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = _pil_to_float_np(image)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _align_pair(reference: Image.Image, candidate: Image.Image) -> Tuple[Image.Image, Image.Image]:
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.BICUBIC)
    return reference, candidate


class _MetricRuntime:
    _lpips_model = None
    _inception_model = None

    @classmethod
    def get_lpips_model(cls):
        if lpips is None:
            raise RuntimeError("lpips package is not installed")
        if cls._lpips_model is None:
            model = lpips.LPIPS(net="alex")
            model.eval()
            cls._lpips_model = model.cpu()
        return cls._lpips_model

    @classmethod
    def get_inception_model(cls):
        if cls._inception_model is None:
            weights = models.Inception_V3_Weights.DEFAULT
            model = models.inception_v3(weights=weights)
            model.fc = nn.Identity()
            if hasattr(model, "AuxLogits"):
                model.AuxLogits = None
            if hasattr(model, "aux_logits"):
                model.aux_logits = False
            model.eval()
            cls._inception_model = model.cpu()
        return cls._inception_model


def _compute_lpips(reference: Image.Image, candidate: Image.Image) -> float:
    model = _MetricRuntime.get_lpips_model()
    ref = (_pil_to_tensor(reference) * 2.0 - 1.0).cpu()
    cand = (_pil_to_tensor(candidate) * 2.0 - 1.0).cpu()
    with torch.no_grad():
        value = model(ref, cand)
    return float(value.item())


def _inception_input(image: Image.Image) -> torch.Tensor:
    image = image.resize((299, 299), Image.BICUBIC)
    tensor = _pil_to_tensor(image).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _extract_inception_features(images: Sequence[Image.Image]) -> np.ndarray:
    if not images:
        return np.empty((0, 2048), dtype=np.float64)
    model = _MetricRuntime.get_inception_model()
    batches = [_inception_input(image) for image in images]
    batch = torch.cat(batches, dim=0)
    with torch.no_grad():
        feats = model(batch)
    if hasattr(feats, "logits"):
        feats = feats.logits
    elif isinstance(feats, tuple):
        feats = feats[0]
    feats = feats.detach().cpu().numpy().astype(np.float64)
    if feats.ndim == 1:
        feats = feats[None, :]
    return feats


def _covariance(features: np.ndarray) -> np.ndarray:
    if features.shape[0] <= 1:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.cov(features, rowvar=False)


def frechet_distance_from_features(ref_features: np.ndarray, cand_features: np.ndarray) -> float:
    if ref_features.size == 0 or cand_features.size == 0:
        raise ValueError("Empty feature set passed to FID computation")

    mu1 = np.mean(ref_features, axis=0)
    mu2 = np.mean(cand_features, axis=0)
    sigma1 = _covariance(ref_features)
    sigma2 = _covariance(cand_features)

    diff = mu1 - mu2
    if not np.any(sigma1) and not np.any(sigma2):
        return float(diff @ diff)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    if fid < 0 and abs(fid) < 1e-8:
        fid = 0.0
    return fid


def compute_pair_metrics(reference_path: Path, candidate_path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "computed_at": _utc_now(),
        "reference_image": str(reference_path),
        "generated_image": str(candidate_path),
        "metrics": {},
        "metric_errors": {},
        "metric_notes": {
            "fid": (
                "Per-image FID is computed as the Fréchet distance between singleton "
                "Inception-V3 feature distributions for the reference and generated image."
            )
        },
    }

    reference_img = _read_rgb(reference_path)
    candidate_img = _read_rgb(candidate_path)
    reference_img, candidate_img = _align_pair(reference_img, candidate_img)

    reference_np = _pil_to_float_np(reference_img)
    candidate_np = _pil_to_float_np(candidate_img)

    payload["reference_size"] = {"width": int(reference_img.width), "height": int(reference_img.height)}
    payload["generated_size"] = {"width": int(candidate_img.width), "height": int(candidate_img.height)}

    try:
        payload["metrics"]["ssim"] = float(
            structural_similarity(reference_np, candidate_np, channel_axis=2, data_range=1.0)
        )
    except Exception as exc:
        payload["metric_errors"]["ssim"] = str(exc)

    try:
        payload["metrics"]["psnr"] = float(peak_signal_noise_ratio(reference_np, candidate_np, data_range=1.0))
    except Exception as exc:
        payload["metric_errors"]["psnr"] = str(exc)

    try:
        payload["metrics"]["lpips"] = _compute_lpips(reference_img, candidate_img)
    except Exception as exc:
        payload["metric_errors"]["lpips"] = str(exc)

    try:
        ref_features = _extract_inception_features([reference_img])
        cand_features = _extract_inception_features([candidate_img])
        payload["metrics"]["fid"] = frechet_distance_from_features(ref_features, cand_features)
    except Exception as exc:
        payload["metric_errors"]["fid"] = str(exc)

    return payload


def write_pair_metrics(
    run_dir: Path,
    reference_relpath: str,
    generated_relpath: str,
    *,
    output_name: str = "image_metrics.json",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = compute_pair_metrics(run_dir / reference_relpath, run_dir / generated_relpath)
    payload["reference_image"] = reference_relpath
    payload["generated_image"] = generated_relpath
    if extra_fields:
        payload.update(extra_fields)
    _write_json(run_dir / output_name, payload)
    return payload


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def aggregate_run_metrics(run_dirs: Sequence[Path]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    source_images: List[Image.Image] = []
    generated_images: List[Image.Image] = []
    numeric_values: Dict[str, List[float]] = {"ssim": [], "psnr": [], "lpips": [], "fid": []}

    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir).resolve()
        metric_path = run_dir / "image_metrics.json"
        if not metric_path.exists():
            continue

        try:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        record: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "reference_image": payload.get("reference_image"),
            "generated_image": payload.get("generated_image"),
            "metrics": metrics,
            "metric_errors": payload.get("metric_errors", {}),
        }

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = None
            if isinstance(metadata, dict):
                record["run_name"] = metadata.get("run_name")
                record["label_index"] = metadata.get("true_label")
                record["role_names"] = metadata.get("role_names")

        runs.append(record)

        ref_rel = payload.get("reference_image")
        gen_rel = payload.get("generated_image")
        if isinstance(ref_rel, str) and isinstance(gen_rel, str):
            ref_path = run_dir / ref_rel
            gen_path = run_dir / gen_rel
            if ref_path.exists() and gen_path.exists():
                try:
                    source_images.append(_read_rgb(ref_path))
                    generated_images.append(_read_rgb(gen_path))
                except Exception:
                    pass

        for key in numeric_values:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))

    summary: Dict[str, Any] = {
        "created_at": _utc_now(),
        "run_count": len(runs),
        "metrics_mean": {},
        "metrics_std": {},
        "runs": runs,
    }

    for key, values in numeric_values.items():
        stats = _mean_std(values)
        summary["metrics_mean"][key] = stats["mean"]
        summary["metrics_std"][key] = stats["std"]
        summary[f"{key}_count"] = len(values)

    if source_images and generated_images and len(source_images) == len(generated_images):
        try:
            ref_features = _extract_inception_features(source_images)
            gen_features = _extract_inception_features(generated_images)
            summary["dataset_fid"] = frechet_distance_from_features(ref_features, gen_features)
        except Exception as exc:
            summary["dataset_fid_error"] = str(exc)

    return summary


def write_aggregate_run_metrics(run_dirs: Sequence[Path], output_path: Path) -> Dict[str, Any]:
    payload = aggregate_run_metrics(run_dirs)
    _write_json(output_path, payload)
    return payload

#!/usr/bin/env python3
"""Evaluate dataset FID and paired DINOv2 CLS cosine similarity from NPZ files.

The clean/adversarial pairing follows the repository's NIPS2017 convention:
rows are natural-sorted by ImageId from data/nips2017/images.csv, exactly as in
eval_fid_from_npz.py and eval_quality_from_npz.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from natsort import natsorted


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.npz_loader import load_adv_images_from_npz
from pytorch_fid_score_new.fid_score_new import calculate_frechet_distance
from pytorch_fid_score_new.inception import InceptionV3


DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
FID_DIMS = 2048


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return value or "item"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest must contain a non-empty items list: {path}")
    seen = set()
    normalized = []
    for raw in items:
        if not isinstance(raw, dict):
            raise TypeError("Every manifest item must be an object")
        item = dict(raw)
        label = str(item.get("label", "")).strip()
        npz_value = str(item.get("npz", "")).strip()
        if not label or not npz_value:
            raise ValueError(f"Manifest item needs label and npz: {raw}")
        if label in seen:
            raise ValueError(f"Duplicate manifest label: {label}")
        seen.add(label)
        npz_path = Path(npz_value).expanduser()
        if not npz_path.is_absolute():
            npz_path = REPO_ROOT / npz_path
        item["label"] = label
        item["npz"] = str(npz_path.resolve())
        normalized.append(item)
    return normalized


def load_image_ids(csv_path: Path) -> List[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "ImageId" not in rows[0]:
        raise ValueError(f"Missing ImageId column or no rows: {csv_path}")
    return natsorted([str(row["ImageId"]) for row in rows])


def load_clean_images(
    image_ids: Sequence[str], images_root: Path, image_size: int
) -> np.ndarray:
    clean = np.empty((len(image_ids), 3, image_size, image_size), dtype=np.uint8)
    for index, image_id in enumerate(image_ids):
        image_path = images_root / f"{image_id}.png"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), resample=Image.BILINEAR)
            clean[index] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
    return clean


def resolve_dinov2_repo() -> Path:
    env_value = str(os.environ.get("DINOV2_REPO_DIR", "")).strip()
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "ckpt" / "torch_hub" / "facebookresearch_dinov2_main",
            REPO_ROOT / "third_party" / "dinov2",
            REPO_ROOT / "third_party" / "DINOv2",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Local DINOv2 repository not found; searched: {searched}")


def load_models(device: torch.device, dino_model_name: str):
    torch_hub_dir = REPO_ROOT / "ckpt" / "torch_hub"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub_dir))

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]])
    inception.eval().to(device)

    dino_repo = resolve_dinov2_repo()
    dino = torch.hub.load(
        str(dino_repo), dino_model_name, source="local", pretrained=True
    )
    dino.eval().to(device)
    return inception, dino, dino_repo


def _to_float_batch(array: np.ndarray, start: int, end: int, device: torch.device):
    batch = torch.from_numpy(np.ascontiguousarray(array[start:end])).to(device)
    if batch.dtype == torch.uint8:
        batch = batch.float().div_(255.0)
    else:
        batch = batch.float()
    return batch.clamp_(0.0, 1.0)


def _dino_input(batch: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        batch, size=(256, 256), mode="bicubic", align_corners=False, antialias=True
    )
    offset = (256 - 224) // 2
    cropped = resized[:, :, offset : offset + 224, offset : offset + 224]
    mean = cropped.new_tensor(DINOV2_MEAN).view(1, 3, 1, 1)
    std = cropped.new_tensor(DINOV2_STD).view(1, 3, 1, 1)
    return (cropped - mean) / std


def _dino_cls_features(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        output = model.forward_features(batch)
    else:
        output = model(batch)
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "cls_token", "features"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError(f"No CLS feature key in DINOv2 output: {sorted(output)}")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Expected 2-D DINOv2 CLS features, got {tuple(output.shape)}")
    return output


def extract_features(
    images: np.ndarray,
    inception: torch.nn.Module,
    dino: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    fid_batches: List[np.ndarray] = []
    dino_batches: List[np.ndarray] = []
    total = int(images.shape[0])
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            batch = _to_float_batch(images, start, end, device)
            fid = inception(batch)[0]
            if fid.shape[-2:] != (1, 1):
                fid = F.adaptive_avg_pool2d(fid, output_size=(1, 1))
            fid_batches.append(fid.flatten(1).float().cpu().numpy())

            dino_input = _dino_input(batch)
            dino_features = _dino_cls_features(dino, dino_input)
            dino_batches.append(dino_features.float().cpu().numpy())
            del batch, fid, dino_input, dino_features
    return np.concatenate(fid_batches, axis=0), np.concatenate(dino_batches, axis=0)


def feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def paired_cosine(clean: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float64)
    adversarial = np.asarray(adversarial, dtype=np.float64)
    clean /= np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-12)
    adversarial /= np.maximum(np.linalg.norm(adversarial, axis=1, keepdims=True), 1e-12)
    return np.sum(clean * adversarial, axis=1)


def bootstrap_mean_ci(values: np.ndarray, seed: int, replicates: int = 10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for start in range(0, replicates, chunk):
        end = min(replicates, start + chunk)
        indices = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_cosine(values: np.ndarray, seed: int) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = bootstrap_mean_ci(values, seed=seed)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
    }


def write_per_sample_csv(
    path: Path, item: Dict[str, Any], image_ids: Sequence[str], cosine: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    metadata_fields = [
        key for key in ("label", "source_model", "method", "verifier") if key in item
    ]
    fields = metadata_fields + ["sample_index", "image_id", "dinov2_cls_cosine"]
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (image_id, value) in enumerate(zip(image_ids, cosine)):
            row = {key: item.get(key) for key in metadata_fields}
            row.update(
                {
                    "sample_index": index,
                    "image_id": image_id,
                    "dinov2_cls_cosine": f"{float(value):.10f}",
                }
            )
            writer.writerow(row)
    temp.replace(path)


def aggregate_outputs(output_dir: Path, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        item_path = output_dir / "items" / f"{slugify(item['label'])}.json"
        if not item_path.is_file():
            continue
        payload = json.loads(item_path.read_text(encoding="utf-8"))
        cosine = payload["dinov2_cls_cosine"]
        row = {key: item.get(key, "") for key in ("label", "source_model", "method", "verifier")}
        row.update(
            {
                "sample_count": payload["sample_count"],
                "fid": payload["fid"],
                "dinov2_cls_cosine_mean": cosine["mean"],
                "dinov2_cls_cosine_std": cosine["std"],
                "dinov2_cls_cosine_median": cosine["median"],
                "dinov2_cls_cosine_p05": cosine["p05"],
                "dinov2_cls_cosine_p95": cosine["p95"],
                "dinov2_cls_cosine_ci95_low": cosine["mean_ci95_low"],
                "dinov2_cls_cosine_ci95_high": cosine["mean_ci95_high"],
                "npz": payload["npz"],
                "npz_sha256": payload["npz_sha256"],
            }
        )
        rows.append(row)
    if rows:
        csv_path = output_dir / "fid_dinov2_raw.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(args.manifest.expanduser().resolve())
    if args.only:
        allowed = set(args.only)
        items = [item for item in items if item["label"] in allowed]
        missing = allowed.difference(item["label"] for item in items)
        if missing:
            raise ValueError(f"Unknown --only labels: {sorted(missing)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    for item in items:
        if not Path(item["npz"]).is_file():
            raise FileNotFoundError(item["npz"])

    first_images = load_adv_images_from_npz(items[0]["npz"], output_layout="nchw")
    full_count = int(first_images.shape[0])
    image_size = int(first_images.shape[-1])
    del first_images
    requested_count = min(full_count, args.limit) if args.limit else full_count

    image_ids_all = load_image_ids(REPO_ROOT / "data" / "nips2017" / "images.csv")
    if full_count > len(image_ids_all):
        raise ValueError(f"NPZ has {full_count} images but CSV has {len(image_ids_all)} rows")
    image_ids = image_ids_all[:requested_count]

    inception, dino, dino_repo = load_models(device, args.dino_model)
    cache_path = output_dir / (
        f"clean_features_n{requested_count}_s{image_size}_{slugify(args.dino_model)}.npz"
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            clean_fid = np.asarray(cache["fid"], dtype=np.float32)
            clean_dino = np.asarray(cache["dino"], dtype=np.float32)
            cached_ids = [str(value) for value in cache["image_ids"].tolist()]
        if cached_ids != image_ids:
            raise ValueError(f"Cached image IDs do not match current CSV: {cache_path}")
    else:
        clean_images = load_clean_images(
            image_ids,
            REPO_ROOT / "data" / "nips2017" / "images",
            image_size,
        )
        clean_fid, clean_dino = extract_features(
            clean_images, inception, dino, device, args.batch_size
        )
        np.savez(
            cache_path,
            fid=clean_fid.astype(np.float32),
            dino=clean_dino.astype(np.float32),
            image_ids=np.asarray(image_ids),
        )
        del clean_images

    clean_mu, clean_sigma = feature_stats(clean_fid)
    run_started = utc_now()
    for item_index, item in enumerate(items):
        label = item["label"]
        item_path = output_dir / "items" / f"{slugify(label)}.json"
        sample_path = output_dir / "per_sample" / f"{slugify(label)}.csv"
        if args.resume and item_path.is_file() and sample_path.is_file():
            print(f"SKIP completed {label}", flush=True)
            continue

        started = time.time()
        print(f"START {label} ({item_index + 1}/{len(items)})", flush=True)
        adv = load_adv_images_from_npz(item["npz"], output_layout="nchw")
        item_full_count = int(adv.shape[0])
        if item_full_count < requested_count:
            raise ValueError(
                f"{label}: requested {requested_count} images but NPZ has {item_full_count}"
            )
        if tuple(adv.shape[1:]) != (3, image_size, image_size):
            raise ValueError(f"{label}: inconsistent image shape {adv.shape}")
        adv = adv[:requested_count]
        quantization_error = float(
            np.max(np.abs(adv[: min(16, requested_count)] * 255.0 - np.round(adv[: min(16, requested_count)] * 255.0)))
        )

        adv_fid, adv_dino = extract_features(
            adv, inception, dino, device, args.batch_size
        )
        adv_mu, adv_sigma = feature_stats(adv_fid)
        fid_value = float(
            calculate_frechet_distance(clean_mu, clean_sigma, adv_mu, adv_sigma)
        )
        cosine = paired_cosine(clean_dino, adv_dino)
        cosine_summary = summarize_cosine(cosine, seed=args.seed + item_index)
        write_per_sample_csv(sample_path, item, image_ids, cosine)

        payload = dict(item)
        payload.update(
            {
                "computed_at": utc_now(),
                "elapsed_seconds": time.time() - started,
                "sample_count": requested_count,
                "full_npz_sample_count": item_full_count,
                "image_size": image_size,
                "fid": fid_value,
                "dinov2_cls_cosine": cosine_summary,
                "npz_sha256": sha256_file(Path(item["npz"])),
                "sample_mapping": "natural-sorted ImageId from data/nips2017/images.csv",
                "sample_image_id_first": image_ids[0],
                "sample_image_id_last": image_ids[-1],
                "sample_quantization_error_8bit_grid": quantization_error,
            }
        )
        write_json(item_path, payload)
        del adv, adv_fid, adv_dino, cosine
        print(
            f"DONE {label} fid={fid_value:.6f} dino_cos={cosine_summary['mean']:.8f}",
            flush=True,
        )

    rows = aggregate_outputs(output_dir, items)
    checkpoint_dir = REPO_ROOT / "ckpt" / "torch_hub" / "checkpoints"
    weight_paths = [
        REPO_ROOT / "pytorch_fid_score_new" / "pt_inception-2015-12-05-6726825d.pth",
        checkpoint_dir / "dinov2_vitb14_pretrain.pth",
    ]
    weights = {}
    for path in weight_paths:
        if path.is_file():
            weights[path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
    run_payload = {
        "started_at": run_started,
        "completed_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "requested_item_count": len(items),
        "sample_count": requested_count,
        "full_npz_sample_count": full_count,
        "limit": args.limit,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "dino_model": args.dino_model,
        "dino_repo": str(dino_repo),
        "dino_preprocess": {
            "resize": [256, 256],
            "center_crop": [224, 224],
            "mean": DINOV2_MEAN,
            "std": DINOV2_STD,
        },
        "dino_feature": "forward_features()['x_norm_clstoken'], L2-normalized before cosine",
        "fid_feature": "canonical pytorch-fid TensorFlow-compatible InceptionV3 pool3 (2048-D)",
        "sample_mapping": "natural-sorted ImageId from data/nips2017/images.csv",
        "weights": weights,
        "seed": args.seed,
    }
    write_json(output_dir / "run_metadata.json", run_payload)
    print(f"COMPLETE rows={len(rows)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

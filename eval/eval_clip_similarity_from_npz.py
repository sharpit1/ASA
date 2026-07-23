#!/usr/bin/env python3
"""Evaluate paired clean/adversarial CLIP image-embedding cosine from NPZ files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from natsort import natsorted
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

import open_clip
from open_clip.pretrained import download_pretrained


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.npz_loader import load_adv_images_from_npz


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_IMAGE_SIZE = 224


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
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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
        npz_path = npz_path.resolve()
        if npz_path.name != "adversarial_examples.npz":
            raise ValueError(f"Only standard adversarial_examples.npz is allowed: {npz_path}")
        if "flux2_vlm_thinking" in str(npz_path):
            raise ValueError(f"Thinking output is excluded: {npz_path}")
        item["label"] = label
        item["npz"] = str(npz_path)
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


def _to_float_batch(
    array: np.ndarray, start: int, end: int, device: torch.device
) -> torch.Tensor:
    batch = torch.from_numpy(np.ascontiguousarray(array[start:end])).to(device)
    if batch.dtype == torch.uint8:
        batch = batch.float().div_(255.0)
    else:
        batch = batch.float()
    return batch.clamp_(0.0, 1.0)


def _clip_input(batch: torch.Tensor) -> torch.Tensor:
    resized = TVF.resize(
        batch,
        size=[CLIP_IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    cropped = TVF.center_crop(resized, [CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE])
    return TVF.normalize(cropped, mean=CLIP_MEAN, std=CLIP_STD)


def extract_features(
    images: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batches: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(images.shape[0]), batch_size):
            end = min(int(images.shape[0]), start + batch_size)
            batch = _to_float_batch(images, start, end, device)
            features = model.encode_image(_clip_input(batch))
            features = F.normalize(features.float(), dim=-1)
            batches.append(features.cpu().numpy())
            del batch, features
    return np.concatenate(batches, axis=0)


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
        key
        for key in ("label", "source_model", "method", "comparison_scope", "verifier")
        if key in item
    ]
    fields = metadata_fields + ["sample_index", "image_id", "clip_image_cosine"]
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (image_id, value) in enumerate(zip(image_ids, cosine)):
            row = {key: item.get(key) for key in metadata_fields}
            row.update(
                {
                    "sample_index": index,
                    "image_id": image_id,
                    "clip_image_cosine": f"{float(value):.10f}",
                }
            )
            writer.writerow(row)
    temp.replace(path)


def aggregate_outputs(
    output_dir: Path, items: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        item_path = output_dir / "items" / f"{slugify(item['label'])}.json"
        if not item_path.is_file():
            continue
        payload = json.loads(item_path.read_text(encoding="utf-8"))
        cosine = payload["clip_image_cosine"]
        row = {
            key: item.get(key, "")
            for key in (
                "label",
                "source_model",
                "method",
                "comparison_scope",
                "verifier",
            )
        }
        row.update(
            {
                "sample_count": payload["sample_count"],
                "clip_image_cosine_mean": cosine["mean"],
                "clip_image_cosine_std": cosine["std"],
                "clip_image_cosine_median": cosine["median"],
                "clip_image_cosine_p05": cosine["p05"],
                "clip_image_cosine_p95": cosine["p95"],
                "clip_image_cosine_ci95_low": cosine["mean_ci95_low"],
                "clip_image_cosine_ci95_high": cosine["mean_ci95_high"],
                "npz": payload["npz"],
                "npz_sha256": payload["npz_sha256"],
            }
        )
        rows.append(row)
    if rows:
        csv_path = output_dir / "clip_similarity_raw.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

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

    cache_dir = REPO_ROOT / "ckpt" / "open_clip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pretrained_cfg = open_clip.get_pretrained_cfg(args.model, args.pretrained)
    if not pretrained_cfg:
        raise ValueError(f"Unknown pretrained pair: {args.model}/{args.pretrained}")
    weight_path = Path(
        download_pretrained(
            pretrained_cfg, prefer_hf_hub=False, cache_dir=str(cache_dir)
        )
    ).resolve()
    weight_sha256 = sha256_file(weight_path)
    model, _, _ = open_clip.create_model_and_transforms(
        args.model,
        pretrained=str(weight_path),
        precision="fp32",
        device=device,
        # OpenAI's canonical .pt is a TorchScript archive. PyTorch 2.6+
        # rejects TorchScript archives with weights_only=True; the file was
        # checksum-verified by open_clip's official URL downloader above.
        weights_only=False,
    )
    model.eval()

    clean_cache = output_dir / (
        f"clean_clip_features_n{requested_count}_s{image_size}_"
        f"{slugify(args.model)}_{weight_sha256[:12]}.npz"
    )
    if clean_cache.is_file():
        with np.load(clean_cache, allow_pickle=False) as cache:
            clean_features = np.asarray(cache["features"], dtype=np.float32)
            cached_ids = [str(value) for value in cache["image_ids"].tolist()]
        if cached_ids != image_ids:
            raise ValueError(f"Cached image IDs do not match current CSV: {clean_cache}")
    else:
        clean_images = load_clean_images(
            image_ids, REPO_ROOT / "data" / "nips2017" / "images", image_size
        )
        clean_features = extract_features(clean_images, model, device, args.batch_size)
        np.savez(
            clean_cache,
            features=clean_features.astype(np.float32),
            image_ids=np.asarray(image_ids),
        )
        del clean_images

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
            np.max(
                np.abs(
                    adv[: min(16, requested_count)] * 255.0
                    - np.round(adv[: min(16, requested_count)] * 255.0)
                )
            )
        )
        adv_features = extract_features(adv, model, device, args.batch_size)
        cosine = paired_cosine(clean_features, adv_features)
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
                "clip_image_cosine": cosine_summary,
                "clip_feature_dim": int(clean_features.shape[1]),
                "npz_sha256": sha256_file(Path(item["npz"])),
                "sample_mapping": "natural-sorted ImageId from data/nips2017/images.csv",
                "sample_image_id_first": image_ids[0],
                "sample_image_id_last": image_ids[-1],
                "sample_quantization_error_8bit_grid": quantization_error,
            }
        )
        write_json(item_path, payload)
        del adv, adv_features, cosine
        print(
            f"DONE {label} clip_image_cos={cosine_summary['mean']:.8f}",
            flush=True,
        )

    rows = aggregate_outputs(output_dir, items)
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
        "open_clip_version": importlib.metadata.version("open-clip-torch"),
        "clip_model": args.model,
        "clip_pretrained": args.pretrained,
        "clip_weight": {
            "path": str(weight_path),
            "sha256": weight_sha256,
            "size": weight_path.stat().st_size,
            "pretrained_cfg": pretrained_cfg,
        },
        "clip_preprocess": {
            "resize_shortest_side": CLIP_IMAGE_SIZE,
            "center_crop": [CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE],
            "interpolation": "bicubic",
            "antialias": True,
            "mean": CLIP_MEAN,
            "std": CLIP_STD,
        },
        "clip_feature": "encode_image output, float32 L2 normalization, paired cosine",
        "tf32_enabled": False,
        "sample_mapping": "natural-sorted ImageId from data/nips2017/images.csv",
        "thinking_excluded": True,
        "standard_npz_only": True,
        "seed": args.seed,
    }
    write_json(output_dir / "run_metadata.json", run_payload)
    print(f"COMPLETE rows={len(rows)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()

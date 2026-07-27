from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "nips2017"
    / "nips2017_selected_20260714_195944"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "nips2017"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis"
    / "nips2017_ablation_and_no_verifier_dinov3_idsim_20260727"
)
DEFAULT_DINOV3_CHECKPOINT = (
    REPO_ROOT / "ckpt" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_IDSIM_CACHE = REPO_ROOT / "ckpt" / "id_sim"
DEFAULT_DINOV3_REPO = REPO_ROOT / "third_party" / "dinov3"
DEFAULT_RUN_NAME = "ablation_and_no_verifier"
DEFAULT_MODELS = ("swin", "vim-small")
DINOV3_CHECKPOINT_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure DINOv3 and ID-Sim on clean-correct, successful NIPS2017 "
            "ablation_and_no_verifier attacks."
        )
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--victim-batch-size", type=int, default=32)
    parser.add_argument("--metric-batch-size", type=int, default=4)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--idsim-cache", type=Path, default=DEFAULT_IDSIM_CACHE)
    parser.add_argument(
        "--stage",
        choices=("selection", "metrics", "all"),
        default="all",
        help="Run victim filtering, identity metrics, or both.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA float16 autocast for identity feature extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-model sample limit for smoke tests only.",
    )
    return parser.parse_args()


def _natural_key(text: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(text))
    ]


def _model_names(raw: str) -> list[str]:
    values = []
    for item in str(raw).split(","):
        name = item.strip()
        if name and name not in values:
            values.append(name)
    unsupported = sorted(set(values) - set(DEFAULT_MODELS))
    if unsupported:
        raise ValueError(f"unsupported model(s): {unsupported}; expected {DEFAULT_MODELS}")
    if not values:
        raise ValueError("at least one model is required")
    return values


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_data_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _load_metadata(dataset_root: Path) -> list[dict[str, object]]:
    csv_path = dataset_root / "images.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing NIPS2017 metadata: {csv_path}")
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "image_id": str(row["ImageId"]),
                    "true_label": int(row["TrueLabel"]) - 1,
                    "target_label": int(row["TargetClass"]) - 1,
                }
            )
    rows.sort(key=lambda item: _natural_key(str(item["image_id"])))
    return rows


def _resolve_attack_image(sample_dir: Path, metrics: dict[str, object]) -> Path:
    relative = str(metrics.get("final_selected_image_path") or "images/final_selected.png")
    path = Path(relative)
    if not path.is_absolute():
        path = sample_dir / path
    return path


def _load_run_rows(
    *,
    model_name: str,
    experiment_root: Path,
    run_name: str,
    dataset_root: Path,
    metadata: Sequence[dict[str, object]],
    limit: int | None,
) -> list[dict[str, object]]:
    run_root = experiment_root / model_name / run_name
    if not run_root.is_dir():
        raise FileNotFoundError(f"missing run root: {run_root}")

    max_count = len(metadata) if limit is None else min(len(metadata), int(limit))
    rows: list[dict[str, object]] = []
    for sample_index in range(max_count):
        meta = metadata[sample_index]
        image_id = str(meta["image_id"])
        source_path = dataset_root / "images" / f"{image_id}.png"
        sample_dir = run_root / f"sample_{sample_index:04d}"
        metrics_path = sample_dir / "metrics.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"missing clean image: {source_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing sample metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        attack_path = _resolve_attack_image(sample_dir, metrics)
        if not attack_path.is_file():
            raise FileNotFoundError(f"missing final attack image: {attack_path}")
        rows.append(
            {
                "model": model_name,
                "sample_index": sample_index,
                "image_id": image_id,
                "true_label": int(meta["true_label"]),
                "target_label": int(meta["target_label"]),
                "clean_path": _portable_path(source_path),
                "attack_path": _portable_path(attack_path),
                "stored_attack_success": metrics.get("final_attack_success") is True,
                "stored_status": str(metrics.get("status") or ""),
                "clean_pred": None,
                "adv_pred": None,
            }
        )
    return rows


def _load_victim_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from eval.attacked_models import model_selection

    model = model_selection(model_name)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_rgb_tensor(path: str | Path, size: int) -> torch.Tensor:
    with Image.open(_resolve_data_path(path)) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    tensor = TF.resize(
        tensor,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return tensor.clamp_(0.0, 1.0)


def _predict_paths(
    model: torch.nn.Module,
    paths: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> list[int]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    predictions: list[int] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([_load_rgb_tensor(path, 224) for path in batch_paths])
        batch = batch.to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.inference_mode():
            logits = model(batch)
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif isinstance(logits, dict):
            logits = logits.get("logits", next(iter(logits.values())))
        elif isinstance(logits, (tuple, list)):
            logits = logits[0]
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        done = min(start + batch_size, len(paths))
        print(f"[victim] predicted {done}/{len(paths)}", flush=True)
    return [int(value) for value in predictions]


def _run_selection(
    rows_by_model: dict[str, list[dict[str, object]]],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for model_name, rows in rows_by_model.items():
        print(f"[selection] loading victim={model_name}", flush=True)
        model = _load_victim_model(model_name, device)
        clean_predictions = _predict_paths(
            model,
            [str(row["clean_path"]) for row in rows],
            device=device,
            batch_size=batch_size,
        )
        adv_predictions = _predict_paths(
            model,
            [str(row["attack_path"]) for row in rows],
            device=device,
            batch_size=batch_size,
        )
        for row, clean_pred, adv_pred in zip(rows, clean_predictions, adv_predictions):
            true_label = int(row["true_label"])
            clean_correct = clean_pred == true_label
            quantized_attack_success = clean_correct and adv_pred != true_label
            selected = clean_correct and bool(row["stored_attack_success"])
            row.update(
                {
                    "clean_pred": clean_pred,
                    "adv_pred": adv_pred,
                    "clean_correct": clean_correct,
                    "quantized_attack_success": quantized_attack_success,
                    "selected_for_metrics": selected,
                    "stored_vs_quantized_success_match": (
                        bool(row["stored_attack_success"]) == (adv_pred != true_label)
                    ),
                }
            )
        all_rows.extend(rows)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


SELECTION_FIELDS = (
    "model",
    "sample_index",
    "image_id",
    "true_label",
    "target_label",
    "clean_pred",
    "adv_pred",
    "clean_correct",
    "stored_attack_success",
    "quantized_attack_success",
    "selected_for_metrics",
    "stored_vs_quantized_success_match",
    "stored_status",
    "clean_path",
    "attack_path",
)


def _read_selection(path: Path) -> list[dict[str, object]]:
    bool_fields = {
        "clean_correct",
        "stored_attack_success",
        "quantized_attack_success",
        "selected_for_metrics",
        "stored_vs_quantized_success_match",
    }
    int_fields = {"sample_index", "true_label", "target_label", "clean_pred", "adv_pred"}
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in bool_fields:
                row[field] = str(row[field]).strip().lower() == "true"
            for field in int_fields:
                row[field] = int(row[field])
            rows.append(row)
    return rows


def _ensure_idsim_backbone(checkpoint: Path, cache_dir: Path) -> Path:
    expected = cache_dir / "checkpoints" / DINOV3_CHECKPOINT_NAME
    expected.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists():
        if expected.stat().st_size != checkpoint.stat().st_size:
            raise ValueError(f"unexpected existing ID-Sim backbone size: {expected}")
        return expected
    try:
        os.link(checkpoint, expected)
    except OSError:
        import shutil

        shutil.copy2(checkpoint, expected)
    return expected


def _metric_transform(path: str | Path, size: int = 448) -> torch.Tensor:
    with Image.open(_resolve_data_path(path)) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
    return TF.resize(
        tensor,
        [size, size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).clamp_(0.0, 1.0)


def _batched_embeddings(
    paths: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    transform_fn: Callable[[str | Path], torch.Tensor],
    amp: bool,
    label: str,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    worker_count = min(8, max(1, int(batch_size)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(list(pool.map(transform_fn, batch_paths)))
            batch = batch.to(device, non_blocking=True)
            amp_enabled = bool(amp and device.type == "cuda")
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                embeddings = embed_fn(batch)
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            outputs.append(embeddings.cpu())
            done = min(start + batch_size, len(paths))
            print(f"[{label}] embedded {done}/{len(paths)}", flush=True)
    if not outputs:
        raise ValueError(f"no paths supplied for {label}")
    return torch.cat(outputs, dim=0)


def _extract_pair_metric(
    selected_rows: Sequence[dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    transform_fn: Callable[[str | Path], torch.Tensor] = _metric_transform,
    amp: bool,
    label: str,
) -> list[float]:
    unique_clean_paths = list(dict.fromkeys(str(row["clean_path"]) for row in selected_rows))
    clean_embeddings = _batched_embeddings(
        unique_clean_paths,
        batch_size=batch_size,
        device=device,
        embed_fn=embed_fn,
        transform_fn=transform_fn,
        amp=amp,
        label=f"{label}:clean",
    )
    clean_by_path = {
        path: clean_embeddings[index] for index, path in enumerate(unique_clean_paths)
    }
    adv_embeddings = _batched_embeddings(
        [str(row["attack_path"]) for row in selected_rows],
        batch_size=batch_size,
        device=device,
        embed_fn=embed_fn,
        transform_fn=transform_fn,
        amp=amp,
        label=f"{label}:adv",
    )
    similarities: list[float] = []
    for index, row in enumerate(selected_rows):
        clean_embedding = clean_by_path[str(row["clean_path"])]
        similarity = torch.dot(clean_embedding, adv_embeddings[index]).item()
        similarities.append(float(similarity))
    return similarities


def _load_dinov3_model(
    checkpoint: Path,
    dinov3_repo: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = torch.hub.load(
        str(dinov3_repo),
        "dinov3_vitl16",
        source="local",
        weights=str(checkpoint),
    )
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_idsim_model(
    *,
    cache_dir: Path,
    dinov3_repo: Path,
    device: torch.device,
):
    if str(REPO_ROOT / "third_party" / "id_sim") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "third_party" / "id_sim"))

    from id_sim import id_sim
    from id_sim.feature_extraction.extractor import ViTExtractor

    original_create_model = ViTExtractor.create_model

    def create_model_from_local_repo(
        model_type: str,
        load_dir: str | Path = "./models",
    ) -> torch.nn.Module:
        if "dinov3" not in model_type:
            return original_create_model(model_type, load_dir)
        checkpoint_path = ViTExtractor._resolve_dinov3_checkpoint(model_type, load_dir)
        return torch.hub.load(
            str(dinov3_repo),
            model_type,
            source="local",
            weights=checkpoint_path,
        )

    ViTExtractor.create_model = staticmethod(create_model_from_local_repo)
    model, preprocess = id_sim(
        pretrained=True,
        device=str(device),
        cache_dir=str(cache_dir),
        normalize_embeds=True,
        id_sim_type="dinov3_vitl16_cls_patch",
    )
    return model.eval(), preprocess


def _run_identity_metrics(
    selected_rows: list[dict[str, object]],
    *,
    checkpoint: Path,
    dinov3_repo: Path,
    idsim_cache: Path,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> None:
    print("[metric] loading DINOv3 ViT-L/16", flush=True)
    dino_model = _load_dinov3_model(checkpoint, dinov3_repo, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def dino_embed(batch: torch.Tensor) -> torch.Tensor:
        return dino_model((batch - mean) / std)

    dino_similarities = _extract_pair_metric(
        selected_rows,
        batch_size=batch_size,
        device=device,
        embed_fn=dino_embed,
        amp=amp,
        label="dinov3",
    )
    for row, similarity in zip(selected_rows, dino_similarities):
        row["dinov3_similarity"] = similarity
        row["dinov3_distance"] = 1.0 - similarity

    del dino_model
    del dino_embed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[metric] loading ID-Sim DINOv3 ViT-L/16 checkpoint", flush=True)
    idsim_model, idsim_preprocess = _load_idsim_model(
        cache_dir=idsim_cache,
        dinov3_repo=dinov3_repo,
        device=device,
    )

    def idsim_embed(batch: torch.Tensor) -> torch.Tensor:
        return idsim_model.embed(batch, mode="cls")["cls"]

    def idsim_transform(path: str | Path) -> torch.Tensor:
        with Image.open(_resolve_data_path(path)) as image:
            return idsim_preprocess(image).squeeze(0)

    idsim_similarities = _extract_pair_metric(
        selected_rows,
        batch_size=batch_size,
        device=device,
        embed_fn=idsim_embed,
        transform_fn=idsim_transform,
        amp=amp,
        label="idsim",
    )
    for row, similarity in zip(selected_rows, idsim_similarities):
        row["idsim_similarity"] = similarity
        row["idsim_distance"] = 1.0 - similarity

    del idsim_model
    del idsim_preprocess
    del idsim_embed
    del idsim_transform
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


METRIC_FIELDS = (
    "model",
    "sample_index",
    "image_id",
    "true_label",
    "target_label",
    "clean_pred",
    "adv_pred",
    "stored_attack_success",
    "quantized_attack_success",
    "dinov3_similarity",
    "dinov3_distance",
    "idsim_similarity",
    "idsim_distance",
    "clean_path",
    "attack_path",
)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _build_summary(
    selection_rows: Sequence[dict[str, object]],
    metric_rows: Sequence[dict[str, object]],
    *,
    model_names: Sequence[str],
    checkpoint: Path,
    checkpoint_sha256: str,
    amp: bool,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "selection_definition": (
            "clean_pred == true_label and stored final_attack_success == true"
        ),
        "quantized_success_definition": (
            "clean_pred == true_label and persisted PNG adv_pred != true_label"
        ),
        "reference_image": "data/nips2017/images/<ImageId>.png",
        "attack_image": "sample_<index>/images/final_selected.png",
        "dinov3": {
            "model": "dinov3_vitl16",
            "checkpoint": _portable_path(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "feature": "normalized final CLS token",
            "distance": "1 - cosine_similarity",
            "image_size": 448,
        },
        "idsim": {
            "model": "dinov3_vitl16_cls_patch",
            "reported_mode": "cls",
            "distance": "official ID-Sim CLS cosine distance (1 - cosine_similarity)",
            "image_size": 448,
            "preprocess": (
                "official PIL RGB -> bicubic Resize -> ToTensor; "
                "ImageNet normalization is applied inside ID-Sim"
            ),
        },
        "amp": bool(amp),
        "models": {},
    }
    models_summary: dict[str, object] = {}
    for model_name in model_names:
        selection = [row for row in selection_rows if row["model"] == model_name]
        metrics = [row for row in metric_rows if row["model"] == model_name]
        quantized_metrics = [
            row for row in metrics if bool(row["quantized_attack_success"])
        ]
        clean_correct_count = sum(bool(row["clean_correct"]) for row in selection)
        selected_count = len(metrics)
        retained_count = len(quantized_metrics)
        models_summary[model_name] = {
            "total_samples": len(selection),
            "clean_correct_count": clean_correct_count,
            "clean_accuracy": clean_correct_count / len(selection) if selection else None,
            "stored_attack_success_total_count": sum(
                bool(row["stored_attack_success"]) for row in selection
            ),
            "selected_for_metrics_count": selected_count,
            "stored_attack_success_rate_on_clean_correct": (
                selected_count / clean_correct_count if clean_correct_count else None
            ),
            "quantized_attack_success_on_clean_correct_count": sum(
                bool(row["quantized_attack_success"]) for row in selection
            ),
            "quantized_success_retained_within_selected_count": retained_count,
            "quantized_success_retention_rate_within_selected": (
                retained_count / selected_count if selected_count else None
            ),
            "stored_vs_quantized_success_mismatch_count": sum(
                not bool(row["stored_vs_quantized_success_match"]) for row in selection
            ),
            "primary_selected_metrics": {
                "dinov3_similarity": _distribution(
                    float(row["dinov3_similarity"]) for row in metrics
                ),
                "dinov3_distance": _distribution(
                    float(row["dinov3_distance"]) for row in metrics
                ),
                "idsim_similarity": _distribution(
                    float(row["idsim_similarity"]) for row in metrics
                ),
                "idsim_distance": _distribution(
                    float(row["idsim_distance"]) for row in metrics
                ),
            },
            "quantized_success_retained_metrics": {
                "dinov3_similarity": _distribution(
                    float(row["dinov3_similarity"]) for row in quantized_metrics
                ),
                "dinov3_distance": _distribution(
                    float(row["dinov3_distance"]) for row in quantized_metrics
                ),
                "idsim_similarity": _distribution(
                    float(row["idsim_similarity"]) for row in quantized_metrics
                ),
                "idsim_distance": _distribution(
                    float(row["idsim_distance"]) for row in quantized_metrics
                ),
            },
        }
    summary["models"] = models_summary
    return summary


def _format_value(value: object, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# NIPS2017 ablation_and_no_verifier: DINOv3 / ID-Sim",
        "",
        "## Evaluation scope",
        "",
        "- Primary selection: `clean_pred == true_label` and stored "
        "`final_attack_success == true`.",
        "- Reference: the original `data/nips2017/images/<ImageId>.png`.",
        "- Attack: each sample's persisted `images/final_selected.png`.",
        "- DINOv3: ViT-L/16 LVD-1689M final normalized CLS cosine similarity.",
        "- ID-Sim: official DINOv3 ViT-L/16 checkpoint, PIL preprocessing, "
        "and CLS mode.",
        "- Similarity is higher-is-better; distance is `1 - similarity` and "
        "lower-is-better.",
        "",
        "## Raw summary",
        "",
        "| Victim | Clean-correct | Selected | PNG-retained | DINOv3 sim (mean±std) | "
        "ID-Sim sim (mean±std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, raw_model_summary in dict(summary["models"]).items():
        model_summary = dict(raw_model_summary)
        primary = dict(model_summary["primary_selected_metrics"])
        dino = dict(primary["dinov3_similarity"])
        idsim = dict(primary["idsim_similarity"])
        lines.append(
            "| {model} | {clean} | {selected} | {retained} | {dino_mean}±{dino_std} | "
            "{idsim_mean}±{idsim_std} |".format(
                model=model_name,
                clean=model_summary["clean_correct_count"],
                selected=model_summary["selected_for_metrics_count"],
                retained=model_summary["quantized_success_retained_within_selected_count"],
                dino_mean=_format_value(dino["mean"]),
                dino_std=_format_value(dino["std"]),
                idsim_mean=_format_value(idsim["mean"]),
                idsim_std=_format_value(idsim["std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Verification notes",
            "",
            "- `PNG-retained` is the subset that remains adversarial after reloading "
            "the persisted 8-bit PNG and rerunning the victim model.",
            "- The JSON summary contains median, quartiles, minimum, maximum, and "
            "separate aggregates for the PNG-retained subset.",
            "- No score is imputed for missing or non-finite samples.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_metric_rows(rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("metric selection is empty")
    metric_names = (
        "dinov3_similarity",
        "dinov3_distance",
        "idsim_similarity",
        "idsim_distance",
    )
    for row in rows:
        for name in metric_names:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite {name} for {row['model']} sample {row['sample_index']}"
                )


def main() -> int:
    args = parse_args()
    started = time.time()
    model_names = _model_names(args.models)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    experiment_root = args.experiment_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = args.dinov3_checkpoint.expanduser().resolve()
    dinov3_repo = args.dinov3_repo.expanduser().resolve()
    idsim_cache = args.idsim_cache.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.csv"

    metadata = _load_metadata(dataset_root)
    if args.stage in {"selection", "all"}:
        rows_by_model = {
            model_name: _load_run_rows(
                model_name=model_name,
                experiment_root=experiment_root,
                run_name=args.run_name,
                dataset_root=dataset_root,
                metadata=metadata,
                limit=args.limit,
            )
            for model_name in model_names
        }
        selection_rows = _run_selection(
            rows_by_model,
            device=device,
            batch_size=max(1, int(args.victim_batch_size)),
        )
        _write_csv(selection_path, selection_rows, SELECTION_FIELDS)
        print(f"[output] wrote {selection_path}", flush=True)
    else:
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"selection stage output is required before --stage metrics: {selection_path}"
            )
        selection_rows = _read_selection(selection_path)
        selection_rows = [
            row for row in selection_rows if str(row["model"]) in set(model_names)
        ]

    if args.stage == "selection":
        for model_name in model_names:
            rows = [row for row in selection_rows if row["model"] == model_name]
            clean_count = sum(bool(row["clean_correct"]) for row in rows)
            selected_count = sum(bool(row["selected_for_metrics"]) for row in rows)
            print(
                f"[selection] {model_name}: clean_correct={clean_count}/{len(rows)} "
                f"selected={selected_count}",
                flush=True,
            )
        return 0

    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing DINOv3 checkpoint: {checkpoint}")
    if not dinov3_repo.is_dir():
        raise FileNotFoundError(f"missing DINOv3 source checkout: {dinov3_repo}")
    checkpoint_sha256 = _sha256(checkpoint)
    expected_prefix = checkpoint.stem.rsplit("-", 1)[-1].lower()
    if not checkpoint_sha256.startswith(expected_prefix):
        raise ValueError(
            f"DINOv3 checkpoint hash mismatch: sha256={checkpoint_sha256}, "
            f"filename prefix={expected_prefix}"
        )
    expected_backbone = _ensure_idsim_backbone(checkpoint, idsim_cache)
    print(f"[checkpoint] sha256={checkpoint_sha256}", flush=True)
    print(f"[checkpoint] ID-Sim backbone={expected_backbone}", flush=True)

    selected_rows = [
        dict(row) for row in selection_rows if bool(row["selected_for_metrics"])
    ]
    if not selected_rows:
        raise ValueError("no clean-correct stored-success samples selected")

    _run_identity_metrics(
        selected_rows,
        checkpoint=checkpoint,
        dinov3_repo=dinov3_repo,
        idsim_cache=idsim_cache,
        device=device,
        batch_size=max(1, int(args.metric_batch_size)),
        amp=bool(args.amp),
    )
    _validate_metric_rows(selected_rows)

    sample_metrics_path = output_dir / "sample_metrics.csv"
    _write_csv(sample_metrics_path, selected_rows, METRIC_FIELDS)
    summary = _build_summary(
        selection_rows,
        selected_rows,
        model_names=model_names,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        amp=bool(args.amp),
    )
    summary["elapsed_seconds"] = round(time.time() - started, 3)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = output_dir / "report.md"
    _write_report(report_path, summary)
    print(f"[output] wrote {sample_metrics_path}", flush=True)
    print(f"[output] wrote {summary_path}", flush=True)
    print(f"[output] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

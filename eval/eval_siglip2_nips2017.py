#!/usr/bin/env python3
"""Evaluate prompt-ensemble ImageNet victims on clean NIPS2017 images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_ROOT = REPO_ROOT / "isolated_vlm_attack"
for import_root in (REPO_ROOT, ISOLATED_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from attack_runner_common import (  # noqa: E402
    OPENAI_CLIP_IMAGENET_MODEL_ID,
    SIGLIP2_IMAGENET_MODEL_ID,
    VictimModelAdapter,
    load_nips_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "nips2017",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--model-id",
        choices=(SIGLIP2_IMAGENET_MODEL_ID, OPENAI_CLIP_IMAGENET_MODEL_ID),
        default=SIGLIP2_IMAGENET_MODEL_ID,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate all samples when set to 0.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_classifier_batch(
    victim: VictimModelAdapter,
    image_root: Path,
    image_ids: List[str],
) -> np.ndarray:
    prepared: List[np.ndarray] = []
    for image_id in image_ids:
        image_path = image_root / f"{image_id}.png"
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            array = np.array(image_rgb, dtype=np.uint8, copy=True)
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            / 255.0
        )
        prepared.append(victim._preprocess(tensor))
    return np.concatenate(prepared, axis=0)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")

    dataset_root = args.dataset_root.expanduser().resolve()
    image_root = dataset_root / "images"
    image_ids, true_labels, _ = load_nips_ground_truth(dataset_root)
    sample_count = len(image_ids)
    if args.max_samples:
        sample_count = min(sample_count, int(args.max_samples))
        image_ids = image_ids[:sample_count]
        true_labels = true_labels[:sample_count]
    missing = [
        image_id
        for image_id in image_ids
        if not (image_root / f"{image_id}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} NIPS2017 images are missing; first={missing[0]}"
        )

    torch.manual_seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    print(
        f"Loading {args.model_id} on {args.device}; "
        "building ImageNet prompt-ensemble prototypes...",
        flush=True,
    )
    victim = VictimModelAdapter(
        model_name=args.model_id,
        device=args.device,
        objective_mode="ce_max",
    )
    model_ready_seconds = time.perf_counter() - started
    print(f"Model ready in {model_ready_seconds:.1f}s", flush=True)

    correct_top1 = 0
    correct_top5 = 0
    pred_confidence_sum = 0.0
    true_confidence_sum = 0.0
    processed = 0
    for start in range(0, sample_count, args.batch_size):
        end = min(start + args.batch_size, sample_count)
        batch_ids = image_ids[start:end]
        labels = np.asarray(true_labels[start:end], dtype=np.int64)
        image_batch = load_classifier_batch(victim, image_root, batch_ids)
        logits = np.asarray(
            victim.f_model.predict(image_batch, batch_size=len(batch_ids)),
            dtype=np.float32,
        )
        if logits.shape != (len(batch_ids), 1000):
            raise ValueError(
                f"expected {(len(batch_ids), 1000)} logits, got {logits.shape}"
            )
        probabilities = stable_softmax(logits)
        predictions = np.argmax(logits, axis=1)
        top5 = np.argpartition(logits, kth=-5, axis=1)[:, -5:]

        correct_top1 += int(np.sum(predictions == labels))
        correct_top5 += int(np.sum(np.any(top5 == labels[:, None], axis=1)))
        pred_confidence_sum += float(
            np.sum(probabilities[np.arange(len(labels)), predictions])
        )
        true_confidence_sum += float(
            np.sum(probabilities[np.arange(len(labels)), labels])
        )
        processed = end
        print(
            f"[{processed:4d}/{sample_count}] "
            f"top1={100.0 * correct_top1 / processed:.2f}% "
            f"top5={100.0 * correct_top5 / processed:.2f}%",
            flush=True,
        )

    elapsed_seconds = time.perf_counter() - started
    result: Dict[str, object] = {
        "model_id": args.model_id,
        "dataset": "NIPS2017",
        "sample_count": sample_count,
        "correct_top1": correct_top1,
        "top1_accuracy": 100.0 * correct_top1 / sample_count,
        "correct_top5": correct_top5,
        "top5_accuracy": 100.0 * correct_top5 / sample_count,
        "mean_pred_confidence": pred_confidence_sum / sample_count,
        "mean_true_confidence": true_confidence_sum / sample_count,
        "confidence_definition": (
            "softmax over 1000 fixed-100x cosine class-prototype logits"
            if args.model_id == OPENAI_CLIP_IMAGENET_MODEL_ID
            else "softmax over 1000 scaled SigLIP2 class-prototype logits"
        ),
        "prompt_templates_per_class": (
            80 if args.model_id == OPENAI_CLIP_IMAGENET_MODEL_ID else 81
        ),
        "label_indexing": "images.csv TrueLabel minus one",
        "model_ready_seconds": model_ready_seconds,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else None
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}", flush=True)


if __name__ == "__main__":
    main()

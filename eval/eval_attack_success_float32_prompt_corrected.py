"""Evaluate untargeted ASR from per-sample float32 classifier-input sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ISOLATED_ROOT = _REPO_ROOT / "isolated_vlm_attack"
for _path in (_REPO_ROOT, _ISOLATED_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from attack_runner_common import SAVED_IMAGE_SIZE, VictimModelAdapter


def load_category_phrases(path: Path) -> Dict[int, List[str]]:
    """Load the same prompt-label aliases used by eval_asr_from_npz.py."""

    excluded_phrases = {"light"}
    category_phrases: Dict[int, List[str]] = {}
    with path.open(encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            try:
                label_idx = int(row["CategoryId"]) - 1
            except Exception:
                continue
            raw_name = str(row.get("CategoryName", "")).strip()
            phrases = [
                phrase.strip().lower()
                for phrase in raw_name.split(",")
                if phrase.strip()
                and phrase.strip().lower() not in excluded_phrases
            ]
            if label_idx >= 0 and phrases:
                category_phrases[label_idx] = phrases
    return category_phrases


def phrase_in_prompt(phrase: str, prompt: str) -> bool:
    """Match a category alias on the same token boundaries as eval_all."""

    pattern = r"(?<![a-z0-9]){}(?![a-z0-9])".format(
        re.escape(phrase.lower())
    )
    return re.search(pattern, prompt.lower()) is not None


def build_prompt_correct_label_set(
    prompt: str,
    category_phrases: Dict[int, List[str]],
) -> set[int]:
    return {
        label_idx
        for label_idx, phrases in category_phrases.items()
        if any(phrase_in_prompt(phrase, prompt) for phrase in phrases)
    }


def load_npz_source_prompts(run_root: Path) -> Dict[int, str]:
    """Map sample indices to the source_prompt values already used by eval_all."""

    npz_path = run_root / "adversarial_examples.npz"
    if not npz_path.is_file():
        return {}
    with np.load(npz_path, allow_pickle=False) as npz_data:
        if "source_prompt" not in npz_data.files:
            return {}
        prompts = np.asarray(npz_data["source_prompt"]).astype(str).reshape(-1)
        sample_names = (
            np.asarray(npz_data["sample_names"]).astype(str).reshape(-1)
            if "sample_names" in npz_data.files
            else None
        )
    if sample_names is not None and sample_names.size != prompts.size:
        raise ValueError(
            f"{npz_path}: source_prompt has {prompts.size} values but "
            f"sample_names has {sample_names.size}"
        )

    prompt_by_index: Dict[int, str] = {}
    for position, prompt in enumerate(prompts):
        sample_index = position
        if sample_names is not None:
            match = re.fullmatch(r"sample_(\d+)", str(sample_names[position]))
            if match is not None:
                sample_index = int(match.group(1))
        if sample_index in prompt_by_index:
            raise ValueError(
                f"{npz_path}: duplicate source prompt for sample {sample_index}"
            )
        prompt_by_index[sample_index] = str(prompt)
    return prompt_by_index


def resolve_source_prompt(
    *,
    sample_index: int,
    sample_dir: Path,
    npz_source_prompts: Dict[int, str],
) -> Optional[str]:
    """Resolve the final attack prompt, preferring eval_all's stored NPZ value."""

    if sample_index in npz_source_prompts:
        return npz_source_prompts[sample_index]
    report_path = sample_dir / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for value in (
        report.get("optimized_prompt"),
        (report.get("early_stop") or {}).get("candidate_prompt")
        if isinstance(report.get("early_stop"), dict)
        else None,
        (report.get("attack_success_candidate") or {}).get("candidate_prompt")
        if isinstance(report.get("attack_success_candidate"), dict)
        else None,
    ):
        prompt = str(value or "").strip()
        if prompt:
            return prompt
    return None


def load_classifier_input(path: Path) -> np.ndarray:
    """Load an exact CHW float32 classifier input without clipping or resizing."""

    array = np.load(path, allow_pickle=False)
    expected_shape = (3, SAVED_IMAGE_SIZE, SAVED_IMAGE_SIZE)
    if array.dtype != np.float32:
        raise ValueError(f"{path}: expected dtype float32, got {array.dtype}")
    if array.shape != expected_shape:
        raise ValueError(f"{path}: expected shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: classifier input contains non-finite values")
    return array


def load_clean_input(path: Path) -> np.ndarray:
    """Load a clean image as an unbatched CHW float32 array in [0, 1]."""

    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (SAVED_IMAGE_SIZE, SAVED_IMAGE_SIZE),
            Image.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32)
    return array.transpose(2, 0, 1) / np.float32(255.0)


def predict_label(victim: VictimModelAdapter, classifier_input: np.ndarray) -> int:
    """Predict from an already prepared classifier input without preprocessing it."""

    scores = np.asarray(
        victim.f_model.predict(classifier_input[None, ...], batch_size=1),
        dtype=np.float32,
    )
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(f"unexpected classifier output shape: {scores.shape}")
    raw_prediction = int(np.argmax(scores[0]))
    return victim._normalize_prediction_index(raw_prediction)


def compute_untarget_asr(samples: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute ASR only over samples the victim classifies correctly when clean."""

    clean_correct = [
        sample
        for sample in samples
        if int(sample["clean_pred_idx"]) == int(sample["true_label"])
    ]
    attack_success = [
        sample
        for sample in clean_correct
        if (
            bool(sample["attack_success"])
            if "attack_success" in sample
            else (
                sample.get("adv_pred_idx") is not None
                and int(sample["adv_pred_idx"]) != int(sample["true_label"])
            )
        )
    ]
    attack_success_query_counts = [
        int(sample["attack_success_query_count"])
        for sample in attack_success
        if sample.get("attack_success_query_count") is not None
    ]
    denominator = len(clean_correct)
    numerator = len(attack_success)
    return {
        "untarget_asr": (
            100.0 * float(numerator) / float(denominator)
            if denominator > 0
            else None
        ),
        "untarget_asr_numerator": numerator,
        "untarget_asr_denominator": denominator,
        "attack_success_query_count_mean": (
            float(sum(attack_success_query_counts))
            / float(len(attack_success_query_counts))
            if attack_success_query_counts
            else None
        ),
        "attack_success_query_count_sample_count": len(
            attack_success_query_counts
        ),
    }


def resolve_clean_image(images_dir: Path, image_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".JPEG"):
        path = images_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"clean image not found for {image_id} under {images_dir}")


def resolve_sample_dir(run_root: Path, result: Dict[str, object]) -> Path:
    sample_index = int(result["sample_index"])
    local_path = run_root / f"sample_{sample_index:04d}"
    if local_path.is_dir():
        return local_path
    recorded_path = str(result.get("sample_dir", "") or "").strip()
    if recorded_path and Path(recorded_path).is_dir():
        return Path(recorded_path)
    raise FileNotFoundError(f"sample directory not found for index {sample_index}")


def resolve_sidecar_path(
    sample_dir: Path,
    result: Dict[str, object],
) -> Optional[Path]:
    relative_path = str(result.get("attack_success_float32_path", "") or "").strip()
    if not relative_path:
        report_path = sample_dir / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            relative_path = str(
                report.get("attack_success_float32_path", "") or ""
            ).strip()

    if relative_path:
        path = Path(relative_path)
        path = path if path.is_absolute() else sample_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"recorded float32 sidecar is missing: {path}")
        return path

    default_path = sample_dir / "images" / "attack_success.float32.npy"
    return default_path if default_path.is_file() else None


def evaluate_run(
    *,
    run_root: Path,
    dataset_root: Optional[Path] = None,
    device: str = "auto",
    prompt_label_correct: bool = True,
) -> Dict[str, object]:
    run_root = run_root.expanduser().resolve()
    summary_path = run_root / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run summary not found: {summary_path}")
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    model_name = str(run_summary.get("victim_model", "") or "").strip()
    if not model_name:
        raise ValueError("run_summary.json does not contain victim_model")
    if dataset_root is None:
        raw_dataset_root = str(run_summary.get("dataset_root", "") or "").strip()
        if not raw_dataset_root:
            raise ValueError(
                "run_summary.json does not contain dataset_root; pass --dataset-root"
            )
        dataset_root = Path(raw_dataset_root)
    dataset_root = dataset_root.expanduser().resolve()
    images_dir = dataset_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"clean image directory not found: {images_dir}")
    category_phrases: Dict[int, List[str]] = {}
    npz_source_prompts: Dict[int, str] = {}
    if prompt_label_correct:
        categories_path = dataset_root / "categories.csv"
        if not categories_path.is_file():
            raise FileNotFoundError(
                f"prompt-label correctness requires categories.csv: {categories_path}"
            )
        category_phrases = load_category_phrases(categories_path)
        npz_source_prompts = load_npz_source_prompts(run_root)

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    victim = VictimModelAdapter(
        model_name=model_name,
        device=resolved_device,
        objective_mode="ce_max",
    )

    raw_results = run_summary.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("run_summary.json contains no sample results")

    samples: List[Dict[str, object]] = []
    seen_indices = set()
    for result in sorted(raw_results, key=lambda item: int(item["sample_index"])):
        sample_index = int(result["sample_index"])
        if sample_index in seen_indices:
            raise ValueError(f"duplicate sample_index in run summary: {sample_index}")
        seen_indices.add(sample_index)

        image_id = str(result["image_id"])
        true_label = int(result["true_label"])
        clean_path = resolve_clean_image(images_dir, image_id)
        clean_pred_idx = predict_label(victim, load_clean_input(clean_path))

        sample_dir = resolve_sample_dir(run_root, result)
        sidecar_path = resolve_sidecar_path(sample_dir, result)
        source_prompt = (
            resolve_source_prompt(
                sample_index=sample_index,
                sample_dir=sample_dir,
                npz_source_prompts=npz_source_prompts,
            )
            if prompt_label_correct
            else None
        )
        prompt_correct_labels = (
            build_prompt_correct_label_set(source_prompt, category_phrases)
            if source_prompt is not None
            else set()
        )
        adv_pred_idx = None
        if sidecar_path is not None:
            adv_pred_idx = predict_label(
                victim,
                load_classifier_input(sidecar_path),
            )

        clean_correct = clean_pred_idx == true_label
        attack_success_without_prompt_correction = (
            clean_correct
            and adv_pred_idx is not None
            and adv_pred_idx != true_label
        )
        prompt_label_match = (
            adv_pred_idx is not None and adv_pred_idx in prompt_correct_labels
        )
        attack_success = (
            attack_success_without_prompt_correction
            and not prompt_label_match
        )
        raw_attack_success_query_count = result.get(
            "attack_success_query_count"
        )
        attack_success_query_count = (
            int(raw_attack_success_query_count)
            if raw_attack_success_query_count is not None
            else None
        )
        samples.append(
            {
                "sample_index": sample_index,
                "image_id": image_id,
                "true_label": true_label,
                "clean_pred_idx": clean_pred_idx,
                "clean_correct": clean_correct,
                "float32_path": (
                    str(sidecar_path.relative_to(run_root))
                    if sidecar_path is not None
                    else None
                ),
                "adv_pred_idx": adv_pred_idx,
                "source_prompt": source_prompt,
                "prompt_correct_labels": sorted(prompt_correct_labels),
                "prompt_label_match": prompt_label_match,
                "attack_success_without_prompt_correction": (
                    attack_success_without_prompt_correction
                ),
                "attack_success": attack_success,
                "attack_success_query_count": attack_success_query_count,
            }
        )

    asr = compute_untarget_asr(samples)
    clean_correct_count = int(asr["untarget_asr_denominator"])
    attack_success_without_prompt_correction_count = sum(
        bool(sample["attack_success_without_prompt_correction"])
        for sample in samples
    )
    attack_success_excluded_by_prompt_correction_count = sum(
        bool(sample["attack_success_without_prompt_correction"])
        and bool(sample["prompt_label_match"])
        for sample in samples
    )
    return {
        "run_root": str(run_root),
        "dataset_root": str(dataset_root),
        "victim_model": model_name,
        "device": resolved_device,
        "sample_count": len(samples),
        "clean_correct_count": clean_correct_count,
        "clean_accuracy": 100.0 * clean_correct_count / len(samples),
        "float32_sidecar_count": sum(
            sample["float32_path"] is not None for sample in samples
        ),
        "prompt_label_correct_enabled": prompt_label_correct,
        "source_prompt_count": sum(
            sample["source_prompt"] is not None for sample in samples
        ),
        "prompt_category_match_count": sum(
            bool(sample["prompt_correct_labels"]) for sample in samples
        ),
        "attack_success_without_prompt_correction_count": (
            attack_success_without_prompt_correction_count
        ),
        "attack_success_excluded_by_prompt_correction_count": (
            attack_success_excluded_by_prompt_correction_count
        ),
        **asr,
        "untarget_asr_denominator_definition": "clean_pred_idx == true_label",
        "untarget_asr_numerator_definition": (
            "clean_pred_idx == true_label and float32 sidecar exists "
            "and adv_pred_idx != true_label and adv_pred_idx is not in "
            "the source-prompt-derived correct-label set"
        ),
        "attack_success_query_count_mean_definition": (
            "mean attack_success_query_count over float32-verified "
            "attack-success samples with a recorded count"
        ),
        "samples": samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="ASA run directory containing run_summary.json and sample_* directories.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional override for the dataset_root recorded in run_summary.json.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--prompt-label-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat predictions named in source_prompt as correct, matching "
            "eval_all.py (enabled by default)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to RUN_ROOT/eval_attack_success_float32.json.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = evaluate_run(
        run_root=args.run_root,
        dataset_root=args.dataset_root,
        device=args.device,
        prompt_label_correct=args.prompt_label_correct,
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.run_root.expanduser().resolve()
        / "eval_attack_success_float32.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

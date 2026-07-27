from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "analysis"
    / "nips2017_ablation_and_no_verifier_dinov3_idsim_20260727"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "prompt_corrected"
DEFAULT_CATEGORIES = REPO_ROOT / "data" / "nips2017" / "categories.csv"
MODEL_NAMES = ("swin", "vim-small")
METRIC_NAMES = (
    "dinov3_similarity",
    "dinov3_distance",
    "idsim_similarity",
    "idsim_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply ASA prompt-label correction to existing NIPS2017 "
            "DINOv3/ID-Sim sample scores."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    return parser.parse_args()


def load_category_phrases(path: Path) -> dict[int, list[str]]:
    """Match eval/eval_attack_success_float32.py exactly."""

    excluded_phrases = {"light"}
    category_phrases: dict[int, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
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
    pattern = r"(?<![a-z0-9]){}(?![a-z0-9])".format(
        re.escape(phrase.lower())
    )
    return re.search(pattern, prompt.lower()) is not None


def build_prompt_correct_label_set(
    prompt: str,
    category_phrases: dict[int, list[str]],
) -> set[int]:
    return {
        label_idx
        for label_idx, phrases in category_phrases.items()
        if any(phrase_in_prompt(phrase, prompt) for phrase in phrases)
    }


def resolve_source_prompt(sample_dir: Path) -> str | None:
    """Match the JSON fallback order in eval_attack_success_float32.py."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
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


def validate_metric_rows(rows: Sequence[dict[str, object]]) -> None:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["model"]), int(row["sample_index"]))
        if key in keys:
            raise ValueError(f"duplicate metric key: {key}")
        keys.add(key)
        for name in METRIC_NAMES:
            if not math.isfinite(float(row[name])):
                raise ValueError(f"non-finite {name}: {key}")


def apply_prompt_correction(
    metric_rows: Sequence[dict[str, str]],
    category_phrases: dict[int, list[str]],
) -> list[dict[str, object]]:
    audited: list[dict[str, object]] = []
    for raw in metric_rows:
        row: dict[str, object] = dict(raw)
        attack_path = Path(str(row["attack_path"]))
        if not attack_path.is_absolute():
            attack_path = REPO_ROOT / attack_path
        sample_dir = attack_path.parent.parent
        source_prompt = resolve_source_prompt(sample_dir)
        prompt_correct_labels = (
            build_prompt_correct_label_set(source_prompt, category_phrases)
            if source_prompt is not None
            else set()
        )
        adv_pred = int(row["adv_pred"])
        prompt_label_match = adv_pred in prompt_correct_labels
        label_names = [
            category_phrases[label][0]
            for label in sorted(prompt_correct_labels)
            if label in category_phrases
        ]
        row.update(
            {
                "source_prompt": source_prompt or "",
                "prompt_correct_labels": json.dumps(
                    sorted(prompt_correct_labels),
                    separators=(",", ":"),
                ),
                "prompt_correct_label_names": json.dumps(
                    label_names,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "prompt_label_match": prompt_label_match,
                "prompt_corrected_success": not prompt_label_match,
            }
        )
        audited.append(row)
    validate_metric_rows(audited)
    return audited


def build_summary(
    audit_rows: Sequence[dict[str, object]],
    selection_rows: Sequence[dict[str, str]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "scope": (
            "Prompt-label correction applied to the existing clean-correct, "
            "stored-success identity-metric set."
        ),
        "prompt_source": (
            "sample report.json optimized_prompt, then early_stop candidate_prompt, "
            "then attack_success_candidate candidate_prompt"
        ),
        "prompt_label_rule": (
            "Predicted labels named in source_prompt are treated as correct, using "
            "categories.csv aliases and eval_attack_success_float32 token boundaries."
        ),
        "prediction_source": (
            "adv_pred from persisted final_selected.png; float32 sidecars are absent "
            "in these copied run directories"
        ),
        "models": {},
    }
    models: dict[str, object] = {}
    for model_name in MODEL_NAMES:
        original = [row for row in audit_rows if row["model"] == model_name]
        corrected = [
            row for row in original if bool(row["prompt_corrected_success"])
        ]
        selection = [
            row for row in selection_rows if row["model"] == model_name
        ]
        clean_correct_count = sum(
            str(row["clean_correct"]).strip().lower() == "true"
            for row in selection
        )
        uncorrected_rate = (
            len(original) / clean_correct_count if clean_correct_count else None
        )
        corrected_rate = (
            len(corrected) / clean_correct_count if clean_correct_count else None
        )
        original_metrics = {
            name: distribution(float(row[name]) for row in original)
            for name in METRIC_NAMES
        }
        corrected_metrics = {
            name: distribution(float(row[name]) for row in corrected)
            for name in METRIC_NAMES
        }
        models[model_name] = {
            "total_samples": len(selection),
            "clean_correct_count": clean_correct_count,
            "uncorrected_selected_count": len(original),
            "source_prompt_count": sum(bool(row["source_prompt"]) for row in original),
            "prompt_category_match_count": sum(
                str(row["prompt_correct_labels"]) != "[]" for row in original
            ),
            "excluded_by_prompt_correction_count": len(original) - len(corrected),
            "prompt_corrected_selected_count": len(corrected),
            "prompt_corrected_retention_rate": (
                len(corrected) / len(original) if original else None
            ),
            "uncorrected_success_rate_on_clean_correct": uncorrected_rate,
            "prompt_corrected_success_rate_on_clean_correct": corrected_rate,
            "prompt_corrected_asr_delta": (
                corrected_rate - uncorrected_rate
                if corrected_rate is not None and uncorrected_rate is not None
                else None
            ),
            "uncorrected_metrics": original_metrics,
            "prompt_corrected_metrics": corrected_metrics,
            "corrected_minus_uncorrected_mean": {
                name: (
                    corrected_metrics[name]["mean"]
                    - original_metrics[name]["mean"]
                )
                for name in METRIC_NAMES
            },
        }
    summary["models"] = models
    return summary


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# NIPS2017 prompt-corrected DINOv3 / ID-Sim",
        "",
        "Prompt-label correction treats a prediction named in the optimized attack "
        "prompt as correct and excludes it from attack successes.",
        "",
        "| Victim | Before | Excluded | After | ASR before | Corrected ASR | Δ ASR (pp) | "
        "DINOv3 sim (mean±std) | ID-Sim sim (mean±std) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        model = dict(dict(summary["models"])[model_name])
        dino = dict(dict(model["prompt_corrected_metrics"])["dinov3_similarity"])
        idsim = dict(dict(model["prompt_corrected_metrics"])["idsim_similarity"])
        lines.append(
            "| {model} | {before} | {excluded} | {after} | {before_asr:.2%} | "
            "{asr:.2%} | {asr_delta_pp:+.2f} pp | "
            "{dino_mean:.6f}±{dino_std:.6f} | "
            "{idsim_mean:.6f}±{idsim_std:.6f} |".format(
                model=model_name,
                before=model["uncorrected_selected_count"],
                excluded=model["excluded_by_prompt_correction_count"],
                after=model["prompt_corrected_selected_count"],
                before_asr=model["uncorrected_success_rate_on_clean_correct"],
                asr=model["prompt_corrected_success_rate_on_clean_correct"],
                asr_delta_pp=100.0 * model["prompt_corrected_asr_delta"],
                dino_mean=dino["mean"],
                dino_std=dino["std"],
                idsim_mean=idsim["mean"],
                idsim_std=idsim["std"],
            )
        )
    lines.extend(
        [
            "",
            "The raw audit CSV records the source prompt, matched category labels, "
            "and exclusion decision for every previously selected sample.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    categories_path = args.categories.resolve()
    metric_rows = read_csv(input_dir / "sample_metrics.csv")
    selection_rows = read_csv(input_dir / "selection.csv")
    category_phrases = load_category_phrases(categories_path)
    audit_rows = apply_prompt_correction(metric_rows, category_phrases)
    corrected_rows = [
        row for row in audit_rows if bool(row["prompt_corrected_success"])
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_fields = list(audit_rows[0].keys())
    write_csv(output_dir / "prompt_correction_audit.csv", audit_rows, audit_fields)
    write_csv(
        output_dir / "prompt_corrected_sample_metrics.csv",
        corrected_rows,
        audit_fields,
    )
    summary = build_summary(audit_rows, selection_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary)
    print(f"[output] audit_rows={len(audit_rows)} corrected_rows={len(corrected_rows)}")
    for model_name in MODEL_NAMES:
        model = dict(dict(summary["models"])[model_name])
        print(
            f"[{model_name}] before={model['uncorrected_selected_count']} "
            f"excluded={model['excluded_by_prompt_correction_count']} "
            f"after={model['prompt_corrected_selected_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

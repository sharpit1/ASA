#!/usr/bin/env python3
"""Independently verify Gaussian-noise vulnerability evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


COUNT_KEYS = (
    "total_count",
    "clean_noisy_correct_count",
    "mist_noisy_correct_count",
    "clean_correct_base_count",
    "clean_noise_failure_on_clean_correct_count",
    "mist_noise_failure_on_clean_correct_count",
    "asa_failed_count",
    "clean_noise_failure_on_asa_failed_count",
    "mist_noise_failure_on_asa_failed_count",
    "paired_both_correct_count",
    "paired_both_failed_count",
    "paired_mist_only_failure_count",
    "paired_clean_only_failure_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--sigmas",
        default="0.005,0.01,0.015,0.02,0.025",
        help="Comma-separated expected sigma values.",
    )
    parser.add_argument("--expected-models", type=int, default=12)
    parser.add_argument("--expected-samples", type=int, default=10_000)
    parser.add_argument("--expected-labels", type=int, default=1_000)
    parser.add_argument("--samples-per-label", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def new_sigma_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def sample_key(record: dict) -> tuple[str, str, int, int]:
    return (
        record["image_path"],
        record["edited_cache_path"],
        int(record["edited_cache_index"]),
        int(record["ground_label"]),
    )


def digest_keys(keys: list[tuple[str, str, int, int]]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def recompute_model(
    model_dir: Path, expected_sigmas: list[float]
) -> tuple[dict, list[tuple[str, str, int, int]]]:
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    per_class = json.loads(
        (model_dir / "per_class_results.json").read_text(encoding="utf-8")
    )
    counts_by_sigma = {sigma: new_sigma_counts() for sigma in expected_sigmas}
    baseline = Counter()
    label_counts: Counter[int] = Counter()
    keys: list[tuple[str, str, int, int]] = []

    with (model_dir / "per_sample_results.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            keys.append(sample_key(record))
            label = int(record["ground_label"])
            label_counts[label] += 1

            clean_correct = bool(record["clean_correct"])
            mist_correct = bool(record["mist_correct"])
            primary = clean_correct and mist_correct
            baseline["total_count"] += 1
            baseline["clean_correct_count"] += clean_correct
            baseline["mist_correct_count"] += mist_correct
            baseline["attack_success_count"] += clean_correct and not mist_correct
            baseline["asa_failed_count"] += primary

            observed_sigmas = [float(item["sigma"]) for item in record["sigma_results"]]
            if observed_sigmas != expected_sigmas:
                raise AssertionError(
                    f"{model_dir.name}:{line_number}: unexpected sigmas {observed_sigmas}"
                )
            for item in record["sigma_results"]:
                sigma = float(item["sigma"])
                counts = counts_by_sigma[sigma]
                clean_noisy_correct = bool(item["clean_noisy_correct"])
                mist_noisy_correct = bool(item["mist_noisy_correct"])
                clean_failed = not clean_noisy_correct
                mist_failed = not mist_noisy_correct

                counts["total_count"] += 1
                counts["clean_noisy_correct_count"] += clean_noisy_correct
                counts["mist_noisy_correct_count"] += mist_noisy_correct
                counts["clean_correct_base_count"] += clean_correct
                counts["clean_noise_failure_on_clean_correct_count"] += (
                    clean_correct and clean_failed
                )
                counts["mist_noise_failure_on_clean_correct_count"] += (
                    clean_correct and mist_failed
                )
                counts["asa_failed_count"] += primary
                counts["clean_noise_failure_on_asa_failed_count"] += (
                    primary and clean_failed
                )
                counts["mist_noise_failure_on_asa_failed_count"] += (
                    primary and mist_failed
                )
                counts["paired_both_correct_count"] += (
                    primary and not clean_failed and not mist_failed
                )
                counts["paired_both_failed_count"] += (
                    primary and clean_failed and mist_failed
                )
                counts["paired_mist_only_failure_count"] += (
                    primary and not clean_failed and mist_failed
                )
                counts["paired_clean_only_failure_count"] += (
                    primary and clean_failed and not mist_failed
                )

    for key in (
        "total_count",
        "clean_correct_count",
        "mist_correct_count",
        "attack_success_count",
        "asa_failed_count",
    ):
        if int(summary["baseline"][key]) != baseline[key]:
            raise AssertionError(
                f"{model_dir.name}: baseline {key}: "
                f"{summary['baseline'][key]} != {baseline[key]}"
            )

    summary_sigmas = [float(item["sigma"]) for item in summary["sigma_results"]]
    if summary_sigmas != expected_sigmas:
        raise AssertionError(
            f"{model_dir.name}: summary sigmas {summary_sigmas} != {expected_sigmas}"
        )
    for item in summary["sigma_results"]:
        sigma = float(item["sigma"])
        for key in COUNT_KEYS:
            if int(item[key]) != counts_by_sigma[sigma][key]:
                raise AssertionError(
                    f"{model_dir.name}: sigma={sigma} {key}: "
                    f"{item[key]} != {counts_by_sigma[sigma][key]}"
                )

    return (
        {
            "model_name": model_dir.name,
            "sample_count": len(keys),
            "label_count": len(label_counts),
            "samples_per_label_min": min(label_counts.values()),
            "samples_per_label_max": max(label_counts.values()),
            "per_class_count": len(per_class),
            "sample_order_sha256": digest_keys(keys),
            "baseline": dict(baseline),
        },
        keys,
    )


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    expected_sigmas = [float(value) for value in args.sigmas.split(",")]
    completion = json.loads(
        (result_dir / "completion.json").read_text(encoding="utf-8")
    )
    if completion["completed_model_count"] != args.expected_models:
        raise AssertionError("unexpected completed_model_count")
    if completion["failure_count"] != 0 or completion["failures"]:
        raise AssertionError("completion.json contains failures")
    if completion["test_count"] != args.expected_models * len(expected_sigmas):
        raise AssertionError("unexpected test_count")

    with (result_dir / "model_sigma_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_keys = [(row["model_name"], float(row["sigma"])) for row in csv_rows]
    if len(csv_rows) != args.expected_models * len(expected_sigmas):
        raise AssertionError("unexpected CSV row count")
    if len(set(csv_keys)) != len(csv_keys):
        raise AssertionError("duplicate model/sigma rows in CSV")

    aggregate = json.loads(
        (result_dir / "aggregate_analysis.json").read_text(encoding="utf-8")
    )
    if aggregate["model_count"] != args.expected_models:
        raise AssertionError("unexpected aggregate model_count")
    if aggregate["test_count"] != args.expected_models * len(expected_sigmas):
        raise AssertionError("unexpected aggregate test_count")

    model_dirs = sorted(
        path.parent
        for path in result_dir.glob("*/summary.json")
        if (path.parent / "per_sample_results.jsonl").is_file()
    )
    if len(model_dirs) != args.expected_models:
        raise AssertionError(f"found {len(model_dirs)} model directories")

    model_reports = []
    reference_keys: list[tuple[str, str, int, int]] | None = None
    for model_dir in model_dirs:
        model_report, keys = recompute_model(model_dir, expected_sigmas)
        if model_report["sample_count"] != args.expected_samples:
            raise AssertionError(f"{model_dir.name}: unexpected sample count")
        if model_report["label_count"] != args.expected_labels:
            raise AssertionError(f"{model_dir.name}: unexpected label count")
        if (
            model_report["samples_per_label_min"] != args.samples_per_label
            or model_report["samples_per_label_max"] != args.samples_per_label
        ):
            raise AssertionError(f"{model_dir.name}: unexpected samples per label")
        if model_report["per_class_count"] != args.expected_labels:
            raise AssertionError(f"{model_dir.name}: unexpected per-class row count")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise AssertionError(f"{model_dir.name}: sample order mismatch")
        model_reports.append(model_report)

    report = {
        "status": "passed",
        "result_dir": str(result_dir),
        "expected_sigmas": expected_sigmas,
        "model_count": len(model_reports),
        "sample_records_recomputed": sum(
            item["sample_count"] for item in model_reports
        ),
        "model_sigma_tests": len(csv_rows),
        "sample_order_consistent_across_models": True,
        "model_reports": model_reports,
    }
    output = args.output or result_dir / "verification_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

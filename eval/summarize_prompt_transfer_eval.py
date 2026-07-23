from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MATPLOTLIB_ERROR: Optional[str] = None
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - depends on local env
    matplotlib = None
    plt = None
    MATPLOTLIB_ERROR = f"{type(exc).__name__}: {exc}"


LABEL_LINE_RE = re.compile(
    r"^label (?P<label>\d+)(?: \((?P<name>.*?)\))? \| "
    r"clean acc: (?P<clean>-?\d+(?:\.\d+)?) \| "
    r"adv acc: (?P<adv>-?\d+(?:\.\d+)?) \| "
    r"asr\(clean-correct only\): (?P<asr>-?\d+(?:\.\d+)?|n/a) \| "
    r"artifact: (?P<artifact>\S+)\s*$"
)


@dataclass
class InputBundle:
    input_path: Path
    run_dir: Path
    log_path: Optional[Path]
    per_class_path: Optional[Path]
    summary_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge prompt-transfer evaluation logs/runs and export CSV, summary JSON, "
            "plots, and a Markdown report."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more eval directories, log.txt files, or per_class_results.json files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to prompt_transfer_summary_<timestamp> in the cwd.",
    )
    parser.add_argument(
        "--categories_csv",
        type=str,
        default=None,
        help="Optional categories.csv used to fill missing class names.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Prompt Transfer Summary",
        help="Title prefix used in plots and report.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top/bottom labels to export in the report tables.",
    )
    parser.add_argument(
        "--high_clean_threshold",
        type=float,
        default=70.0,
        help="Clean-accuracy threshold for the high-confidence top-ASR table.",
    )
    return parser.parse_args()


def resolve_input_bundle(raw_path: str) -> InputBundle:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")

    if path.is_dir():
        run_dir = path
        return InputBundle(
            input_path=path,
            run_dir=run_dir,
            log_path=(run_dir / "log.txt") if (run_dir / "log.txt").is_file() else None,
            per_class_path=(
                (run_dir / "per_class_results.json")
                if (run_dir / "per_class_results.json").is_file()
                else None
            ),
            summary_path=(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else None,
        )

    run_dir = path.parent
    log_path = run_dir / "log.txt"
    per_class_path = run_dir / "per_class_results.json"
    summary_path = run_dir / "summary.json"
    return InputBundle(
        input_path=path,
        run_dir=run_dir,
        log_path=log_path if log_path.is_file() else (path if path.name == "log.txt" else None),
        per_class_path=(
            path if path.name == "per_class_results.json" else (per_class_path if per_class_path.is_file() else None)
        ),
        summary_path=(path if path.name == "summary.json" else (summary_path if summary_path.is_file() else None)),
    )


def default_categories_csv() -> Optional[Path]:
    candidate = _REPO_ROOT / "data" / "nips2017" / "categories.csv"
    return candidate if candidate.is_file() else None


def load_category_names(path: Optional[Path]) -> Dict[int, str]:
    if path is None or not path.is_file():
        return {}

    category_names: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                category_id = int(row["CategoryId"])
            except Exception:
                continue
            label = category_id - 1
            if label < 0:
                continue
            raw_name = str(row.get("CategoryName", "")).strip()
            if not raw_name:
                continue
            category_names[label] = raw_name.split(",")[0].strip() or raw_name
    return category_names


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def try_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def try_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "n/a":
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "n/a"
    return f"{float(value):.{digits}f}"


def parse_log_metadata(log_path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "log_path": str(log_path),
        "selected_label_count_total": None,
        "selected_label_count_evaluated_this_run": None,
        "start_iteration": None,
        "eval_labels_from_ground_truth": None,
        "labels_with_successful_attack_artifacts": None,
        "summary_clean_acc": None,
        "summary_adv_acc": None,
        "summary_attack_success_rate": None,
    }

    summary_line_re = re.compile(
        r"clean acc: (?P<clean>-?\d+(?:\.\d+)?), "
        r"adv acc: (?P<adv>-?\d+(?:\.\d+)?), "
        r"asr\(clean-correct only\): (?P<asr>-?\d+(?:\.\d+)?|n/a)"
    )

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("eval labels from NIPS ground truth"):
                metadata["eval_labels_from_ground_truth"] = try_int(line.rsplit(":", 1)[-1].strip())
            elif line.startswith("labels with successful attack artifacts"):
                metadata["labels_with_successful_attack_artifacts"] = try_int(line.rsplit(":", 1)[-1].strip())
            elif line.startswith("labels selected for transfer eval"):
                metadata["selected_label_count_total"] = try_int(line.rsplit(":", 1)[-1].strip())
            elif line.startswith("resume start_iteration"):
                match = re.search(
                    r"resume start_iteration: (?P<start>\d+) \| labels to evaluate this run: (?P<count>\d+)",
                    line,
                )
                if match:
                    metadata["start_iteration"] = int(match.group("start"))
                    metadata["selected_label_count_evaluated_this_run"] = int(match.group("count"))
            else:
                match = summary_line_re.search(line)
                if match:
                    metadata["summary_clean_acc"] = float(match.group("clean"))
                    metadata["summary_adv_acc"] = float(match.group("adv"))
                    metadata["summary_attack_success_rate"] = try_float(match.group("asr"))
    return metadata


def parse_log_records(log_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = LABEL_LINE_RE.match(line)
            if not match:
                continue
            records.append(
                {
                    "ground_label": int(match.group("label")),
                    "ground_class_name": (match.group("name") or "").strip(),
                    "artifact_type": match.group("artifact"),
                    "clean_acc": float(match.group("clean")),
                    "adv_acc": float(match.group("adv")),
                    "attack_success_rate": try_float(match.group("asr")),
                }
            )
    return records


def load_records_from_per_class_json(per_class_path: Path) -> List[Dict[str, Any]]:
    payload = read_json(per_class_path)
    if not isinstance(payload, list):
        raise ValueError(f"expected list in {per_class_path}")

    records: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        label = try_int(row.get("ground_label"))
        if label is None:
            continue
        records.append(
            {
                "ground_label": label,
                "ground_class_name": str(row.get("ground_class_name") or "").strip(),
                "artifact_type": str(row.get("artifact_type") or "").strip() or "unknown",
                "clean_acc": float(row["clean_acc"]),
                "adv_acc": float(row["adv_acc"]),
                "attack_success_rate": try_float(row.get("attack_success_rate")),
            }
        )
    return records


def enrich_record(
    row: Dict[str, Any],
    source_bundle: InputBundle,
    source_kind: str,
    source_index: int,
    category_names: Dict[int, str],
) -> Dict[str, Any]:
    record = dict(row)
    label = int(record["ground_label"])
    if not record.get("ground_class_name"):
        record["ground_class_name"] = category_names.get(label, "")
    record["clean_acc"] = float(record["clean_acc"])
    record["adv_acc"] = float(record["adv_acc"])
    record["attack_success_rate"] = try_float(record.get("attack_success_rate"))
    record["acc_drop"] = record["clean_acc"] - record["adv_acc"]
    record["source_input_path"] = str(source_bundle.input_path)
    record["source_run_dir"] = str(source_bundle.run_dir)
    record["source_kind"] = source_kind
    record["source_index"] = source_index
    return record


def compute_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "label_count": 0,
            "mean_clean_acc_per_label": None,
            "mean_adv_acc_per_label": None,
            "mean_asr_per_label": None,
            "median_asr_per_label": None,
            "asr_q1_per_label": None,
            "asr_q3_per_label": None,
            "asr_ge_50_count": 0,
            "asr_ge_80_count": 0,
            "asr_le_10_count": 0,
            "acc_drop_ge_40_count": 0,
            "adv_gt_clean_count": 0,
            "adv_eq_clean_count": 0,
            "adv_lt_clean_count": 0,
            "max_asr_record": None,
            "min_asr_record": None,
        }

    clean = np.array([float(row["clean_acc"]) for row in records], dtype=np.float64)
    adv = np.array([float(row["adv_acc"]) for row in records], dtype=np.float64)
    asr_records = [row for row in records if row.get("attack_success_rate") is not None]
    asr = np.array([float(row["attack_success_rate"]) for row in asr_records], dtype=np.float64)

    stats: Dict[str, Any] = {
        "label_count": len(records),
        "mean_clean_acc_per_label": float(clean.mean()),
        "mean_adv_acc_per_label": float(adv.mean()),
        "mean_asr_per_label": float(asr.mean()) if len(asr) else None,
        "median_asr_per_label": float(np.median(asr)) if len(asr) else None,
        "asr_q1_per_label": float(np.percentile(asr, 25)) if len(asr) else None,
        "asr_q3_per_label": float(np.percentile(asr, 75)) if len(asr) else None,
        "asr_ge_50_count": int(np.sum(asr >= 50.0)) if len(asr) else 0,
        "asr_ge_80_count": int(np.sum(asr >= 80.0)) if len(asr) else 0,
        "asr_le_10_count": int(np.sum(asr <= 10.0)) if len(asr) else 0,
        "acc_drop_ge_40_count": int(np.sum((clean - adv) >= 40.0)),
        "adv_gt_clean_count": int(np.sum(adv > clean)),
        "adv_eq_clean_count": int(np.sum(adv == clean)),
        "adv_lt_clean_count": int(np.sum(adv < clean)),
        "max_asr_record": None,
        "min_asr_record": None,
    }

    if len(asr_records):
        stats["max_asr_record"] = max(asr_records, key=lambda row: float(row["attack_success_rate"]))
        stats["min_asr_record"] = min(asr_records, key=lambda row: float(row["attack_success_rate"]))
        stats["asr_ge_50_pct"] = 100.0 * stats["asr_ge_50_count"] / len(asr_records)
        stats["asr_ge_80_pct"] = 100.0 * stats["asr_ge_80_count"] / len(asr_records)
        stats["asr_le_10_pct"] = 100.0 * stats["asr_le_10_count"] / len(asr_records)
    else:
        stats["asr_ge_50_pct"] = None
        stats["asr_ge_80_pct"] = None
        stats["asr_le_10_pct"] = None

    stats["acc_drop_ge_40_pct"] = 100.0 * stats["acc_drop_ge_40_count"] / len(records)
    stats["adv_gt_clean_pct"] = 100.0 * stats["adv_gt_clean_count"] / len(records)
    stats["adv_eq_clean_pct"] = 100.0 * stats["adv_eq_clean_count"] / len(records)
    stats["adv_lt_clean_pct"] = 100.0 * stats["adv_lt_clean_count"] / len(records)
    return stats


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in records:
        normalized.append(
            {
                "ground_label": int(row["ground_label"]),
                "ground_class_name": str(row.get("ground_class_name") or "").strip(),
                "artifact_type": str(row.get("artifact_type") or "unknown"),
                "clean_acc": float(row["clean_acc"]),
                "adv_acc": float(row["adv_acc"]),
                "attack_success_rate": try_float(row.get("attack_success_rate")),
                "acc_drop": float(row.get("acc_drop", float(row["clean_acc"]) - float(row["adv_acc"]))),
                "source_input_path": str(row.get("source_input_path") or ""),
                "source_run_dir": str(row.get("source_run_dir") or ""),
                "source_kind": str(row.get("source_kind") or ""),
                "source_index": int(row.get("source_index", -1)),
            }
        )
    return normalized


def load_input_records(
    bundle: InputBundle,
    source_index: int,
    category_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "input_path": str(bundle.input_path),
        "run_dir": str(bundle.run_dir),
        "used_source_kind": None,
        "log_path": str(bundle.log_path) if bundle.log_path else None,
        "per_class_path": str(bundle.per_class_path) if bundle.per_class_path else None,
        "summary_path": str(bundle.summary_path) if bundle.summary_path else None,
    }

    records: List[Dict[str, Any]]
    if bundle.per_class_path is not None and bundle.per_class_path.is_file():
        records = load_records_from_per_class_json(bundle.per_class_path)
        metadata["used_source_kind"] = "per_class_json"
    elif bundle.log_path is not None and bundle.log_path.is_file():
        records = parse_log_records(bundle.log_path)
        metadata["used_source_kind"] = "log"
    else:
        raise FileNotFoundError(
            f"no usable per_class_results.json or log.txt found for input {bundle.input_path}"
        )

    if bundle.log_path is not None and bundle.log_path.is_file():
        metadata.update(parse_log_metadata(bundle.log_path))
    if bundle.summary_path is not None and bundle.summary_path.is_file():
        summary_payload = read_json(bundle.summary_path)
        if isinstance(summary_payload, dict):
            metadata["summary_json"] = summary_payload

    enriched = [
        enrich_record(
            row=row,
            source_bundle=bundle,
            source_kind=str(metadata["used_source_kind"]),
            source_index=source_index,
            category_names=category_names,
        )
        for row in records
    ]
    metadata["label_count_loaded"] = len(enriched)
    metadata.update(compute_stats(enriched))
    return enriched, metadata


def deduplicate_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}
    duplicates: List[Dict[str, Any]] = []

    for row in records:
        key = (int(row["ground_label"]), str(row["artifact_type"]))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(row)
            continue

        same_metrics = (
            math.isclose(float(existing["clean_acc"]), float(row["clean_acc"]), abs_tol=1e-9)
            and math.isclose(float(existing["adv_acc"]), float(row["adv_acc"]), abs_tol=1e-9)
            and (
                (existing.get("attack_success_rate") is None and row.get("attack_success_rate") is None)
                or (
                    existing.get("attack_success_rate") is not None
                    and row.get("attack_success_rate") is not None
                    and math.isclose(
                        float(existing["attack_success_rate"]),
                        float(row["attack_success_rate"]),
                        abs_tol=1e-9,
                    )
                )
            )
        )

        duplicate_info = {
            "ground_label": int(row["ground_label"]),
            "artifact_type": str(row["artifact_type"]),
            "previous_source_input_path": existing.get("source_input_path"),
            "new_source_input_path": row.get("source_input_path"),
            "resolution": "keep_latest",
            "same_metrics": bool(same_metrics),
        }
        duplicates.append(duplicate_info)

        replacement = dict(row)
        if not replacement.get("ground_class_name") and existing.get("ground_class_name"):
            replacement["ground_class_name"] = existing["ground_class_name"]
        if same_metrics and existing.get("ground_class_name") and not row.get("ground_class_name"):
            replacement = dict(existing)
        merged[key] = replacement

    merged_rows = sorted(
        normalize_records(merged.values()),
        key=lambda row: (int(row["ground_label"]), str(row["artifact_type"]), str(row["source_input_path"])),
    )
    return merged_rows, duplicates


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def save_figure(fig: Any, out_path: Path) -> None:
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))


def plot_asr_histogram(records: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    if plt is None:
        return
    values = [float(row["attack_success_rate"]) for row in records if row.get("attack_success_rate") is not None]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 100.0, 21)
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", linewidth=1.0)
    ax.axvline(np.median(values), color="#E45756", linestyle="--", linewidth=2, label=f"median={np.median(values):.2f}")
    ax.axvline(np.mean(values), color="#54A24B", linestyle="-.", linewidth=2, label=f"mean={np.mean(values):.2f}")
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("ASR (clean-correct only, %)")
    ax.set_ylabel("Label count")
    ax.set_title(f"{title}: ASR Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_asr_sorted(records: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    if plt is None:
        return
    values = sorted(
        [float(row["attack_success_rate"]) for row in records if row.get("attack_success_rate") is not None],
        reverse=True,
    )
    if not values:
        return

    ranks = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ranks, values, color="#F58518", linewidth=2)
    ax.fill_between(ranks, values, 0.0, color="#F58518", alpha=0.18)
    ax.axhline(50.0, color="#E45756", linestyle="--", linewidth=1.5, label="ASR=50")
    ax.axhline(80.0, color="#72B7B2", linestyle="--", linewidth=1.5, label="ASR=80")
    ax.set_xlim(1, len(values))
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Label rank (sorted by ASR)")
    ax.set_ylabel("ASR (clean-correct only, %)")
    ax.set_title(f"{title}: Sorted ASR Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_clean_vs_asr(records: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    if plt is None:
        return
    plotted = [row for row in records if row.get("attack_success_rate") is not None]
    if not plotted:
        return

    clean = np.array([float(row["clean_acc"]) for row in plotted], dtype=np.float64)
    asr = np.array([float(row["attack_success_rate"]) for row in plotted], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(clean, asr, color="#2C7FB8", s=30, alpha=0.8, edgecolors="none")
    x_line = np.array([0.0, 100.0], dtype=np.float64)
    expected_line = 100.0 - x_line
    ax.plot(
        x_line,
        expected_line,
        color="#D95F0E",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label="expected: y = 100 - x",
    )

    slope, intercept = np.polyfit(clean, asr, 1)
    fitted = intercept + slope * x_line
    predicted = intercept + slope * clean
    ss_res = float(np.sum(np.square(asr - predicted)))
    ss_tot = float(np.sum(np.square(asr - np.mean(asr))))
    r_squared = 0.0 if math.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)
    above_expected = int(np.sum(asr > (100.0 - clean)))
    ax.plot(
        x_line,
        fitted,
        color="#1B9E77",
        linestyle="-.",
        linewidth=1.7,
        alpha=0.95,
        label=f"fit: y = {intercept:.1f} {slope:+.3f}x",
    )
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Clean accuracy per label (%)")
    ax.set_ylabel("ASR (clean-correct only, %)")
    ax.set_title(f"{title}: Clean Accuracy vs ASR")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.97,
        f"R^2 = {r_squared:.3f}\nabove y=100-x: {above_expected}/{len(clean)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def top_rows(
    records: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    reverse: bool,
    min_clean_acc: Optional[float] = None,
) -> List[Dict[str, Any]]:
    filtered = [row for row in records if row.get("attack_success_rate") is not None]
    if min_clean_acc is not None:
        filtered = [row for row in filtered if float(row["clean_acc"]) >= float(min_clean_acc)]

    sorted_rows = sorted(
        filtered,
        key=lambda row: (
            float(row["attack_success_rate"]),
            float(row["clean_acc"]),
            -int(row["ground_label"]),
        ),
        reverse=reverse,
    )
    if not reverse:
        sorted_rows = sorted(
            filtered,
            key=lambda row: (
                float(row["attack_success_rate"]),
                -float(row["clean_acc"]),
                int(row["ground_label"]),
            ),
        )
    return sorted_rows[:top_k]


def report_table_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for row in rows:
        table.append(
            {
                "label": int(row["ground_label"]),
                "class_name": row.get("ground_class_name") or "",
                "artifact": row.get("artifact_type") or "",
                "clean_acc": fmt_float(row.get("clean_acc")),
                "adv_acc": fmt_float(row.get("adv_acc")),
                "asr": fmt_float(row.get("attack_success_rate")),
                "acc_drop": fmt_float(row.get("acc_drop")),
                "source": Path(str(row.get("source_input_path") or "")).name,
            }
        )
    return table


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return "_None_\n"

    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body) + "\n"


def write_report(
    out_path: Path,
    title: str,
    merged_stats: Dict[str, Any],
    input_metadata: Sequence[Dict[str, Any]],
    top_asr_rows: Sequence[Dict[str, Any]],
    top_high_clean_rows: Sequence[Dict[str, Any]],
    bottom_asr_rows: Sequence[Dict[str, Any]],
    duplicate_count: int,
    high_clean_threshold: float,
) -> None:
    aggregate_rows = [
        {"metric": "Label count", "value": merged_stats["label_count"]},
        {"metric": "Mean clean acc", "value": fmt_float(merged_stats["mean_clean_acc_per_label"])},
        {"metric": "Mean adv acc", "value": fmt_float(merged_stats["mean_adv_acc_per_label"])},
        {"metric": "Mean ASR", "value": fmt_float(merged_stats["mean_asr_per_label"])},
        {"metric": "Median ASR", "value": fmt_float(merged_stats["median_asr_per_label"])},
        {
            "metric": "ASR IQR",
            "value": f"{fmt_float(merged_stats['asr_q1_per_label'])} - {fmt_float(merged_stats['asr_q3_per_label'])}",
        },
        {
            "metric": "ASR >= 50",
            "value": f"{merged_stats['asr_ge_50_count']} ({fmt_float(merged_stats['asr_ge_50_pct'])}%)",
        },
        {
            "metric": "ASR >= 80",
            "value": f"{merged_stats['asr_ge_80_count']} ({fmt_float(merged_stats['asr_ge_80_pct'])}%)",
        },
        {
            "metric": "ASR <= 10",
            "value": f"{merged_stats['asr_le_10_count']} ({fmt_float(merged_stats['asr_le_10_pct'])}%)",
        },
        {
            "metric": "Acc drop >= 40",
            "value": f"{merged_stats['acc_drop_ge_40_count']} ({fmt_float(merged_stats['acc_drop_ge_40_pct'])}%)",
        },
        {
            "metric": "adv > clean",
            "value": f"{merged_stats['adv_gt_clean_count']} ({fmt_float(merged_stats['adv_gt_clean_pct'])}%)",
        },
        {
            "metric": "Duplicate labels resolved",
            "value": duplicate_count,
        },
    ]

    input_rows = []
    for item in input_metadata:
        input_rows.append(
            {
                "input": item["input_path"],
                "source": item.get("used_source_kind") or "n/a",
                "labels": item.get("label_count_loaded", 0),
                "start_iteration": item.get("start_iteration", "n/a"),
                "selected_total": item.get("selected_label_count_total", "n/a"),
                "selected_this_run": item.get("selected_label_count_evaluated_this_run", "n/a"),
                "mean_asr": fmt_float(item.get("mean_asr_per_label")),
            }
        )

    report = [
        f"# {title}",
        "",
        "## Aggregate Summary",
        markdown_table(aggregate_rows, [("metric", "Metric"), ("value", "Value")]),
        "## Inputs",
        markdown_table(
            input_rows,
            [
                ("input", "Input"),
                ("source", "Source"),
                ("labels", "Labels"),
                ("start_iteration", "Start Iter"),
                ("selected_total", "Selected Total"),
                ("selected_this_run", "Selected This Run"),
                ("mean_asr", "Mean ASR"),
            ],
        ),
        "## Figures",
        "### ASR Histogram",
        "![ASR Histogram](asr_histogram.png)",
        "",
        "### Sorted ASR Curve",
        "![Sorted ASR Curve](asr_sorted_curve.png)",
        "",
        "### Clean Accuracy vs ASR",
        "![Clean Accuracy vs ASR](clean_vs_asr.png)",
        "",
        "## Top ASR Labels",
        markdown_table(
            report_table_rows(top_asr_rows),
            [
                ("label", "Label"),
                ("class_name", "Class"),
                ("artifact", "Artifact"),
                ("clean_acc", "Clean"),
                ("adv_acc", "Adv"),
                ("asr", "ASR"),
                ("acc_drop", "Drop"),
                ("source", "Source"),
            ],
        ),
        f"## Top ASR Labels with Clean >= {fmt_float(high_clean_threshold)}",
        markdown_table(
            report_table_rows(top_high_clean_rows),
            [
                ("label", "Label"),
                ("class_name", "Class"),
                ("artifact", "Artifact"),
                ("clean_acc", "Clean"),
                ("adv_acc", "Adv"),
                ("asr", "ASR"),
                ("acc_drop", "Drop"),
                ("source", "Source"),
            ],
        ),
        "## Lowest ASR Labels",
        markdown_table(
            report_table_rows(bottom_asr_rows),
            [
                ("label", "Label"),
                ("class_name", "Class"),
                ("artifact", "Artifact"),
                ("clean_acc", "Clean"),
                ("adv_acc", "Adv"),
                ("asr", "ASR"),
                ("acc_drop", "Drop"),
                ("source", "Source"),
            ],
        ),
    ]
    out_path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (Path.cwd() / f"prompt_transfer_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    categories_path = (
        Path(args.categories_csv).expanduser().resolve()
        if args.categories_csv is not None
        else default_categories_csv()
    )
    category_names = load_category_names(categories_path)

    all_records: List[Dict[str, Any]] = []
    input_metadata: List[Dict[str, Any]] = []

    for index, raw_input in enumerate(args.inputs):
        bundle = resolve_input_bundle(raw_input)
        records, metadata = load_input_records(bundle, index, category_names)
        all_records.extend(records)
        input_metadata.append(metadata)
        print(
            "[loaded] input={} source={} labels={}".format(
                bundle.input_path,
                metadata.get("used_source_kind"),
                len(records),
            )
        )

    merged_records, duplicate_info = deduplicate_records(all_records)
    merged_stats = compute_stats(merged_records)

    merged_csv_rows = sorted(
        merged_records,
        key=lambda row: (int(row["ground_label"]), str(row["artifact_type"])),
    )
    merged_csv_path = out_dir / "merged_per_class_results.csv"
    write_csv(
        merged_csv_path,
        merged_csv_rows,
        [
            "ground_label",
            "ground_class_name",
            "artifact_type",
            "clean_acc",
            "adv_acc",
            "attack_success_rate",
            "acc_drop",
            "source_input_path",
            "source_run_dir",
            "source_kind",
            "source_index",
        ],
    )

    input_summary_rows = []
    for item in input_metadata:
        input_summary_rows.append(
            {
                "input_path": item["input_path"],
                "run_dir": item["run_dir"],
                "used_source_kind": item.get("used_source_kind"),
                "label_count_loaded": item.get("label_count_loaded"),
                "start_iteration": item.get("start_iteration"),
                "selected_label_count_total": item.get("selected_label_count_total"),
                "selected_label_count_evaluated_this_run": item.get("selected_label_count_evaluated_this_run"),
                "eval_labels_from_ground_truth": item.get("eval_labels_from_ground_truth"),
                "labels_with_successful_attack_artifacts": item.get("labels_with_successful_attack_artifacts"),
                "mean_clean_acc_per_label": item.get("mean_clean_acc_per_label"),
                "mean_adv_acc_per_label": item.get("mean_adv_acc_per_label"),
                "mean_asr_per_label": item.get("mean_asr_per_label"),
                "median_asr_per_label": item.get("median_asr_per_label"),
            }
        )
    write_csv(
        out_dir / "input_summaries.csv",
        input_summary_rows,
        [
            "input_path",
            "run_dir",
            "used_source_kind",
            "label_count_loaded",
            "start_iteration",
            "selected_label_count_total",
            "selected_label_count_evaluated_this_run",
            "eval_labels_from_ground_truth",
            "labels_with_successful_attack_artifacts",
            "mean_clean_acc_per_label",
            "mean_adv_acc_per_label",
            "mean_asr_per_label",
            "median_asr_per_label",
        ],
    )

    top_asr_rows = top_rows(merged_records, top_k=args.top_k, reverse=True)
    top_high_clean_rows = top_rows(
        merged_records,
        top_k=args.top_k,
        reverse=True,
        min_clean_acc=args.high_clean_threshold,
    )
    bottom_asr_rows = top_rows(merged_records, top_k=args.top_k, reverse=False)

    write_csv(
        out_dir / "top_asr.csv",
        top_asr_rows,
        [
            "ground_label",
            "ground_class_name",
            "artifact_type",
            "clean_acc",
            "adv_acc",
            "attack_success_rate",
            "acc_drop",
            "source_input_path",
        ],
    )
    write_csv(
        out_dir / "top_asr_high_clean.csv",
        top_high_clean_rows,
        [
            "ground_label",
            "ground_class_name",
            "artifact_type",
            "clean_acc",
            "adv_acc",
            "attack_success_rate",
            "acc_drop",
            "source_input_path",
        ],
    )
    write_csv(
        out_dir / "bottom_asr.csv",
        bottom_asr_rows,
        [
            "ground_label",
            "ground_class_name",
            "artifact_type",
            "clean_acc",
            "adv_acc",
            "attack_success_rate",
            "acc_drop",
            "source_input_path",
        ],
    )

    if duplicate_info:
        (out_dir / "duplicates.json").write_text(
            json.dumps(duplicate_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    aggregate_summary = {
        "title": args.title,
        "generated_at": dt.datetime.now().isoformat(),
        "categories_csv": str(categories_path) if categories_path else None,
        "input_count": len(args.inputs),
        "duplicate_resolution_count": len(duplicate_info),
        "stats": merged_stats,
        "source_inputs": input_metadata,
        "top_asr_rows": top_asr_rows,
        "top_high_clean_rows": top_high_clean_rows,
        "bottom_asr_rows": bottom_asr_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(aggregate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_asr_histogram(merged_records, out_dir / "asr_histogram.png", args.title)
    plot_asr_sorted(merged_records, out_dir / "asr_sorted_curve.png", args.title)
    plot_clean_vs_asr(merged_records, out_dir / "clean_vs_asr.png", args.title)

    write_report(
        out_path=out_dir / "report.md",
        title=args.title,
        merged_stats=merged_stats,
        input_metadata=input_metadata,
        top_asr_rows=top_asr_rows,
        top_high_clean_rows=top_high_clean_rows,
        bottom_asr_rows=bottom_asr_rows,
        duplicate_count=len(duplicate_info),
        high_clean_threshold=float(args.high_clean_threshold),
    )

    print(f"[done] labels={merged_stats['label_count']} out_dir={out_dir}")
    print(f"[done] merged_csv={merged_csv_path}")
    print(f"[done] report={out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

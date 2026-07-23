#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROLE_ABBR_MAP: Dict[str, str] = {
    "background": "BG",
    "color": "CL",
    "position": "POS",
    "weather": "WTH",
    "contrast": "CTR",
}


@dataclass
class RunSeries:
    run_dir: Path
    role_names: List[str]
    use_div_loss: bool | None
    cls_points: List[Tuple[int, float]]
    opt_points: List[Tuple[int, float]]
    label: str = ""


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _abbr_role(role: str) -> str:
    key = str(role).strip().lower()
    if not key:
        return "TOK"
    if key in ROLE_ABBR_MAP:
        return ROLE_ABBR_MAP[key]
    alnum = "".join(ch for ch in key if ch.isalnum())
    if not alnum:
        return "TOK"
    return alnum[:3].upper()


def _abbr_combo(role_names: Sequence[str]) -> str:
    if not role_names:
        return "UNK"
    return "+".join(_abbr_role(role) for role in role_names)


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _role_names_from_dir_name(run_name: str) -> List[str]:
    if "_tok" not in run_name:
        return []
    token_part = run_name.split("_tok", 1)[1]
    if "_div" in token_part:
        token_part = token_part.split("_div", 1)[0]
    token_part = token_part.strip()
    if not token_part:
        return []
    return [x for x in token_part.split("__") if x]


def _use_div_from_dir_name(run_name: str) -> bool | None:
    if "_divtrue" in run_name:
        return True
    if "_divfalse" in run_name:
        return False
    return None


def load_run_series(run_dir: Path) -> RunSeries | None:
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    metadata = _read_json(run_dir / "metadata.json")
    role_names = metadata.get("role_names")
    if not isinstance(role_names, list) or not role_names:
        role_names = _role_names_from_dir_name(run_dir.name)

    hyper = metadata.get("hyperparameters", {})
    use_div_loss = hyper.get("use_div_loss") if isinstance(hyper, dict) else None
    if use_div_loss is None:
        use_div_loss = _use_div_from_dir_name(run_dir.name)
    elif not isinstance(use_div_loss, bool):
        use_div_loss = None

    cls_points: List[Tuple[int, float]] = []
    opt_points: List[Tuple[int, float]] = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = _safe_int(row.get("step", row.get("iteration")))
            if step is None:
                continue

            cls_loss = _safe_float(row.get("loss_cls", row.get("cls_loss")))
            if cls_loss is not None:
                cls_points.append((step, cls_loss))

            opt_loss = _safe_float(row.get("opt_loss"))
            if opt_loss is not None:
                opt_points.append((step, opt_loss))

    cls_points.sort(key=lambda p: p[0])
    opt_points.sort(key=lambda p: p[0])

    if not cls_points and not opt_points:
        return None

    return RunSeries(
        run_dir=run_dir,
        role_names=[str(x) for x in role_names],
        use_div_loss=use_div_loss,
        cls_points=cls_points,
        opt_points=opt_points,
    )


def assign_labels(series_list: List[RunSeries]) -> None:
    base_labels = [_abbr_combo(s.role_names) for s in series_list]
    counts: Dict[str, int] = {}
    for base in base_labels:
        counts[base] = counts.get(base, 0) + 1

    used: Dict[str, int] = {}
    for idx, series in enumerate(series_list):
        base = base_labels[idx]
        label = base
        if counts.get(base, 0) > 1:
            if series.use_div_loss is True:
                label = f"{base}-D1"
            elif series.use_div_loss is False:
                label = f"{base}-D0"
            else:
                label = f"{base}-D?"

        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label}-{used[label]}"
        series.label = label


def plot_metric(
    series_list: Sequence[RunSeries],
    mode: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 9))

    plotted = 0
    for series in series_list:
        points = series.cls_points if mode == "cls" else series.opt_points
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=2.4, marker="o", markersize=3.8, label=series.label)
        plotted += 1

    y_name = "cls_loss" if mode == "cls" else "opt_loss"
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(f"{y_name} vs Iteration", fontsize=20)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)

    if plotted > 0:
        ax.legend(fontsize=12, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cls_loss and opt_loss from run directories.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories that contain metrics.jsonl and metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Directory where plot PNG files will be saved.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="run_metrics",
        help="Base filename for output plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=1200,
        help="PNG DPI. Higher means sharper and larger files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list: List[RunSeries] = []
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        series = load_run_series(run_dir)
        if series is not None:
            series_list.append(series)

    if not series_list:
        raise RuntimeError("No valid run directories with plot-able metrics were found.")

    assign_labels(series_list)

    cls_out = output_dir / f"{args.base_name}_cls_loss.png"
    opt_out = output_dir / f"{args.base_name}_opt_loss.png"

    plot_metric(series_list, mode="cls", output_path=cls_out, dpi=int(args.dpi))
    plot_metric(series_list, mode="opt", output_path=opt_out, dpi=int(args.dpi))

    print(f"[plot] saved cls_loss plot: {cls_out}")
    print(f"[plot] saved opt_loss plot: {opt_out}")
    print("[plot] legend labels:", ", ".join(s.label for s in series_list))


if __name__ == "__main__":
    main()

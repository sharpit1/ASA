#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.image_quality_metrics import write_aggregate_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-run image quality metrics.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = write_aggregate_run_metrics(run_dirs=run_dirs, output_path=output_path)
    print(f"Saved image metric summary for {payload.get('run_count', 0)} run(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

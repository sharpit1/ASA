import argparse
import datetime
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.eval_prompt_transfer_in_image import (
    evaluate_transfer,
    normalize_model_name,
    prepare_eval_context,
)


DEFAULT_MODEL_NAMES = [
    "resnet50",
    "wrn50",
    "inception_v3",
    "convnext",
    "vgg19",
    "vit",
    "swin",
    "deit",
    "adv_inc",
    "adv_res",
    "vim-small",
    "mambavision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt transfer classifiers from an existing render cache."
    )
    parser.add_argument("--attack_dir", type=str, required=True)
    parser.add_argument(
        "--render_cache_dir",
        type=str,
        required=True,
        help="Path to the existing _render_cache directory produced by eval/eval_prompt_transfer_in_image.py.",
    )
    parser.add_argument("--ground_truth_csv", type=str, default="./data/nips2017/images.csv")
    parser.add_argument("--imagenet_val_dir", type=str, default="./data/ILSVRC2012_img_val")
    parser.add_argument("--artifact_mode", type=str, choices=["auto", "prompt", "cwor"], default="auto")
    parser.add_argument("--eval_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=4,
        help="Kept for compatibility with evaluate_transfer logging. Ignored when reading from render cache.",
    )
    parser.add_argument("--max_images_per_class", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--ground_label", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="prompt_transfer_eval_from_cache",
        help="Root directory for evaluation outputs under attack_dir.",
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=0,
        help="0-based iteration index in selected_labels to resume from.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Save attack-success source/render image pairs under each model result directory.",
    )
    parser.add_argument(
        "--vis_max_images",
        type=int,
        default=0,
        help="Maximum number of source/render pairs to save for each label. 0 means unlimited.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Evaluate a single classifier model. If omitted, use --model_names or the default multi-model list.",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit multi-model list. Example: --model_names resnet50 swin convnext",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort immediately if one model evaluation fails.",
    )
    parser.add_argument(
        "--labels_from_cache",
        action="store_true",
        help=(
            "Evaluate every label_*.npz present in --render_cache_dir instead of "
            "recomputing selected labels from attack_dir success records."
        ),
    )
    return parser.parse_args()


def resolve_render_cache_dir(path_str: str) -> Path:
    cache_dir = Path(path_str).expanduser().resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"render_cache_dir not found: {cache_dir}")
    return cache_dir


def resolve_model_names(args: argparse.Namespace) -> List[str]:
    raw_names: List[str]
    if args.model_name:
        raw_names = [args.model_name]
    elif args.model_names:
        raw_names = list(args.model_names)
    else:
        raw_names = list(DEFAULT_MODEL_NAMES)

    normalized_names: List[str] = []
    for raw_name in raw_names:
        normalized = normalize_model_name(raw_name)
        if normalized is not None:
            normalized_names.append(normalized)

    normalized_names = list(dict.fromkeys(normalized_names))
    if len(normalized_names) == 0:
        raise ValueError("No valid model names resolved.")
    return normalized_names


def build_output_root(attack_dir: Path, save_dir_arg: str) -> Path:
    save_dir = Path(save_dir_arg).expanduser()
    if save_dir.is_absolute():
        root = save_dir.resolve()
    else:
        root = (attack_dir / save_dir).resolve()

    if root.exists():
        timestamp = datetime.date.today().strftime("%Y%m%d") + time.strftime("_%H%M%S")
        root = root.parent / f"{root.name}_{timestamp}"

    root.mkdir(parents=True, exist_ok=True)
    return root


def list_render_cache_labels(cache_dir: Path) -> List[int]:
    labels: List[int] = []
    for cache_path in cache_dir.glob("label_*.npz"):
        label_text = cache_path.stem.removeprefix("label_")
        try:
            labels.append(int(label_text))
        except ValueError:
            continue

    labels = sorted(set(labels))
    if len(labels) == 0:
        raise ValueError(f"No label_*.npz files found in render cache: {cache_dir}")
    return labels


def prepare_cache_label_context(args: argparse.Namespace, attack_dir: Path, render_cache_dir: Path) -> Dict[str, object]:
    if not attack_dir.is_dir():
        raise FileNotFoundError(f"attack_dir not found: {attack_dir}")

    selected_labels = list_render_cache_labels(render_cache_dir)
    if args.ground_label is not None:
        selected_labels = [label for label in selected_labels if int(label) == int(args.ground_label)]
    if int(args.max_classes) > 0:
        selected_labels = selected_labels[: int(args.max_classes)]

    start_iteration = int(args.start_iteration)
    if start_iteration < 0:
        raise ValueError("--start_iteration must be >= 0")
    if start_iteration > len(selected_labels):
        raise ValueError(
            f"--start_iteration {start_iteration} exceeds cache label count {len(selected_labels)}"
        )

    return {
        "attack_dir": attack_dir,
        "category_name_by_label": {},
        "class_dir_map": {},
        "success_records": {},
        "success_source_used": "render_cache",
        "npz_success_summary": None,
        "eval_labels": selected_labels,
        "selected_labels": selected_labels,
        "selected_labels_to_eval": selected_labels[start_iteration:],
        "start_iteration": start_iteration,
    }


def main() -> None:
    args = parse_args()
    attack_dir = Path(args.attack_dir).expanduser().resolve()
    render_cache_dir = resolve_render_cache_dir(args.render_cache_dir)
    if args.labels_from_cache:
        context = prepare_cache_label_context(args, attack_dir, render_cache_dir)
    else:
        context = prepare_eval_context(args)
    model_names = resolve_model_names(args)
    output_root = build_output_root(attack_dir, args.save_dir)

    all_summaries: Dict[str, Dict[str, object]] = {}
    for model_name in model_names:
        run_args = argparse.Namespace(**vars(args))
        run_args.model_name = model_name
        run_args.save_dir = str(output_root / model_name)

        try:
            all_summaries[model_name] = evaluate_transfer(
                run_args,
                shared_context=context,
                cache_dir=render_cache_dir,
            )
        except Exception as exc:
            all_summaries[model_name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[eval_pt_transfer] model '{model_name}' failed: {type(exc).__name__}: {exc}")
            if args.strict:
                raise

    summary_path = output_root / "multi_model_summary.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval_pt_transfer] multi-model summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

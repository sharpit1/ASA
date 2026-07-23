from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS = (
    "resnet50",
    "wrn50",
    "inception_v3",
    "convnext",
    "vgg19",
    "vit",
    "swin",
    "deit",
)

SUPPORTED_MODELS = {
    *DEFAULT_MODELS,
    "vim-small",
    "vim-tiny",
    "mambavision",
    "dinov3_vit7b16_lc",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov1_vitb16",
    "adv_inc",
    "adv_res",
}


def parse_model_list(raw: str) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        model = item.strip()
        if not model or model in seen:
            continue
        if model not in SUPPORTED_MODELS:
            choices = ", ".join(sorted(SUPPORTED_MODELS))
            raise argparse.ArgumentTypeError(
                f"unsupported model {model!r}; choose from: {choices}"
            )
        seen.add(model)
        models.append(model)
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed ASA source attack on its source classifier "
            "and one or more transfer classifiers."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed attack directory containing adversarial_examples.npz.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_list,
        default=list(DEFAULT_MODELS),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report the run as an 8-bit-quantized image evaluation.",
    )
    parser.add_argument(
        "--prompt-label-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat ImageNet category phrases intentionally present in an ASA prompt as correct.",
    )
    return parser.parse_args()


def finite_json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def print_metric_table(
    metrics: dict[str, dict[str, object]],
    *,
    source_model: str,
    requested_models: Iterable[str],
) -> None:
    print("SOURCE_TRANSFER_RESULTS_BEGIN")
    print(
        "model\trole\tclean_acc\tadv_acc\tuntarget_asr\t"
        "clean_correct\tasr_denominator\tsuccess_query_mean"
    )
    for model in requested_models:
        row = metrics[model]
        role = "source" if model == source_model else "transfer"
        query_mean = row.get("success_query_mean")
        query_text = "" if query_mean is None else f"{float(query_mean):.6f}"
        print(
            f"{model}\t{role}\t"
            f"{float(row['clean_acc']):.6f}\t"
            f"{float(row['adv_acc']):.6f}\t"
            f"{float(row['untarget_asr']):.6f}\t"
            f"{int(row['clean_correct_count'])}\t"
            f"{int(row['asr_denominator'])}\t"
            f"{query_text}"
        )
    payload = {
        "source_model": source_model,
        "models": list(requested_models),
        "metrics": metrics,
    }
    print(
        "SOURCE_TRANSFER_RESULT_JSON="
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=finite_json_default,
        )
    )
    print("SOURCE_TRANSFER_RESULTS_END")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    npz_path = run_root / "adversarial_examples.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing adversarial bundle: {npz_path}")

    source_model = run_root.parent.name
    print(f"[source_transfer] run_root={run_root}")
    print(f"[source_transfer] source_model={source_model}")
    print(f"[source_transfer] models={','.join(args.models)}")

    # Keep --help and configuration validation usable before optional FaaS
    # evaluation dependencies are installed.
    from eval.eval_asr_from_npz import eval_asr_from_npz

    result = eval_asr_from_npz(
        file_name=str(run_root),
        re_logger=True,
        quant=bool(args.quant),
        model_name=args.models,
        use_npz_source_query=True,
        use_prompt_label_as_correct=bool(args.prompt_label_correct),
    )
    metrics = result["metrics"]
    print_metric_table(
        metrics,
        source_model=source_model,
        requested_models=args.models,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

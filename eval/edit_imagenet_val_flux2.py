import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.eval_prompt_transfer_in_image import (
    build_imagenet_class_dir_map,
    build_render_cache,
    evaluate_transfer,
    load_flux_pipeline,
    normalize_model_name,
    prepare_render_artifact,
    resolve_device,
    resolve_flux_revision,
    resolve_hf_token,
)


DEFAULT_TRANSFER_MODELS = [
    "resnet50",
    "wrn50",
    "inception_v3",
    "convnext",
    "vgg19",
    "vit",
    "swin",
    "deit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Edit a fixed ImageNet val subset with direct FLUX.2 Klein KV prompts, "
            "cache only adversarial images when requested, and run transfer evaluation."
        )
    )
    parser.add_argument(
        "--imagenet_val_dir",
        type=str,
        default="./data/ILSVRC2012_img_val",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        required=True,
        help="Direct edit prompt. Repeat this option for multiple prompt runs.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(DEFAULT_TRANSFER_MODELS),
        help="Classifier models used for transfer evaluation.",
    )
    parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--start_iteration", type=int, default=0)
    parser.add_argument(
        "--flux_model_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9b-kv",
    )
    parser.add_argument("--flux_revision", type=str, default=None)
    parser.add_argument("--render_size", type=int, default=1024)
    parser.add_argument("--eval_size", type=int, default=224)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--render_batch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--save_clean_batch",
        type=str,
        choices=["on", "off"],
        default="off",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete class caches and completed transfer summaries.",
    )
    parser.add_argument(
        "--skip_transfer_eval",
        action="store_true",
        help="Build edited-image caches only.",
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_max_images", type=int, default=0)
    return parser.parse_args()


def prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(prompt).strip().lower()).strip("_")
    return slug[:80] or "prompt"


def resolve_models(raw_models: List[str]) -> List[str]:
    models: List[str] = []
    for raw_model in raw_models:
        model = normalize_model_name(raw_model)
        if model and model not in models:
            models.append(model)
    if not models:
        raise ValueError("no transfer models were selected")
    return models


def build_direct_prompt_context(
    *,
    imagenet_val_dir: Path,
    prompt_root: Path,
    prompt: str,
    max_classes: int,
    start_iteration: int,
) -> Dict[str, object]:
    class_dir_map = build_imagenet_class_dir_map(imagenet_val_dir)
    selected_labels = sorted(class_dir_map)
    if int(max_classes) > 0:
        selected_labels = selected_labels[: int(max_classes)]
    if int(start_iteration) < 0 or int(start_iteration) > len(selected_labels):
        raise ValueError(
            f"start_iteration={start_iteration} is outside 0..{len(selected_labels)}"
        )

    success_records = {
        int(label): {
            "label": int(label),
            "sample_dir": prompt_root,
            "prompt": str(prompt),
            "prompt_has_cwor": False,
            "best_objective": 0.0,
            "cwor_path": None,
            "attack_mode": "direct_prompt",
            "is_cwor_success": False,
        }
        for label in selected_labels
    }
    category_name_by_label = {
        int(label): class_dir_map[int(label)].name for label in selected_labels
    }
    return {
        "attack_dir": prompt_root,
        "category_name_by_label": category_name_by_label,
        "class_dir_map": class_dir_map,
        "success_records": success_records,
        "success_source_used": "direct_prompt",
        "npz_success_summary": None,
        "eval_labels": selected_labels,
        "selected_labels": selected_labels,
        "selected_labels_to_eval": selected_labels[int(start_iteration) :],
        "start_iteration": int(start_iteration),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    imagenet_val_dir = Path(args.imagenet_val_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples_per_class must be positive")

    prompts = [str(prompt).strip() for prompt in args.prompts if str(prompt).strip()]
    if not prompts:
        raise ValueError("at least one non-empty --prompt is required")
    prompt_slugs = [prompt_slug(prompt) for prompt in prompts]
    if len(set(prompt_slugs)) != len(prompt_slugs):
        raise ValueError("prompt slugs collide; use prompts with distinct normalized text")

    models = resolve_models(list(args.models))
    args.max_images_per_class = int(args.samples_per_class)
    args.artifact_mode = "prompt"
    args.success_source = "direct_prompt"
    args.rerender_cwor_only = False
    args.resume_render_cache = bool(args.resume)
    args.model_name = models[0]
    args.save_dir = str(output_dir)

    manifest = {
        "imagenet_val_dir": str(imagenet_val_dir),
        "output_dir": str(output_dir),
        "prompts": prompts,
        "prompt_slugs": prompt_slugs,
        "samples_per_class": int(args.samples_per_class),
        "max_classes": int(args.max_classes),
        "source_image_count": (
            (int(args.max_classes) if int(args.max_classes) > 0 else 1000)
            * int(args.samples_per_class)
        ),
        "edited_image_count_expected": (
            (int(args.max_classes) if int(args.max_classes) > 0 else 1000)
            * int(args.samples_per_class)
            * len(prompts)
        ),
        "save_clean_batch": str(args.save_clean_batch),
        "transfer_models": models,
        "seed": int(args.seed),
        "render_size": int(args.render_size),
        "eval_size": int(args.eval_size),
        "num_inference_steps": int(args.num_inference_steps),
        "flux_model_path": str(args.flux_model_path),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    pipe = load_flux_pipeline(
        model_path=str(args.flux_model_path),
        revision=resolve_flux_revision(args),
        device=device,
        hf_token=resolve_hf_token(args.hf_token),
        cpu_offload=bool(args.cpu_offload),
    )

    all_results: Dict[str, object] = {}
    for prompt, slug in zip(prompts, prompt_slugs):
        prompt_root = output_dir / slug
        prompt_root.mkdir(parents=True, exist_ok=True)
        context = build_direct_prompt_context(
            imagenet_val_dir=imagenet_val_dir,
            prompt_root=prompt_root,
            prompt=prompt,
            max_classes=int(args.max_classes),
            start_iteration=int(args.start_iteration),
        )
        first_record = context["success_records"][context["selected_labels"][0]]
        context["prepared_render_artifact"] = prepare_render_artifact(
            pipe=pipe,
            record=first_record,
            artifact_type="prompt",
            max_sequence_length=int(args.max_sequence_length),
        )

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        cache_dir = prompt_root / "_render_cache"
        build_render_cache(
            args=args,
            context=context,
            pipe=pipe,
            cache_dir=cache_dir,
        )
        context.pop("prepared_render_artifact", None)

        prompt_results: Dict[str, object] = {
            "prompt": prompt,
            "cache_dir": str(cache_dir),
            "transfer": {},
        }
        if not args.skip_transfer_eval:
            for model_name in models:
                eval_dir = prompt_root / "transfer" / model_name
                summary_path = eval_dir / "summary.json"
                if bool(args.resume) and summary_path.is_file():
                    prompt_results["transfer"][model_name] = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                    continue

                run_args = argparse.Namespace(**vars(args))
                run_args.model_name = model_name
                run_args.save_dir = str(eval_dir)
                prompt_results["transfer"][model_name] = evaluate_transfer(
                    run_args,
                    shared_context=context,
                    cache_dir=cache_dir,
                )

        all_results[slug] = prompt_results
        write_json(prompt_root / "prompt_summary.json", prompt_results)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(output_dir / "all_prompt_summary.json", all_results)
    print(f"[edit_imagenet_val_flux2] results saved to: {output_dir}")


if __name__ == "__main__":
    main()

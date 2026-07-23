from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
LEGACY_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "nips2017"
AUTO_IMAGE_REL_PATHS = (
    Path("images/vlm_final_selection.png"),
    Path("images/final_selected.png"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adversarial_examples.npz from selected images under sample_* folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing sample_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Only valid with a single --root. Defaults to <root>/adversarial_examples.npz.",
    )
    parser.add_argument("--size", type=int, default=224, help="Resize target size.")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU batch size for resize/normalize.")
    parser.add_argument(
        "--image-mode",
        choices=("auto", "none"),
        default="none",
        help=(
            "Image selection mode. 'auto' prefers images/vlm_final_selection.png and falls back to "
            "images/final_selected.png. 'none' uses --image-rel-path."
        ),
    )
    parser.add_argument(
        "--image-rel-path",
        "--image-dir",
        type=Path,
        default="images/final_selected.png",
        help="Relative path from each sample_* folder to the selected image. Required with --image-mode none.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Saved dtype inside the .npz file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def resolve_root_path(root: Path) -> Path:
    if root.is_absolute():
        return root

    cwd_candidate = (Path.cwd() / root).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    legacy_candidate = (LEGACY_OUTPUTS_ROOT / root).resolve()
    if legacy_candidate.exists():
        return legacy_candidate

    return cwd_candidate


def load_image_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_query_count(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def parse_victim_query_exhausted(payload: dict) -> bool:
    early_stop = payload.get("early_stop")
    return (
        isinstance(early_stop, dict)
        and early_stop.get("triggered") is True
        and early_stop.get("reason") == "victim_query_budget_exhausted"
    )


def parse_prompt_artifact_step(value: object) -> int:
    name = Path(str(value or "")).name
    parts = name.split("_", 2)
    if len(parts) < 2 or parts[0] != "step":
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def parse_report_step_count(sample_dir: Path) -> int:
    report_path = sample_dir / "report.json"
    if not report_path.exists():
        return -1

    payload = load_json(report_path)
    early_stop = payload.get("early_stop")
    if isinstance(early_stop, dict):
        step = parse_query_count(early_stop.get("step"))
        if step >= 0:
            return step

    history = payload.get("history")
    if not isinstance(history, list):
        return -1

    max_step = -1
    artifact_keys = (
        "scene_vocab_prompt_text_path",
        "scene_vocab_response_json_path",
        "vlm_query_prompt_text_path",
        "vlm_query_response_json_path",
    )
    for entry in history:
        if not isinstance(entry, dict):
            continue

        entry_step = -1
        has_artifact_record = False
        for key in artifact_keys:
            if key not in entry:
                continue
            has_artifact_record = True
            entry_step = max(entry_step, parse_prompt_artifact_step(entry.get(key)))

        if entry_step < 0 and has_artifact_record:
            try:
                entry_step = int(entry.get("step")) + 1
            except (TypeError, ValueError):
                entry_step = -1

        max_step = max(max_step, entry_step)

    return max_step


def parse_prompt_artifact_step_count(sample_dir: Path) -> int:
    report_step_count = parse_report_step_count(sample_dir)
    if report_step_count >= 0:
        return report_step_count

    artifact_dir = sample_dir / "prompt_artifacts"
    if not artifact_dir.is_dir():
        return -1

    max_step = -1
    for path in artifact_dir.glob("step_*"):
        step = parse_prompt_artifact_step(path.name)
        if step < 0:
            continue
        max_step = max(max_step, step)
    return max_step


def resolve_vlm_final_selection_prompt(sample_dir: Path) -> str | None:
    payload_path = sample_dir / "vlm_first_success.json"
    if not payload_path.exists():
        return None

    payload = load_json(payload_path)
    vlm_final_selection = payload.get("vlm_final_selection")
    if not isinstance(vlm_final_selection, dict):
        return None

    candidate_objective = vlm_final_selection.get("candidate_objective")
    if candidate_objective is None:
        return None
    return str(candidate_objective)


def resolve_image_metadata(sample_dir: Path, image_rel_path: Path) -> tuple[int, str, bool]:
    image_rel_path_posix = image_rel_path.as_posix()

    if image_rel_path_posix in {"images/vlm_final_selection.png", "images/final_selected.png"}:
        report_path = sample_dir / "report.json"
        source_prompt = "none"
        if image_rel_path_posix == "images/vlm_final_selection.png":
            source_prompt = resolve_vlm_final_selection_prompt(sample_dir) or source_prompt
        if not report_path.exists():
            return -1, source_prompt, False
        payload = load_json(report_path)
        early_stop = payload.get("early_stop")
        if source_prompt == "none" and isinstance(early_stop, dict):
            raw_prompt = early_stop.get("candidate_prompt")
            if raw_prompt is not None:
                source_prompt = str(raw_prompt)
        return (
            parse_query_count(payload.get("victim_query_count")),
            source_prompt,
            parse_victim_query_exhausted(payload),
        )

    return -1, "none", False


def resolve_image_path(sample_dir: Path, image_mode: str, image_rel_path: Path | None) -> tuple[Path, Path]:
    if image_mode == "auto":
        for candidate_rel_path in AUTO_IMAGE_REL_PATHS:
            candidate_path = sample_dir / candidate_rel_path
            if candidate_path.exists():
                return candidate_rel_path, candidate_path
        tried = ", ".join(path.as_posix() for path in AUTO_IMAGE_REL_PATHS)
        raise FileNotFoundError(f"Missing selected image under {sample_dir}. Tried: {tried}")

    if image_rel_path is None:
        raise ValueError("--image-rel-path/--image-dir is required when --image-mode none.")

    image_path = sample_dir / image_rel_path
    if not image_path.exists():
        raise FileNotFoundError(f"Missing selected image: {image_path}")
    return image_rel_path, image_path


def collect_image_paths(
    root: Path,
    image_mode: str,
    image_rel_path: Path | None,
) -> tuple[list[str], list[Path], list[str], list[int], list[str], list[bool], list[int]]:
    sample_dirs = sorted(path for path in root.glob("sample_*") if path.is_dir())
    sample_names: list[str] = []
    image_paths: list[Path] = []
    source_rel_paths: list[str] = []
    source_queries: list[int] = []
    source_prompts: list[str] = []
    victim_query_exhausted: list[bool] = []
    gemma_call_counts: list[int] = []

    for sample_dir in sample_dirs:
        resolved_rel_path, image_path = resolve_image_path(sample_dir, image_mode, image_rel_path)
        source_query_count, source_prompt, query_exhausted = resolve_image_metadata(sample_dir, resolved_rel_path)
        sample_names.append(sample_dir.name)
        image_paths.append(image_path)
        source_rel_paths.append(resolved_rel_path.as_posix())
        source_queries.append(source_query_count)
        source_prompts.append(source_prompt)
        victim_query_exhausted.append(query_exhausted)
        gemma_call_counts.append(parse_prompt_artifact_step_count(sample_dir))

    if not image_paths:
        raise FileNotFoundError(f"No sample_* folders found under {root}")

    return (
        sample_names,
        image_paths,
        source_rel_paths,
        source_queries,
        source_prompts,
        victim_query_exhausted,
        gemma_call_counts,
    )


def format_source_rel_paths(source_rel_paths: list[str]) -> np.ndarray:
    unique_paths = sorted(set(source_rel_paths))
    if len(unique_paths) == 1:
        return np.asarray(unique_paths)
    return np.asarray(source_rel_paths)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    return torch.float16 if dtype_name == "float16" else torch.float32


def process_batch(batch_images: list[np.ndarray], size: int, torch_dtype: torch.dtype, device: torch.device) -> np.ndarray:
    resized_tensors = []
    for image in batch_images:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        tensor = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        resized_tensors.append(tensor.squeeze(0))

    tensor = torch.stack(resized_tensors, dim=0).to(dtype=torch_dtype)
    return tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def build_npz_for_root(
    root: Path,
    output_path: Path,
    size: int,
    batch_size: int,
    image_mode: str,
    image_rel_path: Path | None,
    dtype_name: str,
    overwrite: bool,
    device: torch.device,
) -> None:
    if output_path.exists() and not overwrite:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    torch_dtype = resolve_torch_dtype(dtype_name)
    (
        sample_names,
        image_paths,
        source_rel_paths,
        source_queries,
        source_prompts,
        victim_query_exhausted,
        gemma_call_counts,
    ) = collect_image_paths(
        root,
        image_mode,
        image_rel_path,
    )

    output = np.empty((len(image_paths), size, size, 3), dtype=np.float16 if dtype_name == "float16" else np.float32)

    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_images = [load_image_uint8(path) for path in image_paths[start:end]]
        output[start:end] = process_batch(batch_images, size, torch_dtype, device)
        torch.cuda.empty_cache()

    np.savez_compressed(
        output_path,
        adversarial_examples=output,
        sample_names=np.asarray(sample_names),
        resize_size=np.asarray([size, size], dtype=np.int32),
        value_range=np.asarray(["0_to_255_after_resize"]),
        backend=np.asarray(["torch_cuda"]),
        source_rel_path=format_source_rel_paths(source_rel_paths),
        source_query=np.asarray(source_queries, dtype=np.int32),
        source_prompt=np.asarray(source_prompts),
        victim_query_exhausted=np.asarray(victim_query_exhausted, dtype=np.bool_),
        gemma_call_count=np.asarray(gemma_call_counts, dtype=np.int32),
    )


def main() -> None:
    args = parse_args()
    if args.image_mode == "none" and args.image_rel_path is None:
        raise ValueError("--image-rel-path/--image-dir is required when --image-mode none.")

    roots = [resolve_root_path(root) for root in args.root]
    if args.output and len(roots) != 1:
        raise ValueError("--output can only be used when exactly one --root is provided.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

    device = torch.device("cuda")
    for root in roots:
        output_path = args.output.resolve() if args.output else root / "adversarial_examples.npz"
        build_npz_for_root(
            root=root,
            output_path=output_path,
            size=args.size,
            batch_size=args.batch_size,
            image_mode=args.image_mode,
            image_rel_path=args.image_rel_path,
            dtype_name=args.dtype,
            overwrite=args.overwrite,
            device=device,
        )
        print(output_path)


if __name__ == "__main__":
    main()

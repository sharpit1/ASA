from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import torch

from create_adversarial_examples_npz import (
    AUTO_IMAGE_REL_PATHS,
    REPO_ROOT,
    format_source_rel_paths,
    load_image_uint8,
    parse_prompt_artifact_step_count,
    process_batch,
    resolve_image_metadata,
)


DEFAULT_ATTACK_NAME = "flux2_vlm_attack_q100"
DEFAULT_MODEL = "resnet50"
DEFAULT_MODE = "vlm"
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "nips2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an adversarial_examples.npz and replace selected samples with rendered images."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model directory under outputs/nips2017.")
    parser.add_argument("--mode", choices=("vlm", "greedy", "vlm_alter", "greedy_alter"), default=DEFAULT_MODE)
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=None,
        help="Defaults to outputs/nips2017/<model>/flux2_vlm_attack_q100/adversarial_examples.npz.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Defaults to outputs/nips2017/<model>/re_<mode>/adversarial_examples.npz.",
    )
    parser.add_argument(
        "--re-root",
        "--re-vlm-root",
        dest="re_root",
        type=Path,
        default=None,
        help="Defaults to outputs/nips2017/<model>/re_<mode>.",
    )
    parser.add_argument(
        "--sample-index-file",
        type=Path,
        default=None,
        help="Defaults to <re-root>/re_experiment_samples.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("auto", "none"),
        default="auto",
        help="auto prefers images/vlm_final_selection.png and falls back to images/final_selected.png.",
    )
    parser.add_argument(
        "--image-rel-path",
        type=Path,
        default=Path("images/final_selected.png"),
        help="Relative image path used with --image-mode none.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    model_root = OUTPUTS_ROOT / str(args.model)
    re_root = resolve_path(args.re_root) if args.re_root else model_root / f"re_{args.mode}"
    output_npz = (
        resolve_path(args.output_npz)
        if args.output_npz
        else re_root / "adversarial_examples.npz"
    )
    if args.input_npz:
        input_npz = resolve_path(args.input_npz)
    elif output_npz.with_name(f"{output_npz.stem}_new{output_npz.suffix}").exists():
        input_npz = output_npz.with_name(f"{output_npz.stem}_new{output_npz.suffix}")
    else:
        input_npz = model_root / DEFAULT_ATTACK_NAME / "adversarial_examples.npz"
    sample_index_file = (
        resolve_path(args.sample_index_file)
        if args.sample_index_file
        else re_root / "re_experiment_samples"
    )
    return input_npz, output_npz, re_root, sample_index_file


def load_sample_indices(path: Path) -> list[int]:
    value = ast.literal_eval(path.read_text(encoding="utf-8").strip())
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"sample index file must contain an int list: {path}")
    return [int(item) for item in value]


def resolve_replacement_image(
    sample_dir: Path,
    image_mode: str,
    image_rel_path: Path,
) -> tuple[Path, Path]:
    if image_mode == "auto":
        for rel_path in AUTO_IMAGE_REL_PATHS:
            image_path = sample_dir / rel_path
            if image_path.exists():
                return rel_path, image_path
        tried = ", ".join(path.as_posix() for path in AUTO_IMAGE_REL_PATHS)
        raise FileNotFoundError(f"missing replacement image under {sample_dir}. Tried: {tried}")

    image_path = sample_dir / image_rel_path
    if not image_path.exists():
        raise FileNotFoundError(f"missing replacement image: {image_path}")
    return image_rel_path, image_path


def build_sample_name_positions(sample_names: np.ndarray | None) -> dict[str, int]:
    if sample_names is None:
        return {}
    return {str(name): int(pos) for pos, name in enumerate(sample_names)}


def resolve_array_position(
    sample_index: int,
    sample_name_positions: dict[str, int],
    sample_count: int,
) -> int:
    sample_name = f"sample_{sample_index:04d}"
    if sample_name in sample_name_positions:
        return sample_name_positions[sample_name]
    if 0 <= sample_index < sample_count:
        return sample_index
    raise IndexError(f"sample index {sample_index} is outside adversarial_examples length {sample_count}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_name)


def cast_like_array(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.rint(np.clip(array, info.min, info.max)).astype(dtype, copy=False)
    return array.astype(dtype, copy=False)


def widen_unicode_array(array: np.ndarray, values: list[str]) -> np.ndarray:
    if array.dtype.kind != "U":
        return np.array(array, copy=True)
    current_len = array.dtype.itemsize // 4
    needed_len = max([current_len, 1, *(len(value) for value in values)])
    return np.asarray(array, dtype=f"<U{needed_len}")


def resolve_original_sample_root(input_npz: Path, model: str) -> Path:
    if input_npz.parent.name == DEFAULT_ATTACK_NAME:
        return input_npz.parent
    return OUTPUTS_ROOT / str(model) / DEFAULT_ATTACK_NAME


def sample_indices_for_positions(sample_names: np.ndarray | None, sample_count: int) -> list[int]:
    if sample_names is None or len(sample_names) != sample_count:
        return list(range(sample_count))

    sample_indices: list[int] = []
    for position, sample_name in enumerate(sample_names):
        name = str(sample_name)
        if name.startswith("sample_"):
            try:
                sample_indices.append(int(name.rsplit("_", 1)[1]))
                continue
            except (IndexError, ValueError):
                pass
        sample_indices.append(position)
    return sample_indices


def source_rel_paths_for_positions(arrays: dict[str, np.ndarray], sample_count: int) -> list[Path]:
    source_rel_path = arrays.get("source_rel_path")
    if source_rel_path is None:
        return [Path("images/final_selected.png")] * sample_count
    if len(source_rel_path) == 1:
        return [Path(str(source_rel_path[0]))] * sample_count
    if len(source_rel_path) == sample_count:
        return [Path(str(value)) for value in source_rel_path]
    raise ValueError(f"unexpected source_rel_path length: {len(source_rel_path)}")


def build_original_metadata_arrays(
    original_sample_root: Path,
    arrays: dict[str, np.ndarray],
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = sample_indices_for_positions(arrays.get("sample_names"), sample_count)
    source_rel_paths = source_rel_paths_for_positions(arrays, sample_count)
    query_exhausted_array = np.zeros(sample_count, dtype=np.bool_)
    gemma_call_count_array = np.full(sample_count, -1, dtype=np.int32)

    for position, (sample_index, rel_path) in enumerate(zip(sample_indices, source_rel_paths)):
        sample_dir = original_sample_root / f"sample_{sample_index:04d}"
        if not sample_dir.exists():
            continue
        _, _, query_exhausted = resolve_image_metadata(sample_dir, rel_path)
        query_exhausted_array[position] = query_exhausted
        gemma_call_count_array[position] = parse_prompt_artifact_step_count(sample_dir)

    return query_exhausted_array, gemma_call_count_array


def save_npz_atomic(output_path: Path, arrays: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    input_npz, output_npz, re_root, sample_index_file = resolve_default_paths(args)
    original_sample_root = resolve_original_sample_root(input_npz, str(args.model))

    if output_npz.exists() and not args.overwrite and input_npz.resolve() != output_npz.resolve():
        output_npz = output_npz.with_name(f"{output_npz.stem}_new{output_npz.suffix}")
        if output_npz.exists() and input_npz.resolve() != output_npz.resolve():
            raise FileExistsError(f"output already exists: {output_npz}. Use --overwrite to replace it.")

    sample_indices = load_sample_indices(sample_index_file)
    device = resolve_device(args.device)

    with np.load(input_npz, allow_pickle=False) as npz_data:
        arrays = {key: npz_data[key] for key in npz_data.files}

    if "adversarial_examples" not in arrays:
        raise KeyError(f"{input_npz} has no adversarial_examples array")

    adversarial_examples = np.array(arrays["adversarial_examples"], copy=True)
    if adversarial_examples.ndim != 4 or adversarial_examples.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB adversarial_examples, got {adversarial_examples.shape}")
    if adversarial_examples.shape[1] != adversarial_examples.shape[2]:
        raise ValueError("only square adversarial_examples are supported, matching create_adversarial_examples_npz.py")

    sample_name_positions = build_sample_name_positions(arrays.get("sample_names"))
    image_paths: list[Path] = []
    target_positions: list[int] = []
    source_rel_paths: list[str] = []
    source_queries: list[int] = []
    source_prompts: list[str] = []
    victim_query_exhausted: list[bool] = []
    gemma_call_counts: list[int] = []

    for sample_index in sample_indices:
        sample_dir = re_root / f"sample_{sample_index:04d}"
        rel_path, image_path = resolve_replacement_image(sample_dir, args.image_mode, args.image_rel_path)
        source_query, source_prompt, query_exhausted = resolve_image_metadata(sample_dir, rel_path)
        image_paths.append(image_path)
        source_rel_paths.append(rel_path.as_posix())
        source_queries.append(source_query)
        source_prompts.append(source_prompt)
        victim_query_exhausted.append(query_exhausted)
        gemma_call_counts.append(parse_prompt_artifact_step_count(sample_dir))
        target_positions.append(
            resolve_array_position(sample_index, sample_name_positions, len(adversarial_examples))
        )

    torch_dtype = torch.float16 if adversarial_examples.dtype == np.float16 else torch.float32
    resize_size = int(adversarial_examples.shape[1])

    for start in range(0, len(image_paths), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(image_paths))
        batch_images = [load_image_uint8(path) for path in image_paths[start:end]]
        resized = process_batch(batch_images, resize_size, torch_dtype, device)
        resized = cast_like_array(resized, adversarial_examples.dtype)
        adversarial_examples[target_positions[start:end]] = resized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arrays["adversarial_examples"] = adversarial_examples
    if "source_rel_path" in arrays:
        previous_rel_paths = arrays["source_rel_path"]
        if len(previous_rel_paths) == len(adversarial_examples):
            all_rel_paths = [str(value) for value in previous_rel_paths]
        elif len(previous_rel_paths) == 1:
            all_rel_paths = [str(previous_rel_paths[0])] * len(adversarial_examples)
        else:
            raise ValueError(f"unexpected source_rel_path length: {len(previous_rel_paths)}")
        for position, rel_path in zip(target_positions, source_rel_paths):
            all_rel_paths[position] = rel_path
        arrays["source_rel_path"] = format_source_rel_paths(all_rel_paths)
    if "source_query" in arrays and len(arrays["source_query"]) == len(adversarial_examples):
        source_query_array = np.array(arrays["source_query"], copy=True)
        source_query_array[target_positions] = np.asarray(source_queries, dtype=source_query_array.dtype)
        arrays["source_query"] = source_query_array
    if "source_prompt" in arrays and len(arrays["source_prompt"]) == len(adversarial_examples):
        source_prompt_array = widen_unicode_array(arrays["source_prompt"], source_prompts)
        source_prompt_array[target_positions] = np.asarray(source_prompts, dtype=source_prompt_array.dtype)
        arrays["source_prompt"] = source_prompt_array
    if "gemma_call_count" in arrays and len(arrays["gemma_call_count"]) == len(adversarial_examples):
        gemma_call_count_array = np.array(arrays["gemma_call_count"], copy=True)
        missing_gemma_positions = gemma_call_count_array == -1
    else:
        gemma_call_count_array = np.full(len(adversarial_examples), -1, dtype=np.int32)
        missing_gemma_positions = np.ones(len(adversarial_examples), dtype=np.bool_)

    if "victim_query_exhausted" in arrays and len(arrays["victim_query_exhausted"]) == len(adversarial_examples):
        query_exhausted_array = np.array(arrays["victim_query_exhausted"], copy=True)
    else:
        query_exhausted_array = np.zeros(len(adversarial_examples), dtype=np.bool_)

    if bool(np.any(missing_gemma_positions)):
        original_query_exhausted_array, original_gemma_call_count_array = build_original_metadata_arrays(
            original_sample_root,
            arrays,
            len(adversarial_examples),
        )
        query_exhausted_array[missing_gemma_positions] = original_query_exhausted_array[missing_gemma_positions]
        gemma_call_count_array[missing_gemma_positions] = original_gemma_call_count_array[missing_gemma_positions]

    query_exhausted_array[target_positions] = np.asarray(victim_query_exhausted, dtype=query_exhausted_array.dtype)
    arrays["victim_query_exhausted"] = query_exhausted_array
    gemma_call_count_array[target_positions] = np.asarray(gemma_call_counts, dtype=gemma_call_count_array.dtype)
    arrays["gemma_call_count"] = gemma_call_count_array
    save_npz_atomic(output_npz, arrays)

    print(f"input: {input_npz}")
    print(f"replaced: {len(image_paths)} samples")
    print(f"output: {output_npz}")


if __name__ == "__main__":
    main()

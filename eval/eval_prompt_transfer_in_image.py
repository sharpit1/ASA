import argparse
import csv
import datetime
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from art.estimators.classification import PyTorchClassifier
except Exception:  # pragma: no cover
    PyTorchClassifier = None

try:
    from natsort import index_natsorted
except Exception:  # pragma: no cover
    def index_natsorted(values):
        def _natural_key(value: object):
            return [
                int(token) if token.isdigit() else token.lower()
                for token in re.split(r"(\d+)", str(value))
            ]

        return sorted(range(len(values)), key=lambda idx: _natural_key(values[idx]))

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import logger

try:
    from setproctitle import setproctitle as _setproctitle
except Exception:  # pragma: no cover
    _setproctitle = None



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_PREPROCESS = (
    np.array([0.485, 0.456, 0.406], dtype=np.float32),
    np.array([0.229, 0.224, 0.225], dtype=np.float32),
)
DEFAULT_FLUX_FP8_REVISION = "886954cb9d8e8566f6facb7ef61bee7199e5e4bb"
VIS_SAVE_SIZE = 224
_THIRD_PARTY_HF_DIFFUSERS = _REPO_ROOT / "third_party" / "hf_diffusers_git"
MODEL_NAME_ALIASES = {
    "resnet50": "resnet50",
    "convnext": "convnext",
    "swin": "swin",
    "vgg19": "vgg19",
    "inception_v3": "inception_v3",
    "inception-v3": "inception_v3",
    "mobile_v2": "mobile_v2",
    "mobile-v2": "mobile_v2",
    "mobilenet_v2": "mobile_v2",
    "wrn50": "wrn50",
    "vit": "vit",
    "vim-small": "vim-small",
    "vim_small": "vim-small",
    "vim-tiny": "vim-tiny",
    "vim_tiny": "vim-tiny",
    "vit-base": "vit-base",
    "vit_base": "vit-base",
    "deit": "deit",
    "deit-b": "deit-b",
    "deit_b": "deit-b",
    "deit-base": "deit-b",
    "deit_base": "deit-b",
    "mambavision": "mambavision",
    "mamba-vision": "mambavision",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _maybe_disable_cudnn_sdpa_for_vim_small(model_name: Optional[str]) -> None:
    if normalize_model_name(model_name) != "vim-small":
        return

    backend = getattr(getattr(torch, "backends", None), "cuda", None)
    if backend is None or not torch.cuda.is_available():
        return

    try:
        disable_cudnn = (
            not _env_bool("GCG_ENABLE_CUDNN_SDPA", False)
            and _env_bool("GCG_VIM_SMALL_DISABLE_CUDNN_SDPA", True)
            and hasattr(backend, "enable_cudnn_sdp")
        )
        prefer_flash = _env_bool("GCG_VIM_SMALL_PREFER_FLASH_SDPA", True)
        force_flash = _env_bool("GCG_VIM_SMALL_FORCE_FLASH_SDPA", False)

        if prefer_flash and hasattr(backend, "enable_flash_sdp"):
            backend.enable_flash_sdp(True)
        if force_flash:
            if hasattr(backend, "enable_mem_efficient_sdp"):
                backend.enable_mem_efficient_sdp(False)
            if hasattr(backend, "enable_math_sdp"):
                backend.enable_math_sdp(False)

        if disable_cudnn:
            cudnn_enabled = True
            if hasattr(backend, "cudnn_sdp_enabled"):
                cudnn_enabled = bool(backend.cudnn_sdp_enabled())
            if cudnn_enabled:
                backend.enable_cudnn_sdp(False)

        flash_state = (
            bool(backend.flash_sdp_enabled())
            if hasattr(backend, "flash_sdp_enabled")
            else None
        )
        mem_state = (
            bool(backend.mem_efficient_sdp_enabled())
            if hasattr(backend, "mem_efficient_sdp_enabled")
            else None
        )
        math_state = (
            bool(backend.math_sdp_enabled())
            if hasattr(backend, "math_sdp_enabled")
            else None
        )
        cudnn_state = (
            bool(backend.cudnn_sdp_enabled())
            if hasattr(backend, "cudnn_sdp_enabled")
            else None
        )
        print(
            "[eval_prompt_transfer] configured SDPA for vim-small "
            f"(flash={flash_state}, mem_efficient={mem_state}, "
            f"math={math_state}, cudnn={cudnn_state}, force_flash={force_flash})."
        )
    except Exception as exc:
        print(
            "[eval_prompt_transfer] WARNING: failed to configure SDPA for vim-small "
            f"({type(exc).__name__}: {exc})"
        )


def load_ground_truth(csv_filename: str) -> Tuple[List[str], List[int], List[int]]:
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            image_id_list.append(row["ImageId"])
            label_ori_list.append(int(row["TrueLabel"]) - 1)
            label_tar_list.append(int(row["TargetClass"]) - 1)

    # Keep the CSV in natural ImageId order because sample_0000, sample_0001, ...
    # were produced against this ordering convention.
    sorted_indices = index_natsorted(image_id_list)
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]
    return image_id_list, label_ori_list, label_tar_list


def load_category_names(csv_filename: str) -> Dict[int, str]:
    categories_path = Path(csv_filename).expanduser().resolve().parent / "categories.csv"
    if not categories_path.is_file():
        return {}

    category_names: Dict[int, str] = {}
    with categories_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            try:
                category_id = int(row["CategoryId"])
            except Exception:
                continue
            label_idx = int(category_id) - 1
            if label_idx < 0:
                continue
            raw_name = str(row.get("CategoryName", "")).strip()
            if not raw_name:
                continue
            category_names[label_idx] = raw_name.split(",")[0].strip() or raw_name
    return category_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt/CWOR transfer on ImageNet val images with FLUX.2 Klein KV."
    )
    parser.add_argument("--attack_dir", type=str, required=True)
    parser.add_argument("--ground_truth_csv", type=str, default="./data/nips2017/images.csv")
    parser.add_argument("--imagenet_val_dir", type=str, default="./data/ILSVRC2012_img_val")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Classifier model name. If omitted, infer it from attack_dir.",
    )
    parser.add_argument("--flux_model_path", type=str, default="black-forest-labs/FLUX.2-klein-9b-kv")
    parser.add_argument("--flux_revision", type=str, default=None)
    parser.add_argument("--artifact_mode", type=str, choices=["auto", "prompt", "cwor"], default="auto")
    parser.add_argument(
        "--success_source",
        type=str,
        choices=["auto", "report", "npz"],
        default="npz",
        help=(
            "Source used to decide which attack artifacts are successful. "
            "'auto' uses adversarial_examples.npz when present, otherwise report/metrics."
        ),
    )
    parser.add_argument(
        "--vim_small_label_python",
        type=str,
        default=os.environ.get("GCG_VIM_SMALL_LABEL_PYTHON", ""),
        help=(
            "Python executable used only to extract npz success labels for vim-small. "
            "This lets rendering continue in the current Python environment."
        ),
    )
    parser.add_argument(
        "--_npz_success_helper_output",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--eval_size", type=int, default=224)
    parser.add_argument("--render_size", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=4,
        help="Batch size for FLUX render calls. Falls back to single-image rendering if a batch fails.",
    )
    parser.add_argument("--max_images_per_class", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--ground_label", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_dir", type=str, default="prompt_transfer_eval_multi")
    parser.add_argument(
        "--save_clean_batch",
        type=str,
        choices=["on", "off"],
        default="on",
        help=(
            "Store resized clean images in each label cache. 'off' keeps only "
            "adv_batch and reloads clean images from image_paths during evaluation."
        ),
    )
    parser.add_argument(
        "--resume_render_cache",
        action="store_true",
        help="Skip valid label_*.npz files that already exist in the render cache.",
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
        help="Save attack-success source/render image pairs under save_dir/renders.",
    )
    parser.add_argument(
        "--vis_max_images",
        type=int,
        default=0,
        help="Maximum number of source/render pairs to save for each label. 0 means unlimited.",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="Build FLUX render cache and skip classifier evaluation.",
    )
    parser.add_argument(
        "--rerender_cwor_only",
        action="store_true",
        help=(
            "With --render_only, regenerate only labels whose selected successful "
            "artifact is a CWOR success. Existing label_*.npz cache files are overwritten."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        token = str(explicit_token).strip()
        return token if token else None
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            return token
    return None


def _set_eval_process_title(
    label_token: object,
    model_name: Optional[str] = None,
    current_idx: Optional[int] = None,
    total_count: Optional[int] = None,
) -> None:
    if _setproctitle is None:
        return
    model_token = str(model_name or "model").strip()
    model_token = "".join(ch if ch.isalnum() else "_" for ch in model_token)
    model_token = model_token.strip("_") or "model"
    token = str(label_token).strip()
    if not token:
        token = "na"

    suffix = ""
    if current_idx is not None and total_count is not None and int(total_count) > 0:
        total = int(total_count)
        current = max(0, min(int(current_idx), total))
        progress_pct = 100.0 * float(current) / float(total)
        suffix = f"_{progress_pct:05.1f}%"

    _setproctitle(f"{model_token}{suffix}_eval_pt_{token}")


def normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if model_name is None:
        return None
    token = str(model_name).strip()
    if not token:
        return None
    return MODEL_NAME_ALIASES.get(token.lower(), token)


def infer_model_name_from_attack_dir(attack_dir: Path) -> Optional[str]:
    for part in reversed(attack_dir.parts):
        resolved = normalize_model_name(part)
        if resolved in MODEL_NAME_ALIASES.values():
            return resolved
    return None


def resolve_model_name(explicit_model_name: Optional[str], attack_dir: Path) -> str:
    normalized_explicit = normalize_model_name(explicit_model_name)
    if normalized_explicit is not None:
        return normalized_explicit

    inferred = infer_model_name_from_attack_dir(attack_dir)
    if inferred is not None:
        return inferred

    raise ValueError(
        "Unable to infer model_name from attack_dir. "
        "Please pass --model_name explicitly."
    )


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_image_files(class_dir: Path, max_count: int = 0) -> List[Path]:
    image_paths = sorted([path for path in class_dir.iterdir() if is_image_file(path)], key=lambda p: p.name)
    if max_count > 0:
        image_paths = image_paths[:max_count]
    return image_paths


def build_imagenet_class_dir_map(val_dir: Path) -> Dict[int, Path]:
    class_dirs = sorted([path for path in val_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if len(class_dirs) != 1000:
        raise ValueError(f"expected 1000 class directories under {val_dir}, found {len(class_dirs)}")
    return {idx: class_dir for idx, class_dir in enumerate(class_dirs)}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def victim_query_exhausted_from_payload(*payloads: Dict[str, object]) -> bool:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("victim_query_exhausted", False)):
            return True
        if str(payload.get("early_stop_reason", "")).strip() == "victim_query_budget_exhausted":
            return True
        early_stop = payload.get("early_stop")
        if isinstance(early_stop, dict) and early_stop.get("reason") == "victim_query_budget_exhausted":
            return True
    return False


def prompt_mentions_cwor(prompt: object) -> bool:
    return "cwor" in str(prompt or "").strip().lower()


def extract_early_stop_candidate_prompt(report: Dict[str, object]) -> str:
    early_stop = report.get("early_stop")
    if not isinstance(early_stop, dict):
        return ""
    if not bool(early_stop.get("triggered", False)):
        return ""
    if str(early_stop.get("reason", "")).strip() != "attack_success":
        return ""
    candidate_prompt = str(early_stop.get("candidate_prompt") or "").strip()
    if candidate_prompt.lower() == "none":
        return ""
    return candidate_prompt


def report_has_cwor_success(report: Dict[str, object]) -> bool:
    early_stop = report.get("early_stop")
    if isinstance(early_stop, dict):
        if (
            bool(early_stop.get("triggered", False))
            and str(early_stop.get("reason", "")).strip() == "attack_success"
            and str(early_stop.get("candidate_word", "")).strip().upper() == "<CWOR>"
        ):
            return True

    history = report.get("history") or []
    if isinstance(history, list) and len(history) > 0:
        last_step = history[-1]
        if isinstance(last_step, dict):
            return (
                bool(last_step.get("attack_success", False))
                and str(last_step.get("candidate_word", "")).strip().upper() == "<CWOR>"
            )
    return False


def choose_attack_artifact_type(record: Dict[str, object], artifact_mode: str) -> Optional[str]:
    has_prompt = bool(str(record.get("prompt", "")).strip())
    has_cwor = isinstance(record.get("cwor_path"), Path) and Path(record["cwor_path"]).is_file()
    if artifact_mode == "prompt":
        return "prompt" if has_prompt else None
    if artifact_mode == "cwor":
        return "cwor" if has_cwor else None
    if bool(record.get("is_cwor_success", False)):
        if has_cwor:
            return "cwor"
        return "prompt" if has_prompt else None
    if bool(record.get("prompt_has_cwor", False)):
        return "cwor" if has_cwor else None
    if has_prompt:
        return "prompt"
    if has_cwor:
        return "cwor"
    return None


def collect_success_records_by_csv_order(
    attack_dir: Path,
    image_id_list: Sequence[str],
    label_ori_list: Sequence[int],
    allowed_labels: Sequence[int],
    artifact_mode: str,
    success_sample_indices: Optional[Set[int]] = None,
) -> Dict[int, Dict[str, object]]:
    allowed_label_set = set(int(label) for label in allowed_labels)
    best_success_by_label: Dict[int, Dict[str, object]] = {}

    # sample_{idx:04d} is matched against the natural-sorted CSV order from
    # load_ground_truth(), not the raw row order in images.csv. For each label,
    # keep the successful artifact with the largest best_objective value; ties
    # fall back to the earlier CSV/sample order.
    for idx, (image_id, label) in enumerate(zip(image_id_list, label_ori_list)):
        label = int(label)
        if label not in allowed_label_set:
            continue
        if success_sample_indices is not None and int(idx) not in success_sample_indices:
            continue

        sample_dir = attack_dir / f"sample_{idx:04d}"
        report_path = sample_dir / "report.json"
        metrics_path = sample_dir / "metrics.json"
        if not report_path.is_file() or not metrics_path.is_file():
            continue

        try:
            report = read_json(report_path)
            metrics = read_json(metrics_path)
        except Exception as exc:
            logger.log(f"skip invalid sample {sample_dir}: {type(exc).__name__}: {exc}")
            continue

        if victim_query_exhausted_from_payload(metrics, report):
            continue

        report_label = report.get("args", {}).get("classifier_label")
        if report_label is not None and int(report_label) != label:
            logger.log(
                "label mismatch for {}: csv_label={} report_label={}".format(
                    sample_dir.name,
                    label,
                    int(report_label),
                )
            )

        if success_sample_indices is None:
            success = bool(metrics.get("final_attack_success"))
            if not success:
                history = report.get("history") or []
                success = bool(history and history[-1].get("attack_success"))
            if not success:
                continue

        cwor_relpath = metrics.get("cwor_final_embedding_pt_path") or report.get("cwor_final_embedding_pt_path")
        cwor_path = sample_dir / str(cwor_relpath) if cwor_relpath else None
        report_prompt_text = str(report.get("optimized_prompt") or "").strip()
        metrics_prompt_text = str(metrics.get("final_prompt") or "").strip()
        early_stop_prompt_text = extract_early_stop_candidate_prompt(report)
        prompt_text = early_stop_prompt_text or report_prompt_text or metrics_prompt_text
        is_cwor_success = report_has_cwor_success(report)

        record = {
            "label": label,
            "csv_index": int(idx),
            "image_id": str(image_id),
            "sample_dir": sample_dir,
            "prompt": prompt_text,
            "prompt_has_cwor": prompt_mentions_cwor(report_prompt_text or metrics_prompt_text),
            "best_objective": float(metrics.get("best_objective", report.get("best_objective", float("-inf")))),
            "cwor_path": cwor_path,
            "attack_mode": str(metrics.get("attack_mode", report.get("args", {}).get("attack_mode", ""))),
            "is_cwor_success": is_cwor_success,
        }
        if choose_attack_artifact_type(record, artifact_mode) is None:
            continue

        existing = best_success_by_label.get(label)
        if existing is None or float(record["best_objective"]) > float(existing["best_objective"]):
            best_success_by_label[label] = record

    return best_success_by_label


def extract_cwor_snapshot(payload: Dict[str, object]) -> Dict[str, object]:
    step_snapshots = payload.get("step_snapshots")
    if isinstance(step_snapshots, list) and len(step_snapshots) > 0:
        last_step = step_snapshots[-1]
        if isinstance(last_step, dict) and isinstance(last_step.get("snapshot"), dict):
            return last_step["snapshot"]

    final_snapshot = payload.get("final_snapshot")
    if isinstance(final_snapshot, dict):
        return final_snapshot

    if any(torch.is_tensor(payload.get(key)) for key in ("prompt_embeds", "t5_prompt_embeds")):
        return payload

    raise ValueError("unable to locate a valid CWOR embedding snapshot")


def load_cwor_snapshot(pt_path: Path) -> Dict[str, object]:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected CWOR payload type: {type(payload)}")
    return extract_cwor_snapshot(payload)


def resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), resample=Image.BILINEAR)


def pil_to_nchw01(image: Image.Image, size: int) -> np.ndarray:
    image_arr = np.asarray(resize_rgb(image, size), dtype=np.float32) / 255.0
    return image_arr.transpose(2, 0, 1)


def nchw01_to_pil(image_arr: np.ndarray) -> Image.Image:
    image_arr = np.asarray(image_arr, dtype=np.float32)
    if image_arr.ndim != 3 or image_arr.shape[0] != 3:
        raise ValueError(f"expected CHW image array, got shape={list(image_arr.shape)}")
    image_arr = np.clip(image_arr, 0.0, 1.0)
    image_arr = (image_arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_arr, mode="RGB")


def extract_output_images(output: object) -> List[Image.Image]:
    if hasattr(output, "images"):
        images_obj = getattr(output, "images")
    elif isinstance(output, (tuple, list)) and len(output) > 0:
        images_obj = output[0]
    else:
        images_obj = output

    if isinstance(images_obj, Image.Image):
        return [images_obj]
    if isinstance(images_obj, (tuple, list)):
        return [img for img in images_obj if isinstance(img, Image.Image)]
    return []


def save_visualization_artifacts(
    save_dir: Path,
    label: int,
    image_path: Path,
    source_image: Image.Image,
    rendered_image: Image.Image,
    save_size: int,
) -> Dict[str, str]:
    label_dir = save_dir / "renders" / f"label_{int(label):04d}"
    label_dir.mkdir(parents=True, exist_ok=True)

    source_out_path = label_dir / f"{image_path.stem}_source.png"
    render_out_path = label_dir / f"{image_path.stem}_render.png"

    resize_rgb(source_image, save_size).save(source_out_path)
    resize_rgb(rendered_image, save_size).save(render_out_path)
    return {
        "source": str(source_out_path),
        "render": str(render_out_path),
    }


def load_model_via_fallback(model_name: str, device: str):
    from torchvision import models as tv_models

    model_name = normalize_model_name(model_name) or str(model_name)

    if model_name == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == "convnext":
        model = tv_models.convnext_base(weights=tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif model_name == "swin":
        model = tv_models.swin_b(weights=tv_models.Swin_B_Weights.IMAGENET1K_V1)
    elif model_name == "vit-base":
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "vgg19":
        model = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == "mobile_v2":
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2)
    elif model_name == "wrn50":
        model = tv_models.wide_resnet50_2(weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
    else:
        raise RuntimeError(
            f"fallback model loader does not support '{model_name}'. "
            "If you need vim-* models, fix eval/attacked_models.py first."
        )
    return model.to(device)


def load_classifier_model(model_name: str, device: str):
    model_name = normalize_model_name(model_name) or str(model_name)
    try:
        from eval.attacked_models import model_selection  # type: ignore

        model = model_selection(model_name)
        return model.eval()
    except Exception as exc:
        logger.log(
            f"attacked_models.model_selection import failed for '{model_name}', "
            f"falling back to local torchvision loader: {type(exc).__name__}: {exc}"
        )
        model = load_model_via_fallback(model_name, device)
        return model.eval()


def align_classifier_logits_for_eval(model_name: Optional[str], logits: np.ndarray) -> np.ndarray:
    normalized_model_name = normalize_model_name(model_name)
    if normalized_model_name not in {"adv_inc", "adv_res"}:
        return logits
    if logits.ndim != 2:
        return logits

    # The TensorFlow-converted adversarial models expose 1001 logits with a
    # leading background class. Drop it so predictions line up with 0-based
    # ImageNet labels used everywhere else in this evaluator.
    if logits.shape[1] == 1001:
        return logits[:, 1:]
    return logits


def build_art_classifier(model_name: str, res: int, device: str) -> PyTorchClassifier:
    if PyTorchClassifier is None:
        raise ImportError(
            "adversarial-robustness-toolbox is required for transfer evaluation"
        )
    model = load_classifier_model(model_name, device)
    device_type = "gpu" if device.startswith("cuda") else "cpu"
    return PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, res, res),
        nb_classes=1000,
        preprocessing=IMAGENET_PREPROCESS,
        device_type=device_type,
    )


def load_flux_pipeline(
    model_path: str,
    revision: Optional[str],
    device: str,
    hf_token: Optional[str],
    cpu_offload: bool,
):
    if _THIRD_PARTY_HF_DIFFUSERS.is_dir():
        third_party_diffusers = str(_THIRD_PARTY_HF_DIFFUSERS)
        if third_party_diffusers not in sys.path:
            sys.path.insert(0, third_party_diffusers)
    try:
        importlib.import_module("regex")
    except Exception:
        pass
    diffusers_module = importlib.import_module("diffusers")
    pipeline_cls = getattr(diffusers_module, "Flux2KleinKVPipeline", None)
    if pipeline_cls is None:
        raise ImportError("Flux2KleinKVPipeline is not available in the current diffusers build.")

    use_cuda = device.startswith("cuda")
    torch_dtype = torch.float16 if use_cuda else torch.float32
    load_kwargs = {"torch_dtype": torch_dtype}
    if revision:
        load_kwargs["revision"] = str(revision)
    if hf_token:
        load_kwargs["token"] = hf_token
    if use_cuda and not cpu_offload:
        load_kwargs["device_map"] = "balanced"

    pipe = pipeline_cls.from_pretrained(model_path, **load_kwargs)
    if use_cuda and cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif not use_cuda and hasattr(pipe, "to"):
        pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def resolve_flux_revision(args: argparse.Namespace) -> Optional[str]:
    if args.flux_revision:
        return str(args.flux_revision)
    if str(args.flux_model_path).strip().lower().endswith("-fp8"):
        return DEFAULT_FLUX_FP8_REVISION
    return None


def repeat_prompt_embeds(prompt_embeds: torch.Tensor, batch_size: int) -> torch.Tensor:
    batch_size = int(batch_size)
    if prompt_embeds.shape[0] == batch_size:
        return prompt_embeds
    if prompt_embeds.shape[0] != 1:
        raise ValueError(
            "expected prompt_embeds batch dimension to be 1 or equal to render batch size "
            f"(got {list(prompt_embeds.shape)} for batch_size={batch_size})"
        )
    return prompt_embeds.repeat(batch_size, 1, 1)


def is_cuda_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def prepare_render_artifact(
    pipe,
    record: Dict[str, object],
    artifact_type: str,
    max_sequence_length: int,
) -> Dict[str, object]:
    if artifact_type == "cwor":
        snapshot = load_cwor_snapshot(Path(record["cwor_path"]))
        raw_prompt_embeds = snapshot.get("t5_prompt_embeds")
        if not torch.is_tensor(raw_prompt_embeds):
            raw_prompt_embeds = snapshot.get("prompt_embeds")
        if not torch.is_tensor(raw_prompt_embeds):
            raise ValueError("CWOR snapshot does not contain prompt embeddings.")

        reference_prompt = str(snapshot.get("reference_prompt", "")).strip()
        if len(reference_prompt) > 0:
            reference_prompt_embeds, _ = pipe.encode_prompt(
                prompt=reference_prompt,
                max_sequence_length=int(max_sequence_length),
            )
            prompt_embeds = raw_prompt_embeds.to(
                device=reference_prompt_embeds.device,
                dtype=reference_prompt_embeds.dtype,
            )
        else:
            execution_device = getattr(pipe, "_execution_device", None)
            transformer = getattr(pipe, "transformer", None)
            transformer_dtype = getattr(transformer, "dtype", None)
            prompt_embeds = raw_prompt_embeds.to(
                device=execution_device or raw_prompt_embeds.device,
                dtype=transformer_dtype or raw_prompt_embeds.dtype,
            )

        return {
            "artifact_type": "cwor",
            "artifact_prompt_for_log": reference_prompt,
            "prompt_embeds": prompt_embeds.detach(),
        }

    prompt_text = str(record.get("prompt", "")).strip()
    if len(prompt_text) == 0:
        raise ValueError("prompt artifact is missing optimized prompt text")
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt_text,
        max_sequence_length=int(max_sequence_length),
    )
    return {
        "artifact_type": "prompt",
        "artifact_prompt_for_log": prompt_text,
        "prompt_embeds": prompt_embeds.detach(),
    }


def render_with_prompt_embeds_batch(
    pipe,
    source_images: Sequence[Image.Image],
    prompt_embeds: torch.Tensor,
    render_size: int,
    num_inference_steps: int,
    max_sequence_length: int,
) -> List[Image.Image]:
    if len(source_images) == 0:
        return []

    batch_prompt_embeds = repeat_prompt_embeds(prompt_embeds, len(source_images)).contiguous()
    kwargs = {
        "prompt_embeds": batch_prompt_embeds,
        "image": [resize_rgb(image, render_size) for image in source_images],
        "num_inference_steps": int(num_inference_steps),
        "max_sequence_length": int(max_sequence_length),
        "height": int(render_size),
        "width": int(render_size),
        "output_type": "pil",
        "paired_image_batch": len(source_images) > 1,
    }
    output = pipe(**kwargs)
    images = extract_output_images(output)
    if len(images) != len(source_images):
        raise RuntimeError(
            f"FLUX render returned {len(images)} images for {len(source_images)} source images."
        )
    return [image.convert("RGB") for image in images]


def collect_attack_success_indices_from_npz(
    attack_dir: Path,
    image_id_list: Sequence[str],
    label_ori_list: Sequence[int],
    ground_truth_csv: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[Set[int], Dict[str, object]]:
    from eval.npz_loader import load_adv_images_from_npz

    npz_path = attack_dir / "adversarial_examples.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"adversarial_examples.npz not found: {npz_path}")

    adv_batch = load_adv_images_from_npz(npz_path, output_layout="nchw", normalize=True)
    adv_batch = np.asarray(adv_batch, dtype=np.float32)
    if adv_batch.ndim != 4 or adv_batch.shape[1] != 3 or adv_batch.shape[2] != adv_batch.shape[3]:
        raise ValueError(f"unexpected adversarial_examples.npz image shape: {list(adv_batch.shape)}")

    image_count = min(int(adv_batch.shape[0]), len(image_id_list), len(label_ori_list))
    if image_count <= 0:
        raise ValueError(f"no images found in {npz_path}")
    if image_count != int(adv_batch.shape[0]) or image_count != len(label_ori_list):
        print(
            "[eval_prompt_transfer] WARNING: npz/ground-truth length mismatch "
            f"(npz={int(adv_batch.shape[0])}, ground_truth={len(label_ori_list)}); "
            f"using first {image_count} samples."
        )

    victim_query_exhausted_mask = np.zeros((image_count,), dtype=np.bool_)
    with np.load(npz_path, allow_pickle=False) as npz_data:
        if "victim_query_exhausted" in npz_data:
            raw_exhausted = np.asarray(npz_data["victim_query_exhausted"], dtype=np.bool_).reshape(-1)
            exhausted_count = min(int(raw_exhausted.shape[0]), image_count)
            victim_query_exhausted_mask[:exhausted_count] = raw_exhausted[:exhausted_count]
            if int(raw_exhausted.shape[0]) != image_count:
                print(
                    "[eval_prompt_transfer] WARNING: victim_query_exhausted length mismatch "
                    f"(npz={int(raw_exhausted.shape[0])}, used={image_count}); "
                    f"using first {exhausted_count} flags."
                )

    image_size = int(adv_batch.shape[2])
    images_root = Path(ground_truth_csv).expanduser().resolve().parent / "images"
    clean_images: List[np.ndarray] = []
    for idx in tqdm(range(image_count), desc="npz_success_clean_load"):
        image_path = images_root / f"{image_id_list[idx]}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"clean image not found for npz success check: {image_path}")
        with Image.open(image_path) as opened:
            clean_images.append(pil_to_nchw01(opened.convert("RGB"), image_size))

    clean_batch = np.stack(clean_images, axis=0).astype(np.float32)
    labels = np.asarray(label_ori_list[:image_count], dtype=np.int64)

    _maybe_disable_cudnn_sdpa_for_vim_small(model_name)
    classifier = build_art_classifier(model_name, image_size, device)
    clean_pred = classifier.predict(clean_batch, batch_size=int(batch_size))
    clean_pred = align_classifier_logits_for_eval(model_name, clean_pred)
    adv_pred = classifier.predict(adv_batch[:image_count], batch_size=int(batch_size))
    adv_pred = align_classifier_logits_for_eval(model_name, adv_pred)

    clean_idx = np.argmax(clean_pred, axis=1)
    adv_idx = np.argmax(adv_pred, axis=1)
    clean_correct_mask = clean_idx == labels
    success_mask = np.logical_and.reduce(
        (clean_correct_mask, adv_idx != labels, ~victim_query_exhausted_mask)
    )
    success_indices = set(int(idx) for idx in np.nonzero(success_mask)[0].tolist())

    summary = {
        "npz_path": str(npz_path),
        "model_name": model_name,
        "image_count": int(image_count),
        "clean_correct_samples": int(np.sum(clean_correct_mask)),
        "victim_query_exhausted_samples": int(np.sum(victim_query_exhausted_mask)),
        "attack_success_samples": int(np.sum(success_mask)),
        "attack_success_unique_labels": int(len(set(labels[success_mask].astype(int).tolist()))),
    }

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return success_indices, summary


def _tail_process_text(text: str, max_chars: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def collect_attack_success_indices_from_npz_python(
    *,
    python_executable: str,
    attack_dir: Path,
    ground_truth_csv: str,
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[Set[int], Dict[str, object]]:
    helper_output = tempfile.NamedTemporaryFile(
        prefix="eval_prompt_transfer_npz_success_",
        suffix=".json",
        delete=False,
    )
    helper_output_path = Path(helper_output.name)
    helper_output.close()

    cmd = [
        str(Path(python_executable).expanduser()),
        str(Path(__file__).resolve()),
        "--attack_dir",
        str(attack_dir),
        "--ground_truth_csv",
        str(ground_truth_csv),
        "--model_name",
        str(model_name),
        "--device",
        str(device),
        "--batch_size",
        str(int(batch_size)),
        "--_npz_success_helper_output",
        str(helper_output_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "external npz success-label extraction failed with exit code {}.\n"
                "cmd: {}\nstdout:\n{}\nstderr:\n{}".format(
                    completed.returncode,
                    " ".join(cmd),
                    _tail_process_text(completed.stdout),
                    _tail_process_text(completed.stderr),
                )
            )
        if not helper_output_path.is_file() or helper_output_path.stat().st_size == 0:
            raise RuntimeError(
                "external npz success-label extraction did not write output.\n"
                "cmd: {}\nstdout:\n{}\nstderr:\n{}".format(
                    " ".join(cmd),
                    _tail_process_text(completed.stdout),
                    _tail_process_text(completed.stderr),
                )
            )
        payload = json.loads(helper_output_path.read_text(encoding="utf-8"))
    finally:
        try:
            helper_output_path.unlink()
        except FileNotFoundError:
            pass

    success_indices = set(int(idx) for idx in payload.get("success_sample_indices", []))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("external npz success-label extraction returned invalid summary")
    summary = dict(summary)
    summary["label_python"] = str(Path(python_executable).expanduser())
    summary["label_python_mode"] = "subprocess"
    return success_indices, summary


def write_npz_success_helper_output(args: argparse.Namespace) -> None:
    attack_dir = Path(args.attack_dir).expanduser().resolve()
    image_id_list, label_ori_list, _ = load_ground_truth(args.ground_truth_csv)
    success_model_name = infer_model_name_from_attack_dir(attack_dir) or resolve_model_name(
        args.model_name,
        attack_dir,
    )
    success_sample_indices, npz_success_summary = collect_attack_success_indices_from_npz(
        attack_dir=attack_dir,
        image_id_list=image_id_list,
        label_ori_list=label_ori_list,
        ground_truth_csv=args.ground_truth_csv,
        model_name=success_model_name,
        device=resolve_device(args.device),
        batch_size=int(args.batch_size),
    )

    output_path = Path(args._npz_success_helper_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "success_sample_indices": sorted(int(idx) for idx in success_sample_indices),
        "summary": npz_success_summary,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def prepare_eval_context(args: argparse.Namespace) -> Dict[str, object]:
    attack_dir = Path(args.attack_dir).expanduser().resolve()
    if not attack_dir.is_dir():
        raise FileNotFoundError(f"attack_dir not found: {attack_dir}")

    image_id_list, label_ori_list, _ = load_ground_truth(args.ground_truth_csv)
    category_name_by_label = load_category_names(args.ground_truth_csv)
    eval_labels = list(dict.fromkeys(int(label) for label in label_ori_list))
    if args.ground_label is not None:
        eval_labels = [label for label in eval_labels if int(label) == int(args.ground_label)]
    if int(args.max_classes) > 0:
        eval_labels = eval_labels[: int(args.max_classes)]

    class_dir_map = build_imagenet_class_dir_map(Path(args.imagenet_val_dir).expanduser().resolve())
    success_sample_indices = None
    success_source_used = "report"
    npz_success_summary = None
    success_source = str(getattr(args, "success_source", "auto")).strip().lower()
    npz_path = attack_dir / "adversarial_examples.npz"
    if success_source == "npz" or (success_source == "auto" and npz_path.is_file()):
        success_model_name = infer_model_name_from_attack_dir(attack_dir) or resolve_model_name(
            args.model_name,
            attack_dir,
        )
        label_python = str(getattr(args, "vim_small_label_python", "") or "").strip()
        if normalize_model_name(success_model_name) == "vim-small" and label_python:
            print(
                "[eval_prompt_transfer] extracting vim-small npz success labels with "
                f"external python: {label_python}"
            )
            success_sample_indices, npz_success_summary = collect_attack_success_indices_from_npz_python(
                python_executable=label_python,
                attack_dir=attack_dir,
                ground_truth_csv=args.ground_truth_csv,
                model_name=success_model_name,
                device=resolve_device(args.device),
                batch_size=int(args.batch_size),
            )
        else:
            success_sample_indices, npz_success_summary = collect_attack_success_indices_from_npz(
                attack_dir=attack_dir,
                image_id_list=image_id_list,
                label_ori_list=label_ori_list,
                ground_truth_csv=args.ground_truth_csv,
                model_name=success_model_name,
                device=resolve_device(args.device),
                batch_size=int(args.batch_size),
            )
        success_source_used = "npz"
        print(
            "[eval_prompt_transfer] using npz attack-success source | "
            f"model={success_model_name} | "
            f"samples={npz_success_summary['attack_success_samples']} | "
            f"labels={npz_success_summary['attack_success_unique_labels']}"
        )
    elif success_source != "auto" and success_source != "report":
        raise ValueError(f"unknown success_source: {success_source}")

    success_records = collect_success_records_by_csv_order(
        attack_dir=attack_dir,
        image_id_list=image_id_list,
        label_ori_list=label_ori_list,
        allowed_labels=eval_labels,
        artifact_mode=args.artifact_mode,
        success_sample_indices=success_sample_indices,
    )

    selected_labels = [label for label in eval_labels if label in success_records and label in class_dir_map]
    start_iteration = int(args.start_iteration)
    if start_iteration < 0:
        raise ValueError("--start_iteration must be >= 0")
    if start_iteration > len(selected_labels):
        raise ValueError(
            f"--start_iteration {start_iteration} exceeds selected label count {len(selected_labels)}"
        )

    return {
        "attack_dir": attack_dir,
        "category_name_by_label": category_name_by_label,
        "class_dir_map": class_dir_map,
        "success_records": success_records,
        "success_source_used": success_source_used,
        "npz_success_summary": npz_success_summary,
        "eval_labels": eval_labels,
        "selected_labels": selected_labels,
        "selected_labels_to_eval": selected_labels[start_iteration:],
        "start_iteration": start_iteration,
    }


def resolve_save_dir_under_attack_dir(attack_dir: Path, save_dir_arg: Optional[str]) -> Optional[Path]:
    if not save_dir_arg:
        return None
    save_dir = Path(save_dir_arg).expanduser()
    if save_dir.is_absolute():
        return save_dir.resolve()
    return (attack_dir / save_dir).resolve()


def build_empty_label_payload(
    *,
    label: int,
    ground_class_name: Optional[str],
    artifact_type: Optional[str],
    artifact_prompt_for_log: str,
    eval_size: int,
    generation_failures: int = 0,
) -> Dict[str, object]:
    empty_batch = np.empty((0, 3, int(eval_size), int(eval_size)), dtype=np.float32)
    return {
        "ground_label": int(label),
        "ground_class_name": ground_class_name,
        "artifact_type": artifact_type,
        "artifact_prompt_for_log": artifact_prompt_for_log,
        "clean_batch": empty_batch,
        "adv_batch": empty_batch.copy(),
        "label_batch": np.empty((0,), dtype=np.int64),
        "image_paths": [],
        "generation_failures": int(generation_failures),
    }


def generate_label_payload(
    args: argparse.Namespace,
    context: Dict[str, object],
    label: int,
    pipe,
) -> Dict[str, object]:
    category_name_by_label = context["category_name_by_label"]
    class_dir_map = context["class_dir_map"]
    success_records = context["success_records"]

    ground_class_name = category_name_by_label.get(int(label))
    class_dir = class_dir_map[int(label)]
    image_paths = list_image_files(class_dir, max_count=int(args.max_images_per_class))
    if len(image_paths) == 0:
        return build_empty_label_payload(
            label=int(label),
            ground_class_name=ground_class_name,
            artifact_type=None,
            artifact_prompt_for_log="",
            eval_size=int(args.eval_size),
        )

    record = success_records[int(label)]
    artifact_type = choose_attack_artifact_type(record, args.artifact_mode)
    if artifact_type is None:
        return build_empty_label_payload(
            label=int(label),
            ground_class_name=ground_class_name,
            artifact_type=None,
            artifact_prompt_for_log="",
            eval_size=int(args.eval_size),
        )

    render_artifact = context.get("prepared_render_artifact")
    if render_artifact is None:
        render_artifact = prepare_render_artifact(
            pipe=pipe,
            record=record,
            artifact_type=artifact_type,
            max_sequence_length=int(args.max_sequence_length),
        )

    class_clean_images: List[np.ndarray] = []
    class_adv_images: List[np.ndarray] = []
    class_labels: List[int] = []
    rendered_image_paths: List[str] = []
    class_generation_failures = 0
    render_batch_size = max(1, int(args.render_batch_size))

    def _append_rendered(image_path: Path, source_image: Image.Image, adv_image: Image.Image) -> None:
        class_clean_images.append(pil_to_nchw01(source_image, args.eval_size))
        class_adv_images.append(pil_to_nchw01(adv_image, args.eval_size))
        class_labels.append(int(label))
        rendered_image_paths.append(str(image_path))

    for start_idx in range(0, len(image_paths), render_batch_size):
        batch_paths = image_paths[start_idx : start_idx + render_batch_size]
        batch_source_images: List[Image.Image] = []
        for image_path in batch_paths:
            with Image.open(image_path) as opened:
                batch_source_images.append(opened.convert("RGB"))

        try:
            batch_adv_images = render_with_prompt_embeds_batch(
                pipe=pipe,
                source_images=batch_source_images,
                prompt_embeds=render_artifact["prompt_embeds"],
                render_size=int(args.render_size),
                num_inference_steps=int(args.num_inference_steps),
                max_sequence_length=int(args.max_sequence_length),
            )
            for image_path, source_image, adv_image in zip(batch_paths, batch_source_images, batch_adv_images):
                _append_rendered(image_path, source_image, adv_image)
            continue
        except Exception as exc:
            if len(batch_paths) == 1:
                class_generation_failures += 1
                logger.log(
                    f"generation failed label={label} image={batch_paths[0].name} "
                    f"sample={Path(record['sample_dir']).name}: {type(exc).__name__}: {exc}"
                )
                if is_cuda_oom_error(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if len(batch_paths) > 1:
                logger.log(
                    f"batch render fallback label={label} batch={len(batch_paths)} "
                    f"sample={Path(record['sample_dir']).name}: {type(exc).__name__}: {exc}"
                )
                if is_cuda_oom_error(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for image_path, source_image in zip(batch_paths, batch_source_images):
            try:
                adv_image = render_with_prompt_embeds_batch(
                    pipe=pipe,
                    source_images=[source_image],
                    prompt_embeds=render_artifact["prompt_embeds"],
                    render_size=int(args.render_size),
                    num_inference_steps=int(args.num_inference_steps),
                    max_sequence_length=int(args.max_sequence_length),
                )[0]
            except Exception as exc:
                class_generation_failures += 1
                logger.log(
                    f"generation failed label={label} image={image_path.name} "
                    f"sample={Path(record['sample_dir']).name}: {type(exc).__name__}: {exc}"
                )
                if is_cuda_oom_error(exc) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            _append_rendered(image_path, source_image, adv_image)

    if len(class_labels) == 0:
        return build_empty_label_payload(
            label=int(label),
            ground_class_name=ground_class_name,
            artifact_type=artifact_type,
            artifact_prompt_for_log=str(render_artifact["artifact_prompt_for_log"]),
            eval_size=int(args.eval_size),
            generation_failures=class_generation_failures,
        )

    return {
        "ground_label": int(label),
        "ground_class_name": ground_class_name,
        "artifact_type": artifact_type,
        "artifact_prompt_for_log": str(render_artifact["artifact_prompt_for_log"]),
        "clean_batch": np.stack(class_clean_images, axis=0).astype(np.float32),
        "adv_batch": np.stack(class_adv_images, axis=0).astype(np.float32),
        "label_batch": np.asarray(class_labels, dtype=np.int64),
        "image_paths": rendered_image_paths,
        "generation_failures": int(class_generation_failures),
    }


def get_render_cache_path(cache_dir: Path, label: int) -> Path:
    return cache_dir / f"label_{int(label):04d}.npz"


def clean_batch_saving_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() not in {"0", "false", "no", "off"}


def save_label_payload_cache(
    cache_dir: Path,
    payload: Dict[str, object],
    *,
    save_clean_batch: bool = True,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = get_render_cache_path(cache_dir, int(payload["ground_label"]))
    arrays = {
        "adv_batch": np.asarray(payload["adv_batch"], dtype=np.float32),
        "label_batch": np.asarray(payload["label_batch"], dtype=np.int64),
        "image_paths": np.asarray(payload["image_paths"], dtype=np.str_),
        "ground_label": np.asarray([int(payload["ground_label"])], dtype=np.int64),
        "ground_class_name": np.asarray(
            [str(payload.get("ground_class_name") or "")],
            dtype=np.str_,
        ),
        "artifact_type": np.asarray(
            [str(payload.get("artifact_type") or "")],
            dtype=np.str_,
        ),
        "artifact_prompt_for_log": np.asarray(
            [str(payload.get("artifact_prompt_for_log") or "")],
            dtype=np.str_,
        ),
        "generation_failures": np.asarray(
            [int(payload.get("generation_failures", 0))],
            dtype=np.int64,
        ),
        "clean_batch_saved": np.asarray([bool(save_clean_batch)], dtype=np.bool_),
    }
    if save_clean_batch:
        arrays["clean_batch"] = np.asarray(payload["clean_batch"], dtype=np.float32)
    np.savez(cache_path, **arrays)


def load_label_payload_cache(cache_dir: Path, label: int) -> Dict[str, object]:
    cache_path = get_render_cache_path(cache_dir, label)
    with np.load(cache_path, allow_pickle=False) as payload:
        ground_class_name = str(payload["ground_class_name"][0])
        artifact_type = str(payload["artifact_type"][0])
        clean_batch = (
            payload["clean_batch"].astype(np.float32, copy=False)
            if "clean_batch" in payload.files
            else None
        )
        return {
            "ground_label": int(payload["ground_label"][0]),
            "ground_class_name": ground_class_name or None,
            "artifact_type": artifact_type or None,
            "artifact_prompt_for_log": str(payload["artifact_prompt_for_log"][0]),
            "clean_batch": clean_batch,
            "adv_batch": payload["adv_batch"].astype(np.float32, copy=False),
            "label_batch": payload["label_batch"].astype(np.int64, copy=False),
            "image_paths": payload["image_paths"].astype(str).tolist(),
            "generation_failures": int(payload["generation_failures"][0]),
        }


def load_clean_batch_from_image_paths(
    image_paths: Sequence[str],
    eval_size: int,
) -> np.ndarray:
    clean_images: List[np.ndarray] = []
    for image_path_str in image_paths:
        with Image.open(Path(image_path_str)) as opened:
            clean_images.append(pil_to_nchw01(opened.convert("RGB"), int(eval_size)))
    if not clean_images:
        return np.empty((0, 3, int(eval_size), int(eval_size)), dtype=np.float32)
    return np.stack(clean_images, axis=0).astype(np.float32)


def is_valid_label_payload_cache(
    cache_path: Path,
    *,
    expected_count: int = 0,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            required = {"adv_batch", "label_batch", "image_paths", "ground_label"}
            if not required.issubset(set(payload.files)):
                return False
            rendered_count = int(payload["adv_batch"].shape[0])
            if rendered_count != int(payload["label_batch"].shape[0]):
                return False
            return int(expected_count) <= 0 or rendered_count == int(expected_count)
    except Exception:
        return False


def build_render_cache(
    args: argparse.Namespace,
    context: Dict[str, object],
    pipe,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_model_name = resolve_model_name(args.model_name, Path(context["attack_dir"]))
    total_labels = int(len(context["selected_labels"]))
    start_iteration = int(context["start_iteration"])
    rerender_cwor_only = bool(getattr(args, "rerender_cwor_only", False))
    success_records = context.get("success_records", {})
    _set_eval_process_title(
        "rc_all",
        cache_model_name,
        current_idx=start_iteration,
        total_count=total_labels,
    )
    for offset, label in enumerate(
        tqdm(
        context["selected_labels_to_eval"],
        desc="render_cache",
        initial=start_iteration,
        total=total_labels,
        )
    ):
        _set_eval_process_title(
            f"rc_{int(label):04d}",
            cache_model_name,
            current_idx=start_iteration + offset + 1,
            total_count=total_labels,
        )
        cache_path = get_render_cache_path(cache_dir, int(label))
        if bool(getattr(args, "resume_render_cache", False)) and is_valid_label_payload_cache(
            cache_path,
            expected_count=max(0, int(getattr(args, "max_images_per_class", 0))),
        ):
            continue
        if rerender_cwor_only:
            record = success_records.get(int(label)) if isinstance(success_records, dict) else None
            if not isinstance(record, dict) or not bool(record.get("is_cwor_success", False)):
                continue
        payload = generate_label_payload(args=args, context=context, label=int(label), pipe=pipe)
        save_label_payload_cache(
            cache_dir=cache_dir,
            payload=payload,
            save_clean_batch=clean_batch_saving_enabled(
                getattr(args, "save_clean_batch", "on")
            ),
        )
    _set_eval_process_title(
        "rc_done",
        cache_model_name,
        current_idx=total_labels,
        total_count=total_labels,
    )


def render_with_prompt(
    pipe,
    source_image: Image.Image,
    prompt: str,
    render_size: int,
    num_inference_steps: int,
    max_sequence_length: int,
) -> Image.Image:
    kwargs = {
        "prompt": [str(prompt)],
        "image": [resize_rgb(source_image, render_size)],
        "num_inference_steps": int(num_inference_steps),
        "max_sequence_length": int(max_sequence_length),
        "height": int(render_size),
        "width": int(render_size),
        "output_type": "pil",
    }
    output = pipe(**kwargs)
    images = extract_output_images(output)
    if len(images) == 0:
        raise RuntimeError("FLUX prompt render returned no images.")
    return images[-1].convert("RGB")


def render_with_cwor_snapshot(
    pipe,
    source_image: Image.Image,
    snapshot: Dict[str, object],
    render_size: int,
    num_inference_steps: int,
    max_sequence_length: int,
) -> Image.Image:
    raw_prompt_embeds = snapshot.get("t5_prompt_embeds")
    if not torch.is_tensor(raw_prompt_embeds):
        raw_prompt_embeds = snapshot.get("prompt_embeds")
    if not torch.is_tensor(raw_prompt_embeds):
        raise ValueError("CWOR snapshot does not contain prompt embeddings.")

    prompt_embeds = raw_prompt_embeds
    reference_prompt = str(snapshot.get("reference_prompt", "")).strip()
    if len(reference_prompt) > 0:
        ref_prompt_embeds, _ = pipe.encode_prompt(
            prompt=reference_prompt,
            max_sequence_length=int(max_sequence_length),
        )
        prompt_embeds = raw_prompt_embeds.to(
            device=ref_prompt_embeds.device,
            dtype=ref_prompt_embeds.dtype,
        )

    kwargs = {
        "prompt_embeds": prompt_embeds,
        "image": [resize_rgb(source_image, render_size)],
        "num_inference_steps": int(num_inference_steps),
        "max_sequence_length": int(max_sequence_length),
        "height": int(render_size),
        "width": int(render_size),
        "output_type": "pil",
    }
    output = pipe(**kwargs)
    images = extract_output_images(output)
    if len(images) == 0:
        raise RuntimeError("FLUX CWOR render returned no images.")
    return images[-1].convert("RGB")


def evaluate_transfer(
    args: argparse.Namespace,
    shared_context: Optional[Dict[str, object]] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    context = shared_context or prepare_eval_context(args)
    attack_dir = Path(context["attack_dir"])

    save_dir = resolve_save_dir_under_attack_dir(attack_dir, args.save_dir)
    if save_dir is None:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = attack_dir / f"eval_prompt_transfer_{str(today).replace('-', '')}{now}"
    save_dir.mkdir(parents=True, exist_ok=True)
    render_root = save_dir / "renders"
    if args.vis:
        render_root.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(save_dir))

    device = resolve_device(args.device)
    model_name = resolve_model_name(args.model_name, attack_dir)
    _maybe_disable_cudnn_sdpa_for_vim_small(model_name)
    total_labels = int(len(context["selected_labels"]))
    start_iteration = int(context["start_iteration"])
    _set_eval_process_title("all", model_name, current_idx=start_iteration, total_count=total_labels)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    logger.log("******* Evaluating prompt transfer for model: {} *******".format(model_name))
    logger.log(f"attack_dir: {attack_dir}")
    logger.log(f"save_dir: {save_dir}")
    logger.log(f"device: {device}")
    logger.log(f"model_name: {model_name}")
    logger.log(f"artifact_mode: {args.artifact_mode}")
    logger.log(f"success_source: {context.get('success_source_used', getattr(args, 'success_source', 'report'))}")
    if context.get("npz_success_summary") is not None:
        logger.log(f"npz_success_summary: {json.dumps(context['npz_success_summary'], ensure_ascii=False)}")
    logger.log(f"render_batch_size: {max(1, int(args.render_batch_size))}")
    logger.log(f"eval labels from NIPS ground truth (unique, csv order): {len(context['eval_labels'])}")
    logger.log(f"labels with successful attack artifacts: {len(context['success_records'])}")
    logger.log(f"labels selected for transfer eval: {len(context['selected_labels'])}")
    logger.log(
        "resume start_iteration: {} | labels to evaluate this run: {}".format(
            int(context["start_iteration"]),
            len(context["selected_labels_to_eval"]),
        )
    )
    if cache_dir is not None:
        logger.log(f"render cache dir: {cache_dir}")

    classifier = build_art_classifier(model_name, args.eval_size, device)
    pipe = None
    if cache_dir is None:
        pipe = load_flux_pipeline(
            model_path=str(args.flux_model_path),
            revision=resolve_flux_revision(args),
            device=device,
            hf_token=resolve_hf_token(args.hf_token),
            cpu_offload=bool(args.cpu_offload),
        )

    total_images = 0
    total_clean_correct = 0
    total_adv_correct = 0
    total_attack_success = 0
    total_generation_failures = 0
    total_vis_saved = 0
    per_class_results: List[Dict[str, object]] = []
    clean_correct_records: List[Dict[str, object]] = []
    attack_success_records: List[Dict[str, object]] = []
    evaluated_label_clean_accs: List[float] = []
    evaluated_label_adv_accs: List[float] = []
    evaluated_label_attack_success_rates: List[float] = []
    cached_clean_label_count = 0
    reloaded_clean_label_count = 0
    vis_max_images = max(0, int(args.vis_max_images))

    for offset, label in enumerate(
        tqdm(
        context["selected_labels_to_eval"],
        desc="class_eval",
        initial=start_iteration,
        total=total_labels,
        )
        ):
        label = int(label)
        _set_eval_process_title(
            f"{label:04d}",
            model_name,
            current_idx=start_iteration + offset + 1,
            total_count=total_labels,
        )

        if cache_dir is None:
            class_payload = generate_label_payload(args=args, context=context, label=label, pipe=pipe)
        else:
            class_payload = load_label_payload_cache(Path(cache_dir), label)

        ground_class_name = class_payload["ground_class_name"]
        artifact_type = class_payload["artifact_type"]
        artifact_prompt_for_log = class_payload["artifact_prompt_for_log"]
        clean_batch = class_payload["clean_batch"]
        adv_batch = class_payload["adv_batch"]
        label_batch = class_payload["label_batch"]
        image_paths = class_payload["image_paths"]

        total_generation_failures += int(class_payload["generation_failures"])
        class_vis_saved = 0

        if int(label_batch.shape[0]) == 0:
            per_class_results.append(
                {
                    "ground_label": label,
                    "ground_class_name": ground_class_name,
                    "artifact_type": artifact_type,
                    "clean_acc": None,
                    "adv_acc": None,
                    "saved_render_count": int(class_vis_saved),
                    "visualization_dir": str(render_root / f"label_{label:04d}") if args.vis else None,
                }
            )
            continue

        if clean_batch is None:
            clean_batch = load_clean_batch_from_image_paths(
                image_paths=image_paths,
                eval_size=int(args.eval_size),
            )
            reloaded_clean_label_count += 1
        else:
            cached_clean_label_count += 1
        if int(clean_batch.shape[0]) != int(label_batch.shape[0]):
            raise ValueError(
                "clean/image count mismatch for label {}: clean={} labels={}".format(
                    label,
                    int(clean_batch.shape[0]),
                    int(label_batch.shape[0]),
                )
            )

        combined_batch = np.concatenate((clean_batch, adv_batch), axis=0).astype(np.float32, copy=False)
        combined_pred = classifier.predict(combined_batch, batch_size=int(args.batch_size))
        combined_pred = align_classifier_logits_for_eval(model_name, combined_pred)
        split_idx = int(label_batch.shape[0])
        clean_pred = combined_pred[:split_idx]
        adv_pred = combined_pred[split_idx:]

        clean_idx = np.argmax(clean_pred, axis=1)
        adv_idx = np.argmax(adv_pred, axis=1)
        clean_correct_mask = clean_idx == label_batch
        attack_success_mask = np.logical_and(clean_correct_mask, adv_idx != label_batch)

        if args.vis:
            for vis_idx, (is_attack_success, image_path_str) in enumerate(zip(attack_success_mask.tolist(), image_paths)):
                if not is_attack_success:
                    continue
                if vis_max_images > 0 and class_vis_saved >= vis_max_images:
                    break

                image_path = Path(image_path_str)
                with Image.open(image_path) as opened:
                    source_image = opened.convert("RGB")
                rendered_image = nchw01_to_pil(adv_batch[vis_idx])
                save_visualization_artifacts(
                    save_dir=save_dir,
                    label=label,
                    image_path=image_path,
                    source_image=source_image,
                    rendered_image=rendered_image,
                    save_size=VIS_SAVE_SIZE,
                )
                logger.log(
                    "saved attack-success visualization | label={} | image={} | artifact={} | prompt={}".format(
                        label,
                        image_path.name,
                        artifact_type,
                        json.dumps(artifact_prompt_for_log, ensure_ascii=False),
                    )
                )
                class_vis_saved += 1
                total_vis_saved += 1

        class_clean_correct = int(np.sum(clean_idx == label_batch))
        class_adv_correct = int(np.sum(adv_idx == label_batch))
        class_attack_success = int(np.sum(attack_success_mask))
        class_total = int(label_batch.shape[0])

        for sample_offset, image_path_str in enumerate(image_paths):
            if not bool(clean_correct_mask[sample_offset]):
                continue
            sample_record = {
                "image_path": str(image_path_str),
                "edited_cache_path": (
                    str(get_render_cache_path(Path(cache_dir), label))
                    if cache_dir is not None
                    else None
                ),
                "edited_cache_index": int(sample_offset),
                "ground_label": int(label_batch[sample_offset]),
                "ground_class_name": ground_class_name,
                "clean_pred": int(clean_idx[sample_offset]),
                "adv_pred": int(adv_idx[sample_offset]),
                "attack_success": bool(attack_success_mask[sample_offset]),
                "artifact_type": artifact_type,
                "prompt": artifact_prompt_for_log,
            }
            clean_correct_records.append(sample_record)
            if bool(attack_success_mask[sample_offset]):
                attack_success_records.append(sample_record)

        total_images += class_total
        total_clean_correct += class_clean_correct
        total_adv_correct += class_adv_correct
        total_attack_success += class_attack_success

        class_attack_success_rate = None
        if class_clean_correct > 0:
            class_attack_success_rate = 100.0 * float(class_attack_success) / float(class_clean_correct)

        class_result = {
            "ground_label": label,
            "ground_class_name": ground_class_name,
            "artifact_type": artifact_type,
            "total_count": int(class_total),
            "clean_correct_count": int(class_clean_correct),
            "adv_correct_count": int(class_adv_correct),
            "attack_success_count": int(class_attack_success),
            "clean_acc": 100.0 * float(class_clean_correct) / float(class_total),
            "adv_acc": 100.0 * float(class_adv_correct) / float(class_total),
            "attack_success_rate": class_attack_success_rate,
            "saved_render_count": int(class_vis_saved),
            "visualization_dir": str(render_root / f"label_{label:04d}") if args.vis else None,
        }
        evaluated_label_clean_accs.append(float(class_result["clean_acc"]))
        evaluated_label_adv_accs.append(float(class_result["adv_acc"]))
        if class_attack_success_rate is not None:
            evaluated_label_attack_success_rates.append(float(class_attack_success_rate))
        per_class_results.append(class_result)
        logger.log(
            "label {} ({}) | clean acc: {:.2f} | adv acc: {:.2f} | asr(clean-correct only): {} | artifact: {}".format(
                label,
                ground_class_name or "unknown",
                class_result["clean_acc"],
                class_result["adv_acc"],
                (
                    "{:.2f}".format(class_attack_success_rate)
                    if class_attack_success_rate is not None
                    else "n/a"
                ),
                artifact_type,
            )
        )

    if total_images == 0:
        raise RuntimeError("no images were successfully evaluated")

    if len(evaluated_label_clean_accs) > 0:
        per_class_results.append(
            {
                "ground_label": "average",
                "artifact_type": "average",
                "clean_acc": float(np.mean(evaluated_label_clean_accs)),
                "adv_acc": float(np.mean(evaluated_label_adv_accs)),
                "attack_success_rate": (
                    float(np.mean(evaluated_label_attack_success_rates))
                    if len(evaluated_label_attack_success_rates) > 0
                    else None
                ),
            }
        )

    summary = {
        "total_image_count": int(total_images),
        "clean_correct_count": int(total_clean_correct),
        "attack_success_count": int(total_attack_success),
        "clean_acc": 100.0 * float(total_clean_correct) / float(total_images),
        "adv_acc": 100.0 * float(total_adv_correct) / float(total_images),
        "attack_success_rate": (
            100.0 * float(total_attack_success) / float(total_clean_correct)
            if total_clean_correct > 0
            else None
        ),
        "start_iteration": int(context["start_iteration"]),
        "selected_label_count_total": int(len(context["selected_labels"])),
        "selected_label_count_evaluated_this_run": int(len(context["selected_labels_to_eval"])),
        "visualizations_enabled": bool(args.vis),
        "visualization_root": str(render_root) if args.vis else None,
        "saved_visualization_count": int(total_vis_saved),
        "vis_max_images": int(vis_max_images),
        "generation_failures": int(total_generation_failures),
        "cached_clean_label_count": int(cached_clean_label_count),
        "reloaded_clean_label_count": int(reloaded_clean_label_count),
        "attack_success_numerator_definition": (
            "clean_pred == ground_label and adv_pred != ground_label"
        ),
        "attack_success_denominator_definition": "clean_pred == ground_label",
    }

    summary_path = save_dir / "summary.json"
    per_class_path = save_dir / "per_class_results.json"
    clean_correct_path = save_dir / "clean_correct_samples.jsonl"
    attack_success_path = save_dir / "attack_success_samples.jsonl"
    summary["clean_correct_samples_path"] = str(clean_correct_path)
    summary["attack_success_samples_path"] = str(attack_success_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class_path.write_text(json.dumps(per_class_results, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_correct_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in clean_correct_records
        ),
        encoding="utf-8",
    )
    attack_success_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in attack_success_records
        ),
        encoding="utf-8",
    )
    _set_eval_process_title("done", model_name, current_idx=total_labels, total_count=total_labels)

    logger.log("***********************************************************")
    logger.log(
        "clean acc: {:.2f}, adv acc: {:.2f}, asr(clean-correct only): {}".format(
            summary["clean_acc"],
            summary["adv_acc"],
            (
                "{:.2f}".format(summary["attack_success_rate"])
                if summary["attack_success_rate"] is not None
                else "n/a"
            ),
        )
    )
    logger.log(f"summary saved to: {summary_path}")
    logger.log(f"per-class results saved to: {per_class_path}")
    logger.log("***********************************************************")

    del classifier
    if pipe is not None:
        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()

    if args._npz_success_helper_output:
        write_npz_success_helper_output(args)
        return

    if args.render_only:
        context = prepare_eval_context(args)
        attack_dir = Path(args.attack_dir).expanduser().resolve()
        if args.save_dir:
            render_root = resolve_save_dir_under_attack_dir(attack_dir, args.save_dir)
        else:
            today = datetime.date.today()
            now = time.strftime("_%H%M%S")
            render_root = attack_dir / f"render_cache_{str(today).replace('-', '')}{now}"
        render_root.mkdir(parents=True, exist_ok=True)

        logger.configure(dir=str(render_root))
        device = resolve_device(args.device)
        cache_model_name = (
            normalize_model_name(args.model_name)
            or infer_model_name_from_attack_dir(attack_dir)
            or "render_cache"
        )
        _set_eval_process_title("render_only_init", cache_model_name)
        logger.log("******* Building render cache only *******")
        logger.log(f"attack_dir: {attack_dir}")
        logger.log(f"save_dir: {render_root}")
        logger.log(f"device: {device}")
        logger.log(f"artifact_mode: {args.artifact_mode}")
        logger.log(f"success_source: {context.get('success_source_used', getattr(args, 'success_source', 'report'))}")
        if context.get("npz_success_summary") is not None:
            logger.log(f"npz_success_summary: {json.dumps(context['npz_success_summary'], ensure_ascii=False)}")
        logger.log(f"render_batch_size: {max(1, int(args.render_batch_size))}")
        logger.log(f"rerender_cwor_only: {bool(args.rerender_cwor_only)}")
        logger.log(f"labels selected for render cache: {len(context['selected_labels'])}")

        pipe = load_flux_pipeline(
            model_path=str(args.flux_model_path),
            revision=resolve_flux_revision(args),
            device=device,
            hf_token=resolve_hf_token(args.hf_token),
            cpu_offload=bool(args.cpu_offload),
        )
        render_cache_dir = render_root / "_render_cache"
        build_render_cache(args=args, context=context, pipe=pipe, cache_dir=render_cache_dir)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _set_eval_process_title("render_only_done", cache_model_name)
        logger.log(f"render cache saved to: {render_cache_dir}")
        print(f"[eval_prompt_transfer] render cache saved to: {render_cache_dir}")
        return

    # Fill this list in main when you want to evaluate prompt transfer across
    # multiple classifier models in one run. Leave it empty to keep the
    # original single-model behavior driven by --model_name / attack_dir.
    model_names: List[str] = [
        # "resnet50",
        # "wrn50",
        # "inception_v3", 
        # "convnext",
        # "vgg19",
        # "vit",
        # "swin",
        # "deit",
        # "adv_inc",
        # "adv_res",
        # "vim-small",
        # "mambavision"

    ]

    normalized_model_names = []
    for model_name in model_names:
        normalized = normalize_model_name(model_name)
        if normalized is not None:
            normalized_model_names.append(normalized)
    normalized_model_names = list(dict.fromkeys(normalized_model_names))

    if len(normalized_model_names) == 0:
        evaluate_transfer(args, shared_context=prepare_eval_context(args))
        return

    context = prepare_eval_context(args)
    attack_dir = Path(args.attack_dir).expanduser().resolve()
    if args.save_dir:
        multi_model_root = resolve_save_dir_under_attack_dir(attack_dir, args.save_dir)
    else:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        multi_model_root = attack_dir / f"eval_prompt_transfer_multi_{str(today).replace('-', '')}{now}"
    multi_model_root.mkdir(parents=True, exist_ok=True)

    logger.configure(dir=str(multi_model_root))
    device = resolve_device(args.device)
    cache_model_name = (
        normalize_model_name(args.model_name)
        or infer_model_name_from_attack_dir(attack_dir)
        or "render_cache"
    )
    _set_eval_process_title("render_cache_init", cache_model_name)
    pipe = load_flux_pipeline(
        model_path=str(args.flux_model_path),
        revision=resolve_flux_revision(args),
        device=device,
        hf_token=resolve_hf_token(args.hf_token),
        cpu_offload=bool(args.cpu_offload),
    )
    render_cache_dir = multi_model_root / "_render_cache"
    build_render_cache(args=args, context=context, pipe=pipe, cache_dir=render_cache_dir)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _set_eval_process_title("render_cache_ready", cache_model_name)

    all_summaries: Dict[str, Dict[str, object]] = {}
    for model_name in normalized_model_names:
        run_args = argparse.Namespace(**vars(args))
        run_args.model_name = model_name
        run_args.save_dir = str(multi_model_root / model_name)
        all_summaries[model_name] = evaluate_transfer(
            run_args,
            shared_context=context,
            cache_dir=render_cache_dir,
        )

    summary_path = multi_model_root / "multi_model_summary.json"
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval_prompt_transfer] multi-model summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

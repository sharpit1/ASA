import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SAVED_IMAGE_SIZE = 224
SUPPORTED_ATTACK_MODES = ("vlm", "and")
SIGLIP2_IMAGENET_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_IMAGENET_MODEL_ALIASES = frozenset(
    {
        SIGLIP2_IMAGENET_MODEL_ID,
        "siglip2-base-patch16-224",
        "siglip2",
    }
)
OPENAI_CLIP_IMAGENET_MODEL_ID = "openai/clip-vit-base-patch16"
OPENAI_CLIP_IMAGENET_MODEL_ALIASES = frozenset(
    {
        OPENAI_CLIP_IMAGENET_MODEL_ID,
        "clip-vit-base-patch16",
        "clip-vit-b-16",
    }
)


def _is_siglip2_imagenet_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in SIGLIP2_IMAGENET_MODEL_ALIASES


def _is_openai_clip_imagenet_model(model_name: str) -> bool:
    return str(model_name).strip().lower() in OPENAI_CLIP_IMAGENET_MODEL_ALIASES


def _imagenet_zero_shot_prompt_groups() -> List[List[str]]:
    """Return the official Big Vision CLIP prompt ensemble in ILSVRC order."""

    from imagenet_clip_zeroshot import imagenet_clip_prompt_groups

    return imagenet_clip_prompt_groups()


def _openai_clip_zero_shot_prompt_groups() -> List[List[str]]:
    """Return OpenAI's original 80-template ImageNet prompt ensemble."""

    from imagenet_clip_zeroshot import openai_clip_prompt_groups

    return openai_clip_prompt_groups()


class _PromptEnsembleImageNetClassifier(torch.nn.Module):
    """Base predictor for Hugging Face CLIP-family ImageNet classifiers."""

    text_context_length = 64

    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        text_batch_size: int = 256,
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype="auto")
        self.model.to(self.device)
        self.model.eval()

        prompt_groups = self._prompt_groups()
        if len(prompt_groups) != 1000:
            raise ValueError(
                f"{type(self).__name__} ImageNet class count mismatch: "
                f"{len(prompt_groups)}"
            )
        prompts_per_class = len(prompt_groups[0])
        if prompts_per_class <= 0 or any(
            len(group) != prompts_per_class for group in prompt_groups
        ):
            raise ValueError(
                f"{type(self).__name__} ImageNet prompt groups must have equal size"
            )
        prompts = [prompt for group in prompt_groups for prompt in group]
        class_feature_sums: Optional[torch.Tensor] = None
        class_feature_counts = torch.zeros(
            len(prompt_groups),
            dtype=torch.float32,
            device=self.device,
        )
        effective_batch_size = max(1, int(text_batch_size))
        with torch.inference_mode():
            for start in range(0, len(prompts), effective_batch_size):
                end = min(start + effective_batch_size, len(prompts))
                tokens = self.processor.tokenizer(
                    prompts[start:end],
                    padding="max_length",
                    truncation=True,
                    max_length=int(self.text_context_length),
                    return_tensors="pt",
                )
                tokens = {
                    key: value.to(self.device)
                    for key, value in tokens.items()
                    if isinstance(value, torch.Tensor)
                }
                features = self.model.get_text_features(**tokens)
                features = F.normalize(features, dim=-1)
                if class_feature_sums is None:
                    class_feature_sums = torch.zeros(
                        len(prompt_groups),
                        features.shape[-1],
                        dtype=torch.float32,
                        device=features.device,
                    )
                class_indices = (
                    torch.arange(start, end, device=features.device)
                    // prompts_per_class
                )
                class_feature_sums.index_add_(
                    0,
                    class_indices,
                    features.float(),
                )
                class_feature_counts.index_add_(
                    0,
                    class_indices,
                    torch.ones(
                        end - start,
                        dtype=torch.float32,
                        device=features.device,
                    ),
                )
        if class_feature_sums is None or not torch.all(
            class_feature_counts == float(prompts_per_class)
        ):
            raise ValueError(
                f"{type(self).__name__} ImageNet prompt feature count mismatch: "
                f"expected {prompts_per_class} prompts per class"
            )
        cached_text_features = F.normalize(
            class_feature_sums / class_feature_counts.unsqueeze(1),
            dim=-1,
        )
        self.register_buffer(
            "text_features",
            cached_text_features,
            persistent=False,
        )
        # Victim queries only need the vision tower after the fixed ImageNet
        # prompt bank has been encoded. Releasing the text tower saves substantial
        # accelerator memory when a diffusion generator shares the same device.
        if hasattr(self.model, "text_model"):
            self.model.text_model = None

    @staticmethod
    def _prompt_groups() -> List[List[str]]:
        raise NotImplementedError

    def _scale_logits(self, cosine_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_images(self, image_01: np.ndarray) -> Dict[str, torch.Tensor]:
        images_hwc = [
            np.asarray(image, dtype=np.float32).transpose(1, 2, 0)
            for image in image_01
        ]
        processed = self.processor.image_processor(
            images=images_hwc,
            do_rescale=False,
            do_resize=False,
            return_tensors="pt",
        )
        return {
            key: value.to(self.device)
            for key, value in processed.items()
            if isinstance(value, torch.Tensor)
        }

    def predict(self, image_01: np.ndarray, batch_size: int = 1) -> np.ndarray:
        images = np.asarray(image_01, dtype=np.float32)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"{type(self).__name__} expects NCHW RGB input, "
                f"got shape={tuple(images.shape)}"
            )
        if not np.all(np.isfinite(images)):
            raise ValueError(f"{type(self).__name__} input contains non-finite values")

        scores: List[np.ndarray] = []
        chunk_size = max(1, int(batch_size))
        self.eval()
        with torch.inference_mode():
            for start in range(0, images.shape[0], chunk_size):
                vision_inputs = self._prepare_images(
                    np.clip(images[start : start + chunk_size], 0.0, 1.0)
                )
                image_features = self.model.get_image_features(**vision_inputs)
                image_features = F.normalize(image_features, dim=-1)
                text_features = self.text_features.to(dtype=image_features.dtype)
                cosine_logits = image_features @ text_features.transpose(0, 1)
                scores.append(
                    self._scale_logits(cosine_logits).float().cpu().numpy()
                )
        return np.concatenate(scores, axis=0)


class _Siglip2ImageNetClassifier(_PromptEnsembleImageNetClassifier):
    """Expose paper-style SigLIP2 ImageNet scores through `predict`."""

    text_context_length = 64

    @staticmethod
    def _prompt_groups() -> List[List[str]]:
        return _imagenet_zero_shot_prompt_groups()

    def _scale_logits(self, cosine_logits: torch.Tensor) -> torch.Tensor:
        return (
            cosine_logits * self.model.logit_scale.exp()
            + self.model.logit_bias
        )


class _OpenAIClipImageNetClassifier(_PromptEnsembleImageNetClassifier):
    """Expose original-paper CLIP ImageNet scores through `predict`."""

    text_context_length = 77

    @staticmethod
    def _prompt_groups() -> List[List[str]]:
        return _openai_clip_zero_shot_prompt_groups()

    def _scale_logits(self, cosine_logits: torch.Tensor) -> torch.Tensor:
        # OpenAI's ImageNet notebook uses a fixed 100x cosine score.
        return 100.0 * cosine_logits


class VictimModelAdapter:
    """Black-box ImageNet classifier adapter shared by the three generators."""

    IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, *, model_name: str, device: str, objective_mode: str):
        self.model_name = str(model_name)
        self.device = torch.device(str(device))
        self.objective_mode = str(objective_mode).strip().lower()
        self.label: Optional[int] = None
        self.input_res = SAVED_IMAGE_SIZE
        self._evaluation_callback = None
        self._evaluation_attempt_count = 0
        self._last_evaluation_payload: Optional[Dict[str, object]] = None

        if _is_siglip2_imagenet_model(self.model_name):
            self.model = _Siglip2ImageNetClassifier(
                model_id=SIGLIP2_IMAGENET_MODEL_ID,
                device=self.device,
            )
            self.f_model = self.model
        elif _is_openai_clip_imagenet_model(self.model_name):
            self.model = _OpenAIClipImageNetClassifier(
                model_id=OPENAI_CLIP_IMAGENET_MODEL_ID,
                device=self.device,
            )
            self.f_model = self.model
        else:
            from art.estimators.classification import PyTorchClassifier
            from eval.attacked_models import model_selection

            self.model = model_selection(self.model_name).to(self.device)
            self.f_model = PyTorchClassifier(
                model=self.model,
                clip_values=(0, 1),
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=(3, self.input_res, self.input_res),
                nb_classes=1000,
                preprocessing=(self.IMAGENET_MEAN_NP, self.IMAGENET_STD_NP),
                device_type="gpu" if self.device.type == "cuda" else "cpu",
            )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def set_label(self, label: int) -> None:
        self.label = int(label)

    def set_evaluation_callback(self, callback) -> None:
        """Register a lightweight hook invoked after every victim-model query."""

        self._evaluation_callback = callback

    def reset_evaluation_attempt_count(self) -> None:
        """Start a new sample-scoped count of victim inference attempts."""

        self._evaluation_attempt_count = 0
        self._last_evaluation_payload = None

    def get_last_evaluation_payload(self) -> Optional[Dict[str, object]]:
        """Return an isolated copy of the artifacts from the latest completed query."""

        payload = getattr(self, "_last_evaluation_payload", None)
        if payload is None:
            return None
        copied: Dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, Image.Image):
                copied[key] = value.copy()
            elif isinstance(value, np.ndarray):
                copied[key] = np.array(value, copy=True, order="C")
            elif isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().clone()
            else:
                copied[key] = value
        return copied

    def _prediction_index_offset(self) -> int:
        return 1 if self.model_name in {"adv_res", "adv_inc"} else 0

    def _normalize_prediction_index(self, pred_idx: int) -> int:
        return int(pred_idx) - self._prediction_index_offset()

    def _resolve_classifier_index(self, label_idx: int, num_classes: int) -> int:
        resolved_idx = int(label_idx) + self._prediction_index_offset()
        if not 0 <= resolved_idx < int(num_classes):
            raise ValueError(
                "classifier label index out of range after offset: "
                f"label_idx={label_idx}, resolved_idx={resolved_idx}, num_classes={num_classes}"
            )
        return resolved_idx

    def _preprocess(self, image_01: torch.Tensor) -> np.ndarray:
        if image_01.ndim == 3:
            image_01 = image_01.unsqueeze(0)
        x = image_01.float()
        if x.max().item() > 1.0:
            x = x / 255.0
        x = torch.clamp(x, 0.0, 1.0)
        x = F.interpolate(
            x,
            size=(self.input_res, self.input_res),
            mode=(
                "bicubic"
                if _is_openai_clip_imagenet_model(self.model_name)
                else "bilinear"
            ),
            align_corners=False,
            antialias=True,
        )
        x = torch.clamp(x, 0.0, 1.0)
        return x.detach().cpu().numpy().astype(np.float32)

    def objective_and_stats(self, image_01: torch.Tensor, target_label: Optional[int] = None):
        if self.label is None:
            raise ValueError("classifier label is not set for victim model")

        self._last_evaluation_payload = None
        x_np = self._preprocess(image_01)
        self._evaluation_attempt_count = int(
            getattr(self, "_evaluation_attempt_count", 0)
        ) + 1
        logits_or_scores = np.asarray(
            self.f_model.predict(x_np, batch_size=max(1, int(x_np.shape[0]))),
            dtype=np.float32,
        )
        if logits_or_scores.ndim != 2 or int(logits_or_scores.shape[-1]) <= 1:
            raise ValueError(f"invalid classifier output shape: {logits_or_scores.shape}")

        looks_like_prob = bool(
            np.all(logits_or_scores >= 0.0)
            and np.all(logits_or_scores <= 1.0)
            and np.allclose(logits_or_scores.sum(axis=1), 1.0, atol=1e-3, rtol=1e-3)
        )
        if looks_like_prob:
            probs_np = logits_or_scores
            logits_np = np.log(np.clip(probs_np, 1e-12, 1.0))
        else:
            logits_np = logits_or_scores
            shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
            exp_shifted = np.exp(shifted)
            probs_np = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        num_classes = int(logits_np.shape[-1])
        classifier_idx = self._resolve_classifier_index(self.label, num_classes)
        logits = torch.from_numpy(logits_np).to(dtype=torch.float32)
        labels = torch.full((logits.shape[0],), classifier_idx, dtype=torch.long)
        ce = F.cross_entropy(logits, labels)

        true_logits = logits_np[:, classifier_idx]
        non_true_logits = logits_np.copy()
        non_true_logits[:, classifier_idx] = -np.inf
        logit_margin = float(np.mean(np.max(non_true_logits, axis=1) - true_logits))
        if self.objective_mode == "ce_max":
            objective = float(ce.item())
        elif self.objective_mode in {"ce_min", "logit_max"}:
            objective = float(-ce.item())
        elif self.objective_mode == "logit_margin_max":
            objective = logit_margin
        else:
            raise ValueError(f"unsupported objective_mode: {self.objective_mode}")

        mean_logits = np.mean(logits_np, axis=0)
        raw_pred_label = int(np.argmax(mean_logits))
        mean_probs = np.mean(probs_np, axis=0)
        stats = {
            "pred_idx": self._normalize_prediction_index(raw_pred_label),
            "pred_conf": float(np.max(mean_probs)),
            "target_conf": float(mean_probs[classifier_idx]),
            "pred_logit": float(mean_logits[raw_pred_label]),
            "target_logit": float(mean_logits[classifier_idx]),
            "ce": float(ce.item()),
            "logit_margin": logit_margin,
        }
        if target_label is not None and 0 <= int(target_label) < num_classes:
            target_idx = self._resolve_classifier_index(int(target_label), num_classes)
            stats.update(
                target_label=int(target_label),
                target_label_conf=float(mean_probs[target_idx]),
                target_label_logit=float(mean_logits[target_idx]),
            )
        classifier_input_float32 = np.array(
            x_np[0],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        evaluated_array = (
            np.clip(classifier_input_float32, 0.0, 1.0).transpose(1, 2, 0) * 255.0
        ).round().astype(np.uint8)
        evaluation_payload: Dict[str, object] = {
            **stats,
            "candidate_objective": float(objective),
            "candidate_classifier_input_float32": classifier_input_float32,
            "candidate_classifier_image": Image.fromarray(evaluated_array, mode="RGB"),
            "candidate_classifier_image_size": int(self.input_res),
            "candidate_image_source": "victim_classifier_input",
            "victim_query_attempt_count": int(self._evaluation_attempt_count),
        }
        self._last_evaluation_payload = evaluation_payload
        callback = self._evaluation_callback
        if callable(callback):
            callback(self.get_last_evaluation_payload())
        return objective, stats


def parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize_attack_mode(raw: object) -> str:
    mode = str(raw or "vlm").strip().lower()
    if mode not in SUPPORTED_ATTACK_MODES:
        raise ValueError(f"attack_mode must be one of: {', '.join(SUPPORTED_ATTACK_MODES)}")
    return mode


def configure_attack_mode(cfg: argparse.Namespace) -> None:
    """Derive all internal AND settings from the single public attack_mode."""

    cfg.attack_mode = normalize_attack_mode(getattr(cfg, "attack_mode", "vlm"))
    and_enabled = cfg.attack_mode == "and"
    cfg.cwor_enable = and_enabled
    cfg.cwor_reference_mode = "base_prompt"
    cfg.cwor_mode = "untargeted"
    cfg.cwor_target_label = None
    cfg.cwor_embed_inject_mode = "both"
    cfg.cwor_feedback_merge_mode = "accumulate"
    cfg.cwor_accumulate_secondary_ortho = False
    cfg.cwor_accumulate_update_if_improved_only = False
    cfg.cwor_accumulate_delta_use_basis_logit_without_secondary_ortho = False
    cfg.cwor_step_prompt_candidate_ortho = False
    cfg.cwor_step_prompt_flip_alpha_on_regression = False
    cfg.cwor_embed_subtract_scale_by_step = False
    cfg.flux2_strategy_cwor_delta_mode = "score"
    cfg.flux2_strategy_cwor_merge_mode = "and" if and_enabled else "weighted"
    if and_enabled:
        cfg.gcg_candidate_source = "gemma_scene_vocab"
        cfg.gcg_scene_vocab_prompts_per_strategy = max(
            1, int(getattr(cfg, "gcg_scene_vocab_prompts_per_strategy", 0))
        )


def validate_passthrough_core_args(args: List[str]) -> None:
    """Reject removed attack controls when a runner is invoked directly."""

    removed_exact = {
        "--inversion_prompt",
        "--run_mode",
        "--fixed_prompt",
        "--save_intermediate",
        "--save_intermediate_interval",
        "--save_candidate_strips",
        "--capture_classifier_tile_image",
        "--gcg_save_intermediate",
        "--gcg_save_intermediate_interval",
        "--gcg_save_candidate_strips",
        "--gcg_capture_classifier_tile_image",
        "--gcg_early_stop_on_cwor_success_only",
        "--latent_nudging_scalar",
    }
    for raw in args:
        option = str(raw).split("=", 1)[0]
        if (
            option in removed_exact
            or option.startswith("--cwor_")
            or option.startswith("--flux2_strategy_cwor_")
        ):
            raise ValueError(f"removed attack option is not supported: {option}")


def _natural_key(text: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(text))]


def _parse_sample_indices_text(text: str) -> List[int]:
    text = str(text or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, int):
        values = [parsed]
    elif isinstance(parsed, list):
        values = parsed
    else:
        values = re.findall(r"-?\d+", text)
    indices: List[int] = []
    seen = set()
    for value in values:
        idx = int(value)
        if idx < 0:
            raise ValueError(f"sample index must be >= 0: {idx}")
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def resolve_sample_indices(sample_indices: str = "", sample_indices_file: str = "") -> List[int]:
    sources: List[Tuple[str, str]] = []
    if str(sample_indices or "").strip():
        sources.append(("sample_indices", str(sample_indices)))
    if str(sample_indices_file or "").strip():
        path = Path(str(sample_indices_file)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"sample indices file not found: {path}")
        sources.append((f"sample_indices_file:{path}", path.read_text(encoding="utf-8")))
    result: List[int] = []
    seen = set()
    for source_name, source_text in sources:
        try:
            parsed = _parse_sample_indices_text(source_text)
        except Exception as exc:
            raise ValueError(f"failed to parse {source_name}") from exc
        for idx in parsed:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
    return result


def resolve_run_root(cfg: argparse.Namespace, run_name: str) -> Path:
    indices_file = str(getattr(cfg, "sample_indices_file", "") or "").strip()
    if indices_file:
        return Path(indices_file).expanduser().resolve().parent
    return (
        Path(str(cfg.output_root)).expanduser().resolve()
        / str(cfg.dataset_name)
        / str(cfg.victim_model)
        / str(run_name)
    )


def load_nips_ground_truth(dataset_root: Path):
    csv_path = dataset_root / "images.csv"
    image_ids: List[str] = []
    true_labels: List[int] = []
    target_labels: List[int] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=","):
            image_ids.append(str(row["ImageId"]))
            true_labels.append(int(row["TrueLabel"]) - 1)
            target_labels.append(int(row["TargetClass"]) - 1)
    order = sorted(range(len(image_ids)), key=lambda idx: _natural_key(image_ids[idx]))
    return (
        [image_ids[idx] for idx in order],
        [true_labels[idx] for idx in order],
        [target_labels[idx] for idx in order],
    )


def iter_nips_metadata_batches(
    image_ids: Sequence[str],
    true_labels: Sequence[int],
    target_labels: Sequence[int],
    *,
    batch_size: int,
):
    """Yield aligned CSV metadata batches without decoding dataset images."""

    lengths = (len(image_ids), len(true_labels), len(target_labels))
    if len(set(lengths)) != 1:
        raise ValueError(
            "NIPS metadata length mismatch: "
            f"image_ids={lengths[0]} true_labels={lengths[1]} target_labels={lengths[2]}"
        )
    size = max(1, int(batch_size))
    for start in range(0, lengths[0], size):
        end = min(lengths[0], start + size)
        yield (
            image_ids[start:end],
            true_labels[start:end],
            target_labels[start:end],
        )


def resolve_optional_hf_token(explicit: str) -> str:
    token = str(explicit or "").strip()
    if token:
        return token
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def resolve_hf_token(explicit: str) -> str:
    token = resolve_optional_hf_token(explicit)
    if not token:
        raise ValueError("HF token is required. Pass --hf_token or set HF_TOKEN.")
    return token


def apply_manual_seed(manual_seed: int) -> None:
    seed = int(manual_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def prepare_model_input(src_path: Path, dst_path: Path, *, height: int, width: int) -> None:
    """Create a temporary generator condition image; callers must delete it after the sample."""

    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"invalid resize shape: height={height} width={width}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        image.convert("RGB").resize((int(width), int(height)), Image.LANCZOS).save(dst_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NIPS black-box robustness evaluation runner.")
    parser.add_argument("--dataset_root", default="data/nips2017")
    parser.add_argument("--dataset_name", default="nips2017")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--sample_indices", default="")
    parser.add_argument("--sample_indices_file", default="")
    parser.add_argument("--image_size", type=int, default=SAVED_IMAGE_SIZE)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--victim_model", default="resnet50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier_objective", default="ce_max")
    parser.add_argument("--manual_seed", type=int, default=None)
    parser.add_argument("--hf_token", default="")
    parser.add_argument("--prompt", default="a photo of <class> in the background")
    parser.add_argument("--gcg_word", default="background")
    parser.add_argument("--gcg_occurrence", type=int, default=0)
    parser.add_argument("--gcg_steps", type=int, default=10)
    parser.add_argument("--gcg_batch_size", type=int, default=64)
    parser.add_argument("--max_victim_queries", type=int, default=100)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--gcg_scene_vocab_size", type=int, default=100)
    parser.add_argument("--gcg_scene_vocab_prompts_per_strategy", type=int, default=0)
    parser.add_argument("--gcg_scene_vocab_enabled_strategies", default="all")
    parser.add_argument("--gcg_slot_candidate_max_words", type=int, default=5)
    parser.add_argument("--gcg_scene_feedback_limit", type=int, default=1000)
    parser.add_argument(
        "--class_ablation",
        type=parse_bool_flag,
        default=False,
        help=(
            "Remove explicit textual target-class guidance while keeping the "
            "class-dependent victim objective unchanged."
        ),
    )
    parser.add_argument("--gcg_candidate_source", default="vlm_query")
    parser.add_argument("--attack_mode", choices=SUPPORTED_ATTACK_MODES, default="vlm")
    parser.add_argument("--scene_vlm_question", default="What is the background scene in this image? Answer in 1 word.")
    parser.add_argument("--scene_fallback", default="outdoor")
    parser.add_argument("--wandb_enable", type=parse_bool_flag, default=False)
    parser.add_argument("--wandb_project", default="vlm-robustness")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--wandb_mode", default="auto")
    parser.add_argument("--wandb_log_every", type=int, default=1)
    parser.add_argument("--wandb_api_key", default="")
    parser.add_argument("--wandb_api_key_file", default="")
    parser.set_defaults(flux2_strategy_cwor_delta_mode="score")
    parser.add_argument(
        "--saved_image_size",
        type=int,
        choices=[SAVED_IMAGE_SIZE],
        default=SAVED_IMAGE_SIZE,
        help="Successful attack images are always stored at classifier resolution (224x224).",
    )
    return parser


def build_core_cli(
    *,
    cfg: argparse.Namespace,
    hf_token: str,
    sample_image_path: Path,
    output_path: Path,
    report_path: Path,
    classifier_label: int,
    sample_target_label: Optional[int],
    sample_run_name: str,
) -> List[str]:
    del sample_target_label
    cli = [
        "--hf_token", hf_token,
        "--classifier_mode", "black-box",
        "--classifier_name", str(cfg.victim_model),
        "--classifier_label", str(int(classifier_label)),
        "--classifier_objective", str(cfg.classifier_objective),
        "--device", str(cfg.device),
        "--height", str(int(cfg.height)),
        "--width", str(int(cfg.width)),
        "--num_inference_steps", str(int(cfg.num_inference_steps)),
        "--input_img_path", str(sample_image_path),
        "--output_path", str(output_path),
        "--report_path", str(report_path),
        "--prompt", str(cfg.prompt),
        "--gcg_word", str(cfg.gcg_word),
        "--gcg_occurrence", str(int(cfg.gcg_occurrence)),
        "--gcg_steps", str(int(cfg.gcg_steps)),
        "--gcg_batch_size", str(int(cfg.gcg_batch_size)),
        "--max_victim_queries", str(int(cfg.max_victim_queries)),
        "--gcg_scene_vocab_size", str(int(cfg.gcg_scene_vocab_size)),
        "--gcg_scene_vocab_prompts_per_strategy", str(int(cfg.gcg_scene_vocab_prompts_per_strategy)),
        "--gcg_scene_vocab_enabled_strategies", str(cfg.gcg_scene_vocab_enabled_strategies),
        "--gcg_slot_candidate_max_words", str(int(cfg.gcg_slot_candidate_max_words)),
        "--gcg_scene_feedback_limit", str(int(cfg.gcg_scene_feedback_limit)),
        "--class_ablation", "1" if cfg.class_ablation else "0",
        "--gcg_candidate_source", str(cfg.gcg_candidate_source),
        "--attack_mode", str(cfg.attack_mode),
        "--scene_vlm_question", str(cfg.scene_vlm_question),
        "--scene_fallback", str(cfg.scene_fallback),
        "--saved_image_size", str(SAVED_IMAGE_SIZE),
        "--wandb_enable", "1" if cfg.wandb_enable else "0",
        "--wandb_project", str(cfg.wandb_project),
        "--wandb_mode", str(cfg.wandb_mode),
        "--wandb_log_every", str(int(cfg.wandb_log_every)),
        "--wandb_run_name", sample_run_name,
    ]
    if cfg.manual_seed is not None:
        cli.extend(["--seed", str(int(cfg.manual_seed))])
    cli.extend(["--wandb_group", str(cfg.wandb_group or cfg.run_name)])
    for option, value in (
        ("--process_title_backend", getattr(cfg, "process_title_backend", "")),
        ("--wandb_entity", cfg.wandb_entity),
        ("--wandb_tags", cfg.wandb_tags),
        ("--wandb_api_key", cfg.wandb_api_key),
        ("--wandb_api_key_file", cfg.wandb_api_key_file),
    ):
        if str(value or "").strip():
            cli.extend([option, str(value)])
    return cli


def collect_sensitive_values(*values: object) -> List[str]:
    """Collect non-empty credentials that must never enter logs or reports."""

    candidates = [*values]
    for env_name in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
    ):
        candidates.append(os.getenv(env_name, ""))
    output: List[str] = []
    for value in candidates:
        token = str(value or "")
        if token and token not in output:
            output.append(token)
    return output


def redact_transient_paths(
    value: object,
    paths: Sequence[object],
    *,
    sensitive_values: Sequence[object] = (),
) -> str:
    """Remove temporary paths and credentials from persisted diagnostics."""

    text = str(value)
    replacements = set()
    for raw_path in paths:
        raw = str(raw_path or "").strip()
        if not raw:
            continue
        candidates = {raw}
        try:
            candidates.add(str(Path(raw).expanduser().resolve()))
        except Exception:
            pass
        for candidate in candidates:
            replacements.add(candidate)
            replacements.add(candidate.replace("\\", "/"))
            replacements.add(candidate.replace("/", "\\"))
    for candidate in sorted(replacements, key=len, reverse=True):
        if candidate:
            text = text.replace(candidate, "<temporary_input>")
    secrets = {
        str(secret)
        for secret in sensitive_values
        if str(secret or "")
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    return text


def has_valid_saved_attack_image(path: Path) -> bool:
    """Return true only for a complete 224x224 successful-attack artifact."""

    image_path = Path(path)
    if not image_path.is_file():
        return False
    try:
        with Image.open(image_path) as image:
            image.load()
            return image.size == (SAVED_IMAGE_SIZE, SAVED_IMAGE_SIZE)
    except Exception:
        return False


def load_preserved_attack_report(path: Path) -> Optional[dict]:
    """Load core state written after a success but before a later failure."""

    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("final_attack_success") is not True:
        return None
    return payload

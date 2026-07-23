import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from attack_model_registry import validate_generator_model
from attack_runner_common import collect_sensitive_values, redact_transient_paths

try:
    from setproctitle import setproctitle as _setproctitle
except Exception:
    _setproctitle = None

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None


class BlackboxClassifier(Protocol):
    def objective_and_stats(self, image_01, target_label: Optional[int] = None):
        ...


class BlackboxRuntime(Protocol):
    def setup(self, *, args: argparse.Namespace, has_input_image: bool) -> None:
        ...

    def close(self) -> None:
        ...

    def generate_scene_vocab_words(
        self,
        *,
        args: argparse.Namespace,
        step_idx: int,
        current_prompt: str,
        current_word: str,
        slot_kind: str,
        best_objective: float,
        previous_feedback: Sequence[Dict[str, object]],
        reference_image_path: Path,
        fallback_word: str,
    ) -> Tuple[List[str], str, str, Optional[str]]:
        ...

    def query_vlm_word(
        self,
        *,
        image_path: Path,
        args: argparse.Namespace,
        slot_kind: str,
        fallback_word: str,
    ) -> Tuple[str, str, Optional[str]]:
        ...

    def evaluate_candidates(
        self,
        *,
        args: argparse.Namespace,
        classifier: BlackboxClassifier,
        candidate_words: Sequence[str],
        candidate_prompts: Sequence[str],
        has_input_image: bool,
        render_output_path: Optional[Path],
        capture_classifier_tile_image: bool = False,
    ) -> Tuple[List[Dict[str, object]], Optional[str], Optional[Tuple[Path, int]]]:
        ...

    def evaluate_and_candidate(
        self,
        *,
        classifier: BlackboxClassifier,
        cwor_result_prompt: str,
        cwor_base_confidence: Optional[float] = None,
        original_objective: Optional[float] = None,
        prompt_candidates: Sequence[Dict[str, object]],
        cwor_step_index: int = 1,
    ) -> Tuple[List[Dict[str, object]], Optional[str]]:
        ...

    def save_prompt_artifacts(
        self,
        *,
        run_dir: Path,
        step_idx: int,
        candidate_source: str,
        prompt_text: str,
        raw_answer: str,
        feedback_used: Sequence[Dict[str, object]],
        generated_words: Sequence[str],
        filtered_words: Sequence[str],
        scored_candidates: Sequence[Dict[str, object]],
        vlm_error: Optional[str],
        score_error: Optional[str],
    ) -> Dict[str, str]:
        ...

def parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_candidate_source(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token in {"gemma", "gemma_scene_vocab", "gemma-scene-vocab", "scene_vocab", "scene-vocab"}:
        return "gemma_scene_vocab"
    return "vlm_query"


def normalize_attack_mode(raw: object) -> str:
    token = str(raw or "vlm").strip().lower()
    if token not in {"vlm", "and"}:
        raise ValueError("attack_mode must be 'vlm' or 'and'")
    return token


def validate_supported_generator(model_path: object) -> str:
    return validate_generator_model(model_path)


def normalize_cwor_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"target", "targeted"}:
        return "target"
    return "untargeted"


def normalize_cwor_embed_inject_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"clip", "text_encoder1", "te1"}:
        return "clip"
    if token in {"t5", "text_encoder2", "te2"}:
        return "t5"
    return "both"


def is_flux2_klein_model_path(model_path: object) -> bool:
    try:
        validate_generator_model(model_path, expected_family="flux2-klein")
    except ValueError:
        return False
    return True


def normalize_cwor_feedback_merge_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"step_prompt_weighted", "step-weighted", "step_prompt", "per_step_prompt"}:
        return "step_prompt_weighted"
    return "accumulate"


def normalize_cwor_reference_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"best_candidate", "best-candidate", "best", "gs"}:
        return "best_candidate"
    return "base_prompt"


def scene_vocab_strategies_enabled(raw: object) -> bool:
    if raw is None:
        return True
    token = str(raw).strip().lower()
    if not token:
        return False
    return token not in {"none", "off", "0", "false", "no", "disable", "disabled"}


def build_core_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent black-box core attack module.")

    parser.add_argument("--mode", type=str, default="gcg_edit")
    parser.add_argument("--classifier_mode", type=str, default="black-box")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")

    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)

    parser.add_argument("--output_path", type=str, default="outputs/result.png")
    parser.add_argument("--report_path", type=str, default="outputs/gcg_report.json")
    parser.add_argument("--input_img_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    parser.add_argument("--gcg_word", type=str, default="background")
    parser.add_argument("--gcg_occurrence", type=int, default=0)
    parser.add_argument("--gcg_steps", type=int, default=10)
    parser.add_argument("--gcg_batch_size", type=int, default=64)
    parser.add_argument("--max_victim_queries", type=int, default=100)
    parser.add_argument("--gcg_scene_vocab_size", type=int, default=100)
    parser.add_argument("--gcg_scene_vocab_prompts_per_strategy", type=int, default=0)
    parser.add_argument("--gcg_scene_vocab_enabled_strategies", type=str, default="all")
    parser.add_argument("--gcg_slot_candidate_max_words", type=int, default=5)
    parser.add_argument("--gcg_candidate_source", type=str, default="vlm_query")
    parser.add_argument("--attack_mode", choices=["vlm", "and"], default="vlm")
    parser.add_argument("--gcg_scene_vocab_topic", type=str, default=None)
    parser.add_argument("--gcg_scene_feedback_limit", type=int, default=1000)
    parser.add_argument("--gcg_early_stop_on_attack_success", type=parse_bool_flag, default=False)
    parser.add_argument("--gcg_scene_vocab_feedback", type=parse_bool_flag, default=False)

    parser.add_argument("--gcg_scene_llm_model_id", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--gcg_scene_llm_backend", type=str, default="gemma4")
    parser.add_argument("--gcg_scene_llm_device", type=str, default="auto")
    parser.add_argument("--gcg_scene_llm_max_new_tokens", type=int, default=4096)
    parser.add_argument("--gcg_scene_llm_thinking", type=parse_bool_flag, default=False)
    parser.add_argument("--gcg_scene_llm_do_sample", type=parse_bool_flag, default=False)
    parser.add_argument("--scene_vlm_backend", type=str, default="gemma4")
    parser.add_argument("--scene_vlm_model_id", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--scene_vlm_device", type=str, default="auto")
    parser.add_argument("--scene_vlm_max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--scene_vlm_question",
        type=str,
        default="What is the background scene in this image? Answer in 1 word.",
    )
    parser.add_argument("--scene_vlm_thinking", type=parse_bool_flag, default=True)
    parser.add_argument("--scene_vlm_do_sample", type=parse_bool_flag, default=True)
    parser.add_argument("--scene_fallback", type=str, default="outdoor")

    parser.add_argument("--classifier_name", type=str, default="resnet50")
    parser.add_argument("--classifier_objective", type=str, default="ce_max")
    parser.add_argument("--classifier_label", type=int, default=None)
    parser.add_argument("--process_title_backend", type=str, default="")
    parser.add_argument("--saved_image_size", type=int, choices=[224], default=224)

    parser.add_argument("--wandb_enable", type=parse_bool_flag, default=False)
    parser.add_argument("--wandb_project", type=str, default="gcg-flux-edit")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="auto")
    parser.add_argument("--wandb_log_every", type=int, default=1)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_api_key_file", type=str, default=None)
    parser.set_defaults(
        cwor_enable=False,
        cwor_reference_mode="base_prompt",
        cwor_mode="untargeted",
        cwor_embed_inject_mode="both",
        cwor_feedback_merge_mode="accumulate",
        cwor_strategy_feedback_limit=0,
        cwor_accumulate_secondary_ortho=False,
        cwor_accumulate_update_if_improved_only=False,
        cwor_accumulate_delta_use_basis_logit_without_secondary_ortho=False,
        cwor_step_prompt_candidate_ortho=False,
        cwor_step_prompt_flip_alpha_on_regression=False,
        cwor_embed_subtract_scale_by_step=False,
        cwor_target_label=None,
        flux2_strategy_cwor_delta_mode="score",
        flux2_strategy_cwor_merge_mode="weighted",
    )

    return parser


def parse_core_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = build_core_parser()
    args, unknown = parser.parse_known_args(list(argv))
    removed_exact = {
        "--inversion_prompt",
        "--run_mode",
        "--fixed_prompt",
        "--latent_nudging_scalar",
        "--save_intermediate",
        "--save_intermediate_interval",
        "--save_candidate_strips",
        "--capture_classifier_tile_image",
        "--gcg_save_intermediate",
        "--gcg_save_intermediate_interval",
        "--gcg_save_candidate_strips",
        "--gcg_capture_classifier_tile_image",
        "--gcg_eval_naturalness_llm_thinking",
        "--gcg_eval_naturalness_on_attack_success",
        "--gcg_early_stop_on_cwor_success_only",
    }
    for raw in unknown:
        option = str(raw).split("=", 1)[0]
        if (
            option in removed_exact
            or option.startswith("--cwor_")
            or option.startswith("--flux2_strategy_cwor_")
        ):
            raise ValueError(f"removed core option is not supported: {option}")
    return args, unknown


def _process_title_token(raw: object, max_len: int = 32) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = os.path.splitext(os.path.basename(text))[0]
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text[:max_len]


def _infer_process_label_token(args: argparse.Namespace) -> str:
    path_candidates = (
        getattr(args, "output_path", None),
        getattr(args, "report_path", None),
        getattr(args, "input_img_path", None),
    )
    for raw_path in path_candidates:
        text = str(raw_path or "").strip()
        if not text:
            continue
        match = re.search(r"(?:^|[\\/])sample[_-]?0*([0-9]+)(?:[\\/]|$)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1) or "0"
        fallback = re.search(r"sample[_-]?([0-9]+)", text, flags=re.IGNORECASE)
        if fallback:
            return fallback.group(1).lstrip("0") or "0"

    explicit_label = getattr(args, "classifier_label", None)
    if explicit_label is not None:
        return _process_title_token(explicit_label, 16) or "na"

    class_token = _process_title_token(getattr(args, "class_name", None), 24)
    if class_token:
        return class_token

    for raw_path in path_candidates:
        text = str(raw_path or "").strip()
        if not text:
            continue
        match = re.search(r"(?:^|[\\/])label[_-]?0*([0-9]+)(?:[\\/]|$)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1) or "0"
        fallback = re.search(r"label[_-]?([0-9]+)", text, flags=re.IGNORECASE)
        if fallback:
            return fallback.group(1).lstrip("0") or "0"
    return "na"


def _infer_process_model_token(args: argparse.Namespace) -> str:
    classifier_token = _process_title_token(getattr(args, "classifier_name", None), 24)
    if classifier_token:
        return classifier_token
    model_path_token = _process_title_token(getattr(args, "model_path", None), 24)
    if model_path_token:
        return model_path_token
    return "model"


def _infer_process_mode_token(args: argparse.Namespace) -> str:
    mode_raw = str(getattr(args, "classifier_mode", "") or "").strip().lower()
    if "black" in mode_raw:
        return "black"
    if "white" in mode_raw:
        return "white"
    return "black"


def _infer_process_role_token(args: argparse.Namespace) -> str:
    # Prefix role for quick identification in nvidia-smi process list.
    strategy_merge_mode = str(
        getattr(args, "flux2_strategy_cwor_merge_mode", "") or ""
    ).strip().lower()
    attack_mode = str(getattr(args, "attack_mode", "") or "").strip().lower()
    if strategy_merge_mode in {"greedy", "and"}:
        if attack_mode in {"vlm", "ic", "cwor", "gs"}:
            return f"{strategy_merge_mode}_{attack_mode}"
        if bool(getattr(args, "cwor_enable", False)):
            return f"{strategy_merge_mode}_cwor"
        return strategy_merge_mode
    if attack_mode in {"vlm", "ic", "cwor", "gs"}:
        return attack_mode
    if bool(getattr(args, "cwor_enable", False)):
        return "cwor"
    candidate_source = normalize_candidate_source(str(getattr(args, "gcg_candidate_source", "vlm_query")))
    if candidate_source == "gemma_scene_vocab":
        return "gemma"
    return "gemma"


def _infer_process_query_budget_token(args: argparse.Namespace) -> str:
    raw_value = getattr(args, "max_victim_queries", None)
    if raw_value is None:
        return "qna"
    try:
        query_budget = int(raw_value)
    except Exception:
        return "qna"
    if query_budget < 0:
        return "qna"
    return f"q{query_budget}"


def set_process_title_from_args(args: argparse.Namespace) -> None:
    if _setproctitle is None:
        return
    model_token = _infer_process_model_token(args)
    mode_token = _infer_process_mode_token(args)
    role_token = _process_title_token(_infer_process_role_token(args), 24) or "na"
    label_token = _infer_process_label_token(args)
    query_budget_token = _process_title_token(_infer_process_query_budget_token(args), 24) or "qna"
    title = f"{role_token}_{model_token}_{mode_token}_{label_token}_{query_budget_token}"
    backend_token = _process_title_token(getattr(args, "process_title_backend", None), 24)
    if backend_token:
        title = f"{title}_{backend_token}"
    else:
        device_token = _process_title_token(getattr(args, "device", None), 16)
        if device_token:
            title = f"{title}_dev_{device_token}"
    _setproctitle(title)


def parse_csv_tags(raw: object) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    tags: List[str] = []
    for part in text.replace(";", ",").split(","):
        tag = str(part).strip()
        if tag:
            tags.append(tag)
    return tags


def sanitize_diagnostic_text(
    args: argparse.Namespace,
    value: object,
    *extra_sensitive_values: object,
) -> str:
    """Sanitize warnings that are printed instead of persisted."""

    input_path = str(getattr(args, "input_img_path", "") or "").strip()
    transient_paths: List[object] = []
    if input_path:
        transient_paths.extend([input_path, Path(input_path).parent])
    sensitive_values = collect_sensitive_values(
        getattr(args, "hf_token", ""),
        getattr(args, "wandb_api_key", ""),
        *extra_sensitive_values,
    )
    return redact_transient_paths(
        value,
        transient_paths,
        sensitive_values=sensitive_values,
    )


def init_wandb_run(args: argparse.Namespace) -> Tuple[Optional[object], bool]:
    if not bool(args.wandb_enable):
        return None, False

    try:
        import wandb  # type: ignore
    except Exception as exc:
        safe_error = sanitize_diagnostic_text(args, exc)
        print(f"WARNING: wandb import failed ({type(exc).__name__}: {safe_error})")
        return None, False

    api_key = str(args.wandb_api_key or "").strip()
    api_key_file = str(args.wandb_api_key_file or "").strip()
    if not api_key and api_key_file:
        try:
            api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
        except Exception:
            api_key = ""
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    config_payload: Dict[str, object] = dict(vars(args))
    for sensitive_key in (
        "input_img_path",
        "hf_token",
        "wandb_api_key",
        "wandb_api_key_file",
    ):
        config_payload.pop(sensitive_key, None)
    config_payload["mode"] = "vlm_attack_black_box_core"

    init_kwargs: Dict[str, object] = {
        "project": str(args.wandb_project),
        "name": str(args.wandb_run_name or "vlm_attack"),
        "config": config_payload,
        "tags": parse_csv_tags(args.wandb_tags),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = str(args.wandb_entity)
    if args.wandb_group:
        init_kwargs["group"] = str(args.wandb_group)
    if str(args.wandb_mode) != "auto":
        init_kwargs["mode"] = str(args.wandb_mode)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        safe_error = sanitize_diagnostic_text(args, exc, api_key)
        print(f"WARNING: wandb init failed ({type(exc).__name__}: {safe_error})")
        return None, False

    if run is None:
        return None, False
    return run, True


def log_wandb_payload(
    run: object,
    args: argparse.Namespace,
    payload: Dict[str, object],
    step: int,
    *,
    honor_log_every: bool,
) -> bool:
    if honor_log_every:
        log_every = max(1, int(args.wandb_log_every))
        if step > 0 and (step % log_every) != 0:
            return True
    try:
        run.log(payload, step=int(step))
    except Exception as exc:
        safe_error = sanitize_diagnostic_text(args, exc)
        print(f"WARNING: wandb log failed ({type(exc).__name__}: {safe_error})")
        return False
    return True


def log_wandb_final_image(*, run: object, image_path: Path, step: int) -> bool:
    path = Path(image_path)
    if not path.is_file():
        return True
    try:
        import wandb  # type: ignore
    except Exception:
        return True
    try:
        run.log({"media/final_image": wandb.Image(str(path))}, step=int(step))
    except Exception:
        return False
    return True


def finish_wandb_run(run: Optional[object], summary: Dict[str, object]) -> None:
    if run is None:
        return
    try:
        for key, value in summary.items():
            run.summary[key] = value
    except Exception:
        pass
    try:
        run.finish()
    except Exception:
        pass


def attack_success_rule(objective_mode: str) -> str:
    mode = str(objective_mode or "").strip().lower()
    if mode in {"ce_max", "logit_margin_max"}:
        return "pred_idx != classifier_label"
    if mode in {"ce_min", "logit_max"}:
        return "pred_idx == classifier_label"
    return "unknown"


def compute_attack_success(
    *,
    pred_idx: Optional[object],
    classifier_label: Optional[object],
    objective_mode: str,
) -> Optional[bool]:
    if pred_idx is None or classifier_label is None:
        return None
    try:
        pred_i = int(pred_idx)
        label_i = int(classifier_label)
    except Exception:
        return None

    mode = str(objective_mode or "").strip().lower()
    if mode in {"ce_max", "logit_margin_max"}:
        return pred_i != label_i
    if mode in {"ce_min", "logit_max"}:
        return pred_i == label_i
    return None


def compute_victim_query_count(history: Sequence[Dict[str, object]]) -> int:
    total = 0
    for item in history:
        raw_count = item.get("victim_queries_this_step", item.get("candidate_count", 0))
        try:
            count = int(raw_count)
        except Exception:
            count = 0
        if count > 0:
            total += count
    return int(total)


def _to_finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not parsed == parsed:
        return None
    return float(parsed)


def _normalize_cwor_alpha(value: object) -> object:
    if isinstance(value, (list, tuple)):
        values: List[float] = []
        for item in value:
            parsed = _to_finite_float(item)
            if parsed is not None:
                values.append(parsed)
        return values
    return _to_finite_float(value)


def _summarize_objective_change(
    before: object,
    after: object,
) -> Tuple[Optional[float], Optional[str]]:
    before_value = _to_finite_float(before)
    after_value = _to_finite_float(after)
    if before_value is None or after_value is None:
        return None, None
    delta = float(after_value - before_value)
    if delta > 0.0:
        return delta, "up"
    if delta < 0.0:
        return delta, "down"
    return delta, "same"


def _summarize_cwor_strategy_components(value: object) -> List[Dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    summaries: List[Dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        base_objective_before = _to_finite_float(item.get("base_objective_before"))
        trial_objective = _to_finite_float(item.get("trial_objective"))
        objective_delta, objective_change = _summarize_objective_change(
            base_objective_before,
            trial_objective,
        )
        accepted = item.get("accepted")
        skipped_reason = item.get("skipped_reason")
        summaries.append(
            {
                "strategy_name": str(item.get("strategy_name", "")),
                "strategy_title": str(item.get("strategy_title", "")),
                "weight": _to_finite_float(item.get("weight")),
                "merge_alpha": _to_finite_float(item.get("merge_alpha")),
                "delta_sum": _to_finite_float(item.get("delta_sum")),
                "feedback_count": item.get("feedback_count"),
                "reference_prompt": str(item.get("reference_prompt", "")),
                "reference_word": str(item.get("reference_word", "")),
                "reference_objective": _to_finite_float(item.get("reference_objective")),
                "reference_confidence": _to_finite_float(item.get("reference_confidence")),
                "alpha": _normalize_cwor_alpha(item.get("alpha")),
                "merge_anchor": bool(item.get("merge_anchor", False)),
                "accepted": None if accepted is None else bool(accepted),
                "base_objective_before": base_objective_before,
                "trial_objective": trial_objective,
                "objective_delta": objective_delta,
                "objective_change": objective_change,
                "skipped_reason": None if skipped_reason is None else str(skipped_reason),
            }
        )
    return summaries


def dedupe_feedback_candidates_by_prompt(
    feedback_candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    index_by_prompt: Dict[str, int] = {}
    for item in feedback_candidates:
        candidate_prompt = str(item.get("candidate_prompt", "")).strip()
        if not candidate_prompt:
            continue
        existing_idx = index_by_prompt.get(candidate_prompt)
        if existing_idx is None:
            index_by_prompt[candidate_prompt] = len(deduped)
            deduped.append(dict(item))
            continue

        prev_item = deduped[existing_idx]
        prev_score = _to_finite_float(prev_item.get("candidate_objective"))
        new_score = _to_finite_float(item.get("candidate_objective"))
        if prev_score is None and new_score is None:
            continue
        if prev_score is None or (new_score is not None and new_score > prev_score):
            deduped[existing_idx] = dict(item)
    return deduped


def summarize_cwor_feedback_candidates(
    feedback_candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for item in feedback_candidates:
        summaries.append(
            {
                "candidate_word": str(item.get("candidate_word", "")),
                "candidate_prompt": str(item.get("candidate_prompt", "")),
                "candidate_objective": _to_finite_float(item.get("candidate_objective")),
                "pred_idx": item.get("pred_idx"),
                "target_conf": _to_finite_float(item.get("target_conf", item.get("target_logit"))),
                "target_label_conf": _to_finite_float(
                    item.get("target_label_conf", item.get("target_label_logit"))
                ),
            }
        )
    return summaries


def summarize_cwor_candidates(scored_candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for item in scored_candidates:
        if str(item.get("candidate_variant", "")) != "cwor":
            continue
        strategy_components_summary = _summarize_cwor_strategy_components(
            item.get("cwor_strategy_components")
        )
        strategy_base_prompt = None
        strategy_base_word = None
        strategy_base_objective = None
        for component in strategy_components_summary:
            if bool(component.get("merge_anchor", False)):
                strategy_base_prompt = str(component.get("reference_prompt", "")) or None
                strategy_base_word = str(component.get("reference_word", "")) or None
                strategy_base_objective = _to_finite_float(component.get("reference_objective"))
                break
        summaries.append(
            {
                "candidate_word": str(item.get("candidate_word", "")),
                "candidate_prompt": str(item.get("candidate_prompt", "")),
                "candidate_objective": _to_finite_float(item.get("candidate_objective")),
                "pred_idx": item.get("pred_idx"),
                "target_conf": _to_finite_float(item.get("target_conf", item.get("target_logit"))),
                "target_label_conf": _to_finite_float(
                    item.get("target_label_conf", item.get("target_label_logit"))
                ),
                "cwor_alpha": _normalize_cwor_alpha(item.get("cwor_alpha")),
                "cwor_alpha_sum": _to_finite_float(item.get("cwor_alpha_sum")),
                "cwor_feedback_count": item.get("cwor_feedback_count"),
                "cwor_feedback_step_count": item.get("cwor_feedback_step_count"),
                "cwor_state_updated": item.get("cwor_state_updated"),
                "cwor_mode": item.get("cwor_mode"),
                "cwor_embed_inject_mode": item.get("cwor_embed_inject_mode"),
                "cwor_feedback_merge_mode": item.get("cwor_feedback_merge_mode"),
                "cwor_strategy_count": item.get("cwor_strategy_count"),
                "cwor_strategy_delta_mode": item.get("cwor_strategy_delta_mode"),
                "cwor_strategy_merge_mode": item.get("cwor_strategy_merge_mode"),
                "cwor_strategy_merge_alpha_sum": _to_finite_float(
                    item.get("cwor_strategy_merge_alpha_sum")
                ),
                "cwor_strategy_base_prompt": strategy_base_prompt,
                "cwor_strategy_base_word": strategy_base_word,
                "cwor_strategy_base_objective": strategy_base_objective,
                "cwor_strategy_components": strategy_components_summary,
            }
        )
    return summaries


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def redact_transient_payload(
    value: object,
    paths: Sequence[object],
    sensitive_values: Sequence[object] = (),
) -> object:
    """Recursively redact temporary condition-image paths before persistence."""

    if isinstance(value, str):
        return redact_transient_paths(
            value,
            paths,
            sensitive_values=sensitive_values,
        )
    if isinstance(value, dict):
        return {
            key: redact_transient_payload(item, paths, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_transient_payload(item, paths, sensitive_values) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_transient_payload(item, paths, sensitive_values) for item in value)
    return value


def relpath_from_run_dir(run_dir: Path, path: Path) -> str:
    run_dir_abs = run_dir.resolve()
    path_abs = path.resolve()
    try:
        return str(path_abs.relative_to(run_dir_abs))
    except Exception:
        return str(path_abs)


def candidate_image_for_saving(candidate: Dict[str, object]) -> Image.Image:
    """Return the already evaluated candidate image without invoking a generator."""

    for key in ("candidate_classifier_image", "candidate_selected_image"):
        image = candidate.get(key)
        if isinstance(image, Image.Image):
            return image.copy()

    raw_path = str(candidate.get("candidate_precomputed_selected_image_path", "") or "").strip()
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert("RGB").copy()

    raise ValueError("successful candidate has no evaluated image payload")


def save_evaluated_attack_image(
    candidate: Dict[str, object],
    output_path: Path,
    *,
    saved_image_size: int = 224,
) -> Path:
    """Save the exact evaluated candidate at the fixed classifier resolution."""

    size = int(saved_image_size)
    if size != 224:
        raise ValueError("saved attack images must be 224x224")
    image = candidate_image_for_saving(candidate).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def write_report(
    *,
    args: argparse.Namespace,
    unknown_args: List[str],
    original_prompt: str,
    optimized_prompt: str,
    best_objective: float,
    history: List[Dict[str, object]],
    early_stop_event: Optional[Dict[str, object]],
    final_attack_success: Optional[bool],
    attack_success_image_path: Optional[str],
    attack_success_image_error: Optional[str],
    attack_success_query_count: Optional[int],
    attack_success_candidate: Optional[Dict[str, object]],
) -> None:
    transient_input_path = str(getattr(args, "input_img_path", "") or "").strip()
    transient_paths: List[object] = []
    if transient_input_path:
        transient_paths.extend([transient_input_path, Path(transient_input_path).parent])
    sensitive_values = collect_sensitive_values(
        getattr(args, "hf_token", ""),
        getattr(args, "wandb_api_key", ""),
    )
    args_payload: Dict[str, object] = dict(vars(args))
    for sensitive_key in (
        "input_img_path",
        "hf_token",
        "wandb_api_key",
        "wandb_api_key_file",
    ):
        args_payload.pop(sensitive_key, None)
    args_payload["mode"] = "vlm_attack_black_box"
    if unknown_args:
        args_payload["ignored_cli_args"] = unknown_args

    payload = {
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt,
        "best_objective": float(best_objective),
        "gcg_word": str(args.gcg_word),
        "gcg_occurrence": int(args.gcg_occurrence),
        "attack_success_rule": attack_success_rule(str(args.classifier_objective)),
        "victim_query_count": int(compute_victim_query_count(history)),
        "final_attack_success": final_attack_success,
        "early_stop": None if early_stop_event is None else dict(early_stop_event),
        "history": history,
        "args": args_payload,
    }
    if attack_success_image_path:
        payload["attack_success_image_path"] = str(attack_success_image_path)
    if attack_success_image_error:
        payload["attack_success_image_error"] = str(attack_success_image_error)
    if attack_success_query_count is not None:
        payload["attack_success_query_count"] = int(attack_success_query_count)
    if attack_success_candidate is not None:
        payload["attack_success_candidate"] = dict(attack_success_candidate)
    payload = redact_transient_payload(payload, transient_paths, sensitive_values)
    report_path = Path(str(args.report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def contains_class_placeholder(text: str) -> bool:
    prompt = str(text or "")
    return any(marker in prompt for marker in ("<class>", "{class}", "{class_name}"))


def apply_class_placeholder(text: str, class_name: str) -> str:
    prompt = str(text or "")
    value = str(class_name or "").strip()
    if not value:
        return prompt
    replacements = {
        "<class>": value,
        "{class}": value,
        "{class_name}": value,
    }
    for marker, replacement in replacements.items():
        prompt = prompt.replace(marker, replacement)
    return re.sub(r"\s+", " ", prompt).strip()


def get_editable_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None and str(args.prompt).strip():
        return str(args.prompt).strip()
    if args.prompts:
        return str(args.prompts[-1]).strip()
    raise ValueError("Provide --prompt or --prompts.")


def infer_imagenet_class_name(model_name: str, label_idx: int) -> str:
    csv_path = Path(__file__).resolve().parent / "data" / "nips2017" / "categories.csv"
    idx = int(label_idx)
    csv_category_id = idx + 1
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    category_id = int(row.get("CategoryId", ""))
                except Exception:
                    continue
                if category_id != csv_category_id:
                    continue
                raw = str(row.get("CategoryName", "")).strip()
                if not raw:
                    raise ValueError(f"Empty class name for CSV category id: {csv_category_id}")
                return raw.split(",")[0].strip() or raw

    if tv_models is None:
        raise ValueError("torchvision is unavailable")
    token = str(model_name or "").strip().lower().replace("_", "-")
    if token == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT
    elif token == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT
    elif token == "resnet101":
        weights = tv_models.ResNet101_Weights.DEFAULT
    elif token in {"inception-v3", "inceptionv3"}:
        weights = tv_models.Inception_V3_Weights.DEFAULT
    elif token in {"vit-base", "vit-b-16", "vitb16", "vit-base-patch16-224"}:
        weights = tv_models.ViT_B_16_Weights.DEFAULT
    else:
        raise ValueError(f"unsupported model for class-name lookup: {model_name}")

    categories = list(weights.meta.get("categories", []))
    if idx < 0 or idx >= len(categories):
        raise ValueError(f"label index out of range: {idx}")
    raw = str(categories[idx]).strip()
    if not raw:
        raise ValueError("empty class name")
    return raw.split(",")[0].strip() or raw


def infer_slot_kind(prompt: str, question: str) -> str:
    prompt_text = str(prompt or "").strip().lower()
    question_text = str(question or "").strip().lower()
    if "<object>" in prompt_text or "{object}" in prompt_text:
        return "object"
    if "<scene>" in prompt_text or "{scene}" in prompt_text:
        return "scene"
    if any(key in question_text for key in ("object", "next to", "nearby", "adjacent", "beside")):
        return "object"
    return "scene"


def replace_nth_word(prompt: str, current_word: str, replacement: str, occurrence: int) -> Tuple[str, bool]:
    text = str(prompt or "")
    word = str(current_word or "").strip()
    repl = str(replacement or "").strip()
    if not repl:
        return text, False

    if word:
        matches = list(re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))
        if 0 <= int(occurrence) < len(matches):
            match = matches[int(occurrence)]
            updated = text[: match.start()] + repl + text[match.end() :]
            return updated, updated != text

    for marker in ("<scene>", "{scene}", "<object>", "{object}", "<CWOR>", "{CWOR}", "<cwor>", "{cwor}"):
        if marker in text:
            updated = text.replace(marker, repl, 1)
            return updated, updated != text
    return text, False


def image_to_tensor_01(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def evaluate_source_image_baseline(
    *,
    classifier: BlackboxClassifier,
    source_image_path: Path,
    cwor_target_label: Optional[int] = None,
) -> Tuple[float, Dict[str, object], Optional[str]]:
    if not source_image_path.is_file():
        return 0.0, {}, f"source_image_missing:{source_image_path}"
    try:
        with Image.open(source_image_path) as source_img:
            image_01 = image_to_tensor_01(source_img)
        objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
        return float(objective), stats, None
    except Exception as exc:
        return 0.0, {}, f"source_image_eval_failed:{type(exc).__name__}:{exc}"


def scene_feedback_objective(item: Dict[str, object]) -> Optional[float]:
    raw = item.get("objective")
    if raw is None:
        return None
    try:
        objective = float(raw)
    except Exception:
        return None
    if not objective == objective:
        return None
    return objective


def scene_feedback_sort_key(item: Dict[str, object]) -> Tuple[float, int, str]:
    objective = scene_feedback_objective(item)
    objective_rank = objective if objective is not None else float("-inf")
    return (
        objective_rank,
        int(item.get("attempts", 0)),
        str(item.get("scene_word", "")),
    )


def rank_scene_vocab_feedback_entries(
    *,
    feedback_entries: Sequence[Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    if len(feedback_entries) == 0:
        return []
    ranked = sorted(feedback_entries, key=scene_feedback_sort_key, reverse=True)
    k = max(1, int(limit))
    return [dict(item) for item in ranked[:k]]


def normalize_scene_vocab_word(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"^\s*(?:[-*]|\d+[\.\)])\s*", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    out = out.strip(" ,.;:!?")
    return out


def merge_scene_vocab_feedback_history(
    *,
    existing_feedback: Sequence[Dict[str, object]],
    generated_words: Sequence[str],
    scored_candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged_by_word: Dict[str, Dict[str, object]] = {}

    for item in existing_feedback:
        scene_word = normalize_scene_vocab_word(str(item.get("scene_word", "")))
        if len(scene_word) == 0:
            continue
        merged_by_word[scene_word] = {
            "scene_word": scene_word,
            "objective": scene_feedback_objective(item),
            "attempts": max(1, int(item.get("attempts", 1))),
        }

    scored_objective_by_word: Dict[str, float] = {}
    for item in scored_candidates:
        scene_word = normalize_scene_vocab_word(str(item.get("candidate_word", "")))
        if len(scene_word) == 0:
            continue
        objective = float(item.get("candidate_objective", float("-inf")))
        prev_objective = scored_objective_by_word.get(scene_word, float("-inf"))
        if objective > prev_objective:
            scored_objective_by_word[scene_word] = objective

    for raw_word in generated_words:
        scene_word = normalize_scene_vocab_word(str(raw_word))
        if len(scene_word) == 0:
            continue
        objective = scored_objective_by_word.get(scene_word)
        entry = merged_by_word.get(scene_word)
        if entry is None:
            merged_by_word[scene_word] = {
                "scene_word": scene_word,
                "objective": objective,
                "attempts": 1,
            }
            continue
        entry["attempts"] = max(1, int(entry.get("attempts", 1))) + 1
        prev_objective = scene_feedback_objective(entry)
        if objective is not None and (prev_objective is None or objective > prev_objective):
            entry["objective"] = objective

    for scene_word, objective in scored_objective_by_word.items():
        existing = merged_by_word.get(scene_word)
        if existing is not None:
            prev_objective = scene_feedback_objective(existing)
            if objective is not None and (prev_objective is None or objective > prev_objective):
                existing["objective"] = objective
            continue
        merged_by_word[scene_word] = {
            "scene_word": scene_word,
            "objective": objective,
            "attempts": 1,
        }

    return list(merged_by_word.values())


def _resolve_prompt(args: argparse.Namespace) -> str:
    editable_prompt = get_editable_prompt(args)
    has_class_placeholder = contains_class_placeholder(editable_prompt)

    resolved_class_name = str(args.class_name or "").strip()
    if has_class_placeholder and not resolved_class_name:
        try:
            resolved_class_name = infer_imagenet_class_name(
                model_name=str(args.classifier_name),
                label_idx=int(args.classifier_label),
            )
        except Exception:
            resolved_class_name = ""

    if resolved_class_name:
        editable_prompt = apply_class_placeholder(editable_prompt, resolved_class_name)
        args.class_name = resolved_class_name
    return editable_prompt


def run_blackbox_attack_core(
    *,
    args: argparse.Namespace,
    unknown_args: Optional[Sequence[str]],
    classifier: BlackboxClassifier,
    runtime: BlackboxRuntime,
    manage_runtime: bool = True,
) -> Dict[str, object]:
    args.classifier_mode = "black-box"
    args.attack_mode = normalize_attack_mode(getattr(args, "attack_mode", "vlm"))
    args.generator_backend = validate_supported_generator(args.model_path)
    and_mode_enabled = bool(args.attack_mode == "and")
    args.gcg_candidate_source = (
        "gemma_scene_vocab"
        if and_mode_enabled
        else normalize_candidate_source(args.gcg_candidate_source)
    )
    # AND reuses the strategy-merging implementation internally. These are
    # implementation details derived from the public mode, not independent modes.
    args.cwor_enable = and_mode_enabled
    args.cwor_reference_mode = "base_prompt"
    args.cwor_mode = "untargeted"
    args.flux2_strategy_cwor_merge_mode = "and" if and_mode_enabled else "weighted"
    if and_mode_enabled:
        args.gcg_scene_vocab_prompts_per_strategy = max(
            1,
            int(getattr(args, "gcg_scene_vocab_prompts_per_strategy", 0)),
        )
    args.cwor_mode = normalize_cwor_mode(getattr(args, "cwor_mode", "untargeted"))
    args.cwor_embed_inject_mode = normalize_cwor_embed_inject_mode(
        getattr(args, "cwor_embed_inject_mode", "both")
    )
    args.cwor_reference_mode = normalize_cwor_reference_mode(
        getattr(args, "cwor_reference_mode", "base_prompt")
    )
    args.cwor_feedback_merge_mode = normalize_cwor_feedback_merge_mode(
        getattr(args, "cwor_feedback_merge_mode", "accumulate")
    )
    set_process_title_from_args(args)

    if int(args.gcg_steps) < 1:
        raise ValueError("--gcg_steps must be >= 1")
    if int(args.gcg_batch_size) < 1:
        raise ValueError("--gcg_batch_size must be >= 1")
    if int(args.max_victim_queries) < 1:
        raise ValueError("--max_victim_queries must be >= 1")
    if int(args.gcg_scene_vocab_size) < 1:
        raise ValueError("--gcg_scene_vocab_size must be >= 1")
    if int(getattr(args, "gcg_scene_vocab_prompts_per_strategy", 0)) < 0:
        raise ValueError("--gcg_scene_vocab_prompts_per_strategy must be >= 0")
    if int(getattr(args, "cwor_strategy_feedback_limit", 0)) < 0:
        raise ValueError("--cwor_strategy_feedback_limit must be >= 0")
    if int(args.gcg_slot_candidate_max_words) < 1:
        raise ValueError("--gcg_slot_candidate_max_words must be >= 1")
    if args.classifier_label is None:
        raise ValueError("--classifier_label is required")

    output_path = Path(str(args.output_path))
    report_path = Path(str(args.report_path))
    run_dir = report_path.expanduser().resolve().parent
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        output_path.unlink()

    editable_prompt = _resolve_prompt(args)
    current_prompt = editable_prompt
    best_output_prompt = editable_prompt
    current_word = str(args.gcg_word or "").strip()
    attack_mode = str(args.attack_mode)
    flux2_strategy_cwor_merge_mode = str(
        getattr(args, "flux2_strategy_cwor_merge_mode", "weighted") or "weighted"
    ).strip().lower()
    cwor_base_prompt = str(editable_prompt)
    cwor_prompt_template, cwor_prompt_has_slot = replace_nth_word(
        editable_prompt,
        current_word=current_word,
        replacement="<CWOR>",
        occurrence=int(args.gcg_occurrence),
    )
    if cwor_prompt_has_slot and cwor_prompt_template != editable_prompt:
        cwor_base_prompt = cwor_prompt_template
    slot_kind = infer_slot_kind(editable_prompt, args.scene_vlm_question)
    default_fallback = "outdoor" if slot_kind == "scene" else "object"
    fallback_word = str(args.scene_fallback or "").strip().lower() or current_word or default_fallback
    has_input_image = bool(args.input_img_path and str(args.input_img_path).strip())
    if not has_input_image:
        raise ValueError("black-box core requires --input_img_path")
    step_input_img_path = Path(str(args.input_img_path))
    transient_input_paths: List[object] = [step_input_img_path, step_input_img_path.parent]

    wandb_run, wandb_enabled = init_wandb_run(args)
    wandb_run_finished = False
    sensitive_values = collect_sensitive_values(
        getattr(args, "hf_token", ""),
        getattr(args, "wandb_api_key", ""),
    )

    history: List[Dict[str, object]] = []
    early_stop_event: Optional[Dict[str, object]] = None
    best_pred_idx = None
    best_pred_conf = None
    best_target_conf = None
    best_pred_logit = None
    best_target_logit = None
    best_target_label_conf = None
    best_target_label_logit = None
    best_ce = None
    best_objective = float("-inf")
    attack_success_observed = False
    attack_success_image_relpath: Optional[str] = None
    attack_success_image_error: Optional[str] = None
    attack_success_query_count: Optional[int] = None
    attack_success_candidate: Optional[Dict[str, object]] = None
    live_victim_query_count = 0

    def _record_victim_evaluation(candidate: Dict[str, object]) -> None:
        """Persist the first successful classifier input before another query can run."""

        nonlocal attack_success_observed
        nonlocal attack_success_image_relpath
        nonlocal attack_success_image_error
        nonlocal attack_success_query_count
        nonlocal attack_success_candidate
        nonlocal live_victim_query_count

        raw_attempt_count = candidate.get("victim_query_attempt_count")
        try:
            callback_attempt_count = int(raw_attempt_count)
        except Exception:
            callback_attempt_count = None
        if callback_attempt_count is not None and callback_attempt_count > 0:
            live_victim_query_count = max(
                int(live_victim_query_count),
                int(callback_attempt_count),
            )
        else:
            live_victim_query_count += 1
        if compute_attack_success(
            pred_idx=candidate.get("pred_idx"),
            classifier_label=args.classifier_label,
            objective_mode=str(args.classifier_objective),
        ) is not True:
            return
        attack_success_observed = True
        if attack_success_query_count is None:
            attack_success_query_count = int(live_victim_query_count)
            attack_success_candidate = {
                key: value
                for key, value in candidate.items()
                if not isinstance(value, Image.Image)
            }
        if attack_success_image_relpath is not None:
            return
        try:
            saved_path = save_evaluated_attack_image(
                dict(candidate),
                output_path,
                saved_image_size=int(args.saved_image_size),
            )
            attack_success_image_relpath = relpath_from_run_dir(run_dir, saved_path)
            attack_success_image_error = None
        except Exception as exc:
            attack_success_image_error = f"{type(exc).__name__}:{exc}"

    attack_success_rule_text = attack_success_rule(str(args.classifier_objective))
    step_prompts_path = run_dir / "step_prompts.json"
    prompt_trace_steps: List[Dict[str, object]] = []
    prompt_trace_doc: Dict[str, object] = {
        "attack_mode": str(attack_mode),
        "gcg_word": str(args.gcg_word),
        "gcg_occurrence": int(args.gcg_occurrence),
        "initial_prompt": str(editable_prompt),
        "steps": prompt_trace_steps,
    }
    try:
        if bool(manage_runtime):
            runtime.setup(args=args, has_input_image=has_input_image)
        reset_cwor_state = getattr(runtime, "reset_cwor_state", None)
        if callable(reset_cwor_state):
            try:
                reset_cwor_state()
            except Exception:
                pass
        cwor_target_label: Optional[int] = None
        if bool(getattr(args, "cwor_enable", False)) and str(args.cwor_mode) == "target":
            if getattr(args, "cwor_target_label", None) is None:
                raise ValueError("--cwor_target_label is required when --cwor_mode=target")
            cwor_target_label = int(args.cwor_target_label)
        source_objective, source_stats, source_error = evaluate_source_image_baseline(
            classifier=classifier,
            source_image_path=Path(str(args.input_img_path)),
            cwor_target_label=cwor_target_label,
        )
        if source_error is not None:
            raise RuntimeError(f"failed baseline source evaluation: {source_error}")

        reset_attempt_count = getattr(classifier, "reset_evaluation_attempt_count", None)
        if callable(reset_attempt_count):
            reset_attempt_count()
        set_evaluation_callback = getattr(classifier, "set_evaluation_callback", None)
        if callable(set_evaluation_callback):
            set_evaluation_callback(_record_victim_evaluation)

        initial_objective = float(source_objective)
        best_objective = float(source_objective)
        best_pred_idx = source_stats.get("pred_idx")
        best_pred_conf = source_stats.get("pred_conf")
        best_target_conf = source_stats.get("target_conf")
        best_pred_logit = source_stats.get("pred_logit", source_stats.get("pred_conf"))
        best_target_logit = source_stats.get("target_logit", source_stats.get("target_conf"))
        best_target_label_conf = source_stats.get("target_label_conf")
        best_target_label_logit = source_stats.get("target_label_logit", source_stats.get("target_label_conf"))
        best_ce = source_stats.get("ce")
        initial_pred_idx = best_pred_idx
        initial_pred_conf = best_pred_conf
        initial_target_conf = best_target_conf
        initial_pred_logit = best_pred_logit
        initial_target_logit = best_target_logit
        initial_target_label_conf = best_target_label_conf
        initial_target_label_logit = best_target_label_logit
        initial_ce = best_ce
        cwor_base_confidence: Optional[float] = None
        cwor_base_logit: Optional[float] = None
        cwor_mode = normalize_cwor_mode(getattr(args, "cwor_mode", "untargeted"))
        cwor_reference_mode = normalize_cwor_reference_mode(
            getattr(args, "cwor_reference_mode", "base_prompt")
        )

        def _resolve_cwor_feedback_score(
            payload: Dict[str, object],
            *,
            prefer_logit: bool,
        ) -> Optional[float]:
            if cwor_mode == "target":
                keys = (
                    ("target_label_logit", "target_label_conf")
                    if bool(prefer_logit)
                    else ("target_label_conf", "target_label_logit")
                )
            else:
                keys = (
                    ("target_logit", "target_conf")
                    if bool(prefer_logit)
                    else ("target_conf", "target_logit")
                )
            for key in keys:
                raw_value = payload.get(key)
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                if np.isfinite(value):
                    return float(value)
            return None

        def _candidate_objective_for_sort(payload: Dict[str, object]) -> float:
            try:
                objective = float(payload.get("candidate_objective", float("-inf")))
            except Exception:
                objective = float("-inf")
            if not np.isfinite(objective):
                objective = float("-inf")
            return float(objective)

        if cwor_mode == "target":
            raw_cwor_base_conf = initial_target_label_conf
            raw_cwor_base_logit = initial_target_label_logit
        else:
            raw_cwor_base_conf = initial_target_conf
            raw_cwor_base_logit = initial_target_logit
        try:
            if raw_cwor_base_conf is not None:
                cwor_base_confidence = float(raw_cwor_base_conf)
        except Exception:
            cwor_base_confidence = None
        try:
            if raw_cwor_base_logit is not None:
                cwor_base_logit = float(raw_cwor_base_logit)
        except Exception:
            cwor_base_logit = None

        if wandb_enabled and wandb_run is not None:
            baseline_payload: Dict[str, object] = {
                "step": 0,
                "objective/candidate": float(best_objective),
                "objective/best": float(best_objective),
                "loss/cls": float(best_objective),
                "loss/adv": float(best_objective),
                "loss/opt": float(-best_objective),
                "candidate/source": "source_image",
                "candidate/count": 1,
                "flags/accepted_candidate": 1,
                "pred/pred_idx": -1 if best_pred_idx is None else int(best_pred_idx),
            }
            if best_pred_logit is not None:
                baseline_payload["logit/pred"] = float(best_pred_logit)
            if best_target_logit is not None:
                baseline_payload["logit/target"] = float(best_target_logit)
            if best_target_label_logit is not None:
                baseline_payload["logit/cwor_target_label"] = float(best_target_label_logit)
            if best_pred_conf is not None:
                baseline_payload["conf/pred"] = float(best_pred_conf)
            if best_target_conf is not None:
                baseline_payload["conf/target"] = float(best_target_conf)
            if best_target_label_conf is not None:
                baseline_payload["conf/cwor_target_label"] = float(best_target_label_conf)
            if best_ce is not None:
                baseline_payload["ce"] = float(best_ce)
            wandb_enabled = log_wandb_payload(
                run=wandb_run,
                args=args,
                payload=baseline_payload,
                step=0,
                honor_log_every=False,
            )

        init_trace_step: Dict[str, object] = {
            "step": 0,
            "kind": "init_prompt",
            "prompt": str(current_prompt),
            "objective": float(best_objective),
            "pred_idx": best_pred_idx,
            "pred_conf": best_pred_conf,
            "target_conf": best_target_conf,
            "target_label_conf": best_target_label_conf,
            "pred_logit": best_pred_logit,
            "target_logit": best_target_logit,
            "target_label_logit": best_target_label_logit,
            "ce": best_ce,
            "requested_candidate_source": str(args.gcg_candidate_source),
            "used_candidate_source": "source_image",
        }
        prompt_trace_steps.append(init_trace_step)
        write_json(
            step_prompts_path,
            redact_transient_payload(
                prompt_trace_doc,
                transient_input_paths,
                sensitive_values,
            ),
        )

        max_victim_queries = int(args.max_victim_queries)
        victim_query_count_running = 0
        previous_scene_feedback: List[Dict[str, object]] = []
        previous_step_vlm_feedback: List[Dict[str, object]] = []
        for step in range(int(args.gcg_steps)):
            remaining_query_budget_before_step = int(max_victim_queries - victim_query_count_running)
            if remaining_query_budget_before_step <= 0:
                early_stop_event = {
                    "enabled": True,
                    "triggered": True,
                    "reason": "victim_query_budget_exhausted",
                    "step": int(step) + 1,
                    "victim_query_count": int(victim_query_count_running),
                    "max_victim_queries": int(max_victim_queries),
                }
                break

            candidate_source = str(args.gcg_candidate_source)
            feedback_for_generation = rank_scene_vocab_feedback_entries(
                feedback_entries=previous_scene_feedback,
                limit=int(args.gcg_scene_feedback_limit),
            )
            generated_words: List[str] = []
            generation_prompt = ""
            raw_answer = ""
            vlm_error: Optional[str] = None
            candidate_words_for_eval: List[str] = []
            candidate_prompts_for_eval: List[str] = []
            candidate_strategy_names_for_eval: List[str] = []
            candidate_strategy_titles_for_eval: List[str] = []
            generated_strategy_entries: List[Dict[str, object]] = []

            cwor_enabled = bool(getattr(args, "cwor_enable", False))
            strategy_cwor_runtime_enabled = bool(
                is_flux2_klein_model_path(getattr(args, "model_path", None))
                or bool(getattr(args, "qwen_strategy_and_enable", False))
                or (
                    bool(getattr(args, "bernini_edit_prompt_mode", False))
                    and str(flux2_strategy_cwor_merge_mode) == "and"
                )
            )
            flux2_strategy_cwor_enabled = bool(
                cwor_enabled
                and str(candidate_source) == "gemma_scene_vocab"
                and strategy_cwor_runtime_enabled
                and int(getattr(args, "gcg_scene_vocab_prompts_per_strategy", 0)) > 0
                and scene_vocab_strategies_enabled(
                    getattr(args, "gcg_scene_vocab_enabled_strategies", "all")
                )
            )
            cwor_feedback_for_step_raw = [dict(item) for item in previous_step_vlm_feedback]
            cwor_feedback_for_step = dedupe_feedback_candidates_by_prompt(cwor_feedback_for_step_raw)
            cwor_feedback_for_eval = list(cwor_feedback_for_step)
            cwor_base_prompt_for_step = str(cwor_base_prompt)
            cwor_base_confidence_for_step = cwor_base_confidence
            cwor_base_logit_for_step = cwor_base_logit
            cwor_reference_mode_for_step = (
                "strategy_local_best" if flux2_strategy_cwor_enabled else str(cwor_reference_mode)
            )
            cwor_reference_candidate: Optional[Dict[str, object]] = None

            use_best_candidate_reference = bool(
                cwor_enabled
                and not flux2_strategy_cwor_enabled
                and cwor_reference_mode == "best_candidate"
                and str(candidate_source) == "gemma_scene_vocab"
                and len(cwor_feedback_for_step) > 0
            )
            if use_best_candidate_reference:
                best_ref_idx = max(
                    range(len(cwor_feedback_for_step)),
                    key=lambda idx: _candidate_objective_for_sort(cwor_feedback_for_step[idx]),
                )
                cwor_reference_candidate = dict(cwor_feedback_for_step[int(best_ref_idx)])
                ref_prompt = str(cwor_reference_candidate.get("candidate_prompt", "")).strip()
                ref_logit = _resolve_cwor_feedback_score(cwor_reference_candidate, prefer_logit=True)
                if len(ref_prompt) > 0 and ref_logit is not None:
                    cwor_base_prompt_for_step = str(ref_prompt)
                    cwor_feedback_for_eval = [
                        item
                        for idx, item in enumerate(cwor_feedback_for_step)
                        if int(idx) != int(best_ref_idx)
                    ]
                else:
                    cwor_feedback_for_eval = []

            cwor_pre_candidates: List[Dict[str, object]] = []
            cwor_pre_error: Optional[str] = None
            cwor_queries_this_step = 0
            remaining_query_budget = int(remaining_query_budget_before_step)
            skip_text_eval_due_to_cwor_success = False

            prompt_scored_candidates: List[Dict[str, object]] = []
            prompt_query_attempt_count = 0
            prompt_score_error: Optional[str] = None
            if not skip_text_eval_due_to_cwor_success:
                if candidate_source == "gemma_scene_vocab":
                    generated_words, raw_answer, generation_prompt, vlm_error = runtime.generate_scene_vocab_words(
                        args=args,
                        step_idx=step,
                        current_prompt=current_prompt,
                        current_word=current_word,
                        slot_kind=slot_kind,
                        best_objective=best_objective,
                        previous_feedback=feedback_for_generation,
                        reference_image_path=Path(str(step_input_img_path)),
                        fallback_word=fallback_word,
                    )
                    if flux2_strategy_cwor_enabled:
                        raw_strategy_entries = getattr(args, "_scene_vocab_strategy_entries", [])
                        if isinstance(raw_strategy_entries, list):
                            generated_strategy_entries = [
                                dict(item) for item in raw_strategy_entries if isinstance(item, dict)
                            ]

                if candidate_source != "gemma_scene_vocab" or len(generated_words) == 0:
                    candidate_word, raw_answer_simple, vlm_error_simple = runtime.query_vlm_word(
                        image_path=Path(str(step_input_img_path)),
                        args=args,
                        slot_kind=slot_kind,
                        fallback_word=fallback_word,
                    )
                    if len(generated_words) == 0:
                        generated_words = [candidate_word]
                    if not raw_answer:
                        raw_answer = raw_answer_simple
                    if vlm_error is None:
                        vlm_error = vlm_error_simple

                max_eval_candidates_budget = int(remaining_query_budget)
                desired_eval_candidates = max(1, int(args.gcg_batch_size))
                if flux2_strategy_cwor_enabled and len(generated_words) > 0:
                    desired_eval_candidates = max(1, int(len(generated_words)))
                max_eval_candidates = min(int(desired_eval_candidates), int(max_eval_candidates_budget))
                if max_eval_candidates <= 0:
                    prompt_score_error = "victim_query_budget_exhausted"
                else:
                    flux2_prompt_word_only = bool(
                        is_flux2_klein_model_path(getattr(args, "model_path", None))
                        or bool(getattr(args, "bernini_edit_prompt_mode", False))
                        or bool(getattr(args, "qwen_edit_prompt_mode", False))
                        or bool(getattr(args, "qwen_strategy_and_enable", False))
                    )
                    for word_idx, word in enumerate(generated_words):
                        if flux2_prompt_word_only:
                            candidate_prompt_try = str(word).strip()
                            replaced_try = bool(len(candidate_prompt_try) > 0)
                        else:
                            candidate_prompt_try, replaced_try = replace_nth_word(
                                current_prompt,
                                current_word=current_word,
                                replacement=word,
                                occurrence=int(args.gcg_occurrence),
                            )
                        if replaced_try and candidate_prompt_try != current_prompt:
                            candidate_words_for_eval.append(str(word))
                            candidate_prompts_for_eval.append(candidate_prompt_try)
                            strategy_name = ""
                            strategy_title = ""
                            if flux2_strategy_cwor_enabled and int(word_idx) < len(generated_strategy_entries):
                                strategy_name = str(
                                    generated_strategy_entries[int(word_idx)].get("strategy_name", "")
                                ).strip()
                                strategy_title = str(
                                    generated_strategy_entries[int(word_idx)].get(
                                        "strategy_title",
                                        strategy_name,
                                    )
                                ).strip()
                            candidate_strategy_names_for_eval.append(strategy_name)
                            candidate_strategy_titles_for_eval.append(strategy_title)
                            if len(candidate_prompts_for_eval) >= max_eval_candidates:
                                break

                    if len(candidate_prompts_for_eval) > 0:
                        prompt_score_errors: List[str] = []
                        args.input_img_path = str(step_input_img_path)
                        runtime.setup(args=args, has_input_image=has_input_image)

                        branch_results, branch_error, _ = runtime.evaluate_candidates(
                            args=args,
                            classifier=classifier,
                            candidate_words=candidate_words_for_eval,
                            candidate_prompts=candidate_prompts_for_eval,
                            has_input_image=has_input_image,
                            render_output_path=None,
                            capture_classifier_tile_image=True,
                        )
                        raw_prompt_query_count = getattr(
                            runtime,
                            "last_prompt_query_count",
                            len(branch_results),
                        )
                        try:
                            prompt_query_attempt_count += max(
                                0,
                                int(raw_prompt_query_count),
                            )
                        except Exception:
                            prompt_query_attempt_count += int(len(branch_results))
                        for result_idx, item in enumerate(branch_results):
                            if int(result_idx) < len(candidate_strategy_names_for_eval):
                                strategy_name = str(candidate_strategy_names_for_eval[int(result_idx)]).strip()
                                strategy_title = str(candidate_strategy_titles_for_eval[int(result_idx)]).strip()
                                if strategy_name:
                                    item["candidate_strategy_name"] = strategy_name
                                    item["candidate_strategy_title"] = strategy_title or strategy_name
                        prompt_scored_candidates.extend(branch_results)
                        if branch_error:
                            prompt_score_errors.append(str(branch_error))

                        if len(prompt_score_errors) > 0:
                            prompt_score_error = " | ".join(prompt_score_errors)
                    else:
                        prompt_score_error = "no_candidates"
            else:
                vlm_error = "skipped_text_eval_due_to_cwor_success"

            def _json_safe_candidate(candidate_item: Dict[str, object]) -> Dict[str, object]:
                return {
                    key: value
                    for key, value in dict(candidate_item).items()
                    if key not in {
                        "candidate_classifier_image",
                        "candidate_selected_image",
                        "candidate_precomputed_selected_image_path",
                    }
                }

            def _early_prompt_success_is_acceptable(candidate_item: Dict[str, object]) -> bool:
                return compute_attack_success(
                    pred_idx=candidate_item.get("pred_idx"),
                    classifier_label=args.classifier_label,
                    objective_mode=str(args.classifier_objective),
                ) is True

            prompt_success_early_stop = False
            if (
                bool(args.gcg_early_stop_on_attack_success)
            ):
                prompt_success_early_stop = any(
                    _early_prompt_success_is_acceptable(item)
                    for item in prompt_scored_candidates
                )

            strategy_weighted_prompt_candidates: List[Dict[str, object]] = list(prompt_scored_candidates)
            strategy_merge_mode = str(
                getattr(args, "flux2_strategy_cwor_merge_mode", "weighted") or "weighted"
            ).strip().lower()
            if (
                flux2_strategy_cwor_enabled
                and strategy_merge_mode not in {"greedy", "and"}
                and len(cwor_feedback_for_eval) > 0
            ):
                strategy_weighted_prompt_candidates = dedupe_feedback_candidates_by_prompt(
                    [*cwor_feedback_for_eval, *prompt_scored_candidates]
                )

            strategy_query_budget = int(
                max(0, int(remaining_query_budget_before_step - prompt_query_attempt_count))
            )
            if (
                not prompt_success_early_stop
                and flux2_strategy_cwor_enabled
                and len(strategy_weighted_prompt_candidates) > 0
                and strategy_query_budget > 0
            ):
                strategy_delta_mode = str(
                    getattr(args, "flux2_strategy_cwor_delta_mode", "score") or "score"
                ).strip().lower()
                strategy_cwor_base_score = cwor_base_logit_for_step
                if strategy_delta_mode == "conf":
                    strategy_cwor_base_score = cwor_base_confidence_for_step
                setattr(args, "_flux2_strategy_cwor_query_budget", int(strategy_query_budget))
                strategy_cwor_candidates, strategy_cwor_error = runtime.evaluate_and_candidate(
                    classifier=classifier,
                    cwor_result_prompt=cwor_base_prompt_for_step,
                    cwor_base_confidence=strategy_cwor_base_score,
                    original_objective=initial_objective,
                    prompt_candidates=strategy_weighted_prompt_candidates,
                    cwor_step_index=int(step) + 1,
                )
                strategy_query_count = 0
                raw_runtime_query_count = getattr(runtime, "last_and_query_count", None)
                try:
                    parsed_runtime_query_count = int(raw_runtime_query_count)
                except Exception:
                    parsed_runtime_query_count = None
                if parsed_runtime_query_count is not None and parsed_runtime_query_count >= 0:
                    strategy_query_count = int(parsed_runtime_query_count)
                elif len(strategy_cwor_candidates) > 0:
                    raw_strategy_query_count = strategy_cwor_candidates[0].get("cwor_strategy_query_count")
                    try:
                        parsed_strategy_query_count = int(raw_strategy_query_count)
                    except Exception:
                        parsed_strategy_query_count = None
                    if parsed_strategy_query_count is not None and parsed_strategy_query_count >= 0:
                        strategy_query_count = int(parsed_strategy_query_count)
                cwor_queries_this_step += int(strategy_query_count)
                cwor_pre_candidates.extend(strategy_cwor_candidates)
                if strategy_cwor_error:
                    cwor_pre_error = str(strategy_cwor_error)

            scored_candidates = list(prompt_scored_candidates)
            if len(cwor_pre_candidates) > 0:
                scored_candidates.extend(cwor_pre_candidates)

            score_errors: List[str] = []
            if prompt_score_error:
                score_errors.append(str(prompt_score_error))
            if cwor_pre_error:
                score_errors.append(str(cwor_pre_error))
            score_error: Optional[str] = None
            if len(score_errors) > 0:
                score_error = " | ".join(score_errors)

            prompt_queries_this_step = int(prompt_query_attempt_count)
            # Count only actual victim scoring calls (exclude baseline and reserved CWOR budget slots).
            victim_queries_this_step = int(int(cwor_queries_this_step) + prompt_queries_this_step)
            victim_query_count_before_step = int(victim_query_count_running)
            victim_query_count_running += int(victim_queries_this_step)
            remaining_query_budget_after_step = int(max(0, max_victim_queries - victim_query_count_running))

            cwor_candidates_for_step = summarize_cwor_candidates(cwor_pre_candidates)
            feedback_previous_summary = summarize_cwor_feedback_candidates(cwor_feedback_for_step_raw)
            feedback_used_summary = summarize_cwor_feedback_candidates(cwor_feedback_for_eval)
            cwor_step_record: Dict[str, object] = {
                "step": int(step) + 1,
                "mode": {
                    "enabled": bool(cwor_enabled),
                    "strategy_weighted": bool(flux2_strategy_cwor_enabled),
                    "reference_mode": str(cwor_reference_mode_for_step),
                },
                "queries": {
                    "budget_before": int(remaining_query_budget_before_step),
                    "budget_after": int(remaining_query_budget_after_step),
                    "victim": int(victim_queries_this_step),
                    "cwor": int(cwor_queries_this_step),
                },
                "base": {
                    "prompt": str(cwor_base_prompt_for_step),
                    "confidence": (
                        None if cwor_base_confidence_for_step is None else float(cwor_base_confidence_for_step)
                    ),
                    "logit": None if cwor_base_logit_for_step is None else float(cwor_base_logit_for_step),
                },
                "feedback": {
                    "previous": feedback_previous_summary,
                },
                "cwor": {
                    "candidates": cwor_candidates_for_step,
                },
            }
            if feedback_used_summary and feedback_used_summary != feedback_previous_summary:
                cwor_step_record["feedback"]["used_for_cwor"] = feedback_used_summary
            if flux2_strategy_cwor_enabled:
                strategy_current_summary = summarize_cwor_feedback_candidates(prompt_scored_candidates)
                strategy_used_summary = summarize_cwor_feedback_candidates(strategy_weighted_prompt_candidates)
                cwor_step_record["strategy_pool"] = {
                    "current_step": strategy_current_summary,
                }
                if strategy_used_summary and strategy_used_summary != strategy_current_summary:
                    cwor_step_record["strategy_pool"]["used_for_cwor"] = strategy_used_summary
            if cwor_reference_candidate is not None:
                cwor_step_record["cwor"]["reference_candidate"] = summarize_cwor_feedback_candidates(
                    [cwor_reference_candidate]
                )[0]
            if cwor_pre_error is not None:
                cwor_step_record["score_error"] = str(cwor_pre_error)
            cwor_step_status = "ok"
            cwor_step_reason: Optional[str] = None
            strategy_requires_feedback = bool(
                flux2_strategy_cwor_enabled and strategy_merge_mode not in {"greedy", "and"}
            )
            if not cwor_enabled:
                cwor_step_status = "disabled"
            elif flux2_strategy_cwor_enabled and len(prompt_scored_candidates) == 0:
                cwor_step_status = "skipped"
                cwor_step_reason = "no_current_step_strategy_candidates"
            elif flux2_strategy_cwor_enabled and len(cwor_candidates_for_step) == 0:
                cwor_step_status = "empty"
            elif cwor_base_logit_for_step is None:
                cwor_step_status = "skipped"
                cwor_step_reason = "base_confidence_missing"
            elif strategy_requires_feedback and len(cwor_feedback_for_step) == 0:
                cwor_step_status = "skipped"
                cwor_step_reason = "no_previous_step_feedback"
            elif strategy_requires_feedback and len(cwor_feedback_for_eval) == 0:
                cwor_step_status = "skipped"
                cwor_step_reason = "no_feedback_after_reference_selection"
            elif len(cwor_candidates_for_step) == 0:
                cwor_step_status = "empty"
            cwor_step_record["status"] = cwor_step_status
            if cwor_step_reason:
                cwor_step_record["reason"] = cwor_step_reason

            prompt_feedback_candidates = [
                dict(item)
                for item in scored_candidates
                if str(item.get("candidate_variant", "prompt")) == "prompt"
            ]
            feedback_candidates_for_next_step = list(prompt_feedback_candidates)
            if str(cwor_reference_mode_for_step) == "best_candidate" and len(cwor_pre_candidates) > 0:
                best_prompt_objective = float("-inf")
                for item in prompt_feedback_candidates:
                    parsed = _to_finite_float(item.get("candidate_objective"))
                    if parsed is not None and parsed > best_prompt_objective:
                        best_prompt_objective = float(parsed)

                best_cwor_candidate_for_feedback = max(
                    cwor_pre_candidates,
                    key=lambda item: float(item.get("candidate_objective", float("-inf"))),
                )
                best_cwor_objective = _to_finite_float(
                    best_cwor_candidate_for_feedback.get("candidate_objective")
                )
                if best_cwor_objective is not None and float(best_cwor_objective) >= float(best_prompt_objective):
                    feedback_candidates_for_next_step.append(dict(best_cwor_candidate_for_feedback))

            only_improve_feedback_for_step_prompt = bool(
                bool(getattr(args, "cwor_accumulate_update_if_improved_only", False))
                and str(getattr(args, "cwor_feedback_merge_mode", "accumulate")) == "step_prompt_weighted"
            )
            filtered_feedback_candidates_for_next_step = [
                dict(item) for item in feedback_candidates_for_next_step
            ]
            merged_feedback_candidates_for_next_step = dedupe_feedback_candidates_by_prompt(
                [*cwor_feedback_for_step_raw, *filtered_feedback_candidates_for_next_step]
            )
            cwor_state_updated_this_step: Optional[bool] = None
            if only_improve_feedback_for_step_prompt:
                cwor_state_update_flags = [
                    bool(item.get("cwor_state_updated"))
                    for item in cwor_pre_candidates
                    if (
                        str(item.get("candidate_variant", "")) == "cwor"
                        and item.get("cwor_state_updated") is not None
                    )
                ]
                if len(cwor_state_update_flags) > 0:
                    cwor_state_updated_this_step = bool(any(cwor_state_update_flags))
                if cwor_state_updated_this_step is False:
                    previous_step_vlm_feedback = []
                else:
                    # Keep bootstrap behavior when CWOR was not evaluated yet.
                    previous_step_vlm_feedback = merged_feedback_candidates_for_next_step
            else:
                previous_step_vlm_feedback = merged_feedback_candidates_for_next_step

            cwor_step_record["feedback"]["next_step"] = summarize_cwor_feedback_candidates(previous_step_vlm_feedback)
            if only_improve_feedback_for_step_prompt:
                cwor_step_record["feedback"]["gate"] = "cwor_state_updated"
                cwor_step_record["feedback"]["cwor_state_updated"] = cwor_state_updated_this_step

            candidate_word = fallback_word
            candidate_prompt = current_prompt
            candidate_objective = float(best_objective)
            candidate_pred_idx = None
            candidate_pred_conf = None
            candidate_target_conf = None
            candidate_target_label_conf = None
            candidate_pred_logit = None
            candidate_target_logit = None
            candidate_target_label_logit = None
            candidate_ce = None
            best_candidate_idx: Optional[int] = None
            best_candidate: Dict[str, object] = {}

            def _absolute_success_query_count(
                success_item: Dict[str, object],
                fallback_index: int,
            ) -> int:
                raw_query_offset = success_item.get("cwor_strategy_query_offset")
                try:
                    cwor_query_offset = int(raw_query_offset)
                except Exception:
                    cwor_query_offset = None
                if cwor_query_offset is not None and cwor_query_offset > 0:
                    return int(
                        victim_query_count_before_step
                        + prompt_query_attempt_count
                        + cwor_query_offset
                    )
                raw_prompt_offset = success_item.get(
                    "candidate_strip_index",
                    fallback_index,
                )
                try:
                    prompt_query_offset = max(0, int(raw_prompt_offset)) + 1
                except Exception:
                    prompt_query_offset = int(fallback_index) + 1
                return int(victim_query_count_before_step + prompt_query_offset)

            # The synchronous classifier callback saves the image before the
            # runtime can attach prompt/strategy metadata. Enrich that record
            # from the ordered returned candidates without regenerating it.
            if attack_success_observed and len(scored_candidates) > 0:
                if not attack_success_candidate or not attack_success_candidate.get("candidate_prompt"):
                    for success_idx, success_item in enumerate(scored_candidates):
                        if compute_attack_success(
                            pred_idx=success_item.get("pred_idx"),
                            classifier_label=args.classifier_label,
                            objective_mode=str(args.classifier_objective),
                        ) is True:
                            attack_success_candidate = _json_safe_candidate(dict(success_item))
                            attack_success_query_count = _absolute_success_query_count(
                                success_item,
                                success_idx,
                            )
                            break

            if not attack_success_observed and len(scored_candidates) > 0:
                success_save_errors: List[str] = []
                for success_idx, success_item in enumerate(scored_candidates):
                    if compute_attack_success(
                        pred_idx=success_item.get("pred_idx"),
                        classifier_label=args.classifier_label,
                        objective_mode=str(args.classifier_objective),
                    ) is not True:
                        continue
                    attack_success_observed = True
                    if attack_success_query_count is None:
                        attack_success_query_count = _absolute_success_query_count(
                            success_item,
                            success_idx,
                        )
                        attack_success_candidate = _json_safe_candidate(dict(success_item))
                    try:
                        saved_path = save_evaluated_attack_image(
                            dict(success_item),
                            output_path,
                            saved_image_size=int(args.saved_image_size),
                        )
                        attack_success_image_relpath = relpath_from_run_dir(run_dir, saved_path)
                        attack_success_image_error = None
                        break
                    except Exception as exc:
                        success_save_errors.append(f"{type(exc).__name__}:{exc}")
                if attack_success_observed and attack_success_image_relpath is None:
                    attack_success_image_error = " | ".join(success_save_errors) or "candidate_image_missing"

            if len(scored_candidates) > 0:
                successful_indices = [
                    idx
                    for idx, item in enumerate(scored_candidates)
                    if compute_attack_success(
                        pred_idx=item.get("pred_idx"),
                        classifier_label=args.classifier_label,
                        objective_mode=str(args.classifier_objective),
                    ) is True
                ]
                if bool(args.gcg_early_stop_on_attack_success) and successful_indices:
                    # The callback saved the first successful query, so keep
                    # prompt/report metadata aligned with that exact image.
                    best_candidate_idx = int(successful_indices[0])
                else:
                    best_candidate_idx = max(
                        range(len(scored_candidates)),
                        key=lambda idx: float(scored_candidates[idx]["candidate_objective"]),
                    )
                best_candidate = scored_candidates[int(best_candidate_idx)]
                candidate_word = str(best_candidate["candidate_word"])
                candidate_prompt = str(best_candidate["candidate_prompt"])
                candidate_objective = float(best_candidate["candidate_objective"])
                candidate_pred_idx = best_candidate.get("pred_idx")
                candidate_pred_conf = best_candidate.get("pred_conf")
                candidate_target_conf = best_candidate.get("target_conf")
                candidate_target_label_conf = best_candidate.get("target_label_conf")
                candidate_pred_logit = best_candidate.get("pred_logit", best_candidate.get("pred_conf"))
                candidate_target_logit = best_candidate.get("target_logit", best_candidate.get("target_conf"))
                candidate_target_label_logit = best_candidate.get(
                    "target_label_logit",
                    best_candidate.get("target_label_conf"),
                )
                candidate_ce = best_candidate.get("ce")

            objective_improved = bool(len(scored_candidates) > 0 and candidate_objective > float(best_objective))
            prediction_changed = bool(
                candidate_pred_idx is not None
                and best_pred_idx is not None
                and int(candidate_pred_idx) != int(best_pred_idx)
            )
            candidate_attack_success = compute_attack_success(
                pred_idx=candidate_pred_idx,
                classifier_label=args.classifier_label,
                objective_mode=str(args.classifier_objective),
            )
            early_stop_triggered = bool(
                bool(args.gcg_early_stop_on_attack_success)
                and candidate_attack_success is True
            )
            accepted = bool(
                len(scored_candidates) > 0
                and (
                    objective_improved
                    or not prediction_changed
                    or early_stop_triggered
                )
            )

            if objective_improved or early_stop_triggered:
                best_objective = float(candidate_objective)
                best_output_prompt = str(candidate_prompt)
                candidate_word_norm = str(candidate_word).strip().lower()
                candidate_prompt_text = str(candidate_prompt)
                # GS mode can produce CWOR candidates whose prompt no longer contains
                # a replaceable slot marker (<CWOR>/<scene>/...). In that case, keep
                # the previous text prompt anchor to avoid `no_candidates` loops.
                has_replaceable_slot = any(
                    marker in candidate_prompt_text
                    for marker in ("<scene>", "{scene}", "<object>", "{object}", "<CWOR>", "{CWOR}", "<cwor>", "{cwor}")
                )
                if not (candidate_word_norm == "<cwor>" and not has_replaceable_slot):
                    current_prompt = candidate_prompt
                    current_word = candidate_word
                best_pred_idx = candidate_pred_idx
                best_pred_conf = candidate_pred_conf
                best_target_conf = candidate_target_conf
                best_target_label_conf = candidate_target_label_conf
                best_pred_logit = candidate_pred_logit
                best_target_logit = candidate_target_logit
                best_target_label_logit = candidate_target_label_logit
                best_ce = candidate_ce

            entry: Dict[str, object] = {
                "step": int(step),
                "accepted": accepted,
                "candidate_word": str(candidate_word),
                "raw_vlm_answer": str(raw_answer),
                "candidate_prompt": str(candidate_prompt),
                "current_prompt": str(current_prompt),
                "candidate_objective": float(candidate_objective),
                "best_objective": float(best_objective),
                "token_update_method": "vlm_black_box",
                "requested_candidate_source": candidate_source,
                "scene_vocab_selected_words": list(generated_words),
                "candidate_count": int(len(scored_candidates)),
                "candidate_text_count": int(len(candidate_prompts_for_eval)),
                "next_step_prompt_feedback_candidate_count": int(len(prompt_feedback_candidates)),
                "next_step_prompt_feedback_included_count": int(len(previous_step_vlm_feedback)),
                "victim_queries_this_step": int(victim_queries_this_step),
                "scored_candidates": [_json_safe_candidate(item) for item in scored_candidates],
                "pred_idx": candidate_pred_idx,
                "pred_conf": candidate_pred_conf,
                "target_conf": candidate_target_conf,
                "target_label_conf": candidate_target_label_conf,
                "pred_logit": candidate_pred_logit,
                "target_logit": candidate_target_logit,
                "target_label_logit": candidate_target_label_logit,
                "ce": candidate_ce,
                "attack_success": candidate_attack_success,
                "attack_success_rule": attack_success_rule_text,
                "early_stop_triggered": bool(early_stop_triggered),
            }
            if cwor_base_confidence_for_step is not None:
                entry["cwor_base_confidence"] = float(cwor_base_confidence_for_step)
                entry["cwor_mode"] = str(cwor_mode)
            if bool(getattr(args, "cwor_enable", False)):
                entry["cwor_reference_mode"] = str(cwor_reference_mode_for_step)
                entry["cwor_embed_inject_mode"] = str(getattr(args, "cwor_embed_inject_mode", "both"))
                entry["cwor_feedback_merge_mode"] = str(getattr(args, "cwor_feedback_merge_mode", "accumulate"))
            if cwor_target_label is not None:
                entry["cwor_target_label"] = int(cwor_target_label)
            entry["cwor_enable"] = bool(getattr(args, "cwor_enable", False))
            if "cwor_greedy_margin_shadow_candidate" in best_candidate:
                entry["cwor_greedy_margin_shadow_candidate"] = bool(
                    best_candidate.get("cwor_greedy_margin_shadow_candidate")
                )
            if "cwor_greedy_margin_shadow_order" in best_candidate:
                entry["cwor_greedy_margin_shadow_order"] = int(
                    best_candidate.get("cwor_greedy_margin_shadow_order")
                )
            if "cwor_state_updated" in best_candidate:
                entry["cwor_state_updated"] = bool(best_candidate.get("cwor_state_updated"))
            if only_improve_feedback_for_step_prompt:
                entry["cwor_state_updated_this_step"] = cwor_state_updated_this_step
            if generation_prompt:
                entry["scene_vocab_generation_prompt"] = generation_prompt
            if feedback_for_generation:
                entry["scene_vocab_feedback_used"] = feedback_for_generation
            safe_vlm_error = (
                redact_transient_paths(
                    vlm_error,
                    transient_input_paths,
                    sensitive_values=sensitive_values,
                )
                if vlm_error
                else None
            )
            safe_score_error = (
                redact_transient_paths(
                    score_error,
                    transient_input_paths,
                    sensitive_values=sensitive_values,
                )
                if score_error
                else None
            )
            if safe_vlm_error:
                entry["vlm_error"] = safe_vlm_error
            if safe_score_error:
                entry["score_error"] = safe_score_error
            if early_stop_triggered:
                entry["early_stop_reason"] = "attack_success"

            prompt_text_for_artifact = generation_prompt if generation_prompt else str(args.scene_vlm_question)
            prompt_artifact_paths = runtime.save_prompt_artifacts(
                run_dir=run_dir,
                step_idx=step,
                candidate_source=candidate_source,
                prompt_text=str(
                    redact_transient_payload(
                        prompt_text_for_artifact,
                        transient_input_paths,
                        sensitive_values,
                    )
                ),
                raw_answer=str(
                    redact_transient_payload(
                        str(raw_answer),
                        transient_input_paths,
                        sensitive_values,
                    )
                ),
                feedback_used=redact_transient_payload(
                    feedback_for_generation,
                    transient_input_paths,
                    sensitive_values,
                ),
                generated_words=redact_transient_payload(
                    generated_words,
                    transient_input_paths,
                    sensitive_values,
                ),
                filtered_words=redact_transient_payload(
                    candidate_words_for_eval,
                    transient_input_paths,
                    sensitive_values,
                ),
                scored_candidates=redact_transient_payload(
                    scored_candidates,
                    transient_input_paths,
                    sensitive_values,
                ),
                vlm_error=safe_vlm_error,
                score_error=safe_score_error,
            )
            if candidate_source == "gemma_scene_vocab":
                entry["scene_vocab_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                entry["scene_vocab_response_json_path"] = prompt_artifact_paths.get("response_json_path")
            else:
                entry["vlm_query_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                entry["vlm_query_response_json_path"] = prompt_artifact_paths.get("response_json_path")

            history.append(entry)

            trace_step: Dict[str, object] = {
                "step": int(step) + 1,
                "kind": "gcg_step",
                "accepted": bool(accepted),
                "prompt": str(current_prompt),
                "candidate_prompt": str(candidate_prompt),
                "objective": float(best_objective),
                "candidate_objective": float(candidate_objective),
                "candidate_word": str(candidate_word),
                "requested_candidate_source": str(candidate_source),
                "used_candidate_source": str(candidate_source),
                "candidate_count": int(len(scored_candidates)),
                "candidate_text_count": int(len(candidate_prompts_for_eval)),
                "next_step_prompt_feedback_candidate_count": int(len(prompt_feedback_candidates)),
                "next_step_prompt_feedback_included_count": int(len(previous_step_vlm_feedback)),
                "victim_queries_this_step": int(victim_queries_this_step),
                "pred_idx": candidate_pred_idx,
                "pred_conf": candidate_pred_conf,
                "target_conf": candidate_target_conf,
                "target_label_conf": candidate_target_label_conf,
                "pred_logit": candidate_pred_logit,
                "target_logit": candidate_target_logit,
                "target_label_logit": candidate_target_label_logit,
                "ce": candidate_ce,
                "attack_success": candidate_attack_success,
                "attack_success_rule": attack_success_rule_text,
                "early_stop_triggered": bool(early_stop_triggered),
            }
            if cwor_base_confidence_for_step is not None:
                trace_step["cwor_base_confidence"] = float(cwor_base_confidence_for_step)
                trace_step["cwor_mode"] = str(cwor_mode)
            if bool(getattr(args, "cwor_enable", False)):
                trace_step["cwor_reference_mode"] = str(cwor_reference_mode_for_step)
                trace_step["cwor_embed_inject_mode"] = str(getattr(args, "cwor_embed_inject_mode", "both"))
                trace_step["cwor_feedback_merge_mode"] = str(getattr(args, "cwor_feedback_merge_mode", "accumulate"))
            if cwor_target_label is not None:
                trace_step["cwor_target_label"] = int(cwor_target_label)
            if "cwor_greedy_margin_shadow_candidate" in best_candidate:
                trace_step["cwor_greedy_margin_shadow_candidate"] = bool(
                    best_candidate.get("cwor_greedy_margin_shadow_candidate")
                )
            if "cwor_greedy_margin_shadow_order" in best_candidate:
                trace_step["cwor_greedy_margin_shadow_order"] = int(
                    best_candidate.get("cwor_greedy_margin_shadow_order")
                )
            trace_step["cwor_enable"] = bool(getattr(args, "cwor_enable", False))
            if "cwor_state_updated" in entry:
                trace_step["cwor_state_updated"] = entry["cwor_state_updated"]
            if "cwor_state_updated_this_step" in entry:
                trace_step["cwor_state_updated_this_step"] = entry["cwor_state_updated_this_step"]
            if "step_image_path" in entry:
                trace_step["step_image_path"] = entry["step_image_path"]
            if "cwor_step_image_path" in entry:
                trace_step["cwor_step_image_path"] = entry["cwor_step_image_path"]
            if "step_image_reason" in entry:
                trace_step["step_image_reason"] = entry["step_image_reason"]
            prompt_trace_steps.append(trace_step)
            write_json(
                step_prompts_path,
                redact_transient_payload(
                    prompt_trace_doc,
                    transient_input_paths,
                    sensitive_values,
                ),
            )

            if wandb_enabled and wandb_run is not None:
                wandb_payload: Dict[str, object] = {
                    "step": int(step) + 1,
                    "objective/candidate": float(candidate_objective),
                    "objective/best": float(best_objective),
                    "loss/cls": float(candidate_objective),
                    "loss/adv": float(candidate_objective),
                    "loss/opt": float(-candidate_objective),
                    "candidate/source": str(candidate_source),
                    "candidate/count": int(len(scored_candidates)),
                    "flags/accepted_candidate": 1 if accepted else 0,
                    "pred/pred_idx": -1 if candidate_pred_idx is None else int(candidate_pred_idx),
                }
                if candidate_attack_success is not None:
                    wandb_payload["flags/attack_success"] = 1 if bool(candidate_attack_success) else 0
                if early_stop_triggered:
                    wandb_payload["flags/early_stop"] = 1
                if candidate_pred_logit is not None:
                    wandb_payload["logit/pred"] = float(candidate_pred_logit)
                if candidate_target_logit is not None:
                    wandb_payload["logit/target"] = float(candidate_target_logit)
                if candidate_target_label_logit is not None:
                    wandb_payload["logit/cwor_target_label"] = float(candidate_target_label_logit)
                if candidate_pred_conf is not None:
                    wandb_payload["conf/pred"] = float(candidate_pred_conf)
                if candidate_target_conf is not None:
                    wandb_payload["conf/target"] = float(candidate_target_conf)
                if candidate_target_label_conf is not None:
                    wandb_payload["conf/cwor_target_label"] = float(candidate_target_label_conf)
                if candidate_ce is not None:
                    wandb_payload["ce"] = float(candidate_ce)
                wandb_enabled = log_wandb_payload(
                    run=wandb_run,
                    args=args,
                    payload=wandb_payload,
                    step=int(step) + 1,
                    honor_log_every=True,
                )

            if candidate_source == "gemma_scene_vocab":
                previous_scene_feedback = merge_scene_vocab_feedback_history(
                    existing_feedback=previous_scene_feedback,
                    generated_words=generated_words,
                    scored_candidates=scored_candidates,
                )

            if early_stop_triggered:
                early_stop_event = {
                    "enabled": bool(args.gcg_early_stop_on_attack_success),
                    "triggered": True,
                    "reason": "attack_success",
                    "step": int(step) + 1,
                    "candidate_word": str(candidate_word),
                    "candidate_prompt": str(candidate_prompt),
                    "candidate_objective": float(candidate_objective),
                    "pred_idx": None if candidate_pred_idx is None else int(candidate_pred_idx),
                    "classifier_label": int(args.classifier_label),
                    "attack_success_rule": attack_success_rule_text,
                }
                break

            if not accepted:
                # Keep exploring in black-box mode even when a step is rejected.
                continue

        # Only generated/evaluated candidates count as an attack success.  A
        # source image that is already misclassified is a baseline condition,
        # not an image produced by this attack.
        final_attack_success = bool(attack_success_observed)
        if attack_success_observed and attack_success_image_relpath is None:
            attack_success_image_error = attack_success_image_error or "successful_candidate_image_not_saved"
        if attack_success_image_relpath is not None:
            if not output_path.is_file():
                attack_success_image_error = "saved_attack_image_missing"
                attack_success_image_relpath = None
            else:
                with Image.open(output_path) as saved_image:
                    if saved_image.size != (224, 224):
                        raise RuntimeError(
                            f"saved attack image must be 224x224, got {saved_image.size}"
                        )

        if wandb_enabled and wandb_run is not None and output_path.is_file():
            wandb_enabled = log_wandb_final_image(
                run=wandb_run,
                image_path=output_path,
                step=int(len(history)),
            )

        write_report(
            args=args,
            unknown_args=list(unknown_args or []),
            original_prompt=editable_prompt,
            optimized_prompt=best_output_prompt,
            best_objective=best_objective,
            history=history,
            early_stop_event=early_stop_event,
            final_attack_success=final_attack_success,
            attack_success_image_path=attack_success_image_relpath,
            attack_success_image_error=attack_success_image_error,
            attack_success_query_count=attack_success_query_count,
            attack_success_candidate=attack_success_candidate,
        )

        accepted_steps = sum(1 for item in history if bool(item.get("accepted", False)))
        victim_query_count = compute_victim_query_count(history)
        finish_wandb_run(
            run=wandb_run,
            summary={
                "final_best_objective": float(best_objective),
                "final_pred_idx": None if best_pred_idx is None else int(best_pred_idx),
                "final_pred_conf": None if best_pred_conf is None else float(best_pred_conf),
                "final_pred_logit": None if best_pred_logit is None else float(best_pred_logit),
                "final_target_logit": None if best_target_logit is None else float(best_target_logit),
                "final_ce": None if best_ce is None else float(best_ce),
                "initial_pred_idx": None if initial_pred_idx is None else int(initial_pred_idx),
                "initial_pred_conf": None if initial_pred_conf is None else float(initial_pred_conf),
                "initial_pred_logit": None if initial_pred_logit is None else float(initial_pred_logit),
                "initial_target_logit": None if initial_target_logit is None else float(initial_target_logit),
                "initial_ce": None if initial_ce is None else float(initial_ce),
                "history_len": int(len(history)),
                "accepted_steps": int(accepted_steps),
                "victim_query_count": int(victim_query_count),
                "final_attack_success": final_attack_success,
                "attack_success_rule": attack_success_rule_text,
                "early_stop_enabled": bool(args.gcg_early_stop_on_attack_success),
                "early_stop_triggered": bool(early_stop_event is not None),
                "early_stop_reason": None if early_stop_event is None else early_stop_event.get("reason"),
                "classifier_mode": "black-box",
                "candidate_source": str(args.gcg_candidate_source),
                "attack_mode": str(attack_mode),
                "original_prompt": str(editable_prompt),
                "optimized_prompt": str(best_output_prompt),
            },
        )
        wandb_run_finished = True

        result: Dict[str, object] = {
            "status": "ok",
            "history_len": int(len(history)),
            "accepted_steps": int(accepted_steps),
            "victim_query_count": int(victim_query_count),
            "final_attack_success": final_attack_success,
            "early_stop_enabled": bool(args.gcg_early_stop_on_attack_success),
            "early_stop_triggered": bool(early_stop_event is not None),
            "early_stop_reason": None if early_stop_event is None else early_stop_event.get("reason"),
            "best_objective": float(best_objective),
            "final_prompt": str(best_output_prompt),
            "attack_mode": str(attack_mode),
            "report_path": str(report_path),
            "step_trace_path": str(step_prompts_path),
        }
        if attack_success_image_relpath is not None:
            result["attack_success_image_path"] = str(attack_success_image_relpath)
            result["output_path"] = str(output_path)
        if attack_success_image_error is not None:
            result["attack_success_image_error"] = str(attack_success_image_error)
        if attack_success_query_count is not None:
            result["attack_success_query_count"] = int(attack_success_query_count)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception:
        # If a later artifact/runtime operation fails after the synchronous
        # success save, preserve the success state before the runner records
        # the sample-level failure.
        if attack_success_observed and output_path.is_file():
            try:
                emergency_history = list(history)
                accounted_queries = compute_victim_query_count(emergency_history)
                if int(live_victim_query_count) > int(accounted_queries):
                    emergency_history.append(
                        {
                            "kind": "interrupted_after_success",
                            "accepted": False,
                            "victim_queries_this_step": int(
                                live_victim_query_count - accounted_queries
                            ),
                        }
                    )
                write_report(
                    args=args,
                    unknown_args=list(unknown_args or []),
                    original_prompt=editable_prompt,
                    optimized_prompt=best_output_prompt,
                    best_objective=best_objective,
                    history=emergency_history,
                    early_stop_event=early_stop_event,
                    final_attack_success=True,
                    attack_success_image_path=attack_success_image_relpath,
                    attack_success_image_error=attack_success_image_error,
                    attack_success_query_count=attack_success_query_count,
                    attack_success_candidate=attack_success_candidate,
                )
            except Exception:
                pass
        raise
    finally:
        if not wandb_run_finished and wandb_run is not None:
            finish_wandb_run(
                run=wandb_run,
                summary={
                    "status": "failed",
                    "final_attack_success": bool(attack_success_observed),
                    "victim_query_count": int(live_victim_query_count),
                },
            )
        set_evaluation_callback = getattr(classifier, "set_evaluation_callback", None)
        if callable(set_evaluation_callback):
            set_evaluation_callback(None)
        if bool(manage_runtime):
            runtime.close()

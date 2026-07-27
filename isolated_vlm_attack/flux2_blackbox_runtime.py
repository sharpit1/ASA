"""Minimal FLUX.2 Klein runtime for the supported VLM and AND attacks."""

import gc
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from attack_model_registry import validate_generator_model


_ISOLATED_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(os.getenv("ASA_PROJECT_ROOT", str(_ISOLATED_DIR.parent))).resolve()
_THIRD_PARTY_HF_DIFFUSERS = _PROJECT_ROOT / "third_party" / "hf_diffusers_git"


def _extract_output_images(output: object) -> List[Image.Image]:
    if hasattr(output, "images"):
        images_obj = getattr(output, "images")
    elif isinstance(output, (tuple, list)) and output:
        images_obj = output[0]
    else:
        images_obj = output
    if isinstance(images_obj, Image.Image):
        return [images_obj]
    if isinstance(images_obj, (tuple, list)):
        return [image for image in images_obj if isinstance(image, Image.Image)]
    return []


def _safe_float(value: object, default: float = float("-inf")) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


class Flux2KleinRenderSession:
    def __init__(self, *, args, has_input_image: bool):
        self.args = args
        self.has_input_image = bool(has_input_image)
        self.pipe = None
        self.last_and_query_count = 0
        self._load_pipeline()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))

    def _load_pipeline(self) -> None:
        model_path = str(getattr(self.args, "model_path", "") or "").strip()
        validate_generator_model(model_path, expected_family="flux2-klein")
        model_path_lower = model_path.lower()

        # Delay the pinned diffusers import until model setup so parser/help,
        # static validation, and mock tests do not load GPU/native libraries.
        if _THIRD_PARTY_HF_DIFFUSERS.is_dir():
            third_party_diffusers = str(_THIRD_PARTY_HF_DIFFUSERS)
            if third_party_diffusers not in sys.path:
                sys.path.insert(0, third_party_diffusers)
        try:
            importlib.import_module("regex")
        except Exception:
            pass
        diffusers_module = importlib.import_module("diffusers")
        pipeline_cls = diffusers_module.Flux2KleinPipeline
        flux2_kv_pipeline = getattr(diffusers_module, "Flux2KleinKVPipeline", None)

        if "-kv" in model_path_lower or model_path_lower.endswith("kv"):
            if flux2_kv_pipeline is None:
                raise ImportError("the bundled diffusers build does not provide Flux2KleinKVPipeline")
            pipeline_cls = flux2_kv_pipeline

        load_kwargs = {
            "torch_dtype": torch.float16,
        }
        hf_token = str(getattr(self.args, "hf_token", "") or "").strip()
        if hf_token:
            load_kwargs["token"] = hf_token
        if not bool(getattr(self.args, "cpu_offload", False)):
            load_kwargs["device_map"] = "balanced"
        self.pipe = pipeline_cls.from_pretrained(model_path, **load_kwargs)
        if bool(getattr(self.args, "cpu_offload", False)):
            self.pipe.enable_sequential_cpu_offload()

    def _load_condition_image(self) -> Optional[Image.Image]:
        if not self.has_input_image:
            return None
        raw_path = str(getattr(self.args, "input_img_path", "") or "").strip()
        if not raw_path:
            raise ValueError("FLUX.2 Klein edit mode requires input_img_path")
        image_path = Path(raw_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing input image: {image_path}")
        with Image.open(image_path) as opened:
            return opened.convert("RGB")

    def _pipeline_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "num_inference_steps": int(getattr(self.args, "num_inference_steps", 20)),
            "max_sequence_length": int(getattr(self.args, "max_sequence_length", 512)),
            "output_type": "pil",
        }
        if self.pipe is not None:
            call_parameters = inspect.signature(self.pipe.__call__).parameters
            if "guidance_scale" in call_parameters:
                kwargs["guidance_scale"] = float(getattr(self.args, "guidance_scale", 3.0))
        height = int(getattr(self.args, "height", 0) or 0)
        width = int(getattr(self.args, "width", 0) or 0)
        if height > 0:
            kwargs["height"] = height
        if width > 0:
            kwargs["width"] = width
        condition_image = self._load_condition_image()
        if condition_image is not None:
            kwargs["image"] = [condition_image]
        return kwargs

    @torch.no_grad()
    def _render_prompt_image(self, prompt: str) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("FLUX.2 Klein render session is not initialized")
        self._seed_everything(int(getattr(self.args, "seed", 42)))
        output = self.pipe(prompt=[str(prompt)], **self._pipeline_kwargs())
        images = _extract_output_images(output)
        if len(images) != 1:
            raise RuntimeError(f"FLUX.2 Klein returned {len(images)} images for one prompt")
        return images[0].convert("RGB")

    @torch.no_grad()
    def render_images(
        self,
        *,
        prompts: Sequence[str],
        mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    ) -> List[Image.Image]:
        if mixed_initial_edit_cache is not None:
            raise ValueError("FLUX.2 Klein does not support mixed edit caches")
        if self.pipe is None:
            raise RuntimeError("FLUX.2 Klein render session is not initialized")
        prompt_list = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not prompt_list:
            raise ValueError("render requires at least one prompt")

        self._seed_everything(int(getattr(self.args, "seed", 42)))
        output = self.pipe(prompt=prompt_list, **self._pipeline_kwargs())
        images = _extract_output_images(output)
        if len(images) != len(prompt_list):
            raise RuntimeError(
                f"FLUX.2 Klein returned {len(images)} images for {len(prompt_list)} prompts"
            )
        return [image.convert("RGB").copy() for image in images]

    @staticmethod
    def _candidate_prompt(candidate: Dict[str, object]) -> str:
        return str(candidate.get("candidate_prompt", "") or "").strip()

    @staticmethod
    def _candidate_strategy(candidate: Dict[str, object], index: int) -> str:
        strategy = str(candidate.get("candidate_strategy_name", "") or "").strip()
        return strategy or f"candidate_{int(index):03d}"

    @staticmethod
    def _join_and_prompts(prompts: Sequence[str]) -> str:
        output: List[str] = []
        seen = set()
        for raw_prompt in prompts:
            prompt = str(raw_prompt or "").strip()
            key = prompt.lower()
            if prompt and key not in seen:
                seen.add(key)
                output.append(prompt)
        return " and ".join(output)

    def evaluate_and_candidate(
        self,
        *,
        classifier,
        cwor_result_prompt: str,
        cwor_base_confidence: Optional[float] = None,
        original_objective: Optional[float] = None,
        prompt_candidates: Sequence[Dict[str, object]],
        cwor_step_index: int = 1,
    ) -> Tuple[List[Dict[str, object]], Optional[str]]:
        self.last_and_query_count = 0
        del cwor_result_prompt, cwor_step_index
        if str(getattr(self.args, "flux2_strategy_cwor_merge_mode", "")).strip().lower() != "and":
            return [], "flux2_runtime_supports_only_and_merge"

        prompt_items = [
            dict(item)
            for item in prompt_candidates
            if str(item.get("candidate_variant", "prompt")).strip().lower() == "prompt"
            and self._candidate_prompt(dict(item))
        ]
        best_by_strategy: Dict[str, Dict[str, object]] = {}
        for index, item in enumerate(prompt_items):
            strategy = self._candidate_strategy(item, index)
            item["candidate_strategy_name"] = strategy
            previous = best_by_strategy.get(strategy)
            if previous is None or _safe_float(item.get("candidate_objective")) > _safe_float(
                previous.get("candidate_objective")
            ):
                best_by_strategy[strategy] = item
        ranked = sorted(
            best_by_strategy.values(),
            key=lambda item: _safe_float(item.get("candidate_objective")),
            reverse=True,
        )
        unique_ranked: List[Dict[str, object]] = []
        seen_prompts = set()
        for item in ranked:
            prompt_key = self._candidate_prompt(item).casefold()
            if not prompt_key or prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            unique_ranked.append(item)
        ranked = unique_ranked
        if len(ranked) < 2:
            return [], "flux2_strategy_and_requires_multiple_strategy_candidates"

        query_budget = max(0, int(getattr(self.args, "_flux2_strategy_cwor_query_budget", 1)))
        if query_budget <= 0:
            return [], "flux2_strategy_and_query_budget_exhausted"

        module = __import__("vlm_attack")
        target_label = getattr(self.args, "cwor_target_label", None)
        target_label = None if target_label is None else int(target_label)
        selected_items: List[Dict[str, object]] = [ranked[0]]
        selected_prompts = [self._candidate_prompt(ranked[0])]
        current_objective = _safe_float(ranked[0].get("candidate_objective"))
        evaluated_results: List[Dict[str, object]] = []
        accepted_count = 0
        query_count = 0

        for candidate in ranked[1:]:
            if len(evaluated_results) >= query_budget:
                break
            trial_prompt = self._join_and_prompts(
                [*selected_prompts, self._candidate_prompt(candidate)]
            )
            if not trial_prompt:
                continue
            trial_items = [*selected_items, candidate]
            try:
                trial_image = self._render_prompt_image(trial_prompt)
                image_01 = module.image_to_tensor_01(trial_image).to(
                    device=str(getattr(self.args, "device", "cuda"))
                )
                with torch.no_grad():
                    query_count += 1
                    self.last_and_query_count = int(query_count)
                    objective, stats = classifier.objective_and_stats(
                        image_01,
                        target_label=target_label,
                    )
                exact_artifacts = module.classifier_evaluation_artifacts(classifier)
            except Exception as exc:
                for result in evaluated_results:
                    result["cwor_strategy_query_count"] = int(query_count)
                return evaluated_results, f"flux2_strategy_and:{type(exc).__name__}:{exc}"

            if not exact_artifacts:
                evaluated_image = module.classifier_input_image(trial_image, classifier)
                exact_artifacts = {
                    "candidate_classifier_image": evaluated_image.copy(),
                    "candidate_classifier_image_size": int(evaluated_image.size[0]),
                }
            components = [
                {
                    "strategy_name": str(item.get("candidate_strategy_name", "")),
                    "strategy_title": str(item.get("candidate_strategy_title", "")),
                    "reference_prompt": self._candidate_prompt(item),
                    "reference_objective": item.get("candidate_objective"),
                    "merge_anchor": bool(index == 0),
                }
                for index, item in enumerate(trial_items)
            ]
            evaluated_results.append(
                {
                    "candidate_word": "<CWOR>",
                    "candidate_prompt": trial_prompt,
                    "candidate_objective": float(objective),
                    "pred_idx": stats.get("pred_idx"),
                    "pred_conf": stats.get("pred_conf"),
                    "pred_logit": stats.get("pred_logit"),
                    "target_conf": stats.get("target_conf"),
                    "target_logit": stats.get("target_logit"),
                    "target_label_conf": stats.get("target_label_conf"),
                    "target_label_logit": stats.get("target_label_logit"),
                    "ce": stats.get("ce"),
                    "candidate_variant": "cwor",
                    "candidate_selected_image": trial_image.copy(),
                    **exact_artifacts,
                    "candidate_selected_image_source": "flux2_strategy_and",
                    "cwor_strategy_merge_mode": "and",
                    "cwor_strategy_components": components,
                    "cwor_strategy_query_offset": int(query_count),
                    "flux2_strategy_and_original_objective": original_objective,
                    "flux2_strategy_and_base_confidence": cwor_base_confidence,
                }
            )
            if float(objective) > float(current_objective):
                selected_items.append(candidate)
                selected_prompts.append(self._candidate_prompt(candidate))
                current_objective = float(objective)
                accepted_count += 1

        if not evaluated_results:
            return [], "flux2_strategy_and_no_merge_trial_executed"
        total_queries = int(query_count)
        for result in evaluated_results:
            result["cwor_strategy_query_count"] = total_queries
            result["flux2_strategy_and_accepted_component_count"] = int(accepted_count)
        return evaluated_results, None

    def reset_cwor_aggregate_state(self) -> None:
        return None

    def close(self) -> None:
        pipe = self.pipe
        self.pipe = None
        if pipe is not None:
            try:
                if hasattr(pipe, "to"):
                    pipe.to("cpu")
            except Exception:
                pass
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


class Flux2AttackRuntimeAdapter:
    def __init__(self) -> None:
        self._module = None
        self._runtime_cache = None
        self._render_session: Optional[Flux2KleinRenderSession] = None
        self.last_and_query_count = 0
        self.last_prompt_query_count = 0

    def _get_module(self):
        if self._module is None:
            import vlm_attack as vlm_attack_module

            self._module = vlm_attack_module
        return self._module

    @staticmethod
    def _ensure_args(args) -> None:
        if not str(getattr(args, "model_path", "") or "").strip():
            args.model_path = "black-forest-labs/FLUX.2-klein-9B"
        if not hasattr(args, "cpu_offload"):
            args.cpu_offload = False
        if not hasattr(args, "gcg_scene_vocab_prompts_per_strategy"):
            args.gcg_scene_vocab_prompts_per_strategy = 0
        if not hasattr(args, "gcg_scene_vocab_enabled_strategies"):
            args.gcg_scene_vocab_enabled_strategies = "all"

    def setup(self, *, args, has_input_image: bool) -> None:
        mode = str(getattr(args, "attack_mode", "vlm") or "vlm").strip().lower()
        if mode not in {"vlm", "and"}:
            raise ValueError(f"FLUX.2 runtime does not support attack_mode={mode}")
        self._ensure_args(args)
        validate_generator_model(args.model_path, expected_family="flux2-klein")
        module = self._get_module()
        if self._runtime_cache is None:
            self._runtime_cache = module.PersistentVLMRuntimeCache()

        rebuild = self._render_session is None
        if self._render_session is not None:
            previous_args = self._render_session.args
            rebuild = (
                bool(self._render_session.has_input_image) != bool(has_input_image)
                or str(getattr(previous_args, "model_path", ""))
                != str(getattr(args, "model_path", ""))
                or str(getattr(previous_args, "device", "")) != str(getattr(args, "device", ""))
            )
        if rebuild:
            if self._render_session is not None:
                self._render_session.close()
            self._render_session = Flux2KleinRenderSession(
                args=args,
                has_input_image=bool(has_input_image),
            )
        else:
            self._render_session.args = args
            self._render_session.has_input_image = bool(has_input_image)

    def close(self) -> None:
        render_session = self._render_session
        runtime_cache = self._runtime_cache
        self._render_session = None
        self._runtime_cache = None
        if render_session is not None:
            render_session.close()
        if runtime_cache is not None:
            runtime_cache.close()

    def reset_cwor_state(self) -> None:
        if self._render_session is not None:
            self._render_session.reset_cwor_aggregate_state()

    def generate_scene_vocab_words(self, **kwargs):
        module = self._get_module()
        kwargs["runtime_cache"] = self._runtime_cache
        return module.generate_scene_vocab_words(**kwargs)

    def query_vlm_word(self, **kwargs):
        module = self._get_module()
        kwargs["runtime_cache"] = self._runtime_cache
        return module.query_vlm_word(**kwargs)

    def evaluate_naturalness(self, **kwargs):
        module = self._get_module()
        kwargs["runtime_cache"] = self._runtime_cache
        return module.evaluate_attack_success_naturalness(**kwargs)

    def evaluate_candidates(self, **kwargs):
        module = self._get_module()
        kwargs["render_session"] = self._render_session
        kwargs["mixed_initial_edit_cache"] = None
        self.last_prompt_query_count = 0
        result = module.evaluate_attack_candidates(**kwargs)
        if self._render_session is not None:
            self.last_prompt_query_count = int(
                getattr(self._render_session, "last_prompt_query_count", len(result[0]))
            )
        return result

    def evaluate_and_candidate(self, **kwargs):
        if self._render_session is None:
            self.last_and_query_count = 0
            return [], "flux2_strategy_and_session_not_initialized"
        result = self._render_session.evaluate_and_candidate(**kwargs)
        self.last_and_query_count = int(self._render_session.last_and_query_count)
        return result

    def save_prompt_artifacts(self, **kwargs) -> Dict[str, str]:
        return self._get_module().save_blackbox_prompt_artifacts(**kwargs)

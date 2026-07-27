import gc
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch

from attack_model_registry import validate_generator_model
from qwen_image_edit_batch import render_qwen_image_edit_batch


def _extract_output_images(output: object) -> List[Image.Image]:
    images_obj: object
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


def _parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean flag: {value}")


def _safe_float(value: object, default: float = float("-inf")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class QwenImageEditRenderSession:
    def __init__(self, *, args, has_input_image: bool):
        self.args = args
        self.has_input_image = bool(has_input_image)
        self.pipe = None
        self.last_and_query_count = 0
        self._load_pipeline()

    def _device(self) -> str:
        requested = str(getattr(self.args, "device", "cuda") or "cuda").strip()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested or "cuda"

    def _load_pipeline(self) -> None:
        model_path = str(getattr(self.args, "model_path", "") or "").strip()
        if not model_path:
            model_path = "Qwen/Qwen-Image-Edit-2511"
        validate_generator_model(model_path, expected_family="qwen-image-edit")

        try:
            from diffusers import QwenImageEditPlusPipeline
        except Exception as exc:
            raise ImportError(
                "QwenImageEditPlusPipeline is required for Qwen-Image-Edit-2511. "
                "Install a recent diffusers build, e.g. `pip install git+https://github.com/huggingface/diffusers`."
            ) from exc

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
        }
        hf_token = str(getattr(self.args, "hf_token", "") or "").strip()
        if hf_token:
            load_kwargs["token"] = hf_token
        try:
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, **load_kwargs)
        except TypeError:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, **load_kwargs)

        if bool(getattr(self.args, "cpu_offload", False)):
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self._device())
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=None)

    def _load_condition_image(self) -> Image.Image:
        if not self.has_input_image:
            raise ValueError("Qwen Image Edit requires input_img_path.")
        raw_path = str(getattr(self.args, "input_img_path", "") or "").strip()
        if not raw_path:
            raise ValueError("Qwen Image Edit requires input_img_path.")
        image_path = Path(raw_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing input image: {image_path}")
        with Image.open(image_path) as opened:
            return opened.convert("RGB")

    def _generator(self, seed: int):
        try:
            return torch.Generator(device=self._device()).manual_seed(int(seed))
        except Exception:
            return torch.Generator().manual_seed(int(seed))

    def _pipe_call(self, *, prompt: str, image: Image.Image, seed: int) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("Qwen Image Edit render session is not initialized.")

        kwargs = {
            "image": [image],
            "prompt": str(prompt),
            "generator": self._generator(seed),
            "true_cfg_scale": float(getattr(self.args, "qwen_true_cfg_scale", 4.0)),
            "negative_prompt": str(getattr(self.args, "qwen_negative_prompt", " ") or " "),
            "num_inference_steps": int(getattr(self.args, "num_inference_steps", 4)),
            "max_sequence_length": int(getattr(self.args, "max_sequence_length", 512)),
            "guidance_scale": float(getattr(self.args, "guidance_scale", 1.0)),
            "num_images_per_prompt": int(getattr(self.args, "qwen_num_images_per_prompt", 1)),
        }

        optional_keys = ["true_cfg_scale", "negative_prompt", "num_images_per_prompt"]
        with torch.inference_mode():
            while True:
                try:
                    output = self.pipe(**kwargs)
                    break
                except TypeError as exc:
                    removed = False
                    message = str(exc)
                    for key in list(optional_keys):
                        if key in kwargs and key in message:
                            kwargs.pop(key, None)
                            optional_keys.remove(key)
                            removed = True
                            break
                    if not removed:
                        raise

        images = _extract_output_images(output)
        if len(images) == 0:
            raise RuntimeError("Qwen Image Edit returned no images.")
        return images[0].convert("RGB")

    def _batch_render_supported(self, *, prompt_count: int) -> bool:
        if int(prompt_count) <= 1:
            return False
        if self.pipe is None:
            return False
        if int(getattr(self.args, "qwen_batch_size", 1)) <= 1:
            return False
        if bool(getattr(self.args, "cpu_offload", False)):
            return False
        return True

    def _pipe_call_batch(
        self,
        *,
        prompts: Sequence[str],
        image: Image.Image,
        seeds: Sequence[int],
    ) -> List[Image.Image]:
        if self.pipe is None:
            raise RuntimeError("Qwen Image Edit render session is not initialized.")
        return render_qwen_image_edit_batch(
            pipe=self.pipe,
            prompts=list(prompts),
            image=image,
            generators=[self._generator(int(seed)) for seed in seeds],
            true_cfg_scale=float(getattr(self.args, "qwen_true_cfg_scale", 4.0)),
            negative_prompt=str(getattr(self.args, "qwen_negative_prompt", " ") or " "),
            num_inference_steps=int(getattr(self.args, "num_inference_steps", 4)),
            max_sequence_length=int(getattr(self.args, "max_sequence_length", 512)),
            guidance_scale=float(getattr(self.args, "guidance_scale", 1.0)),
        )

    @staticmethod
    def _clear_failed_batch_allocations() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def render_images(
        self,
        *,
        prompts: Sequence[str],
        mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    ) -> List[Image.Image]:
        if mixed_initial_edit_cache is not None:
            raise ValueError("Qwen Image Edit runtime does not support mixed initial edit caches.")
        prompt_list = [str(p).strip() for p in prompts if str(p).strip()]
        if len(prompt_list) == 0:
            raise ValueError("render requires at least one prompt.")

        condition_image = self._load_condition_image()
        base_seed = int(getattr(self.args, "seed", 42))
        images: List[Image.Image] = []
        if self._batch_render_supported(prompt_count=len(prompt_list)):
            batch_size = max(2, int(getattr(self.args, "qwen_batch_size", 1)))
            try:
                for start in range(0, len(prompt_list), batch_size):
                    chunk = prompt_list[start : start + batch_size]
                    chunk_seeds = [base_seed + idx for idx in range(start, start + len(chunk))]
                    if len(chunk) == 1:
                        images.append(
                            self._pipe_call(
                                prompt=chunk[0],
                                image=condition_image,
                                seed=chunk_seeds[0],
                            )
                        )
                        continue
                    print(
                        f"[qwen_batch] rendering {len(chunk)} prompts in one denoising batch",
                        file=sys.stderr,
                    )
                    images.extend(
                        self._pipe_call_batch(
                            prompts=chunk,
                            image=condition_image,
                            seeds=chunk_seeds,
                        )
                    )
            except Exception as exc:
                if not bool(getattr(self.args, "qwen_batch_fallback", True)):
                    raise
                print(
                    "[qwen_batch] batch render failed; falling back to sequential render: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                images = []
                self._clear_failed_batch_allocations()
        if len(images) == 0:
            images = [
                self._pipe_call(prompt=prompt, image=condition_image, seed=base_seed + idx)
                for idx, prompt in enumerate(prompt_list)
            ]
        if len(images) != len(prompt_list):
            raise RuntimeError(
                f"Qwen Image Edit returned {len(images)} images for {len(prompt_list)} prompts."
            )
        return [image.convert("RGB").copy() for image in images]

    def _score_image(
        self,
        *,
        image: Image.Image,
        classifier,
        cwor_target_label: Optional[int],
    ) -> Tuple[float, Dict[str, object], Dict[str, object]]:
        import vlm_attack as vlm_attack_module

        image_01 = vlm_attack_module.image_to_tensor_01(image).to(device=str(getattr(self.args, "device", "cuda")))
        self.last_and_query_count += 1
        with torch.no_grad():
            objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
        exact_artifacts = vlm_attack_module.classifier_evaluation_artifacts(classifier)
        return float(objective), dict(stats), exact_artifacts

    @staticmethod
    def _candidate_prompt(candidate: Dict[str, object]) -> str:
        return str(candidate.get("candidate_prompt", "") or "").strip()

    @staticmethod
    def _candidate_strategy_name(candidate: Dict[str, object], fallback_idx: int) -> str:
        strategy_name = str(candidate.get("candidate_strategy_name", "") or "").strip()
        if strategy_name:
            return strategy_name
        return f"candidate_{int(fallback_idx):03d}"

    @staticmethod
    def _join_and_prompts(prompts: Sequence[str]) -> str:
        cleaned: List[str] = []
        seen = set()
        for prompt in prompts:
            text = str(prompt or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return " and ".join(cleaned)

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
        del cwor_result_prompt
        if self.pipe is None:
            return [], "qwen_image_edit_session_not_initialized"

        prompt_items = [
            dict(item)
            for item in prompt_candidates
            if str(item.get("candidate_variant", "prompt")).strip().lower() == "prompt"
            and self._candidate_prompt(dict(item))
        ]
        if len(prompt_items) < 2:
            return [], "qwen_strategy_and_requires_at_least_two_prompt_candidates"

        best_by_strategy: Dict[str, Dict[str, object]] = {}
        for idx, item in enumerate(prompt_items):
            strategy_name = self._candidate_strategy_name(item, idx)
            previous = best_by_strategy.get(strategy_name)
            if previous is None or _safe_float(item.get("candidate_objective")) > _safe_float(
                previous.get("candidate_objective")
            ):
                item["candidate_strategy_name"] = strategy_name
                best_by_strategy[strategy_name] = item

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
            return [], "qwen_strategy_and_requires_multiple_strategy_candidates"

        query_budget = max(0, int(getattr(self.args, "_flux2_strategy_cwor_query_budget", 1)))
        if query_budget <= 0:
            return [], "qwen_strategy_and_query_budget_exhausted"

        cwor_target_label = getattr(self.args, "cwor_target_label", None)
        try:
            cwor_target_label = None if cwor_target_label is None else int(cwor_target_label)
        except Exception:
            cwor_target_label = None

        condition_image = self._load_condition_image()
        base_seed = int(getattr(self.args, "seed", 42))
        selected_items: List[Dict[str, object]] = [ranked[0]]
        selected_prompts: List[str] = [self._candidate_prompt(ranked[0])]
        current_objective = _safe_float(ranked[0].get("candidate_objective"))
        query_count = 0
        attempt_count = 0
        accepted_count = 0
        attempted_prompts: List[str] = []
        evaluated_results: List[Dict[str, object]] = []
        import vlm_attack as vlm_attack_module

        for candidate in ranked[1:]:
            if query_count >= query_budget:
                break
            trial_prompt = self._join_and_prompts([*selected_prompts, self._candidate_prompt(candidate)])
            if not trial_prompt:
                continue
            attempt_count += 1
            attempted_prompts.append(trial_prompt)
            trial_items = [*selected_items]
            if all(
                self._candidate_prompt(existing) != self._candidate_prompt(candidate)
                for existing in trial_items
            ):
                trial_items.append(candidate)
            trial_seed = base_seed + 1000 + int(cwor_step_index) * 97 + int(attempt_count)
            try:
                trial_image = self._pipe_call(prompt=trial_prompt, image=condition_image, seed=trial_seed)
                trial_objective, trial_stats, exact_artifacts = self._score_image(
                    image=trial_image,
                    classifier=classifier,
                    cwor_target_label=cwor_target_label,
                )
                query_count = int(self.last_and_query_count)
            except Exception as exc:
                query_count = int(self.last_and_query_count)
                for result in evaluated_results:
                    result["cwor_strategy_query_count"] = int(query_count)
                return evaluated_results, f"qwen_strategy_and:{type(exc).__name__}:{exc}"
            if not exact_artifacts:
                evaluated_image = vlm_attack_module.classifier_input_image(
                    trial_image,
                    classifier,
                )
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
                    "merge_anchor": bool(idx == 0),
                }
                for idx, item in enumerate(trial_items)
            ]
            evaluated_results.append(
                {
                    "candidate_word": "<CWOR>",
                    "candidate_prompt": str(trial_prompt),
                    "candidate_objective": float(trial_objective),
                    "pred_idx": trial_stats.get("pred_idx"),
                    "pred_conf": trial_stats.get("pred_conf"),
                    "pred_logit": trial_stats.get("pred_logit"),
                    "target_conf": trial_stats.get("target_conf"),
                    "target_logit": trial_stats.get("target_logit"),
                    "target_label_conf": trial_stats.get("target_label_conf"),
                    "target_label_logit": trial_stats.get("target_label_logit"),
                    "ce": trial_stats.get("ce"),
                    "candidate_variant": "cwor",
                    "candidate_selected_image": trial_image.copy(),
                    **exact_artifacts,
                    "candidate_selected_image_source": "qwen_strategy_and",
                    "cwor_strategy_merge_mode": "and",
                    "cwor_strategy_components": components,
                    "cwor_strategy_query_offset": int(query_count),
                    "qwen_strategy_and_original_objective": original_objective,
                    "qwen_strategy_and_base_confidence": cwor_base_confidence,
                }
            )
            if trial_objective > current_objective:
                selected_items.append(candidate)
                selected_prompts.append(self._candidate_prompt(candidate))
                current_objective = float(trial_objective)
                accepted_count += 1

        if query_count <= 0:
            return [], "qwen_strategy_and_no_merge_trial_executed"
        for result in evaluated_results:
            result["cwor_strategy_query_count"] = int(query_count)
            result["qwen_strategy_and_accepted_component_count"] = int(accepted_count)
            result["qwen_strategy_and_attempted_prompts"] = list(attempted_prompts)
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


class Qwen2AttackRuntimeAdapter:
    def __init__(self) -> None:
        self._module = None
        self._runtime_cache = None
        self._render_session = None
        self.last_and_query_count = 0
        self.last_prompt_query_count = 0

    def _get_module(self):
        if self._module is None:
            import vlm_attack as vlm_attack_module

            self._module = vlm_attack_module
        return self._module

    @staticmethod
    def _ensure_qwen_args(args) -> None:
        if not hasattr(args, "model_path") or not str(getattr(args, "model_path", "") or "").strip():
            setattr(args, "model_path", "Qwen/Qwen-Image-Edit-2511")
        if not hasattr(args, "cpu_offload"):
            setattr(args, "cpu_offload", False)
        if not hasattr(args, "qwen_true_cfg_scale"):
            setattr(args, "qwen_true_cfg_scale", 4.0)
        if not hasattr(args, "qwen_negative_prompt"):
            setattr(args, "qwen_negative_prompt", " ")
        if not hasattr(args, "qwen_num_images_per_prompt"):
            setattr(args, "qwen_num_images_per_prompt", 1)
        if int(getattr(args, "qwen_num_images_per_prompt", 1)) != 1:
            raise ValueError("Qwen Image Edit attack runtime requires one image per prompt")
        if not hasattr(args, "qwen_batch_size"):
            setattr(args, "qwen_batch_size", 1)
        if int(getattr(args, "qwen_batch_size", 1)) < 1:
            raise ValueError("Qwen Image Edit qwen_batch_size must be >= 1")
        if not hasattr(args, "qwen_batch_fallback"):
            setattr(args, "qwen_batch_fallback", True)
        if not hasattr(args, "gcg_scene_vocab_prompts_per_strategy"):
            setattr(args, "gcg_scene_vocab_prompts_per_strategy", 0)
        if not hasattr(args, "gcg_scene_vocab_enabled_strategies"):
            setattr(args, "gcg_scene_vocab_enabled_strategies", "all")

    @staticmethod
    def _render_session_rebuild_required(*, render_session, args, has_input_image: bool) -> bool:
        if render_session is None:
            return True
        if bool(getattr(render_session, "has_input_image", False)) != bool(has_input_image):
            return True
        prev_args = getattr(render_session, "args", None)
        if prev_args is None:
            return True
        static_keys = (
            "model_path",
            "hf_token",
            "device",
            "cpu_offload",
            "qwen_true_cfg_scale",
            "qwen_negative_prompt",
            "qwen_num_images_per_prompt",
        )
        for key in static_keys:
            if str(getattr(prev_args, key, "")) != str(getattr(args, key, "")):
                return True
        return False

    def setup(self, *, args, has_input_image: bool) -> None:
        self._ensure_qwen_args(args)
        validate_generator_model(args.model_path, expected_family="qwen-image-edit")
        attack_mode = str(getattr(args, "attack_mode", "vlm") or "vlm").strip().lower()
        if attack_mode not in {"vlm", "and"}:
            raise ValueError(f"Qwen Image Edit runtime does not support attack_mode={attack_mode}")
        if bool(getattr(args, "cwor_enable", False)) and not bool(
            getattr(args, "qwen_strategy_and_enable", False)
        ):
            raise ValueError("Qwen Image Edit runner supports CWOR only through attack_mode=and.")

        module = self._get_module()
        if self._runtime_cache is None:
            self._runtime_cache = module.PersistentVLMRuntimeCache()

        rebuild_session = self._render_session_rebuild_required(
            render_session=self._render_session,
            args=args,
            has_input_image=has_input_image,
        )
        if rebuild_session:
            if self._render_session is not None:
                try:
                    self._render_session.close()
                except Exception:
                    pass
            self._render_session = QwenImageEditRenderSession(
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
            try:
                render_session.close()
            except Exception:
                pass
        if runtime_cache is not None:
            try:
                runtime_cache.close()
            except Exception:
                pass


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
        kwargs.pop("ic_current_branch_mix_initial_cache", None)
        kwargs["render_session"] = self._render_session
        kwargs["mixed_initial_edit_cache"] = None
        self.last_prompt_query_count = 0
        result = module.evaluate_attack_candidates(**kwargs)
        if self._render_session is not None:
            self.last_prompt_query_count = int(
                getattr(self._render_session, "last_prompt_query_count", len(result[0]))
            )
        return result

    def evaluate_and_candidate(self, **_kwargs):
        if self._render_session is None:
            self.last_and_query_count = 0
            return [], "qwen_image_edit_session_not_initialized"
        result = self._render_session.evaluate_and_candidate(**_kwargs)
        self.last_and_query_count = int(self._render_session.last_and_query_count)
        return result

    def save_prompt_artifacts(self, **kwargs) -> Dict[str, str]:
        module = self._get_module()
        return module.save_blackbox_prompt_artifacts(**kwargs)

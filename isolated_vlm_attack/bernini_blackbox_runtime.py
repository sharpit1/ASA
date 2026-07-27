import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from attack_model_registry import validate_generator_model


_ISOLATED_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(os.getenv("ASA_PROJECT_ROOT", str(_ISOLATED_DIR.parent))).resolve()
_DEFAULT_BERNINI_ROOT = _REPO_ROOT / "third_party" / "bernini"
_DEFAULT_BERNINI_CONFIG = _DEFAULT_BERNINI_ROOT / "Bernini-R-Diffusers"


def _resolve_existing_or_relative(path_value: object, *, base_dir: Optional[Path] = None) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise ValueError("empty path")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, _REPO_ROOT / path]
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_local_path_or_raw(path_value: object, *, base_dir: Optional[Path] = None) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        raise ValueError("empty path")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [Path.cwd() / path, _REPO_ROOT / path]
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def _ensure_bernini_on_path(bernini_root: Path) -> None:
    root = str(Path(bernini_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class BerniniRenderSession:
    def __init__(self, *, args, has_input_image: bool):
        self.args = args
        self.has_input_image = bool(has_input_image)
        self.pipeline = None
        self._bernini_cli = None
        self._load_pipeline()

    @staticmethod
    def _requested_device(args) -> torch.device:
        requested = str(getattr(args, "device", "cuda") or "cuda").strip()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested or "cuda")

    def _bernini_root(self) -> Path:
        raw_root = getattr(self.args, "bernini_root", str(_DEFAULT_BERNINI_ROOT))
        return _resolve_existing_or_relative(raw_root)

    def _config_path(self) -> str:
        raw_config = getattr(self.args, "bernini_config", str(_DEFAULT_BERNINI_CONFIG))
        return _resolve_local_path_or_raw(raw_config, base_dir=self._bernini_root())

    def _optional_path_or_none(self, attr_name: str) -> Optional[str]:
        raw = str(getattr(self.args, attr_name, "") or "").strip()
        if not raw:
            return None
        return _resolve_local_path_or_raw(raw, base_dir=self._bernini_root())

    def _load_pipeline(self) -> None:
        bernini_root = self._bernini_root()
        _ensure_bernini_on_path(bernini_root)

        from bernini import cli as bernini_cli

        self._bernini_cli = bernini_cli
        bernini_cli.setup_logging()

        device = self._requested_device(self.args)
        if device.type == "cuda":
            torch.cuda.set_device(device if device.index is not None else 0)

        pipe_args = type(
            "BerniniPipelineArgs",
            (),
            {
                "config": str(self._config_path()),
                "high_noise_ckpt": self._optional_path_or_none("bernini_high_noise_ckpt"),
                "low_noise_ckpt": self._optional_path_or_none("bernini_low_noise_ckpt"),
                "use_unipc": bool(getattr(self.args, "bernini_use_unipc", True)),
                "use_src_tgt_id": bool(getattr(self.args, "bernini_use_src_tgt_id", True)),
            },
        )()
        self.pipeline = bernini_cli.build_pipeline(pipe_args, device)

    def _load_condition_image_path(self) -> Optional[str]:
        if not self.has_input_image:
            return None
        raw_path = str(getattr(self.args, "input_img_path", "") or "").strip()
        if not raw_path:
            raise ValueError("Bernini i2i rendering requires input_img_path.")
        image_path = Path(raw_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing input image: {image_path}")
        return str(image_path)

    def _generation_kwargs(self) -> Dict[str, object]:
        if self._bernini_cli is None:
            raise RuntimeError("Bernini CLI module is not initialized.")
        num_frames = int(getattr(self.args, "bernini_num_frames", 1))
        if num_frames != 1:
            raise ValueError("Bernini black-box image attack runner requires --bernini_num_frames=1.")

        neg_prompt = str(getattr(self.args, "bernini_neg_prompt", "") or "")
        if not neg_prompt:
            neg_prompt = str(self._bernini_cli.DEFAULT_NEG_PROMPT)

        norm_threshold = getattr(self.args, "bernini_norm_threshold", (50.0, 50.0, 50.0))
        if isinstance(norm_threshold, str):
            norm_threshold = [float(item) for item in norm_threshold.replace(",", " ").split() if item]
        return {
            "neg_prompt": neg_prompt,
            "num_frames": num_frames,
            "max_image_size": int(getattr(self.args, "bernini_max_image_size", 848)),
            "height": int(getattr(self.args, "height", 480)),
            "width": int(getattr(self.args, "width", 848)),
            "num_inference_steps": int(getattr(self.args, "num_inference_steps", 40)),
            "guidance_mode": str(getattr(self.args, "bernini_guidance_mode", "v2v") or "v2v"),
            "omega_V": float(getattr(self.args, "bernini_omega_V", 1.25)),
            "omega_I": float(getattr(self.args, "bernini_omega_I", 4.5)),
            "omega_TI": float(getattr(self.args, "bernini_omega_TI", 4.0)),
            "omega_scale": float(getattr(self.args, "bernini_omega_scale", 0.8)),
            "flow_shift": float(getattr(self.args, "bernini_flow_shift", 5.0)),
            "seed": int(getattr(self.args, "seed", 42)),
            "fps": int(getattr(self.args, "bernini_fps", 16)),
            "eta": float(getattr(self.args, "bernini_eta", 0.5)),
            "norm_threshold": tuple(float(item) for item in norm_threshold),
            "momentum": float(getattr(self.args, "bernini_momentum", 0.0)),
            "keep_models_on_gpu": not bool(getattr(self.args, "cpu_offload", False)),
        }

    def _system_prompt(self) -> str:
        if self._bernini_cli is None:
            raise RuntimeError("Bernini CLI module is not initialized.")
        task_type = str(getattr(self.args, "bernini_task_type", "i2i") or "i2i")
        explicit = str(getattr(self.args, "bernini_system_prompt", "") or "")
        prompt_args = type(
            "BerniniPromptArgs",
            (),
            {"task_type": task_type, "system_prompt": explicit},
        )()
        return self._bernini_cli.resolve_system_prompt({"task_type": task_type}, prompt_args)

    def _render_one(self, *, prompt: str, output_path: Path) -> Image.Image:
        if self.pipeline is None:
            raise RuntimeError("Bernini render session is not initialized.")

        task_type = str(getattr(self.args, "bernini_task_type", "i2i") or "i2i")
        image_path = self._load_condition_image_path()
        generation_kwargs = self._generation_kwargs()
        generation_kwargs.pop("keep_models_on_gpu", None)
        with torch.inference_mode():
            self.pipeline(
                str(prompt),
                image=image_path,
                images=None,
                video=None,
                output_path=str(output_path),
                system_prompt=self._system_prompt(),
                **generation_kwargs,
            )
        if not output_path.is_file():
            raise RuntimeError(f"Bernini did not write output for task_type={task_type}: {output_path}")
        with Image.open(output_path) as rendered:
            return rendered.convert("RGB").copy()

    def _batch_render_supported(self, *, prompt_count: int) -> bool:
        if int(prompt_count) <= 1:
            return False
        if not bool(getattr(self.args, "bernini_batch_render", True)):
            return False
        if not self.has_input_image:
            return False
        if str(getattr(self.args, "bernini_task_type", "i2i") or "i2i").strip().lower() != "i2i":
            return False
        if str(getattr(self.args, "bernini_guidance_mode", "v2v") or "v2v").strip().lower() != "v2v":
            return False
        if int(getattr(self.args, "bernini_num_frames", 1)) != 1:
            return False
        return True

    def _render_batch(self, *, prompts: Sequence[str]) -> List[Image.Image]:
        if self.pipeline is None:
            raise RuntimeError("Bernini render session is not initialized.")
        from bernini_batch_pipeline import render_bernini_i2i_v2v_batch

        return render_bernini_i2i_v2v_batch(
            pipeline=self.pipeline,
            prompts=prompts,
            image_path=self._load_condition_image_path(),
            system_prompt=self._system_prompt(),
            generation_kwargs=self._generation_kwargs(),
        )

    def render_images(
        self,
        *,
        prompts: Sequence[str],
        mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    ) -> List[Image.Image]:
        if mixed_initial_edit_cache is not None:
            raise ValueError("Bernini runtime does not support mixed initial edit caches.")
        prompt_list = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if len(prompt_list) == 0:
            raise ValueError("render requires at least one prompt.")

        with tempfile.TemporaryDirectory(prefix="bernini_render_") as tmpdir:
            tmp_root = Path(tmpdir)
            images: List[Image.Image] = []
            if self._batch_render_supported(prompt_count=len(prompt_list)):
                batch_limit = int(getattr(self.args, "bernini_batch_size", 0) or 0)
                if batch_limit > 0:
                    chunks = [
                        prompt_list[idx : idx + batch_limit]
                        for idx in range(0, len(prompt_list), batch_limit)
                    ]
                else:
                    chunks = [prompt_list]
                try:
                    for chunk in chunks:
                        print(
                            f"[bernini_batch] rendering {len(chunk)} prompts as one i2i/v2v batch",
                            file=sys.stderr,
                        )
                        images.extend(self._render_batch(prompts=chunk))
                except Exception as exc:
                    if not bool(getattr(self.args, "bernini_batch_fallback", True)):
                        raise
                    print(
                        f"[bernini_batch] batch render failed; falling back to sequential render: {exc}",
                        file=sys.stderr,
                    )
                    images = []
            if len(images) == 0:
                for idx, prompt in enumerate(prompt_list):
                    images.append(self._render_one(prompt=prompt, output_path=tmp_root / f"{idx:03d}.png"))

        if len(images) == 0:
            raise RuntimeError("Bernini returned no images.")
        base_size = images[0].size
        normalized = [
            image.convert("RGB")
            if image.size == base_size
            else image.resize(base_size, Image.LANCZOS).convert("RGB")
            for image in images
        ]
        return [image.copy() for image in normalized]

    def reset_cwor_aggregate_state(self) -> None:
        return None

    def close(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is not None:
            for attr_name in ("model", "vae"):
                component = getattr(pipeline, attr_name, None)
                if component is None:
                    continue
                try:
                    component.to("cpu")
                except Exception:
                    pass
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


class BerniniAttackRuntimeAdapter:
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
    def _ensure_bernini_args(args) -> None:
        if not hasattr(args, "model_path") or not str(getattr(args, "model_path", "") or "").strip():
            setattr(args, "model_path", "bernini")
        if not hasattr(args, "bernini_root"):
            setattr(args, "bernini_root", str(_DEFAULT_BERNINI_ROOT))
        if not hasattr(args, "bernini_config"):
            setattr(args, "bernini_config", str(_DEFAULT_BERNINI_CONFIG))
        if not hasattr(args, "bernini_high_noise_ckpt"):
            setattr(args, "bernini_high_noise_ckpt", "")
        if not hasattr(args, "bernini_low_noise_ckpt"):
            setattr(args, "bernini_low_noise_ckpt", "")
        if not hasattr(args, "bernini_task_type"):
            setattr(args, "bernini_task_type", "i2i")
        if not hasattr(args, "bernini_guidance_mode"):
            setattr(args, "bernini_guidance_mode", "v2v")
        if not hasattr(args, "bernini_num_frames"):
            setattr(args, "bernini_num_frames", 1)
        if not hasattr(args, "bernini_max_image_size"):
            setattr(args, "bernini_max_image_size", 848)
        if not hasattr(args, "bernini_use_unipc"):
            setattr(args, "bernini_use_unipc", True)
        if not hasattr(args, "bernini_use_src_tgt_id"):
            setattr(args, "bernini_use_src_tgt_id", True)
        if not hasattr(args, "bernini_neg_prompt"):
            setattr(args, "bernini_neg_prompt", "")
        if not hasattr(args, "bernini_system_prompt"):
            setattr(args, "bernini_system_prompt", "")
        if not hasattr(args, "bernini_omega_V"):
            setattr(args, "bernini_omega_V", 1.25)
        if not hasattr(args, "bernini_omega_I"):
            setattr(args, "bernini_omega_I", 4.5)
        if not hasattr(args, "bernini_omega_TI"):
            setattr(args, "bernini_omega_TI", 4.0)
        if not hasattr(args, "bernini_omega_scale"):
            setattr(args, "bernini_omega_scale", 0.8)
        if not hasattr(args, "bernini_flow_shift"):
            setattr(args, "bernini_flow_shift", 5.0)
        if not hasattr(args, "bernini_fps"):
            setattr(args, "bernini_fps", 16)
        if not hasattr(args, "bernini_eta"):
            setattr(args, "bernini_eta", 0.5)
        if not hasattr(args, "bernini_norm_threshold"):
            setattr(args, "bernini_norm_threshold", [50.0, 50.0, 50.0])
        if not hasattr(args, "bernini_momentum"):
            setattr(args, "bernini_momentum", 0.0)
        if not hasattr(args, "bernini_batch_render"):
            setattr(args, "bernini_batch_render", True)
        if not hasattr(args, "bernini_batch_fallback"):
            setattr(args, "bernini_batch_fallback", True)
        if not hasattr(args, "bernini_batch_size"):
            setattr(args, "bernini_batch_size", 0)
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
            "device",
            "bernini_root",
            "bernini_config",
            "bernini_high_noise_ckpt",
            "bernini_low_noise_ckpt",
            "bernini_use_unipc",
            "bernini_use_src_tgt_id",
        )
        for key in static_keys:
            if str(getattr(prev_args, key, "")) != str(getattr(args, key, "")):
                return True
        return False

    def setup(self, *, args, has_input_image: bool) -> None:
        self._ensure_bernini_args(args)
        validate_generator_model(args.model_path, expected_family="bernini")
        attack_mode = str(getattr(args, "attack_mode", "vlm") or "vlm").strip().lower()
        if attack_mode not in {"vlm", "and"}:
            raise ValueError(f"Bernini runtime does not support attack_mode={attack_mode}")
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
            self._render_session = BerniniRenderSession(
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

    def evaluate_and_candidate(self, **kwargs):
        self.last_and_query_count = 0
        render_session = self._render_session
        if render_session is None:
            return [], "bernini_strategy_and_session_not_initialized"

        args = getattr(render_session, "args", None)
        if args is None:
            return [], "bernini_strategy_and_args_missing"

        strategy_merge_mode = str(
            getattr(args, "flux2_strategy_cwor_merge_mode", "weighted") or "weighted"
        ).strip().lower()
        if strategy_merge_mode != "and":
            return [], "bernini_strategy_cwor_supports_only_and"

        classifier = kwargs.get("classifier")
        if classifier is None:
            return [], "bernini_strategy_and_classifier_missing"

        prompt_candidates = kwargs.get("prompt_candidates") or []
        prompt_items = [
            dict(item)
            for item in prompt_candidates
            if str(item.get("candidate_variant", "prompt")).strip().lower() == "prompt"
            and str(item.get("candidate_prompt", "") or "").strip()
        ]
        if len(prompt_items) < 2:
            return [], "bernini_strategy_and_requires_at_least_two_prompt_candidates"

        def _safe_float(raw, default=float("-inf")) -> float:
            try:
                value = float(raw)
            except Exception:
                return float(default)
            if not np.isfinite(value):
                return float(default)
            return float(value)

        def _candidate_prompt(candidate: Dict[str, object]) -> str:
            return str(candidate.get("candidate_prompt", "") or "").strip()

        def _candidate_strategy_name(candidate: Dict[str, object], fallback_idx: int) -> str:
            strategy_name = str(candidate.get("candidate_strategy_name", "") or "").strip()
            if strategy_name:
                return strategy_name
            return f"candidate_{int(fallback_idx):03d}"

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

        best_by_strategy: Dict[str, Dict[str, object]] = {}
        for idx, item in enumerate(prompt_items):
            strategy_name = _candidate_strategy_name(item, idx)
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
            prompt_key = _candidate_prompt(item).casefold()
            if not prompt_key or prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            unique_ranked.append(item)
        ranked = unique_ranked
        if len(ranked) < 2:
            return [], "bernini_strategy_and_requires_multiple_strategy_candidates"

        try:
            query_budget = int(getattr(args, "_flux2_strategy_cwor_query_budget", 1))
        except Exception:
            query_budget = 1
        query_budget = max(0, int(query_budget))
        if query_budget <= 0:
            return [], "bernini_strategy_and_query_budget_exhausted"

        cwor_target_label = getattr(args, "cwor_target_label", None)
        try:
            cwor_target_label = None if cwor_target_label is None else int(cwor_target_label)
        except Exception:
            cwor_target_label = None

        def _score_image(
            image: Image.Image,
        ) -> Tuple[float, Dict[str, object], Dict[str, object]]:
            nonlocal query_count
            module = self._get_module()
            image_01 = module.image_to_tensor_01(image).to(
                device=str(getattr(args, "device", "cuda") or "cuda")
            )
            query_count += 1
            self.last_and_query_count = int(query_count)
            with torch.no_grad():
                objective, stats = classifier.objective_and_stats(
                    image_01,
                    target_label=cwor_target_label,
                )
            exact_artifacts = module.classifier_evaluation_artifacts(classifier)
            return float(objective), dict(stats), exact_artifacts

        selected_items: List[Dict[str, object]] = [ranked[0]]
        selected_prompts: List[str] = [_candidate_prompt(ranked[0])]
        current_objective = _safe_float(ranked[0].get("candidate_objective"))

        query_count = 0
        accepted_count = 0
        attempted_prompts: List[str] = []
        failures: List[str] = []
        evaluated_results: List[Dict[str, object]] = []

        remaining_candidates: List[Dict[str, object]] = list(ranked[1:])
        wave_idx = 0
        while remaining_candidates and query_count < query_budget:
            wave_idx += 1
            available_budget = max(0, int(query_budget - query_count))
            trial_plans: List[Dict[str, object]] = []
            for candidate in remaining_candidates[:available_budget]:
                trial_prompt = _join_and_prompts([*selected_prompts, _candidate_prompt(candidate)])
                if not trial_prompt:
                    continue
                trial_items = [*selected_items]
                if all(_candidate_prompt(existing) != _candidate_prompt(candidate) for existing in trial_items):
                    trial_items.append(candidate)
                trial_plans.append(
                    {
                        "candidate": candidate,
                        "prompt": trial_prompt,
                        "items": trial_items,
                    }
                )
            if not trial_plans:
                break

            trial_prompts = [str(plan["prompt"]) for plan in trial_plans]
            attempted_prompts.extend(trial_prompts)
            try:
                trial_images = render_session.render_images(prompts=trial_prompts)
                if len(trial_images) != len(trial_prompts):
                    raise RuntimeError(
                        f"image_count_mismatch:{len(trial_images)}!={len(trial_prompts)}"
                    )
            except Exception as exc:
                failures.append(f"batch_{int(wave_idx)}:{type(exc).__name__}:{exc}")
                break

            best_improvement: Optional[Dict[str, object]] = None
            for plan, trial_image in zip(trial_plans, trial_images):
                try:
                    trial_image = trial_image.convert("RGB").copy()
                    trial_objective, trial_stats, exact_artifacts = _score_image(trial_image)
                except Exception as exc:
                    failures.append(f"trial_{int(query_count)}:{type(exc).__name__}:{exc}")
                    continue

                module = self._get_module()
                if not exact_artifacts:
                    evaluated_image = module.classifier_input_image(
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
                        "reference_prompt": _candidate_prompt(item),
                        "reference_objective": item.get("candidate_objective"),
                        "merge_anchor": bool(idx == 0),
                    }
                    for idx, item in enumerate(plan["items"])
                ]
                evaluated_results.append(
                    {
                        "candidate_word": "<CWOR>",
                        "candidate_prompt": str(plan["prompt"]),
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
                        "candidate_selected_image_source": "bernini_strategy_and",
                        "cwor_strategy_merge_mode": "and",
                        "cwor_strategy_components": components,
                        "cwor_strategy_query_offset": int(query_count),
                        "bernini_strategy_and_original_objective": kwargs.get("original_objective"),
                        "bernini_strategy_and_base_confidence": kwargs.get("cwor_base_confidence"),
                    }
                )
                if trial_objective > current_objective and (
                    best_improvement is None
                    or trial_objective > float(best_improvement["objective"])
                ):
                    best_improvement = {
                        "candidate": plan["candidate"],
                        "objective": float(trial_objective),
                    }

            if best_improvement is None:
                break
            accepted_candidate = dict(best_improvement["candidate"])
            selected_items.append(accepted_candidate)
            selected_prompts.append(_candidate_prompt(accepted_candidate))
            current_objective = float(best_improvement["objective"])
            accepted_count += 1
            accepted_prompt = _candidate_prompt(accepted_candidate)
            remaining_candidates = [
                item
                for item in remaining_candidates
                if _candidate_prompt(item) != accepted_prompt
            ]

        if query_count <= 0:
            return [], "bernini_strategy_and_no_merge_trial_executed"
        if not evaluated_results:
            if failures:
                short = "; ".join(failures[:3])
                if len(failures) > 3:
                    short = f"{short}; ..."
                return [], f"bernini_strategy_and_failures={len(failures)} ({short})"
            return [], "bernini_strategy_and_no_image"

        for result in evaluated_results:
            result["cwor_strategy_query_count"] = int(query_count)
            result["bernini_strategy_and_accepted_component_count"] = int(accepted_count)
            result["bernini_strategy_and_attempted_prompts"] = list(attempted_prompts)
        error = None
        if failures:
            short = "; ".join(failures[:3])
            if len(failures) > 3:
                short = f"{short}; ..."
            error = f"bernini_strategy_and_failures={len(failures)} ({short})"
        return evaluated_results, error

    def save_prompt_artifacts(self, **kwargs) -> Dict[str, str]:
        module = self._get_module()
        return module.save_blackbox_prompt_artifacts(**kwargs)

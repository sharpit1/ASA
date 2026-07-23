#!/usr/bin/env python3
"""Archived implementation retained only as support code for the slim runtime facade.

The supported launcher never executes this module's legacy CLI, inversion, fixed-prompt,
or rerendering flows. New code must import the public symbols from ``vlm_attack``.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vlm_runtime import infer_vlm_backend, load_vlm_runtime

try:
    from setproctitle import setproctitle as _setproctitle
except Exception:  # pragma: no cover
    _setproctitle = None

try:
    import torchvision
    from torchvision import models
except Exception as exc:  # pragma: no cover
    torchvision = None
    models = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "and",
    "to",
    "is",
    "are",
    "this",
    "that",
    "it",
    "image",
    "photo",
    "picture",
    "scene",
    "background",
}
_CUDNN_SDPA_VIM_SMALL_CONFIGURED: Optional[bool] = None
_NIPS2017_CATEGORY_NAMES_CACHE: Optional[List[str]] = None


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
    path_candidates = [
        getattr(args, "output_path", None),
        getattr(args, "report_path", None),
        getattr(args, "input_img_path", None),
    ]
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


def _set_process_title_from_args(args: argparse.Namespace) -> None:
    if _setproctitle is None:
        return
    model_token = _infer_process_model_token(args)
    mode_token = _infer_process_mode_token(args)
    role_token = _process_title_token(getattr(args, "gcg_word", None), 24) or "na"
    label_token = _infer_process_label_token(args)
    title = f"{model_token}_{mode_token}_role_{role_token}_label_{label_token}"
    device_token = _process_title_token(getattr(args, "device", None), 16)
    if device_token:
        title = f"{title}_dev_{device_token}"
    _setproctitle(title)


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


def _is_vim_small_classifier(name: object) -> bool:
    text = str(name or "").strip().lower().replace("_", "-")
    return text in {"vim-small", "mambavision"}


def _maybe_disable_cudnn_sdpa_for_vim_small(classifier_name: object, *, context: str) -> None:
    global _CUDNN_SDPA_VIM_SMALL_CONFIGURED
    if not _is_vim_small_classifier(classifier_name):
        return
    if _CUDNN_SDPA_VIM_SMALL_CONFIGURED is True:
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
            "[vlm_attack] configured SDPA for vim-small/mambavision "
            f"(context='{context}', flash={flash_state}, mem_efficient={mem_state}, "
            f"math={math_state}, cudnn={cudnn_state}, force_flash={force_flash})."
        )
        _CUDNN_SDPA_VIM_SMALL_CONFIGURED = True
    except Exception as exc:
        print(
            "[vlm_attack] WARNING: failed to configure SDPA for vim-small "
            f"(context='{context}'; {type(exc).__name__}: {exc})"
        )
        _CUDNN_SDPA_VIM_SMALL_CONFIGURED = False


class PersistentVLMRuntimeCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    def query(
        self,
        *,
        image_path: Path,
        question: str,
        vlm_backend: str,
        vlm_model_id: str,
        vlm_device_raw: str,
        max_new_tokens: int,
        enable_thinking: bool,
        do_sample: bool,
    ) -> Tuple[str, Optional[str]]:
        device = resolve_vlm_device(vlm_device_raw)
        backend = infer_vlm_backend(
            vlm_backend=str(vlm_backend),
            model_id=str(vlm_model_id),
            allow_blip=True,
        )
        cache_key = (backend, str(vlm_model_id), str(device))
        runtime = self._cache.get(cache_key)

        if runtime is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            vlm_model, vlm_processor, ask_fn, uses_pipeline_backend = load_vlm_runtime(
                backend=backend,
                model_id=str(vlm_model_id),
                vlm_dtype=dtype,
                vlm_device=device,
                allow_blip=True,
            )
            if not uses_pipeline_backend and hasattr(vlm_model, "to"):
                vlm_model = vlm_model.to(device)
                if hasattr(vlm_model, "eval"):
                    vlm_model.eval()
            runtime = {
                "model": vlm_model,
                "processor": vlm_processor,
                "ask_fn": ask_fn,
                "uses_pipeline_backend": bool(uses_pipeline_backend),
                "device": device,
            }
            self._cache[cache_key] = runtime

        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                raw_answer = runtime["ask_fn"](
                    image=image,
                    question=str(question),
                    model=runtime["model"],
                    processor=runtime["processor"],
                    device=runtime["device"],
                    max_new_tokens=int(max_new_tokens),
                    enable_thinking=bool(enable_thinking),
                    do_sample=bool(do_sample),
                )
            return str(raw_answer or "").strip(), None
        except Exception as exc:
            return "", str(exc)

    def close(self) -> None:
        for runtime in self._cache.values():
            model = runtime.get("model")
            uses_pipeline_backend = bool(runtime.get("uses_pipeline_backend", False))
            if model is not None and hasattr(model, "to") and not uses_pipeline_backend:
                try:
                    model.to("cpu")
                except Exception:
                    pass
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


class PersistentFluxRenderSession:
    def __init__(self, *, args: argparse.Namespace, has_input_image: bool):
        self.args = args
        self.has_input_image = bool(has_input_image)
        self._stable_flow = None
        self._edit_cache: Optional[Dict[str, object]] = None
        self._edit_cache_key: Optional[Tuple[object, ...]] = None
        self._cwor_aggregate_state: Optional[Dict[str, object]] = None
        self._cwor_embed_step_counter: int = 0
        self._initialize()

    def reset_cwor_aggregate_state(self) -> None:
        self._cwor_aggregate_state = None
        self._cwor_embed_step_counter = 0

    def _initialize(self) -> None:
        _maybe_disable_cudnn_sdpa_for_vim_small(
            getattr(self.args, "classifier_name", None),
            context="persistent_flux_render_session_init",
        )
        module = importlib.import_module("third_party.gcg_flux_edit")
        stable_cls = getattr(module, "StableFlowGCG")
        mode = "edit" if self.has_input_image else "infer"

        argv = [
            "gcg_flux_edit.py",
            "--mode",
            mode,
            "--classifier_mode",
            str(self.args.classifier_mode),
            "--classifier_name",
            str(self.args.classifier_name),
            "--model_path",
            str(self.args.model_path),
            "--hf_token",
            str(self.args.hf_token),
            "--output_path",
            str(Path(self.args.output_path)),
            "--seed",
            str(int(self.args.seed)),
            "--device",
            str(self.args.device),
            "--height",
            str(int(self.args.height)),
            "--width",
            str(int(self.args.width)),
            "--num_inference_steps",
            str(int(self.args.num_inference_steps)),
            "--max_sequence_length",
            str(int(self.args.max_sequence_length)),
            "--guidance_scale",
            str(float(self.args.guidance_scale)),
            "--latent_nudging_scalar",
            str(float(self.args.latent_nudging_scalar)),
            "--visualize_attention",
            "1" if bool(getattr(self.args, "visualize_attention", False)) else "0",
            "--prompts",
            "bootstrap",
        ]
        if getattr(self.args, "classifier_label", None) is not None:
            argv.extend(["--classifier_label", str(int(self.args.classifier_label))])
        if self.has_input_image:
            argv.extend(["--input_img_path", str(self.args.input_img_path)])
        if bool(self.args.cpu_offload):
            argv.append("--cpu_offload")
        if bool(self.args.gradient_checkpointing):
            argv.append("--gradient_checkpointing")

        old_argv = list(sys.argv)
        try:
            sys.argv = argv
            self._stable_flow = stable_cls()
        finally:
            sys.argv = old_argv

    def render(
        self,
        *,
        prompts: Sequence[str],
        output_path: str,
        mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    ) -> None:
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        prompt_list = [str(p).strip() for p in prompts if str(p).strip()]
        if len(prompt_list) == 0:
            raise ValueError("render requires at least one prompt.")

        self._stable_flow.args.output_path = str(output_path)
        self._stable_flow.args.prompts = prompt_list
        self._stable_flow.args.prompt = prompt_list[-1]
        Path(str(output_path)).parent.mkdir(parents=True, exist_ok=True)

        if self.has_input_image:
            cache = self._ensure_edit_cache(inversion_prompt=prompt_list[0])
            if cache is not None:
                if mixed_initial_edit_cache is not None and len(prompt_list) > 1:
                    self._render_edit_from_mixed_caches(
                        prompts=prompt_list,
                        output_path=str(output_path),
                        current_cache=cache,
                        initial_cache=mixed_initial_edit_cache,
                    )
                else:
                    self._render_edit_from_cache(prompts=prompt_list, output_path=str(output_path), cache=cache)
            else:
                self._stable_flow.invert_and_save(prompts=prompt_list)
        else:
            self._stable_flow.infer_and_save(prompts=prompt_list)

    def capture_edit_cache(self, *, inversion_prompt: str) -> Optional[Dict[str, object]]:
        return self._ensure_edit_cache(inversion_prompt=str(inversion_prompt))

    def _build_edit_cache_key(self, *, inversion_prompt: str) -> Tuple[object, ...]:
        return (
            str(getattr(self.args, "input_img_path", "") or ""),
            str(inversion_prompt),
            int(self.args.height),
            int(self.args.width),
            int(self.args.num_inference_steps),
            int(self.args.max_sequence_length),
            float(self.args.latent_nudging_scalar),
            int(self.args.seed),
        )

    def _ensure_edit_cache(self, *, inversion_prompt: str) -> Optional[Dict[str, object]]:
        if not self.has_input_image:
            self._edit_cache = None
            self._edit_cache_key = None
            return None
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        if not hasattr(self._stable_flow, "prepare_edit_cache"):
            return None

        # Keep inversion prompt aligned with the current render request.
        self._stable_flow.args.inversion_prompt = str(inversion_prompt)
        cache_key = self._build_edit_cache_key(inversion_prompt=str(inversion_prompt))
        if self._edit_cache is not None and self._edit_cache_key == cache_key:
            return self._edit_cache

        cache = self._stable_flow.prepare_edit_cache()
        if not isinstance(cache, dict):
            return None
        self._edit_cache = cache
        self._edit_cache_key = cache_key
        return cache

    def _render_edit_from_cache(
        self,
        *,
        prompts: Sequence[str],
        output_path: str,
        cache: Dict[str, object],
    ) -> None:
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        prompt_list = [str(p) for p in prompts]
        if len(prompt_list) == 0:
            raise ValueError("render requires at least one prompt.")
        inverted_latent_list = cache.get("inverted_latent_list")
        start_latents = cache.get("start_latents")
        if inverted_latent_list is None or not torch.is_tensor(start_latents):
            raise RuntimeError("invalid edit cache payload for persistent render session")

        prompt_count = len(prompt_list)
        guidance_scale = [1.0] + [float(self.args.guidance_scale)] * (prompt_count - 1)
        images = self._stable_flow.pipe(
            prompt_list,
            height=self.args.height,
            width=self.args.width,
            guidance_scale=guidance_scale,
            output_type="pil",
            num_inference_steps=self.args.num_inference_steps,
            max_sequence_length=self.args.max_sequence_length,
            latents=start_latents.tile(prompt_count, 1, 1),
            inverted_latent_list=inverted_latent_list,
            mm_copy_blocks=self._stable_flow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=self._stable_flow.SINGLE_MODAL_VITAL_LAYERS,
        ).images

        orig_size = cache.get("orig_size")
        if isinstance(orig_size, tuple) and len(orig_size) == 2:
            images = [img.resize(orig_size, Image.LANCZOS) for img in images]
        images = [np.array(img) for img in images]
        Image.fromarray(np.hstack(images)).save(output_path)

    @staticmethod
    def _compose_mixed_edit_tensor(
        *,
        current_tensor: torch.Tensor,
        initial_tensor: torch.Tensor,
        prompt_count: int,
    ) -> torch.Tensor:
        current_first = current_tensor[:1]
        initial_first = initial_tensor[:1].to(
            device=current_first.device,
            dtype=current_first.dtype,
        )
        # Pull the current slot toward the initial inversion while leaving
        # the remaining slots fixed to the initial cache.
        alpha = 0.5
        mixed_first = (
            float(alpha) * initial_first.to(torch.float32)
            + (1.0 - float(alpha)) * current_first.to(torch.float32)
        ).to(dtype=current_first.dtype)
        if int(prompt_count) <= 1:
            return mixed_first
        initial_rest = initial_first.expand(int(prompt_count) - 1, *initial_first.shape[1:])
        return torch.cat([mixed_first, initial_rest], dim=0)

    def _render_edit_from_mixed_caches(
        self,
        *,
        prompts: Sequence[str],
        output_path: str,
        current_cache: Dict[str, object],
        initial_cache: Dict[str, object],
    ) -> None:
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        prompt_list = [str(p) for p in prompts]
        if len(prompt_list) == 0:
            raise ValueError("render requires at least one prompt.")

        current_inverted_latent_list = current_cache.get("inverted_latent_list")
        current_start_latents = current_cache.get("start_latents")
        initial_inverted_latent_list = initial_cache.get("inverted_latent_list")
        initial_start_latents = initial_cache.get("start_latents")
        if (
            current_inverted_latent_list is None
            or not torch.is_tensor(current_start_latents)
            or initial_inverted_latent_list is None
            or not torch.is_tensor(initial_start_latents)
        ):
            raise RuntimeError("invalid mixed edit cache payload for persistent render session")
        if len(current_inverted_latent_list) != len(initial_inverted_latent_list):
            raise RuntimeError(
                "mixed edit caches have different inversion lengths: "
                f"{len(current_inverted_latent_list)} != {len(initial_inverted_latent_list)}"
            )

        prompt_count = len(prompt_list)
        mixed_start_latents = self._compose_mixed_edit_tensor(
            current_tensor=current_start_latents,
            initial_tensor=initial_start_latents,
            prompt_count=prompt_count,
        )
        mixed_inverted_latent_list: List[torch.Tensor] = []
        for current_latents, initial_latents in zip(current_inverted_latent_list, initial_inverted_latent_list):
            if not torch.is_tensor(current_latents) or not torch.is_tensor(initial_latents):
                raise RuntimeError("mixed edit caches contain non-tensor inversion latents")
            mixed_inverted_latent_list.append(
                self._compose_mixed_edit_tensor(
                    current_tensor=current_latents,
                    initial_tensor=initial_latents,
                    prompt_count=prompt_count,
                )
            )

        guidance_scale = [1.0] + [float(self.args.guidance_scale)] * (prompt_count - 1)
        images = self._stable_flow.pipe(
            prompt_list,
            height=self.args.height,
            width=self.args.width,
            guidance_scale=guidance_scale,
            output_type="pil",
            num_inference_steps=self.args.num_inference_steps,
            max_sequence_length=self.args.max_sequence_length,
            latents=mixed_start_latents,
            inverted_latent_list=mixed_inverted_latent_list,
            mm_copy_blocks=self._stable_flow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=self._stable_flow.SINGLE_MODAL_VITAL_LAYERS,
        ).images

        orig_size = current_cache.get("orig_size")
        if isinstance(orig_size, tuple) and len(orig_size) == 2:
            images = [img.resize(orig_size, Image.LANCZOS) for img in images]
        images = [np.array(img) for img in images]
        Image.fromarray(np.hstack(images)).save(output_path)

    @staticmethod
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

    @staticmethod
    def _orthogonal_component(
        *,
        base_embed: torch.Tensor,
        fail_embed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if base_embed.shape != fail_embed.shape:
            raise ValueError(
                f"embedding shape mismatch for CWOR: base={list(base_embed.shape)} fail={list(fail_embed.shape)}"
            )
        base_vec = base_embed.float().reshape(base_embed.shape[0], -1)
        fail_vec = fail_embed.float().reshape(fail_embed.shape[0], -1)
        denom = (base_vec * base_vec).sum(dim=1, keepdim=True).clamp_min(float(eps))
        proj = ((fail_vec * base_vec).sum(dim=1, keepdim=True) / denom) * base_vec
        fail_perp = (fail_vec - proj).reshape_as(base_embed)
        return fail_perp.to(device=base_embed.device, dtype=base_embed.dtype)

    @staticmethod
    def _normalize_cwor_update_sum(
        *,
        update_sum: Optional[torch.Tensor],
        delta_sum: object,
        eps: float = 1e-12,
    ) -> Optional[torch.Tensor]:
        if update_sum is None:
            return None
        try:
            denom = abs(float(delta_sum))
        except Exception:
            denom = 0.0
        if not np.isfinite(denom) or denom < float(eps):
            return update_sum.to(torch.float32)
        return update_sum.to(torch.float32) / float(denom)

    @torch.no_grad()
    def _encode_prompt_embeds(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        prompt_text = str(prompt or "")
        prompt_embeds = self._stable_flow.t5_prompt_embeds(prompt_text)
        pooled_prompt_embeds = self._stable_flow.clip_pooled_prompt_embeds(prompt_text)
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def get_cwor_embedding_snapshot(self) -> Optional[Dict[str, object]]:
        state = self._cwor_aggregate_state
        if not isinstance(state, dict):
            return None

        reference_prompt = str(state.get("reference_prompt", "")).strip()
        if len(reference_prompt) == 0:
            return None

        try:
            ref_t5_prompt_embeds, ref_clip_pooled_prompt_embeds = self._encode_prompt_embeds(reference_prompt)
        except Exception:
            return None

        raw_t5_update_sum = state.get("prompt_update_sum")
        raw_clip_update_sum = state.get("pooled_update_sum")
        t5_update_sum = raw_t5_update_sum if torch.is_tensor(raw_t5_update_sum) else None
        clip_update_sum = raw_clip_update_sum if torch.is_tensor(raw_clip_update_sum) else None
        delta_sum = state.get("delta_sum", 0.0)
        normalized_t5_update_sum = self._normalize_cwor_update_sum(
            update_sum=t5_update_sum,
            delta_sum=delta_sum,
        )
        normalized_clip_update_sum = self._normalize_cwor_update_sum(
            update_sum=clip_update_sum,
            delta_sum=delta_sum,
        )

        if normalized_t5_update_sum is not None:
            merged_t5_prompt_embeds = (
                ref_t5_prompt_embeds.to(torch.float32) - normalized_t5_update_sum
            ).to(device=ref_t5_prompt_embeds.device, dtype=ref_t5_prompt_embeds.dtype)
        else:
            merged_t5_prompt_embeds = ref_t5_prompt_embeds

        if normalized_clip_update_sum is not None:
            merged_clip_pooled_prompt_embeds = (
                ref_clip_pooled_prompt_embeds.to(torch.float32) - normalized_clip_update_sum
            ).to(device=ref_clip_pooled_prompt_embeds.device, dtype=ref_clip_pooled_prompt_embeds.dtype)
        else:
            merged_clip_pooled_prompt_embeds = ref_clip_pooled_prompt_embeds

        state_key = state.get("key")
        cwor_mode = None
        cwor_embed_inject_mode = None
        cwor_feedback_merge_mode = str(state.get("cwor_feedback_merge_mode", "accumulate"))
        cwor_target_label = None
        cwor_base_confidence = None
        if isinstance(state_key, tuple):
            if len(state_key) > 2:
                cwor_mode = state_key[2]
            if len(state_key) > 3:
                cwor_embed_inject_mode = state_key[3]
            has_feedback_mode_slot = False
            if len(state_key) > 4:
                key4 = str(state_key[4]).strip().lower()
                if key4 in {"accumulate", "step_prompt_weighted"}:
                    cwor_feedback_merge_mode = key4
                    has_feedback_mode_slot = True
            if has_feedback_mode_slot:
                if len(state_key) > 5:
                    cwor_target_label = state_key[5]
                if len(state_key) > 6:
                    cwor_base_confidence = state_key[6]
            else:
                if len(state_key) > 4:
                    cwor_target_label = state_key[4]
                if len(state_key) > 5:
                    cwor_base_confidence = state_key[5]

        payload: Dict[str, object] = {
            "reference_prompt": str(reference_prompt),
            "reference_word": str(state.get("reference_word", "")),
            "cwor_mode": None if cwor_mode is None else str(cwor_mode),
            "cwor_embed_inject_mode": (
                None if cwor_embed_inject_mode is None else str(cwor_embed_inject_mode)
            ),
            "cwor_feedback_merge_mode": str(cwor_feedback_merge_mode),
            "cwor_target_label": cwor_target_label,
            "cwor_base_confidence": cwor_base_confidence,
            "feedback_count": int(state.get("count", 0)),
            "conf_sum": float(state.get("conf_sum", 0.0)),
            "delta_sum": float(state.get("delta_sum", 0.0)),
            "t5_prompt_embeds": merged_t5_prompt_embeds.detach().cpu(),
            "clip_pooled_prompt_embeds": merged_clip_pooled_prompt_embeds.detach().cpu(),
            "tensor_schema": {
                "t5_prompt_embeds": "accumulated CWOR prompt embeddings (T5)",
                "clip_pooled_prompt_embeds": "accumulated CWOR pooled prompt embeddings (CLIP)",
            },
        }
        return payload

    @torch.no_grad()
    def save_cwor_embedding_snapshot(self, *, output_path: Path) -> Optional[str]:
        snapshot = self.get_cwor_embedding_snapshot()
        if snapshot is None:
            return None
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, out_path)
        return str(out_path)

    @torch.no_grad()
    def get_prompt_embedding_snapshot(self, *, prompt: str, source: str = "base_prompt") -> Optional[Dict[str, object]]:
        prompt_text = str(prompt or "").strip()
        if len(prompt_text) == 0:
            return None
        try:
            t5_prompt_embeds, clip_pooled_prompt_embeds = self._encode_prompt_embeds(prompt_text)
        except Exception:
            return None
        payload: Dict[str, object] = {
            "source": str(source),
            "reference_prompt": prompt_text,
            "t5_prompt_embeds": t5_prompt_embeds.detach().cpu(),
            "clip_pooled_prompt_embeds": clip_pooled_prompt_embeds.detach().cpu(),
            "tensor_schema": {
                "t5_prompt_embeds": "prompt embeddings (T5)",
                "clip_pooled_prompt_embeds": "pooled prompt embeddings (CLIP)",
            },
        }
        return payload

    @torch.no_grad()
    def save_prompt_embedding_snapshot(
        self,
        *,
        output_path: Path,
        prompt: str,
        source: str = "base_prompt",
    ) -> Optional[str]:
        snapshot = self.get_prompt_embedding_snapshot(prompt=prompt, source=source)
        if snapshot is None:
            return None
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, out_path)
        return str(out_path)

    @torch.no_grad()
    def _render_candidate_tensor_from_embeds(
        self,
        *,
        inversion_prompt: str,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self._stable_flow is None:
            raise RuntimeError("Persistent FLUX render session is not initialized.")
        cache = self._ensure_edit_cache(inversion_prompt=str(inversion_prompt))
        if cache is None:
            raise RuntimeError("CWOR requires edit cache, but edit cache is unavailable.")

        kwargs = self._stable_flow.make_generation_kwargs(
            cache=cache,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type="pil",
            use_edit_reference_batch=bool(cache.get("kind") == "edit"),
        )
        output = self._stable_flow.pipe(**kwargs)
        images = self._extract_output_images(output)
        if len(images) == 0:
            raise RuntimeError("CWOR render returned no images.")

        image_index = None
        if hasattr(self._stable_flow, "edit_target_image_index"):
            try:
                image_index = self._stable_flow.edit_target_image_index(cache)
            except Exception:
                image_index = None
        if image_index is None:
            image_index = len(images) - 1
        idx = int(image_index)
        if idx < 0:
            idx += len(images)
        if idx < 0 or idx >= len(images):
            raise IndexError(f"CWOR rendered image index out of range: idx={idx}, count={len(images)}")
        return image_to_tensor_01(images[idx].convert("RGB"))

    @torch.no_grad()
    def evaluate_cwor_candidates(
        self,
        *,
        inversion_prompt: str,
        cwor_base_prompt: Optional[str],
        base_candidates: Sequence[Dict[str, object]],
        classifier: "TorchvisionClassifier",
        base_confidence: Optional[float],
        cwor_mode: str,
        cwor_embed_inject_mode: str,
        cwor_target_label: Optional[int],
        aggregate_feedback: bool = False,
        cwor_feedback_merge_mode: str = "accumulate",
        cwor_step_index: int = 1,
    ) -> Tuple[List[Dict[str, object]], Optional[str]]:
        if self._stable_flow is None:
            return [], "cwor_session_unavailable"
        if not self.has_input_image:
            return [], "cwor_requires_input_image"
        if len(base_candidates) == 0:
            return [], None
        cwor_placeholder_word = "<CWOR>"

        try:
            c_base = float(base_confidence) if base_confidence is not None else None
        except Exception:
            c_base = None
        if c_base is None or not np.isfinite(c_base):
            return [], "cwor_base_confidence_missing"

        mode = str(cwor_mode or "").strip().lower()
        if mode not in {"untargeted", "target"}:
            mode = "untargeted"
        embed_inject_mode = normalize_cwor_embed_inject_mode(cwor_embed_inject_mode)
        feedback_merge_mode = normalize_cwor_feedback_merge_mode(cwor_feedback_merge_mode)
        try:
            accumulate_secondary_ortho = bool(
                parse_bool_flag(getattr(self.args, "cwor_accumulate_secondary_ortho", False))
            )
        except Exception:
            accumulate_secondary_ortho = bool(getattr(self.args, "cwor_accumulate_secondary_ortho", False))
        try:
            accumulate_update_if_improved_only = bool(
                parse_bool_flag(getattr(self.args, "cwor_accumulate_update_if_improved_only", False))
            )
        except Exception:
            accumulate_update_if_improved_only = bool(
                getattr(self.args, "cwor_accumulate_update_if_improved_only", False)
            )
        try:
            delta_use_basis_logit_without_secondary_ortho = bool(
                parse_bool_flag(
                    getattr(
                        self.args,
                        "cwor_accumulate_delta_use_basis_logit_without_secondary_ortho",
                        False,
                    )
                )
            )
        except Exception:
            delta_use_basis_logit_without_secondary_ortho = bool(
                getattr(
                    self.args,
                    "cwor_accumulate_delta_use_basis_logit_without_secondary_ortho",
                    False,
                )
            )
        try:
            step_prompt_candidate_ortho = bool(
                parse_bool_flag(getattr(self.args, "cwor_step_prompt_candidate_ortho", False))
            )
        except Exception:
            step_prompt_candidate_ortho = bool(
                getattr(self.args, "cwor_step_prompt_candidate_ortho", False)
            )
        try:
            step_prompt_flip_alpha_on_regression = bool(
                parse_bool_flag(
                    getattr(self.args, "cwor_step_prompt_flip_alpha_on_regression", False)
                )
            )
        except Exception:
            step_prompt_flip_alpha_on_regression = bool(
                getattr(self.args, "cwor_step_prompt_flip_alpha_on_regression", False)
            )
        try:
            cwor_embed_subtract_scale_by_step = bool(
                parse_bool_flag(
                    getattr(self.args, "cwor_embed_subtract_scale_by_step", False)
                )
            )
        except Exception:
            cwor_embed_subtract_scale_by_step = bool(
                getattr(self.args, "cwor_embed_subtract_scale_by_step", False)
            )
        if cwor_embed_subtract_scale_by_step:
            try:
                cwor_embed_step_counter = int(getattr(self, "_cwor_embed_step_counter", 0))
            except Exception:
                cwor_embed_step_counter = 0
            if cwor_embed_step_counter < 0:
                cwor_embed_step_counter = 0
            cwor_embed_step_counter += 1
            self._cwor_embed_step_counter = int(cwor_embed_step_counter)
            embed_subtract_scale = float(cwor_embed_step_counter)
        else:
            embed_subtract_scale = 1.0

        try:
            fail_prompt_embeds, fail_pooled_prompt_embeds = self._encode_prompt_embeds(str(inversion_prompt))
        except Exception as exc:
            return [], f"cwor_inversion_embed_failed:{type(exc).__name__}:{exc}"

        prompt_embed_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        def _get_cached_prompt_embeds(candidate_prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
            cached = prompt_embed_cache.get(candidate_prompt)
            if cached is not None:
                return cached
            cached = self._encode_prompt_embeds(candidate_prompt)
            prompt_embed_cache[candidate_prompt] = cached
            return cached

        def _resolve_conf_from_payload(payload: Dict[str, object]) -> Optional[float]:
            if mode == "target":
                raw_conf = payload.get("target_label_logit")
                if raw_conf is None:
                    raw_conf = payload.get("target_label_conf")
            else:
                raw_conf = payload.get("target_logit")
                if raw_conf is None:
                    raw_conf = payload.get("target_conf")
            try:
                c_i = float(raw_conf)
            except Exception:
                return None
            if not np.isfinite(c_i):
                return None
            return float(c_i)

        def _resolve_candidate_conf(base_item: Dict[str, object]) -> Optional[float]:
            return _resolve_conf_from_payload(base_item)

        def _build_cwor_embeds(candidate_prompt: str, delta_c: float) -> Tuple[torch.Tensor, torch.Tensor]:
            base_prompt_embeds, base_pooled_prompt_embeds = _get_cached_prompt_embeds(candidate_prompt)

            if embed_inject_mode in {"t5", "both"}:
                fail_prompt_perp = self._orthogonal_component(
                    base_embed=base_prompt_embeds,
                    fail_embed=fail_prompt_embeds,
                )
                new_prompt_embeds = (
                    base_prompt_embeds.to(torch.float32)
                    - (float(delta_c) * float(embed_subtract_scale)) * fail_prompt_perp.to(torch.float32)
                ).to(device=base_prompt_embeds.device, dtype=base_prompt_embeds.dtype)
            else:
                new_prompt_embeds = base_prompt_embeds

            if embed_inject_mode in {"clip", "both"}:
                fail_pooled_perp = self._orthogonal_component(
                    base_embed=base_pooled_prompt_embeds,
                    fail_embed=fail_pooled_prompt_embeds,
                )
                new_pooled_prompt_embeds = (
                    base_pooled_prompt_embeds.to(torch.float32)
                    - (float(delta_c) * float(embed_subtract_scale)) * fail_pooled_perp.to(torch.float32)
                ).to(device=base_pooled_prompt_embeds.device, dtype=base_pooled_prompt_embeds.dtype)
            else:
                new_pooled_prompt_embeds = base_pooled_prompt_embeds
            return new_prompt_embeds, new_pooled_prompt_embeds

        cwor_results: List[Dict[str, object]] = []
        failures: List[str] = []
        if bool(aggregate_feedback):
            reference_prompt = str(cwor_base_prompt or "").strip()
            if len(reference_prompt) == 0:
                reference_prompt = str(inversion_prompt or "").strip()
            if len(reference_prompt) == 0:
                return [], "cwor_reference_prompt_missing"
            try:
                reference_prompt_embeds, reference_pooled_prompt_embeds = _get_cached_prompt_embeds(reference_prompt)
            except Exception as exc:
                return [], f"cwor_reference_embed_failed:{type(exc).__name__}:{exc}"

            agg_prompt_update_sum: Optional[torch.Tensor] = None
            agg_pooled_update_sum: Optional[torch.Tensor] = None
            agg_count = 0
            agg_conf_sum = 0.0
            agg_delta_sum = 0.0
            agg_delta_signed_sum = 0.0
            agg_delta_values: List[float] = []
            agg_candidate_word = ""

            state_key: Tuple[object, ...] = (
                str(inversion_prompt),
                str(reference_prompt),
                str(mode),
                str(embed_inject_mode),
                str(feedback_merge_mode),
                None if cwor_target_label is None else int(cwor_target_label),
                float(c_base),
                bool(delta_use_basis_logit_without_secondary_ortho),
            )
            if self._cwor_aggregate_state is not None:
                prev_key = self._cwor_aggregate_state.get("key")
                if prev_key != state_key:
                    self._cwor_aggregate_state = None

            use_secondary_ortho = bool(
                accumulate_secondary_ortho and feedback_merge_mode != "step_prompt_weighted"
            )
            use_step_prompt_candidate_ortho = bool(
                feedback_merge_mode == "step_prompt_weighted" and step_prompt_candidate_ortho
            )
            cwor_prompt_basis_embeds: Optional[torch.Tensor] = None
            cwor_pooled_basis_embeds: Optional[torch.Tensor] = None
            if use_secondary_ortho and self._cwor_aggregate_state is not None:
                prev_delta_sum = self._cwor_aggregate_state.get("delta_sum", 0.0)
                prev_prompt_update_sum_raw = self._cwor_aggregate_state.get("prompt_update_sum")
                prev_pooled_update_sum_raw = self._cwor_aggregate_state.get("pooled_update_sum")
                prev_prompt_update_sum = prev_prompt_update_sum_raw if torch.is_tensor(prev_prompt_update_sum_raw) else None
                prev_pooled_update_sum = (
                    prev_pooled_update_sum_raw if torch.is_tensor(prev_pooled_update_sum_raw) else None
                )
                prev_normalized_prompt_update_sum = self._normalize_cwor_update_sum(
                    update_sum=prev_prompt_update_sum,
                    delta_sum=prev_delta_sum,
                )
                prev_normalized_pooled_update_sum = self._normalize_cwor_update_sum(
                    update_sum=prev_pooled_update_sum,
                    delta_sum=prev_delta_sum,
                )
                if embed_inject_mode in {"t5", "both"} and prev_normalized_prompt_update_sum is not None:
                    cwor_prompt_basis_embeds = (
                        reference_prompt_embeds.to(torch.float32) - prev_normalized_prompt_update_sum.to(torch.float32)
                    ).to(device=reference_prompt_embeds.device, dtype=reference_prompt_embeds.dtype)
                if embed_inject_mode in {"clip", "both"} and prev_normalized_pooled_update_sum is not None:
                    cwor_pooled_basis_embeds = (
                        reference_pooled_prompt_embeds.to(torch.float32)
                        - prev_normalized_pooled_update_sum.to(torch.float32)
                    ).to(
                        device=reference_pooled_prompt_embeds.device,
                        dtype=reference_pooled_prompt_embeds.dtype,
                    )

            delta_reference_conf = float(c_base)
            use_prev_basis_conf_for_delta = bool(use_secondary_ortho)
            if (
                not accumulate_secondary_ortho
                and delta_use_basis_logit_without_secondary_ortho
            ):
                use_prev_basis_conf_for_delta = True
            if use_prev_basis_conf_for_delta and self._cwor_aggregate_state is not None:
                prev_basis_conf_raw = self._cwor_aggregate_state.get("basis_confidence")
                try:
                    prev_basis_conf = float(prev_basis_conf_raw)
                except Exception:
                    prev_basis_conf = None
                if prev_basis_conf is not None and np.isfinite(prev_basis_conf):
                    delta_reference_conf = float(prev_basis_conf)

            indexed_base_candidates: List[Tuple[int, Dict[str, object]]] = [
                (int(idx), base_item) for idx, base_item in enumerate(base_candidates)
            ]
            if use_step_prompt_candidate_ortho:
                def _objective_sort_key(indexed_item: Tuple[int, Dict[str, object]]) -> Tuple[float, int]:
                    raw_objective = indexed_item[1].get("candidate_objective")
                    try:
                        objective = float(raw_objective)
                    except Exception:
                        objective = float("-inf")
                    if not np.isfinite(objective):
                        objective = float("-inf")
                    # Tie-break on original order to keep behavior deterministic.
                    return objective, -int(indexed_item[0])

                indexed_base_candidates = sorted(indexed_base_candidates, key=_objective_sort_key, reverse=True)

            step_prompt_higher_prompt_perps: List[torch.Tensor] = []
            step_prompt_higher_pooled_perps: List[torch.Tensor] = []
            for idx, base_item in indexed_base_candidates:
                candidate_prompt = str(base_item.get("candidate_prompt", "")).strip()
                candidate_word = str(base_item.get("candidate_word", "")).strip()
                if len(candidate_prompt) == 0:
                    continue
                c_i = _resolve_candidate_conf(base_item)
                if c_i is None:
                    continue
                delta_c = float(c_i - delta_reference_conf)
                try:
                    fail_i_prompt_embeds, fail_i_pooled_prompt_embeds = _get_cached_prompt_embeds(candidate_prompt)
                    if feedback_merge_mode == "step_prompt_weighted":
                        if embed_inject_mode in {"t5", "both"}:
                            fail_prompt_perp = self._orthogonal_component(
                                base_embed=reference_prompt_embeds,
                                fail_embed=fail_i_prompt_embeds,
                            )
                            if use_step_prompt_candidate_ortho:
                                for higher_prompt_perp in step_prompt_higher_prompt_perps:
                                    fail_prompt_perp = self._orthogonal_component(
                                        base_embed=higher_prompt_perp,
                                        fail_embed=fail_prompt_perp,
                                    )
                            update_prompt = float(delta_c) * fail_prompt_perp.to(torch.float32)
                            if agg_prompt_update_sum is None:
                                agg_prompt_update_sum = update_prompt
                            else:
                                agg_prompt_update_sum = agg_prompt_update_sum + update_prompt
                            if use_step_prompt_candidate_ortho:
                                step_prompt_higher_prompt_perps.append(fail_prompt_perp)
                        if embed_inject_mode in {"clip", "both"}:
                            fail_pooled_perp = self._orthogonal_component(
                                base_embed=reference_pooled_prompt_embeds,
                                fail_embed=fail_i_pooled_prompt_embeds,
                            )
                            if use_step_prompt_candidate_ortho:
                                for higher_pooled_perp in step_prompt_higher_pooled_perps:
                                    fail_pooled_perp = self._orthogonal_component(
                                        base_embed=higher_pooled_perp,
                                        fail_embed=fail_pooled_perp,
                                    )
                            update_pooled = float(delta_c) * fail_pooled_perp.to(torch.float32)
                            if agg_pooled_update_sum is None:
                                agg_pooled_update_sum = update_pooled
                            else:
                                agg_pooled_update_sum = agg_pooled_update_sum + update_pooled
                            if use_step_prompt_candidate_ortho:
                                step_prompt_higher_pooled_perps.append(fail_pooled_perp)
                    else:
                        if embed_inject_mode in {"t5", "both"}:
                            fail_prompt_perp = self._orthogonal_component(
                                base_embed=reference_prompt_embeds,
                                fail_embed=fail_i_prompt_embeds,
                            )
                            if cwor_prompt_basis_embeds is not None:
                                fail_prompt_perp = self._orthogonal_component(
                                    base_embed=cwor_prompt_basis_embeds,
                                    fail_embed=fail_prompt_perp,
                                )
                            update_prompt = float(delta_c) * fail_prompt_perp.to(torch.float32)
                            if agg_prompt_update_sum is None:
                                agg_prompt_update_sum = update_prompt
                            else:
                                agg_prompt_update_sum = agg_prompt_update_sum + update_prompt
                        if embed_inject_mode in {"clip", "both"}:
                            fail_pooled_perp = self._orthogonal_component(
                                base_embed=reference_pooled_prompt_embeds,
                                fail_embed=fail_i_pooled_prompt_embeds,
                            )
                            if cwor_pooled_basis_embeds is not None:
                                fail_pooled_perp = self._orthogonal_component(
                                    base_embed=cwor_pooled_basis_embeds,
                                    fail_embed=fail_pooled_perp,
                                )
                            update_pooled = float(delta_c) * fail_pooled_perp.to(torch.float32)
                            if agg_pooled_update_sum is None:
                                agg_pooled_update_sum = update_pooled
                            else:
                                agg_pooled_update_sum = agg_pooled_update_sum + update_pooled
                    if len(agg_candidate_word) == 0:
                        agg_candidate_word = candidate_word
                    agg_count += 1
                    agg_conf_sum += float(c_i)
                    agg_delta_sum += abs(float(delta_c))
                    agg_delta_signed_sum += float(delta_c)
                    agg_delta_values.append(float(delta_c))
                except Exception as exc:
                    failures.append(f"{idx}:{type(exc).__name__}:{exc}")

            if agg_count > 0:
                cum_count = int(agg_count)
                cum_conf_sum = float(agg_conf_sum)
                cum_delta_sum = float(agg_delta_sum)
                cum_delta_values: List[float] = list(agg_delta_values)
                cum_prompt_update_sum = agg_prompt_update_sum
                cum_pooled_update_sum = agg_pooled_update_sum
                state_word = agg_candidate_word

                if feedback_merge_mode != "step_prompt_weighted":
                    delta_denom = float(agg_delta_sum)
                    if not np.isfinite(delta_denom) or abs(delta_denom) < 1e-12:
                        failures.append("agg:accumulate_zero_delta_sum")
                        cum_prompt_update_sum = None
                        cum_pooled_update_sum = None
                    else:
                        if cum_prompt_update_sum is not None:
                            cum_prompt_update_sum = cum_prompt_update_sum.to(torch.float32) / float(delta_denom)
                        if cum_pooled_update_sum is not None:
                            cum_pooled_update_sum = cum_pooled_update_sum.to(torch.float32) / float(delta_denom)

                if feedback_merge_mode != "step_prompt_weighted" and self._cwor_aggregate_state is not None:
                    prev_count = self._cwor_aggregate_state.get("count")
                    prev_conf_sum = self._cwor_aggregate_state.get("conf_sum")
                    prev_word = str(self._cwor_aggregate_state.get("reference_word", "")).strip()
                    prev_delta_sum = self._cwor_aggregate_state.get("delta_sum", 1.0)
                    prev_prompt_update_sum_raw = self._cwor_aggregate_state.get("prompt_update_sum")
                    prev_pooled_update_sum_raw = self._cwor_aggregate_state.get("pooled_update_sum")
                    prev_prompt_update_sum = self._normalize_cwor_update_sum(
                        update_sum=(
                            prev_prompt_update_sum_raw
                            if torch.is_tensor(prev_prompt_update_sum_raw)
                            else None
                        ),
                        delta_sum=prev_delta_sum,
                    )
                    prev_pooled_update_sum = self._normalize_cwor_update_sum(
                        update_sum=(
                            prev_pooled_update_sum_raw
                            if torch.is_tensor(prev_pooled_update_sum_raw)
                            else None
                        ),
                        delta_sum=prev_delta_sum,
                    )
                    if isinstance(prev_count, int) and prev_count > 0:
                        try:
                            cum_count = int(prev_count) + int(agg_count)
                            cum_conf_sum = float(prev_conf_sum) + float(agg_conf_sum)
                            # Keep report delta_sum step-local (current step only).
                            cum_delta_sum = float(agg_delta_sum)
                            state_word = prev_word
                            if embed_inject_mode in {"t5", "both"} and prev_prompt_update_sum is not None:
                                if cum_prompt_update_sum is None:
                                    cum_prompt_update_sum = prev_prompt_update_sum
                                else:
                                    cum_prompt_update_sum = prev_prompt_update_sum + cum_prompt_update_sum.to(torch.float32)
                            if embed_inject_mode in {"clip", "both"} and prev_pooled_update_sum is not None:
                                if cum_pooled_update_sum is None:
                                    cum_pooled_update_sum = prev_pooled_update_sum
                                else:
                                    cum_pooled_update_sum = prev_pooled_update_sum + cum_pooled_update_sum.to(torch.float32)
                        except Exception as exc:
                            failures.append(f"agg_state_merge:{type(exc).__name__}:{exc}")

                alpha_sum_for_report = float(cum_delta_sum)
                if feedback_merge_mode == "step_prompt_weighted":
                    delta_denom = float(agg_delta_sum)
                    if not np.isfinite(delta_denom) or abs(delta_denom) < 1e-12:
                        failures.append("agg:step_prompt_weighted_zero_delta_sum")
                        normalized_prompt_update_sum = None
                        normalized_pooled_update_sum = None
                        state_delta_sum = float(cum_delta_sum)
                    else:
                        normalized_prompt_update_sum = (
                            None
                            if cum_prompt_update_sum is None
                            else (cum_prompt_update_sum.to(torch.float32) / float(delta_denom))
                        )
                        normalized_pooled_update_sum = (
                            None
                            if cum_pooled_update_sum is None
                            else (cum_pooled_update_sum.to(torch.float32) / float(delta_denom))
                        )
                        cum_prompt_update_sum = normalized_prompt_update_sum
                        cum_pooled_update_sum = normalized_pooled_update_sum
                        state_delta_sum = 1.0
                        alpha_sum_for_report = float(delta_denom)
                else:
                    normalized_prompt_update_sum = (
                        None if cum_prompt_update_sum is None else cum_prompt_update_sum.to(torch.float32)
                    )
                    normalized_pooled_update_sum = (
                        None if cum_pooled_update_sum is None else cum_pooled_update_sum.to(torch.float32)
                    )
                    state_delta_sum = 1.0

                previous_aggregate_state = self._cwor_aggregate_state
                self._cwor_aggregate_state = {
                    "key": state_key,
                    "reference_prompt": str(reference_prompt),
                    "reference_word": str(state_word),
                    "count": int(cum_count),
                    "conf_sum": float(cum_conf_sum),
                    "delta_sum": float(state_delta_sum),
                    "delta_values": list(cum_delta_values),
                    "prompt_update_sum": None if cum_prompt_update_sum is None else cum_prompt_update_sum.detach(),
                    "pooled_update_sum": None if cum_pooled_update_sum is None else cum_pooled_update_sum.detach(),
                    "cwor_feedback_merge_mode": str(feedback_merge_mode),
                }

                try:

                    if embed_inject_mode in {"t5", "both"} and normalized_prompt_update_sum is not None:
                        merged_prompt_embeds = (
                            reference_prompt_embeds.to(torch.float32)
                            - float(embed_subtract_scale) * normalized_prompt_update_sum.to(torch.float32)
                        ).to(device=reference_prompt_embeds.device, dtype=reference_prompt_embeds.dtype)
                    else:
                        merged_prompt_embeds = reference_prompt_embeds
                    if embed_inject_mode in {"clip", "both"} and normalized_pooled_update_sum is not None:
                        merged_pooled_prompt_embeds = (
                            reference_pooled_prompt_embeds.to(torch.float32)
                            - float(embed_subtract_scale) * normalized_pooled_update_sum.to(torch.float32)
                        ).to(
                            device=reference_pooled_prompt_embeds.device,
                            dtype=reference_pooled_prompt_embeds.dtype,
                        )
                    else:
                        merged_pooled_prompt_embeds = reference_pooled_prompt_embeds
                    image_01 = self._render_candidate_tensor_from_embeds(
                        inversion_prompt=str(inversion_prompt),
                        prompt_embeds=merged_prompt_embeds,
                        pooled_prompt_embeds=merged_pooled_prompt_embeds,
                    ).to(device=str(self.args.device))
                    objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
                    objective_value = float(objective)
                    should_update_aggregate_state = True
                    previous_aggregate_objective: Optional[float] = None
                    if isinstance(previous_aggregate_state, dict):
                        previous_objective_raw = previous_aggregate_state.get("aggregate_objective")
                        try:
                            previous_aggregate_objective = float(previous_objective_raw)
                        except Exception:
                            previous_aggregate_objective = None
                    cwor_alpha_sign_flipped = False
                    if (
                        step_prompt_flip_alpha_on_regression
                        and feedback_merge_mode == "step_prompt_weighted"
                        and previous_aggregate_objective is not None
                        and np.isfinite(previous_aggregate_objective)
                        and objective_value < float(previous_aggregate_objective)
                    ):
                        flipped_prompt_update_sum = (
                            None
                            if normalized_prompt_update_sum is None
                            else (-normalized_prompt_update_sum.to(torch.float32))
                        )
                        flipped_pooled_update_sum = (
                            None
                            if normalized_pooled_update_sum is None
                            else (-normalized_pooled_update_sum.to(torch.float32))
                        )
                        if embed_inject_mode in {"t5", "both"} and flipped_prompt_update_sum is not None:
                            flipped_prompt_embeds = (
                                reference_prompt_embeds.to(torch.float32)
                                - float(embed_subtract_scale) * flipped_prompt_update_sum.to(torch.float32)
                            ).to(device=reference_prompt_embeds.device, dtype=reference_prompt_embeds.dtype)
                        else:
                            flipped_prompt_embeds = reference_prompt_embeds
                        if embed_inject_mode in {"clip", "both"} and flipped_pooled_update_sum is not None:
                            flipped_pooled_embeds = (
                                reference_pooled_prompt_embeds.to(torch.float32)
                                - float(embed_subtract_scale) * flipped_pooled_update_sum.to(torch.float32)
                            ).to(
                                device=reference_pooled_prompt_embeds.device,
                                dtype=reference_pooled_prompt_embeds.dtype,
                            )
                        else:
                            flipped_pooled_embeds = reference_pooled_prompt_embeds
                        flipped_image_01 = self._render_candidate_tensor_from_embeds(
                            inversion_prompt=str(inversion_prompt),
                            prompt_embeds=flipped_prompt_embeds,
                            pooled_prompt_embeds=flipped_pooled_embeds,
                        ).to(device=str(self.args.device))
                        objective, stats = classifier.objective_and_stats(
                            flipped_image_01,
                            target_label=cwor_target_label,
                        )
                        image_01 = flipped_image_01
                        objective_value = float(objective)
                        normalized_prompt_update_sum = flipped_prompt_update_sum
                        normalized_pooled_update_sum = flipped_pooled_update_sum
                        cum_prompt_update_sum = normalized_prompt_update_sum
                        cum_pooled_update_sum = normalized_pooled_update_sum
                        cum_delta_values = [-float(value) for value in cum_delta_values]
                        cwor_alpha_sign_flipped = True
                        if self._cwor_aggregate_state is not None:
                            self._cwor_aggregate_state["delta_values"] = list(cum_delta_values)
                            self._cwor_aggregate_state["prompt_update_sum"] = (
                                None if cum_prompt_update_sum is None else cum_prompt_update_sum.detach()
                            )
                            self._cwor_aggregate_state["pooled_update_sum"] = (
                                None if cum_pooled_update_sum is None else cum_pooled_update_sum.detach()
                            )
                    if (
                        accumulate_update_if_improved_only
                        and previous_aggregate_objective is not None
                        and np.isfinite(previous_aggregate_objective)
                    ):
                        if (
                            not (objective_value > float(previous_aggregate_objective))
                        ):
                            should_update_aggregate_state = False

                    if should_update_aggregate_state:
                        if self._cwor_aggregate_state is not None:
                            self._cwor_aggregate_state["aggregate_objective"] = float(objective_value)
                    else:
                        self._cwor_aggregate_state = previous_aggregate_state

                    basis_conf = _resolve_conf_from_payload(stats if isinstance(stats, dict) else {})
                    if should_update_aggregate_state and basis_conf is not None and self._cwor_aggregate_state is not None:
                        self._cwor_aggregate_state["basis_confidence"] = float(basis_conf)
                    cwor_selected_image = image_tensor_01_to_pil(image_01).copy()
                    cwor_results.append(
                        {
                            "candidate_word": cwor_placeholder_word,
                            "candidate_prompt": str(reference_prompt),
                            "candidate_objective": float(objective_value),
                            "pred_idx": stats.get("pred_idx"),
                            "pred_conf": stats.get("pred_conf"),
                            "pred_logit": stats.get("pred_logit"),
                            "target_conf": stats.get("target_conf"),
                            "target_logit": stats.get("target_logit"),
                            "target_label_conf": stats.get("target_label_conf"),
                            "target_label_logit": stats.get("target_label_logit"),
                            "ce": stats.get("ce"),
                            "candidate_variant": "cwor",
                            "candidate_selected_image": cwor_selected_image,
                            "candidate_selected_image_width": int(cwor_selected_image.size[0]),
                            "candidate_selected_image_height": int(cwor_selected_image.size[1]),
                            "candidate_selected_image_source": "cwor_tensor",
                            "cwor_base_candidate_index": -1,
                            "cwor_mode": mode,
                            "cwor_embed_inject_mode": embed_inject_mode,
                            "cwor_target_label": cwor_target_label,
                            "cwor_feedback_merge_mode": str(feedback_merge_mode),
                            "cwor_candidate_confidence": float(cum_conf_sum / float(cum_count)),
                            "cwor_base_confidence": float(c_base),
                            "cwor_candidate_logit": float(cum_conf_sum / float(cum_count)),
                            "cwor_base_logit": float(c_base),
                            "cwor_alpha": list(cum_delta_values),
                            "cwor_alpha_sum": float(alpha_sum_for_report),
                            "cwor_alpha_sign_flipped": bool(cwor_alpha_sign_flipped),
                            "cwor_feedback_count": int(cum_count),
                            "cwor_feedback_step_count": int(agg_count),
                            "cwor_state_updated": bool(should_update_aggregate_state),
                            "cwor_aggregate_feedback": True,
                        }
                    )
                except Exception as exc:
                    failures.append(f"agg:{type(exc).__name__}:{exc}")

            if len(failures) == 0:
                return cwor_results, None
            short = "; ".join(failures[:3])
            if len(failures) > 3:
                short = f"{short}; ..."
            return cwor_results, f"cwor_failures={len(failures)} ({short})"

        for idx, base_item in enumerate(base_candidates):
            candidate_prompt = str(base_item.get("candidate_prompt", "")).strip()
            candidate_word = str(base_item.get("candidate_word", "")).strip()
            if len(candidate_prompt) == 0:
                continue

            c_i = _resolve_candidate_conf(base_item)
            if c_i is None:
                continue

            delta_c = float(c_i - c_base)
            try:
                new_prompt_embeds, new_pooled_prompt_embeds = _build_cwor_embeds(candidate_prompt, delta_c)
                image_01 = self._render_candidate_tensor_from_embeds(
                    inversion_prompt=str(inversion_prompt),
                    prompt_embeds=new_prompt_embeds,
                    pooled_prompt_embeds=new_pooled_prompt_embeds,
                ).to(device=str(self.args.device))
                objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
                cwor_selected_image = image_tensor_01_to_pil(image_01).copy()
                cwor_results.append(
                    {
                        "candidate_word": cwor_placeholder_word,
                        "candidate_prompt": candidate_prompt,
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
                        "candidate_selected_image": cwor_selected_image,
                        "candidate_selected_image_width": int(cwor_selected_image.size[0]),
                        "candidate_selected_image_height": int(cwor_selected_image.size[1]),
                        "candidate_selected_image_source": "cwor_tensor",
                        "cwor_base_candidate_index": int(idx),
                        "cwor_mode": mode,
                        "cwor_embed_inject_mode": embed_inject_mode,
                        "cwor_target_label": cwor_target_label,
                        "cwor_candidate_confidence": float(c_i),
                        "cwor_base_confidence": float(c_base),
                        "cwor_candidate_logit": float(c_i),
                        "cwor_base_logit": float(c_base),
                        "cwor_alpha": float(delta_c),
                    }
                )
            except Exception as exc:
                failures.append(f"{idx}:{type(exc).__name__}:{exc}")

        if len(failures) == 0:
            return cwor_results, None
        short = "; ".join(failures[:3])
        if len(failures) > 3:
            short = f"{short}; ..."
        return cwor_results, f"cwor_failures={len(failures)} ({short})"

    def close(self) -> None:
        stable_flow = self._stable_flow
        if stable_flow is None:
            return
        try:
            stable_flow.close()
        except Exception:
            pass
        pipe = getattr(stable_flow, "pipe", None)
        if pipe is not None and hasattr(pipe, "to"):
            try:
                pipe.to("cpu")
            except Exception:
                pass
        self._stable_flow = None
        self._edit_cache = None
        self._edit_cache_key = None
        self._cwor_aggregate_state = None
        self._cwor_embed_step_counter = 0
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


class TorchvisionClassifier(nn.Module):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @staticmethod
    def _canonical_torchvision_model_name(model_name: str) -> str:
        text = str(model_name or "").strip().lower()
        aliases = {
            "vit-base": "vit_b_16",
            "vit_base": "vit_b_16",
            "vit-b-16": "vit_b_16",
            "vitb16": "vit_b_16",
            "vit-base-patch16-224": "vit_b_16",
        }
        return aliases.get(text, text)

    def __init__(
        self,
        model_name: str,
        device: str,
        checkpoint_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        input_size: int = 224,
        objective_mode: str = "ce_max",
        label: Optional[int] = None,
    ):
        super().__init__()
        if torchvision is None:
            raise ImportError(
                f"torchvision import failed: {_TORCHVISION_IMPORT_ERROR}. Install torchvision first."
            )
        self.device_name = str(device)
        self.input_size = int(input_size)
        self.objective_mode = str(objective_mode or "ce_max").strip().lower()
        self.label = label
        self.requested_model_name = str(model_name)
        self.model_name = self._canonical_torchvision_model_name(model_name)
        self.model, self.weights = self._build_model(self.model_name, checkpoint_path, num_classes)
        self.preprocess_transform = self.weights.transforms() if self.weights is not None else None
        self.model.eval().to(self.device_name)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _build_model(self, model_name: str, checkpoint_path: Optional[str], num_classes: Optional[int]):
        model = None
        weights = None
        if checkpoint_path is None:
            try:
                weights_enum = models.get_model_weights(model_name)
                weights = weights_enum.DEFAULT
                model = models.get_model(model_name, weights=weights)
            except Exception:
                pass

        if model is None:
            ctor = getattr(models, model_name)
            if checkpoint_path is None:
                try:
                    model = ctor(weights="DEFAULT")
                except Exception:
                    try:
                        model = ctor(pretrained=True)
                    except Exception:
                        model = ctor()
            else:
                model = ctor()

        if checkpoint_path is not None:
            if num_classes is not None:
                self._replace_head(model, int(num_classes))
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)
        elif num_classes is not None:
            self._replace_head(model, int(num_classes))
        return model, weights

    @staticmethod
    def _replace_head(model: nn.Module, num_classes: int) -> None:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                return
            if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
                last = model.classifier[-1]
                if isinstance(last, nn.Linear):
                    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
                    return
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
            return
        if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return
        raise ValueError("Could not automatically replace classifier head for this model.")

    def preprocess(self, image_01: torch.Tensor) -> torch.Tensor:
        if self.preprocess_transform is not None:
            try:
                x = self.preprocess_transform(image_01)
            except Exception:
                x = torch.stack([self.preprocess_transform(img) for img in image_01], dim=0)
        else:
            x = F.interpolate(
                image_01,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            mean = self.IMAGENET_MEAN.to(x.device, x.dtype)
            std = self.IMAGENET_STD.to(x.device, x.dtype)
            x = (x - mean) / std

        param = next(self.model.parameters())
        return x.to(device=param.device, dtype=param.dtype)

    def logits(self, image_01: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(image_01))

    def _prediction_index_offset(self) -> int:
        if self.requested_model_name.strip().lower() in {"adv_res", "adv_inc"}:
            return 1
        return 0

    def _normalize_prediction_index(self, pred_idx: int) -> int:
        return int(pred_idx) - int(self._prediction_index_offset())

    def objective_and_stats(
        self,
        image_01: torch.Tensor,
        target_label: Optional[int] = None,
    ) -> Tuple[float, Dict[str, object]]:
        if self.label is None:
            raise ValueError("--classifier_label is required for CE score-based black-box mode.")
        logits = self.logits(image_01)
        labels = torch.full((logits.shape[0],), fill_value=int(self.label), device=logits.device, dtype=torch.long)
        ce = F.cross_entropy(logits, labels)
        if int(logits.shape[-1]) <= 1:
            raise ValueError("classifier logits must have at least 2 classes for margin objective.")
        true_logits = logits[:, int(self.label)]
        non_true_logits = logits.clone()
        non_true_logits[:, int(self.label)] = float("-inf")
        max_non_true_logits = non_true_logits.max(dim=1).values
        logit_margin = max_non_true_logits - true_logits
        logit_margin_value = float(logit_margin.mean().detach().item())
        if self.objective_mode == "ce_max":
            objective = float(ce.detach().item())
        elif self.objective_mode in {"ce_min", "logit_max"}:
            # Keep score-based formulation CE-only as requested.
            objective = float((-ce).detach().item())
        elif self.objective_mode == "logit_margin_max":
            objective = logit_margin_value
        else:
            raise ValueError(f"Unsupported classifier objective for black-box CE scoring: {self.objective_mode}")

        mean_logits = logits.float().mean(dim=0)
        pred_logit, raw_pred_label = mean_logits.max(dim=0)
        pred_label = self._normalize_prediction_index(int(raw_pred_label.item()))
        target_logit = float(mean_logits[int(self.label)].item())
        mean_probs = logits.float().softmax(dim=-1).mean(dim=0)
        pred_confidence, _ = mean_probs.max(dim=0)
        target_conf = float(mean_probs[int(self.label)].item())
        stats: Dict[str, object] = {
            "pred_idx": pred_label,
            "pred_conf": float(pred_confidence.item()),
            "target_conf": target_conf,
            "pred_logit": float(pred_logit.item()),
            "target_logit": target_logit,
            "ce": float(ce.detach().item()),
            "logit_margin": logit_margin_value,
        }
        if target_label is not None:
            target_idx = int(target_label)
            if 0 <= target_idx < int(mean_logits.shape[0]):
                target_label_logit = float(mean_logits[target_idx].item())
                target_label_conf = float(mean_probs[target_idx].item())
                stats["target_label"] = int(target_idx)
                stats["target_label_conf"] = target_label_conf
                stats["target_label_logit"] = target_label_logit
        return objective, stats


def parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_cwor_embed_inject_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"clip", "text_encoder1", "te1"}:
        return "clip"
    if token in {"t5", "text_encoder2", "te2"}:
        return "t5"
    return "both"


def normalize_cwor_feedback_merge_mode(raw: object) -> str:
    token = str(raw or "").strip().lower()
    if token in {"step_prompt_weighted", "step-weighted", "step_prompt", "per_step_prompt"}:
        return "step_prompt_weighted"
    return "accumulate"


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Black-box VLM word-substitution attack (no gradient)."
    )
    parser.add_argument("--mode", type=str, default="gcg_edit")
    parser.add_argument("--classifier_mode", type=str, default=os.getenv("CLASSIFIER_MODE", "black-box"))
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--hf_token", type=str, required=True)

    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--inversion_prompt", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)

    parser.add_argument("--output_path", type=str, default="outputs/result.png")
    parser.add_argument("--report_path", type=str, default="outputs/gcg_report.json")
    parser.add_argument("--input_img_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--latent_nudging_scalar", type=float, default=1.15)

    parser.add_argument("--gcg_word", type=str, default="background")
    parser.add_argument("--gcg_occurrence", type=int, default=0)
    parser.add_argument("--gcg_steps", type=int, default=10)
    parser.add_argument("--gcg_batch_size", type=int, default=64)
    parser.add_argument("--gcg_topk", type=int, default=128)
    parser.add_argument("--gcg_candidate_source", type=str, default="vlm_query")
    parser.add_argument("--cwor_enable", type=parse_bool_flag, default=parse_bool_flag(os.getenv("CWOR_ENABLE", "0")))
    parser.add_argument(
        "--cwor_mode",
        type=str,
        choices=["untargeted", "target"],
        default=os.getenv("CWOR_MODE", "untargeted"),
    )
    parser.add_argument(
        "--cwor_embed_inject_mode",
        type=str,
        choices=["clip", "t5", "both"],
        default=os.getenv("CWOR_EMBED_INJECT_MODE", "both"),
    )
    parser.add_argument(
        "--cwor_target_label",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cwor_feedback_merge_mode",
        type=str,
        choices=["accumulate", "step_prompt_weighted"],
        default=os.getenv("CWOR_FEEDBACK_MERGE_MODE", "accumulate"),
    )
    parser.add_argument(
        "--cwor_accumulate_secondary_ortho",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("CWOR_ACCUMULATE_SECONDARY_ORTHO", "0")),
    )
    parser.add_argument(
        "--cwor_accumulate_update_if_improved_only",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("CWOR_ACCUMULATE_UPDATE_IF_IMPROVED_ONLY", "0")),
    )
    parser.add_argument(
        "--cwor_accumulate_delta_use_basis_logit_without_secondary_ortho",
        type=parse_bool_flag,
        default=parse_bool_flag(
            os.getenv("CWOR_ACCUMULATE_DELTA_USE_BASIS_LOGIT_WITHOUT_SECONDARY_ORTHO", "0")
        ),
    )
    parser.add_argument(
        "--cwor_step_prompt_candidate_ortho",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("CWOR_STEP_PROMPT_CANDIDATE_ORTHO", "0")),
    )
    parser.add_argument(
        "--cwor_step_prompt_flip_alpha_on_regression",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("CWOR_STEP_PROMPT_FLIP_ALPHA_ON_REGRESSION", "0")),
    )
    parser.add_argument(
        "--cwor_embed_subtract_scale_by_step",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("CWOR_EMBED_SUBTRACT_SCALE_BY_STEP", "0")),
    )
    parser.add_argument("--gcg_scene_vocab_size", type=int, default=100)
    parser.add_argument(
        "--gcg_scene_vocab_prompts_per_strategy",
        type=int,
        default=int(os.getenv("GCG_SCENE_VOCAB_PROMPTS_PER_STRATEGY", "0")),
    )
    parser.add_argument(
        "--gcg_scene_vocab_enabled_strategies",
        type=str,
        default=os.getenv("GCG_SCENE_VOCAB_ENABLED_STRATEGIES", "all"),
    )
    parser.add_argument(
        "--gcg_slot_candidate_max_words",
        type=int,
        default=int(os.getenv("GCG_SLOT_CANDIDATE_MAX_WORDS", "5")),
    )
    parser.add_argument("--gcg_scene_vocab_topic", type=str, default=None)
    parser.add_argument("--gcg_save_intermediate", action="store_true")
    parser.add_argument("--gcg_save_intermediate_interval", type=int, default=1)
    parser.add_argument(
        "--gcg_early_stop_on_attack_success",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("GCG_EARLY_STOP_ON_ATTACK_SUCCESS", "0")),
    )
    parser.add_argument("--gcg_scene_vocab_feedback", action="store_true")
    parser.add_argument("--gcg_scene_feedback_limit", type=int, default=100)
    parser.add_argument(
        "--gcg_scene_llm_model_id",
        type=str,
        default=os.getenv(
            "GCG_SCENE_LLM_MODEL_ID",
            os.getenv("SCENE_VLM_MODEL_ID", "google/gemma-3-4b-it"),
        ),
    )
    parser.add_argument(
        "--gcg_scene_llm_backend",
        type=str,
        default=os.getenv(
            "GCG_SCENE_LLM_BACKEND",
            "gemma3",
        ),
    )
    parser.add_argument(
        "--gcg_scene_llm_device",
        type=str,
        default=os.getenv("GCG_SCENE_LLM_DEVICE", os.getenv("SCENE_VLM_DEVICE", "auto")),
    )
    parser.add_argument(
        "--gcg_scene_llm_max_new_tokens",
        type=int,
        default=int(
            os.getenv(
                "GCG_SCENE_LLM_MAX_NEW_TOKENS",
                os.getenv("SCENE_VLM_MAX_NEW_TOKENS", "384"),
            )
        ),
    )
    parser.add_argument(
        "--gcg_scene_llm_thinking",
        type=parse_bool_flag,
        default=parse_bool_flag(
            os.getenv("GCG_SCENE_LLM_THINKING", os.getenv("SCENE_VLM_THINKING", "0"))
        ),
    )
    parser.add_argument(
        "--gcg_eval_naturalness_llm_thinking",
        type=parse_bool_flag,
        default=parse_bool_flag(
            os.getenv(
                "GCG_EVAL_NATURALNESS_LLM_THINKING",
                os.getenv("GCG_SCENE_LLM_THINKING", os.getenv("SCENE_VLM_THINKING", "0")),
            )
        ),
    )
    parser.add_argument(
        "--gcg_scene_llm_do_sample",
        type=parse_bool_flag,
        default=parse_bool_flag(
            os.getenv("GCG_SCENE_LLM_DO_SAMPLE", os.getenv("SCENE_VLM_DO_SAMPLE", "0"))
        ),
    )

    parser.add_argument("--scene_vlm_backend", type=str, default=os.getenv("SCENE_VLM_BACKEND", "gemma3"))
    parser.add_argument(
        "--scene_vlm_model_id",
        type=str,
        default=os.getenv("SCENE_VLM_MODEL_ID", "google/gemma-3-4b-it"),
    )
    parser.add_argument("--scene_vlm_device", type=str, default=os.getenv("SCENE_VLM_DEVICE", "auto"))
    parser.add_argument(
        "--scene_vlm_max_new_tokens",
        type=int,
        default=int(os.getenv("SCENE_VLM_MAX_NEW_TOKENS", "256")),
    )
    parser.add_argument(
        "--scene_vlm_question",
        type=str,
        default=os.getenv("SCENE_VLM_QUESTION", "What is the background scene in this image? Answer in 1 word."),
    )
    parser.add_argument(
        "--scene_vlm_thinking",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("SCENE_VLM_THINKING", "0")),
    )
    parser.add_argument(
        "--scene_vlm_do_sample",
        type=parse_bool_flag,
        default=parse_bool_flag(os.getenv("SCENE_VLM_DO_SAMPLE", "0")),
    )
    parser.add_argument("--scene_fallback", type=str, default=os.getenv("SCENE_FALLBACK", "outdoor"))

    parser.add_argument("--classifier_name", type=str, default="resnet50")
    parser.add_argument("--classifier_ckpt", type=str, default=None)
    parser.add_argument("--classifier_num_classes", type=int, default=None)
    parser.add_argument("--classifier_input_size", type=int, default=224)
    parser.add_argument("--classifier_objective", type=str, default="ce_max")
    parser.add_argument("--classifier_label", type=int, default=None)

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

    args, unknown = parser.parse_known_args()
    return args, unknown


def normalize_classifier_mode(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token in {"white-box", "white_box", "whitebox", "white"}:
        return "white-box"
    if token in {"black-box", "black_box", "blackbox", "black"}:
        return "black-box"
    raise ValueError(f"--classifier_mode must be white-box or black-box (got '{raw}')")


def normalize_candidate_source(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token in {"gemma", "gemma_scene_vocab", "gemma-scene-vocab", "scene_vocab", "scene-vocab"}:
        return "gemma_scene_vocab"
    return "vlm_query"


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


def resolve_default_wandb_run_name(args: argparse.Namespace) -> str:
    explicit = str(args.wandb_run_name or "").strip()
    if explicit:
        return explicit

    report_path = Path(str(args.report_path or "")).expanduser()
    report_parent = str(report_path.parent.name or "").strip()
    if report_parent and report_parent not in {".", "/"}:
        return report_parent
    report_stem = str(report_path.stem or "").strip()
    if report_stem:
        return report_stem

    output_stem = str(Path(str(args.output_path or "")).stem or "").strip()
    if output_stem:
        return output_stem
    return "vlm_attack"


def init_wandb_run(args: argparse.Namespace) -> Tuple[Optional[object], bool]:
    if not bool(args.wandb_enable):
        return None, False

    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(
            "WARNING: wandb_enable=true but wandb import failed; "
            f"continuing without W&B logging ({type(exc).__name__}: {exc})"
        )
        return None, False

    api_key = str(args.wandb_api_key or "").strip()
    api_key_file = str(args.wandb_api_key_file or "").strip()
    if not api_key and api_key_file:
        try:
            api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"WARNING: failed to read wandb_api_key_file='{api_key_file}' ({type(exc).__name__}: {exc})")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    config_payload: Dict[str, object] = dict(vars(args))
    config_payload.pop("wandb_api_key", None)
    config_payload["mode"] = "vlm_attack_black_box"
    config_payload["resolved_output_path"] = str(Path(str(args.output_path)).resolve())
    config_payload["resolved_report_path"] = str(Path(str(args.report_path)).resolve())

    run_name = resolve_default_wandb_run_name(args)
    init_kwargs: Dict[str, object] = {
        "project": str(args.wandb_project),
        "name": run_name,
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
        print(
            "WARNING: Failed to initialize W&B; "
            f"continuing without W&B logging ({type(exc).__name__}: {exc})"
        )
        return None, False

    if run is None:
        print("WARNING: wandb.init returned None; W&B logging is disabled.")
        return None, False

    print(
        "W&B logging enabled: "
        f"project={args.wandb_project}, run={run_name}, "
        f"mode={args.wandb_mode}, log_every={int(args.wandb_log_every)}"
    )
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
        print(
            "WARNING: W&B log call failed; disabling further W&B logs "
            f"({type(exc).__name__}: {exc})"
        )
        return False
    return True


def finish_wandb_run(run: Optional[object], summary: Dict[str, object]) -> None:
    if run is None:
        return
    try:
        for key, value in summary.items():
            run.summary[key] = value
    except Exception as exc:
        print(f"WARNING: failed to write W&B summary ({type(exc).__name__}: {exc})")
    try:
        run.finish()
    except Exception as exc:
        print(f"WARNING: failed to finish W&B run ({type(exc).__name__}: {exc})")


def log_wandb_final_image(
    *,
    run: object,
    args: argparse.Namespace,
    image_path: Path,
    step: int,
) -> bool:
    path = Path(image_path)
    if not path.is_file():
        print(f"WARNING: final image not found for W&B upload: {path}")
        return True

    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"WARNING: failed to import wandb for final image upload ({type(exc).__name__}: {exc})")
        return True

    payload: Dict[str, object] = {
        "media/final_image": wandb.Image(str(path)),
        "media/final_image_path": str(path.resolve()),
    }
    return log_wandb_payload(
        run=run,
        args=args,
        payload=payload,
        step=int(step),
        honor_log_every=False,
    )


def resolve_vlm_device(raw: str) -> torch.device:
    token = str(raw or "").strip().lower()
    if token in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(token)


def _imagenet_weights_for_classifier(model_name: str):
    if models is None:
        raise RuntimeError(
            "torchvision is unavailable; cannot auto-resolve <class> placeholder "
            f"({_TORCHVISION_IMPORT_ERROR})"
        )
    token = str(model_name or "").strip().lower().replace("_", "-")
    if token == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if token == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if token == "resnet101":
        return models.ResNet101_Weights.DEFAULT
    if token in {"inception-v3", "inceptionv3"}:
        return models.Inception_V3_Weights.DEFAULT
    if token in {"vit-base", "vit-b-16", "vitb16"}:
        return models.ViT_B_16_Weights.DEFAULT
    raise ValueError(f"Unsupported model for ImageNet class lookup: {model_name}")


def infer_imagenet_class_name(model_name: str, label_idx: int) -> str:
    raw = infer_imagenet_category_name(model_name, label_idx)
    return raw.split(",")[0].strip() or raw


def infer_imagenet_category_name(model_name: str, label_idx: int) -> str:
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
                return raw

    weights = _imagenet_weights_for_classifier(model_name)
    categories = list(weights.meta.get("categories", []))
    if idx < 0 or idx >= len(categories):
        raise ValueError(f"label index out of range for {model_name}: {idx}")
    raw = str(categories[idx]).strip()
    if not raw:
        raise ValueError(f"Empty class name for label index: {idx}")
    return raw


def load_nips2017_category_names() -> List[str]:
    global _NIPS2017_CATEGORY_NAMES_CACHE
    if _NIPS2017_CATEGORY_NAMES_CACHE is not None:
        return list(_NIPS2017_CATEGORY_NAMES_CACHE)

    csv_path = Path(__file__).resolve().parent / "data" / "nips2017" / "categories.csv"
    names: List[str] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str(row.get("CategoryName", "")).strip()
                name = raw.split(",")[0].strip() or raw
                if name:
                    names.append(name)

    _NIPS2017_CATEGORY_NAMES_CACHE = names
    return list(_NIPS2017_CATEGORY_NAMES_CACHE)


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


def get_inversion_prompt(args: argparse.Namespace, editable_prompt: str) -> str:
    if args.inversion_prompt is not None and str(args.inversion_prompt).strip():
        return str(args.inversion_prompt).strip()
    if args.prompts:
        return str(args.prompts[0]).strip()
    return editable_prompt


def is_flux2_klein_model_path(model_path: object) -> bool:
    token = str(model_path or "").strip().lower()
    if not token:
        return False
    return ("flux.2" in token or "flux2" in token) and "klein" in token


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pick_generic_word(answer: str, extra_stopwords: Optional[set[str]] = None) -> Optional[str]:
    stopwords = STOPWORDS if not extra_stopwords else STOPWORDS.union(extra_stopwords)
    for token in answer.replace("-", " ").split():
        if len(token) >= 3 and token not in stopwords:
            return token
    return None


def normalize_scene(raw_answer: str, fallback: str) -> str:
    answer = normalize_text(raw_answer)
    if not answer:
        return fallback
    for key, value in [
        ("indoor", "indoor"),
        ("inside", "indoor"),
        ("outdoor", "outdoor"),
        ("beach", "beach"),
        ("forest", "forest"),
        ("mountain", "mountain"),
        ("street", "street"),
        ("city", "city"),
        ("park", "park"),
        ("field", "field"),
        ("desert", "desert"),
        ("snow", "snow"),
        ("night", "night"),
    ]:
        if key in answer:
            return value
    return pick_generic_word(answer) or fallback


def normalize_object(raw_answer: str, fallback: str) -> str:
    answer = normalize_text(raw_answer)
    if not answer:
        return fallback
    return pick_generic_word(
        answer,
        extra_stopwords={"main", "subject", "next", "nearby", "adjacent", "beside", "near"},
    ) or fallback


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


def normalize_slot_value(raw_answer: str, fallback: str, slot_kind: str) -> str:
    if slot_kind == "scene":
        return normalize_scene(raw_answer, fallback)
    return normalize_object(raw_answer, fallback)


def _strip_balanced_quotes(text: str) -> str:
    out = str(text or "").strip()
    if len(out) >= 2 and out[0] in {"'", '"', "`"} and out[-1] == out[0]:
        out = out[1:-1].strip()
    return out


def normalize_scene_vocab_word(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"^\s*(?:[-*]|\d+[\.\)])\s*", "", out)
    out = _strip_balanced_quotes(out)
    out = re.sub(r"\s+", " ", out).strip()
    out = out.strip(" ,.;:!?")
    return out


def format_forbidden_categories_instruction() -> str:
    category_names = load_nips2017_category_names()
    if len(category_names) == 0:
        return ""
    forbidden_list = "; ".join(f'"{name}"' for name in category_names)
    return (
        "- Do NOT use any ImageNet label name from this forbidden list in the candidates: "
        f"{forbidden_list}.\n"
    )


def slot_prompt_spec(slot_kind: str, max_words: int = 5) -> Dict[str, str]:
    candidate_length_text = "1 word" if int(max_words) == 1 else f"1~{int(max_words)} words"
    if slot_kind == "scene":
        return {
            "kind": "scene",
            "marker": "<scene>",
            "anchor_label": "Scene/topic anchor",
            "candidate_requirement": (
                f"- Each candidate must be a concise English phrase of {candidate_length_text}.\n"
            ),
            # scene/background/location word or
            "relevance_requirement": (
                "- Candidates should stay visually meaningful in relation to the current prompt context.\n"
            ),
        }
    return {
        "kind": "object",
        "marker": "<object>",
        "anchor_label": "Object/topic anchor",
        "candidate_requirement": (
            f"- Each candidate must be a concise English physical-object noun or phrase of {candidate_length_text}.\n"
        ),
        "relevance_requirement": (
            "- Candidates should stay visually meaningful in relation to the current prompt context.\n"
        ),
    }


def scene_vocab_strategy_specs() -> List[Dict[str, str]]:
    return [
        {
            "name": "background_shift",
            "title": "Background Shift",
            "description": "Commands that modify or replace the surrounding environment.",
        },
        {
            "name": "weather_atmosphere",
            "title": "Weather & Atmosphere",
            "description": "Commands that introduce weather, haze, lighting, or atmospheric effects.",
        },
        {
            "name": "texture_material",
            "title": "Texture & Material",
            "description": "Commands that alter surface properties, materials, or texture cues.",
        },
    ]


def _match_scene_vocab_strategy_name(raw: object) -> Optional[str]:
    token = normalize_text(str(raw or ""))
    if not token:
        return None
    alias_map = {
        "background": "background_shift",
        "background shift": "background_shift",
        "background_shift": "background_shift",
        "environment": "background_shift",
        "environment shift": "background_shift",
        "weather": "weather_atmosphere",
        "atmosphere": "weather_atmosphere",
        "weather atmosphere": "weather_atmosphere",
        "weather_atmosphere": "weather_atmosphere",
        "atmospheric": "weather_atmosphere",
        "texture": "texture_material",
        "material": "texture_material",
        "texture material": "texture_material",
        "texture_material": "texture_material",
        "surface": "texture_material",
    }
    if token in alias_map:
        return alias_map[token]
    return None


def resolve_scene_vocab_strategy_specs(raw: object = "all") -> List[Dict[str, str]]:
    specs = scene_vocab_strategy_specs()
    if raw is None:
        return list(specs)
    if isinstance(raw, (list, tuple, set)):
        tokens: List[str] = []
        for item in raw:
            tokens.extend(re.split(r"[,;|]+", str(item or "")))
    else:
        text = str(raw).strip()
        if not text:
            return []
        token = text.lower()
        if token in {"all", "default", "1", "true", "yes", "on"}:
            return list(specs)
        if token in {"none", "off", "0", "false", "no", "disable", "disabled"}:
            return []
        tokens = re.split(r"[,;|]+", text)

    selected: List[Dict[str, str]] = []
    selected_names = set()
    for raw_token in tokens:
        token = str(raw_token or "").strip()
        if not token:
            continue
        token_l = token.lower()
        if token_l in {"all", "default", "1", "true", "yes", "on"}:
            return list(specs)
        if token_l in {"none", "off", "0", "false", "no", "disable", "disabled"}:
            continue
        name = _match_scene_vocab_strategy_name(token)
        if name is None:
            valid = ", ".join(str(item["name"]) for item in specs)
            raise ValueError(
                f"--gcg_scene_vocab_enabled_strategies contains unknown strategy '{token}'. "
                f"Valid values are: all, none, {valid}."
            )
        if name in selected_names:
            continue
        selected_names.add(name)
        for spec in specs:
            if str(spec["name"]) == name:
                selected.append(dict(spec))
                break
    return selected


def format_scene_vocab_strategy_lines(
    strategy_specs: Sequence[Dict[str, str]],
    *,
    step_idx: int,
) -> str:
    initial_lines = {
        "background_shift": "Commands that entirely replace or subtly alter the surrounding environment (e.g., 'move the subject to a dusty warehouse', 'place the scene in a dense forest').",
        "weather_atmosphere": "Commands that introduce meteorological changes or atmospheric conditions affecting global lighting (e.g., 'add a thick morning fog', 'turn the weather into a torrential downpour').",
        "texture_material": "Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps using modifiers like 'coated in', 'wrapped in', or 'painted with' (e.g., 'coated in iridescent bioluminescent scales', 'wrapped in Vantablack velvet', 'painted with glitching neon chrome'). Do NOT alter the photo medium itself (no film grain, no scratched lenses) and do NOT change what the object fundamentally is.",
    }
    iterative_lines = {
        "background_shift": "Generate commands that modify the environment. If previous background shifts failed, try contrasting indoor/outdoor settings or changing the era/time period.",
        "weather_atmosphere": "Generate commands introducing atmospheric effects. Focus on conditions that logically conflict with the original lighting (e.g., adding heavy rain to a sunny scene).",
        "texture_material": "Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps using modifiers like 'coated in', 'wrapped in', or 'painted with' (e.g., 'coated in iridescent bioluminescent scales', 'wrapped in Vantablack velvet', 'painted with glitching neon chrome'). Do NOT alter the photo medium itself (no film grain, no scratched lenses) and do NOT change what the object fundamentally is.",
    }
    line_map = initial_lines if int(step_idx) == 0 else iterative_lines
    lines: List[str] = []
    for idx, spec in enumerate(strategy_specs, start=1):
        name = str(spec.get("name", "")).strip()
        title = str(spec.get("title", name)).strip()
        description = line_map.get(name, str(spec.get("description", "")).strip())
        lines.append(f"  * Strategy {int(idx)} ({title}): {description}\n")
    return "".join(lines)


def format_flux2_attribute_constraint(strategy_specs: Sequence[Dict[str, str]]) -> str:
    enabled_terms = set()
    for spec in strategy_specs:
        name = str(spec.get("name", "")).strip()
        if name == "background_shift":
            enabled_terms.add("background")
        elif name == "weather_atmosphere":
            enabled_terms.add("weather")
        elif name == "texture_material":
            enabled_terms.update(["color", "texture"])
    terms = [term for term in ("background", "color", "texture", "weather") if term in enabled_terms]
    if len(terms) == 0:
        return (
            "We constrain prompts to attribute-level edits. "
            "Do Not use Prompts that instantiate a new named entity, new category, or class-specific material. "
            "This preserves the attack mechanism while preventing explicit class-concept injection. \n"
        )
    if len(terms) == 1:
        term_list = terms[0]
    elif len(terms) == 2:
        term_list = " and ".join(terms)
    else:
        term_list = ", ".join(terms[:-1]) + f", and {terms[-1]}"
    return (
        f"We allow verb-driven transformations of {term_list}, but constrain them to attribute-level edits. "
        "Do Not use Prompts that instantiate a named object, location, scene category, or class-specific material. "
        "This preserves the attack mechanism while preventing explicit class-concept injection. \n"
    )


def extract_json_payload(raw: str) -> Optional[Any]:
    text = str(raw or "").strip()
    if not text:
        return None

    decoder = json.JSONDecoder()
    parsed_candidates: List[Any] = []
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        parsed_candidates.append(payload)

    def looks_like_candidate_payload(payload: Any) -> bool:
        if isinstance(payload, list):
            return True
        if isinstance(payload, dict):
            if isinstance(payload.get("strategies"), list):
                return True
            for key in ("candidates", "object_words", "objects", "scene_words", "words", "vocab"):
                if isinstance(payload.get(key), list):
                    return True
        return False

    def looks_like_strategy_group_payload(payload: Any) -> bool:
        return isinstance(payload, dict) and isinstance(payload.get("strategies"), list)

    for payload in reversed(parsed_candidates):
        if looks_like_strategy_group_payload(payload):
            return payload
    for payload in reversed(parsed_candidates):
        if looks_like_candidate_payload(payload):
            return payload
    for payload in reversed(parsed_candidates):
        if isinstance(payload, (dict, list)):
            return payload
    return None


def _normalize_feedback_note(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,.;:!?")


def _coerce_optional_bool(raw: object) -> Optional[bool]:
    if isinstance(raw, bool):
        return bool(raw)
    text = str(raw or "").strip().lower()
    if text in {"1", "true", "yes", "y", "natural", "plausible", "realistic"}:
        return True
    if text in {"0", "false", "no", "n", "unnatural", "implausible", "not natural"}:
        return False
    return None


def _feedback_suggests_unnatural(raw: object) -> bool:
    text = _normalize_feedback_note(raw).lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "unnatural",
            "not natural",
            "implausible",
            "not plausible",
            "unrealistic",
            "not realistic",
            "artificial",
            "fake",
            "deformed",
            "distorted",
            "warped",
            "artifact",
            "artifacts",
            "glitch",
        )
    )


def parse_naturalness_eval_answer(raw_answer: str) -> Tuple[Optional[bool], str]:
    payload = extract_json_payload(raw_answer)
    natural: Optional[bool] = None
    feedback = ""

    if isinstance(payload, dict):
        for key in ("natural", "is_natural", "looks_natural", "naturalness", "verdict", "answer"):
            if key in payload:
                natural = _coerce_optional_bool(payload.get(key))
                if natural is not None:
                    break
        for key in ("feedback", "reason", "explanation", "note", "why"):
            if key in payload:
                feedback = _normalize_feedback_note(payload.get(key))
                if feedback:
                    break

    text = _normalize_feedback_note(raw_answer)
    text_lower = text.lower()
    if natural is None:
        if _feedback_suggests_unnatural(feedback) or _feedback_suggests_unnatural(text):
            natural = False
        elif any(token in text_lower for token in ("looks natural", "is natural", "plausible", "realistic")):
            natural = True
        elif text_lower.startswith(("yes", "true")):
            natural = True
        elif text_lower.startswith(("no", "false")):
            natural = False

    if natural is False and not feedback:
        feedback = text or "The rendered image looks unnatural."
    return natural, feedback


def parse_scene_vocab_words(raw_answer: str, limit: int) -> List[str]:
    payload = extract_json_payload(raw_answer)
    raw_items: Optional[Sequence[object]] = None

    if isinstance(payload, dict):
        for key in ("candidates", "object_words", "objects", "scene_words", "words", "vocab"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break
    elif isinstance(payload, list):
        raw_items = payload

    if raw_items is None:
        # Fallback for non-JSON outputs: prioritize quoted tokens before broad splitting.
        quoted_tokens = re.findall(r"[\"']([^\"']+)[\"']", str(raw_answer or ""))
        if len(quoted_tokens) > 0:
            raw_items = quoted_tokens
        else:
            raw_items = re.split(r"[\n,;]+", str(raw_answer or ""))

    words: List[str] = []
    seen = set()
    parser_meta_tokens = {"json", "candidates", "object_words", "scene_words", "words", "vocab"}
    for item in raw_items:
        candidate = item
        if isinstance(item, dict):
            for key in ("word", "candidate", "text", "object", "object_word", "scene_word"):
                if key in item:
                    candidate = item[key]
                    break
        normalized = normalize_scene_vocab_word(str(candidate))
        if len(normalized) == 0:
            continue
        key = normalized.lower()
        if key in parser_meta_tokens:
            continue
        if key in seen:
            continue
        seen.add(key)
        words.append(normalized)
        if len(words) >= int(limit):
            break
    return words


def parse_scene_vocab_strategy_groups(
    raw_answer: str,
    *,
    prompts_per_strategy: int,
    strategy_specs: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, object]]:
    specs = list(strategy_specs) if strategy_specs is not None else scene_vocab_strategy_specs()
    if len(specs) == 0:
        return []
    spec_by_name = {str(item["name"]): item for item in specs}
    payload = extract_json_payload(raw_answer)
    groups_by_name: Dict[str, List[str]] = {str(item["name"]): [] for item in specs}

    def _normalize_candidates(raw_candidates: object) -> List[str]:
        if not isinstance(raw_candidates, list):
            return []
        words: List[str] = []
        seen = set()
        for item in raw_candidates:
            candidate = item
            if isinstance(item, dict):
                for key in ("word", "candidate", "text", "object", "object_word", "scene_word"):
                    if key in item:
                        candidate = item[key]
                        break
            normalized = normalize_scene_vocab_word(str(candidate))
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(normalized)
            if len(words) >= int(prompts_per_strategy):
                break
        return words

    if isinstance(payload, dict):
        raw_strategies = payload.get("strategies")
        if isinstance(raw_strategies, list):
            for item in raw_strategies:
                if not isinstance(item, dict):
                    continue
                strategy_name = _match_scene_vocab_strategy_name(
                    item.get("name", item.get("strategy", item.get("title")))
                )
                if strategy_name is None or strategy_name not in spec_by_name:
                    continue
                groups_by_name[strategy_name] = _normalize_candidates(item.get("candidates"))
        for spec in specs:
            strategy_name = str(spec["name"])
            if len(groups_by_name[strategy_name]) > 0:
                continue
            raw_candidates = payload.get(strategy_name)
            if raw_candidates is None:
                raw_candidates = payload.get(str(spec["title"]))
            if raw_candidates is None:
                raw_candidates = payload.get(str(spec["title"]).lower())
            if raw_candidates is None:
                continue
            groups_by_name[strategy_name] = _normalize_candidates(raw_candidates)

    groups: List[Dict[str, object]] = []
    for spec in specs:
        strategy_name = str(spec["name"])
        candidates = groups_by_name[strategy_name]
        groups.append(
            {
                "name": strategy_name,
                "title": str(spec["title"]),
                "candidates": list(candidates),
            }
        )

    if any(len(group["candidates"]) > 0 for group in groups):
        return groups

    flat_words = parse_scene_vocab_words(
        str(raw_answer or ""),
        limit=max(1, int(prompts_per_strategy)) * len(specs),
    )
    if len(flat_words) == 0:
        return []

    groups = []
    cursor = 0
    for spec in specs:
        next_cursor = cursor + max(1, int(prompts_per_strategy))
        groups.append(
            {
                "name": str(spec["name"]),
                "title": str(spec["title"]),
                "candidates": list(flat_words[cursor:next_cursor]),
            }
        )
        cursor = next_cursor
    return groups


def flatten_scene_vocab_strategy_groups(
    groups: Sequence[Dict[str, object]],
) -> List[Dict[str, str]]:
    normalized_groups: List[Dict[str, object]] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        strategy_name = str(item.get("name", "")).strip()
        strategy_title = str(item.get("title", strategy_name)).strip()
        raw_candidates = item.get("candidates")
        if not strategy_name or not isinstance(raw_candidates, list):
            continue
        normalized_groups.append(
            {
                "name": strategy_name,
                "title": strategy_title,
                "candidates": [str(candidate).strip() for candidate in raw_candidates if str(candidate).strip()],
            }
        )

    if len(normalized_groups) == 0:
        return []

    entries: List[Dict[str, str]] = []
    max_len = max(len(item["candidates"]) for item in normalized_groups)
    for candidate_idx in range(int(max_len)):
        for item in normalized_groups:
            candidates = item["candidates"]
            if candidate_idx >= len(candidates):
                continue
            entries.append(
                {
                    "strategy_name": str(item["name"]),
                    "strategy_title": str(item["title"]),
                    "word": str(candidates[candidate_idx]),
                }
            )
    return entries


def evaluate_attack_success_naturalness(
    *,
    image_path: Path,
    candidate_prompt: str,
    args: argparse.Namespace,
    is_source_vs_edited_comparison: bool = False,
    runtime_cache: Optional[PersistentVLMRuntimeCache] = None,
) -> Tuple[Optional[bool], str, str, Optional[str]]:
    class_name = str(getattr(args, "class_name", "") or "").strip()
    if class_name:
        target_class_name = class_name
        base_description = f"a photo of {class_name}"
    else:
        target_class_name = "the original main subject"
        base_description = "a photo of the original main subject"
    if bool(is_source_vs_edited_comparison):
        # eval_prompt = (
        #     "You are judging whether an edited image is a natural and plausible result of a requested change.\n"
        #     "The attached image is a side-by-side comparison.\n"
        #     "- Left: source image\n"
        #     "- Right: edited image\n"
        #     f'Base description of the source image: "{base_description}"\n'
        #     "Judge whether the edited image on the right looks like a natural and plausible edit of the source "
        #     "image on the left.\n"
        #     "Judge only visual naturalness, plausibility, editing artifacts, and whether the main subject remains "
        #     "consistent with the base description.\n"
        #     "If the edited result looks natural and plausible, set natural=true.\n"
        #     "If the edited result looks visually unnatural or implausible, set natural=false and write one "
        #     "short feedback sentence about the main visual problem.\n"
        #     'Return ONLY valid JSON with the schema {"natural": true, "feedback": ""}.'
        # )
        eval_prompt = (
            "You are judging whether an edited image is a natural and plausible result of a requested change.\n"
            "The attached image is a side-by-side comparison.\n"
            "- Left: source image\n"
            "- Right: edited image\n"
            f'Base description of the source image: "{base_description}"\n'
            f'The main_object is the target_class_name: "{target_class_name}".\n'
            "Your primary goal is to verify if the MAIN OBJECT from the source image is preserved and if the overall edit makes semantic sense.\n"
            "CRITICAL EVALUATION CRITERIA:\n"
            "1. Focus heavily on SEMANTIC PLAUSIBILITY and the structural integrity of the main subject.\n"
            "2. BE LENIENT with minor editing artifacts, slight lighting mismatches, or background changes. Do NOT reject the image for small imperfections as long as the main object is clearly recognizable and consistent with the base description.\n"
            "If the main object is maintained and the edit is logically possible in the real world (or the given context), set natural=true.\n"
            "If the edited result is completely structurally broken, the main object is destroyed, or it contains severe, immersion-breaking distortions, set natural=false and write one short feedback sentence about the main visual problem.\n"
            'Return ONLY valid JSON with the schema {"natural": true, "feedback": ""}.'
        )
    else:
        eval_prompt = (
            "You are judging whether an edited image is natural and human-plausible.\n"
            f'Base description of the image: "{base_description}"\n'
            f'The main_object is the target_class_name: "{target_class_name}".\n'
            "Judge only visual naturalness and plausibility of the attached edited image.\n"
            "If the edited result looks natural and plausible, set natural=true.\n"
            "If the edited result looks visually unnatural or implausible, set natural=false and write one "
            "short feedback sentence about the main visual problem.\n"
            'Return ONLY valid JSON with the schema {"natural": true, "feedback": ""}.'
        )
    raw_answer, error = query_vlm_text(
        image_path=Path(image_path),
        question=eval_prompt,
        vlm_backend=str(getattr(args, "gcg_scene_llm_backend", "gemma4")),
        vlm_model_id=str(getattr(args, "gcg_scene_llm_model_id", "google/gemma-4-e4b-it")),
        vlm_device_raw=str(getattr(args, "gcg_scene_llm_device", "auto")),
        max_new_tokens=min(512, max(64, int(getattr(args, "gcg_scene_llm_max_new_tokens", 512)))),
        enable_thinking=bool(
            getattr(
                args,
                "gcg_eval_naturalness_llm_thinking",
                getattr(args, "gcg_scene_llm_thinking", False),
            )
        ),
        do_sample=False,
        classifier_name=str(getattr(args, "classifier_name", "")),
        runtime_cache=runtime_cache,
    )
    if error is not None:
        return None, "", str(raw_answer or ""), error

    natural, feedback = parse_naturalness_eval_answer(str(raw_answer or ""))
    return natural, feedback, str(raw_answer or ""), None


def format_scene_vocab_feedback(
    *,
    feedback_entries: Sequence[Dict[str, object]],
    enabled: bool,
    limit: int,
) -> str:
    if not enabled or len(feedback_entries) == 0:
        return "No previous-step feedback is available yet."

    k = min(max(1, int(limit)), len(feedback_entries))
    ranked = sorted(
        feedback_entries,
        key=scene_feedback_sort_key,
        reverse=True,
    )[:k]

    lines = ["Previously tried candidates and best objectives:"]
    for item in ranked:
        word = str(item.get("scene_word", "")).strip()
        if len(word) == 0:
            continue
        objective = scene_feedback_objective(item)
        attempts = max(1, int(item.get("attempts", 1)))
        naturalness_note = _normalize_feedback_note(item.get("naturalness_feedback"))
        naturalness_is_natural = item.get("naturalness_is_natural")
        if objective is not None:
            line = f"- {word} | best_objective={objective + 10.0:.6f} | attempts={attempts}"
        else:
            line = f"- {word} | best_objective=unscored | attempts={attempts}"
        if naturalness_is_natural is False:
            line += " | naturalness=unnatural"
        if naturalness_note:
            line += f" | evaluator_feedback={naturalness_note}"
        lines.append(line)
    if len(lines) == 1:
        return "No previous-step feedback is available yet."
    return "\n".join(lines)


def merge_scene_vocab_feedback_history(
    *,
    existing_feedback: Sequence[Dict[str, object]],
    generated_words: Sequence[str],
    scored_candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    def _copy_naturalness_fields(dst: Dict[str, object], src: Dict[str, object]) -> None:
        naturalness = src.get("naturalness_is_natural")
        if naturalness is not None:
            dst["naturalness_is_natural"] = bool(naturalness)
        naturalness_feedback = _normalize_feedback_note(src.get("naturalness_feedback"))
        if naturalness_feedback:
            dst["naturalness_feedback"] = naturalness_feedback

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
        _copy_naturalness_fields(merged_by_word[scene_word], item)

    scored_objective_by_word: Dict[str, float] = {}
    scored_feedback_by_word: Dict[str, Dict[str, object]] = {}
    for item in scored_candidates:
        scene_word = normalize_scene_vocab_word(str(item.get("candidate_word", "")))
        if len(scene_word) == 0:
            continue
        objective = float(item.get("candidate_objective", float("-inf")))
        if not np.isfinite(objective):
            continue
        prev_objective = scored_objective_by_word.get(scene_word, float("-inf"))
        if objective > prev_objective:
            scored_objective_by_word[scene_word] = objective
        note = _normalize_feedback_note(item.get("naturalness_feedback"))
        naturalness = item.get("naturalness_is_natural")
        if naturalness is not None or note:
            scored_feedback = scored_feedback_by_word.setdefault(scene_word, {})
            if naturalness is not None:
                scored_feedback["naturalness_is_natural"] = bool(naturalness)
            if note:
                scored_feedback["naturalness_feedback"] = note

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
            if scene_word in scored_feedback_by_word:
                _copy_naturalness_fields(merged_by_word[scene_word], scored_feedback_by_word[scene_word])
            continue
        entry["attempts"] = max(1, int(entry.get("attempts", 1))) + 1
        prev_objective = scene_feedback_objective(entry)
        if objective is not None and (prev_objective is None or objective > prev_objective):
            entry["objective"] = objective
        if scene_word in scored_feedback_by_word:
            _copy_naturalness_fields(entry, scored_feedback_by_word[scene_word])

    for scene_word, objective in scored_objective_by_word.items():
        if scene_word in merged_by_word:
            if scene_word != "<CWOR>":
                if scene_word in scored_feedback_by_word:
                    _copy_naturalness_fields(merged_by_word[scene_word], scored_feedback_by_word[scene_word])
                continue
            prev_objective = scene_feedback_objective(merged_by_word[scene_word])
            if prev_objective is None or objective > prev_objective:
                merged_by_word[scene_word]["objective"] = objective
            if scene_word in scored_feedback_by_word:
                _copy_naturalness_fields(merged_by_word[scene_word], scored_feedback_by_word[scene_word])
            continue
        merged_by_word[scene_word] = {
            "scene_word": scene_word,
            "objective": objective,
            "attempts": 1,
        }
        if scene_word in scored_feedback_by_word:
            _copy_naturalness_fields(merged_by_word[scene_word], scored_feedback_by_word[scene_word])

    return list(merged_by_word.values())


def rank_scene_vocab_feedback_entries(
    *,
    feedback_entries: Sequence[Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    if len(feedback_entries) == 0:
        return []
    ranked = sorted(
        feedback_entries,
        key=scene_feedback_sort_key,
        reverse=True,
    )
    k = max(1, int(limit))
    return [dict(item) for item in ranked[:k]]


def scene_feedback_objective(item: Dict[str, object]) -> Optional[float]:
    raw = item.get("objective")
    if raw is None:
        return None
    try:
        objective = float(raw)
    except Exception:
        return None
    if not np.isfinite(objective):
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


def build_marked_prompt(current_prompt: str, current_word: str, occurrence: int, marker: str) -> str:
    text = str(current_prompt or "")
    word = str(current_word or "").strip()
    if word:
        matches = list(re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))
        if 0 <= int(occurrence) < len(matches):
            match = matches[int(occurrence)]
            return text[: match.start()] + marker + text[match.end() :]
    for slot_marker in ("<scene>", "{scene}", "<object>", "{object}"):
        if slot_marker in text:
            return text.replace(slot_marker, marker, 1)
    return text


def generate_scene_vocab_words(
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
    runtime_cache: Optional[PersistentVLMRuntimeCache] = None,
) -> Tuple[List[str], str, str, Optional[str]]:
    _maybe_disable_cudnn_sdpa_for_vim_small(
        getattr(args, "classifier_name", None),
        context="generate_scene_vocab_words",
    )
    setattr(args, "_scene_vocab_strategy_groups", [])
    setattr(args, "_scene_vocab_strategy_entries", [])
    slot_candidate_max_words = max(1, int(getattr(args, "gcg_slot_candidate_max_words", 5)))
    prompts_per_strategy = max(0, int(getattr(args, "gcg_scene_vocab_prompts_per_strategy", 0)))
    strategy_specs = resolve_scene_vocab_strategy_specs(
        getattr(args, "gcg_scene_vocab_enabled_strategies", "all")
    )
    strategy_prompt_model_enabled = bool(
        is_flux2_klein_model_path(getattr(args, "model_path", None))
        or bool(getattr(args, "bernini_edit_prompt_mode", False))
        or bool(getattr(args, "qwen_edit_prompt_mode", False))
        or bool(getattr(args, "qwen_strategy_and_enable", False))
    )
    flux2_strategy_prompt_mode = bool(
        prompts_per_strategy > 0
        and len(strategy_specs) > 0
        and strategy_prompt_model_enabled
    )
    all_strategy_specs = scene_vocab_strategy_specs()
    using_default_strategy_set = [str(item["name"]) for item in strategy_specs] == [
        str(item["name"]) for item in all_strategy_specs
    ]
    slot_spec = slot_prompt_spec(slot_kind, max_words=slot_candidate_max_words)
    slot_marker = slot_spec["marker"]
    marked_prompt = build_marked_prompt(current_prompt, current_word, int(args.gcg_occurrence), slot_marker)
    current_value = normalize_scene_vocab_word(str(current_word))
    if not current_value:
        current_value = normalize_scene_vocab_word(str(fallback_word))
    slot_topic = str(args.gcg_scene_vocab_topic or current_value).strip()
    if not slot_topic:
        slot_topic = fallback_word

    target_class_name = str(args.class_name or "").strip() or None
    class_requirement = (
        f'- Every candidate must be semantically related to the target class "<class>" (for this example: "{target_class_name}").\n'
        if target_class_name
        else '- Every candidate must be semantically related to the target class "<class>".\n'
    )
    has_naturalness_feedback = any(
        item.get("naturalness_is_natural") is False
        or bool(_normalize_feedback_note(item.get("naturalness_feedback")))
        for item in previous_feedback
    )
    feedback_block = format_scene_vocab_feedback(
        feedback_entries=previous_feedback,
        enabled=bool(args.gcg_scene_vocab_feedback),
        limit=int(args.gcg_scene_feedback_limit),
    )
    visual_feedback_block = (
        "Adversarial visual feedback:\n"
        "- The reference image is attached and represents the most recent generated result.\n"
    )
    cwor_instruction_block = ""
    if bool(getattr(args, "cwor_enable", False)) and not flux2_strategy_prompt_mode:
        cwor_instruction_block = (
            "- EXCEPTION (OVERRIDE): If '<CWOR>' is identified as the best candidate from the previous step, STRICTLY IGNORE Strategy 1. Instead, use Strategy 2 exclusively to generate exactly 2 novel words that are completely unrelated to '<CWOR>'.\n"
            "- Crucially, analyze ALL previous-step candidates (both successes and failures), EXCLUDING '<CWOR>':\n"
            "  * STRICTLY IGNORE '<CWOR>' during your analysis. Do not extract or learn any traits from it.\n"
            "  * Winners: Extract successful semantic/visual traits from the REMAINING best candidates to fuel Strategy 1 (unless the override applies).\n"
        )
    naturalness_feedback_instruction = (
        "- If a feedback entry says 'naturalness=unnatural', treat it as a warning and avoid reusing the artifact pattern described in its evaluator_feedback.\n"
        if has_naturalness_feedback
        else ""
    )

    # if step_idx == 0:
    #     strategy_instruction = (
    #         "- Since this is the initial step (no previous feedback), focus entirely on EXPLORATION.\n"
    #         "- Propose a diverse, wide-ranging set of out-of-domain, unconventional, or visually disruptive concepts to establish a baseline of vulnerabilities.\n"
    #     )
    # else:
    #     strategy_instruction = (
    #         "- Employ the following TWO distinct generation strategies to ensure both depth and breadth in the attack:\n"
    #         "  * Strategy 1 (Exploit Winners): You MUST use the best candidate from feedback as the base and create variant(s) by appending or prepending disruptive modifiers, conflicting materials, or compounding traits.\n"
    #         "  * Strategy 2 (Explore Novelty): Propose completely new, out-of-domain, or unconventional visual concepts to discover entirely new vulnerabilities.\n"
    #         f"{cwor_instruction_block}"
    #         "  * Losers: Identify the underlying characteristics of the REMAINING other candidates (e.g., concepts that the target model easily harmonized, ignored, or rendered safely). STRICTLY AVOID generating new candidates that share these ineffective traits.\n"
    #     )
    if len(strategy_specs) == 0:
        strategy_instruction = ""
    elif using_default_strategy_set:
        if step_idx == 0:
            strategy_instruction = (
                "- Since this is the initial step (no previous feedback), establish a baseline by equally distributing your candidates across THREE distinct attack vectors:\n"
                "  * Strategy 1 (Background Shift): Commands that entirely replace or subtly alter the surrounding environment (e.g., 'move the subject to a dusty warehouse', 'place the scene in a dense forest').\n"
                "  * Strategy 2 (Weather & Atmosphere): Commands that introduce meteorological changes or atmospheric conditions affecting global lighting (e.g., 'add a thick morning fog', 'turn the weather into a torrential downpour').\n"
                "  * Strategy 3 (Material & Color Reskinning): Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps. CRITICAL SYNTAX: You MUST start your candidate with a past participle (e.g., 'coated in', 'wrapped in', 'painted with', 'covered in'). Do NOT start with a noun or a base verb (e.g., NEVER output 'coat in dark bronze', strictly output 'coated in dark bronze'). Do NOT alter the photo medium itself and do NOT change what the object fundamentally is.\n"
                "- Propose diverse concepts within these three categories to discover which dimension the target model is most sensitive to.\n"
            )
        else:
            strategy_instruction = (
                "- Employ the following THREE distinct generation strategies to systematically test the model's vulnerabilities. Use the feedback to refine your attacks:\n"
                "  * Strategy 1 (Background Shift): Generate commands that modify the environment. If previous background shifts failed, try contrasting indoor/outdoor settings or changing the era/time period.\n"
                "  * Strategy 2 (Weather & Atmosphere): Generate commands introducing atmospheric effects. Focus on conditions that logically conflict with the original lighting (e.g., adding heavy rain to a sunny scene).\n"
                "  * Strategy 3 (Material & Color Reskinning): Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps. CRITICAL SYNTAX: You MUST start your candidate with a past participle (e.g., 'coated in', 'wrapped in', 'painted with', 'covered in'). Do NOT start with a noun or a base verb (e.g., NEVER output 'coat in dark bronze', strictly output 'coated in dark bronze'). Do NOT alter the photo medium itself and do NOT change what the object fundamentally is.\n"
                f"{cwor_instruction_block}"
                "  * Exploit Winners: Identify which of the three categories (Background, Weather, Texture) produced the best candidate in the feedback. Generate more compounded variants within that winning category.\n"
                "  * Avoid Losers: Identify the underlying characteristics of the remaining ineffective candidates. STRICTLY AVOID generating new commands that share these traits or fall into the weakest category.\n"
            )
    else:
        strategy_count = int(len(strategy_specs))
        strategy_lines = format_scene_vocab_strategy_lines(strategy_specs, step_idx=int(step_idx))
        if step_idx == 0:
            strategy_instruction = (
                f"- Since this is the initial step (no previous feedback), establish a baseline using the enabled {strategy_count} strategy bucket(s):\n"
                f"{strategy_lines}"
                "- Propose diverse concepts within the enabled categories to discover which dimension the target model is most sensitive to.\n"
            )
        else:
            strategy_instruction = (
                f"- Employ the enabled {strategy_count} generation strategy bucket(s) to test the model's vulnerabilities. Use the feedback to refine your attacks:\n"
                f"{strategy_lines}"
                f"{cwor_instruction_block}"
                "  * Exploit Winners: Identify which enabled category produced the best candidate in the feedback and generate stronger variants within it.\n"
                "  * Avoid Losers: Identify ineffective traits in the remaining candidates. STRICTLY AVOID generating new commands that share these traits or fall into disabled or weak categories.\n"
            )

    if strategy_prompt_model_enabled:
        if flux2_strategy_prompt_mode:
            if using_default_strategy_set:
                strategy_instruction = (
                    f"{strategy_instruction}"
                    "- Keep the candidates separated by strategy in the final JSON output.\n"
                    f"- For EACH strategy, generate exactly {int(prompts_per_strategy)} unique candidates.\n"
                    "- Keep the three strategy buckets balanced. Do not move candidates between buckets.\n"
                )
            else:
                strategy_instruction = (
                    f"{strategy_instruction}"
                    "- Keep the candidates separated by enabled strategy in the final JSON output.\n"
                    f"- For EACH enabled strategy, generate exactly {int(prompts_per_strategy)} unique candidates.\n"
                    "- Do not invent disabled strategy buckets or move candidates between buckets.\n"
                )
            total_candidate_count = int(prompts_per_strategy) * int(len(strategy_specs))
            if using_default_strategy_set:
                output_schema_line = (
                    "- Output ONLY valid JSON with the schema "
                    "{\"strategies\": ["
                    "{\"name\": \"background_shift\", \"candidates\": [\"...\", \"...\"]}, "
                    "{\"name\": \"weather_atmosphere\", \"candidates\": [\"...\", \"...\"]}, "
                    "{\"name\": \"texture_material\", \"candidates\": [\"...\", \"...\"]}"
                    "]}.\n"
                )
            else:
                schema_parts = [
                    "{\"name\": \""
                    + str(spec["name"])
                    + "\", \"candidates\": [\"...\", \"...\"]}"
                    for spec in strategy_specs
                ]
                output_schema_line = (
                    "- Output ONLY valid JSON with the schema "
                    + "{\"strategies\": ["
                    + ", ".join(schema_parts)
                    + "]}.\n"
                )
        else:
            total_candidate_count = int(args.gcg_scene_vocab_size)
            if len(strategy_specs) == 0:
                output_schema_line = (
                    "- Output ONLY valid JSON with the schema "
                    "{\"candidates\": [\"...\", \"...\"]}.\n"
                )
            else:
                output_schema_line = (
                    "- Output ONLY valid JSON with the schema "
                    "{\"candidates\": [\"turn the scene into...\", \"add a...\", ...]}.\n"
                )
        # make the gemma4 avoid to use specific-class
        # forbidden_categories_instruction = format_forbidden_categories_instruction()
        forbidden_categories_instruction = format_flux2_attribute_constraint(strategy_specs)
        strategy_free_prompt_mode = bool(len(strategy_specs) == 0)
        if strategy_free_prompt_mode:
            intro_text = (
                "You are a visual consistency tester. Your goal is to find 'Trojan' editing commands for a text-to-image model. "
                "These commands must be contextually plausible and benign, yet act as strong latent attractors that subtly disrupt the generated output."
            )
            context_block = f"Current best objective score: {float(best_objective) + 10.0:.6f}.\n\n"
            task_target = "the editable slot"
            action_verb_instruction = (
                "- CRITICAL: ALL candidates MUST be formulated as concise direct editing instructions starting with action verbs.\n"
            )
            candidate_style_instruction = ""
            disallowed_terms_instruction = ""
        else:
            intro_text = (
                "You are a visual consistency tester. Your goal is to find 'Trojan' editing commands for a text-to-image model. "
                "These commands must be so contextually perfect and benign that they seamlessly camouflage into the original scene, yet act as strong latent attractors that subtly disrupt the generated output."
            )
            context_block = (
                f"Step: {int(step_idx)}\n"
                f"Prompt template with slot marker: \"{marked_prompt}\"\n"
                f"Current {slot_marker} value: \"{current_value}\"\n"
                f"{slot_spec['anchor_label']}: \"{slot_topic}\"\n"
                f"Current best objective score: {float(best_objective) + 10.0:.6f}.\n\n"
            )
            task_target = slot_marker
            action_verb_instruction = (
                "- CRITICAL: ALL candidates MUST be formulated as direct editing instructions starting with action verbs, such as 'turn the scene into', 'add', 'change the lighting to', or 'apply a'.\n "
            )
            candidate_style_instruction = (
                "- Favor candidates that request subtle, plausible modifications (e.g., 'turn the scene into a misty morning', 'add soft shadows', 'change the material to brushed metal') rather than overt or destructive visual conflicts.\n"
            )
            disallowed_terms_instruction = (
                "- Do NOT use the word 'overlay' and words related to 'cybernetic', 'neon' and strong 'fog'.\n"
            )

        generation_prompt = (
            f"{intro_text}\n\n"
            f"{visual_feedback_block}"
            f"{context_block}"
            f"Feedback from previous steps:\n"
            f"{feedback_block}\n\n"
            "Task:\n"
            f"- Generate exactly {int(total_candidate_count)} unique candidate replacement instructions for {task_target}.\n"
            "- Prefer short, concrete, highly visual commands.\n"
            f"{action_verb_instruction}"
            "- Do not output standalone nouns, adjectives, or simple prepositions.\n"
            f"{candidate_style_instruction}"
            f"{disallowed_terms_instruction}"
            f"{forbidden_categories_instruction}"
            "- The instructions MUST make logical and semantic sense for the original context. They should appear as perfectly normal user requests to a human observer.\n"
            f"{naturalness_feedback_instruction}"
            f"{strategy_instruction}"
            "- Do NOT repeat any exact candidate already listed in the feedback block.\n"
            f"{class_requirement}"
            f"{slot_spec['candidate_requirement']}"
            f"{slot_spec['relevance_requirement']}"
            "- Do not write explanations, numbering, markdown, or sentences.\n"
            f"{output_schema_line}"
        )
    else:
        generation_prompt = (
            f"You are an AI red-teamer and iterative optimizer. Your goal is to maximize objective and to discover adversarial replacement words for the {slot_marker} slot in an image-edit prompt that cause the target image generation model to fail, produce unexpected artifacts, or significantly deviate from its intended behavior.\n\n"
            f"{visual_feedback_block}"
            f"Step: {int(step_idx)}\n"
            f"Prompt template with slot marker: \"{marked_prompt}\"\n"
            f"Current {slot_marker} value: \"{current_value}\"\n"
            f"{slot_spec['anchor_label']}: \"{slot_topic}\"\n"
            f"Current best objective score: {float(best_objective) + 10.0:.6f}.\n\n"
            f"Feedback from previous steps:\n"
            f"{feedback_block}\n\n"
            "Task:\n"
            f"- Generate exactly {int(args.gcg_scene_vocab_size)} unique candidate replacements for {slot_marker}.\n"
            f"{naturalness_feedback_instruction}"
            f"{strategy_instruction}"
            "- Do NOT repeat any exact candidate already listed in the feedback block.\n"
            f"{class_requirement}"
            "- The candidates do NOT need to make logical sense in the original context. In fact, unusual, contradictory, or visually disruptive objects/attributes often yield better adversarial results.\n"
            "- Focus on concrete, visualizable terms (objects, textures, lighting, strange combinations) rather than abstract concepts.\n"
            f"{slot_spec['candidate_requirement']}"
            f"{slot_spec['relevance_requirement']}"
            "- Do not write explanations, numbering, markdown, or sentences.\n"
            "- Output ONLY valid JSON with the schema {\"candidates\": [\"word1\", \"word2\", ...]}.\n"
        )

    raw_answer, error = query_vlm_text(
        image_path=reference_image_path,
        question=generation_prompt,
        vlm_backend=str(args.gcg_scene_llm_backend),
        vlm_model_id=str(args.gcg_scene_llm_model_id),
        vlm_device_raw=str(args.gcg_scene_llm_device),
        max_new_tokens=int(args.gcg_scene_llm_max_new_tokens),
        enable_thinking=bool(args.gcg_scene_llm_thinking),
        do_sample=bool(args.gcg_scene_llm_do_sample),
        classifier_name=str(getattr(args, "classifier_name", "")),
        runtime_cache=runtime_cache,
    )
    if error is not None:
        return [], str(raw_answer or ""), generation_prompt, error

    if flux2_strategy_prompt_mode:
        strategy_groups = parse_scene_vocab_strategy_groups(
            str(raw_answer or ""),
            prompts_per_strategy=int(prompts_per_strategy),
            strategy_specs=strategy_specs,
        )
        strategy_entries = flatten_scene_vocab_strategy_groups(strategy_groups)
        words = [str(item["word"]) for item in strategy_entries]
        if len(words) == 0:
            fallback_candidate = normalize_slot_value(str(raw_answer or ""), fallback_word, slot_kind)
            if fallback_candidate:
                first_spec = strategy_specs[0]
                strategy_groups = [
                    {
                        "name": str(first_spec["name"]),
                        "title": str(first_spec["title"]),
                        "candidates": [fallback_candidate],
                    },
                    *[
                        {
                            "name": str(spec["name"]),
                            "title": str(spec["title"]),
                            "candidates": [],
                        }
                        for spec in strategy_specs[1:]
                    ],
                ]
                strategy_entries = flatten_scene_vocab_strategy_groups(strategy_groups)
                words = [str(item["word"]) for item in strategy_entries]
        setattr(args, "_scene_vocab_strategy_groups", list(strategy_groups))
        setattr(args, "_scene_vocab_strategy_entries", list(strategy_entries))
    else:
        parsed_limit = max(1, int(args.gcg_scene_vocab_size))
        words = parse_scene_vocab_words(str(raw_answer or ""), limit=parsed_limit)
        if len(words) == 0:
            fallback_candidate = normalize_slot_value(str(raw_answer or ""), fallback_word, slot_kind)
            if fallback_candidate:
                words = [fallback_candidate]
    return words, str(raw_answer or ""), generation_prompt, None


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

    for marker in (
        "<scene>",
        "{scene}",
        "<object>",
        "{object}",
        "<CWOR>",
        "{CWOR}",
        "<cwor>",
        "{cwor}",
    ):
        if marker in text:
            updated = text.replace(marker, repl, 1)
            return updated, updated != text
    return text, False


def query_vlm_text(
    *,
    image_path: Path,
    question: str,
    vlm_backend: str,
    vlm_model_id: str,
    vlm_device_raw: str,
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    classifier_name: str = "",
    runtime_cache: Optional[PersistentVLMRuntimeCache] = None,
) -> Tuple[str, Optional[str]]:
    _maybe_disable_cudnn_sdpa_for_vim_small(classifier_name, context="query_vlm_text")
    if runtime_cache is not None:
        return runtime_cache.query(
            image_path=image_path,
            question=question,
            vlm_backend=vlm_backend,
            vlm_model_id=vlm_model_id,
            vlm_device_raw=vlm_device_raw,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
        )

    vlm_model = None
    vlm_processor = None
    ask_fn = None
    uses_pipeline_backend = False
    device = resolve_vlm_device(vlm_device_raw)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    try:
        backend = infer_vlm_backend(
            vlm_backend=str(vlm_backend),
            model_id=str(vlm_model_id),
            allow_blip=True,
        )
        vlm_model, vlm_processor, ask_fn, uses_pipeline_backend = load_vlm_runtime(
            backend=backend,
            model_id=str(vlm_model_id),
            vlm_dtype=dtype,
            vlm_device=device,
            allow_blip=True,
        )
        if not uses_pipeline_backend:
            vlm_model = vlm_model.to(device)
            vlm_model.eval()

        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            raw_answer = ask_fn(
                image=image,
                question=str(question),
                model=vlm_model,
                processor=vlm_processor,
                device=device,
                max_new_tokens=int(max_new_tokens),
                enable_thinking=bool(enable_thinking),
                do_sample=bool(do_sample),
            )
        return str(raw_answer or "").strip(), None
    except Exception as exc:
        return "", str(exc)
    finally:
        if vlm_model is not None and hasattr(vlm_model, "to") and not uses_pipeline_backend:
            try:
                vlm_model.to("cpu")
            except Exception:
                pass
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


def query_vlm_word(
    *,
    image_path: Path,
    args: argparse.Namespace,
    slot_kind: str,
    fallback_word: str,
    runtime_cache: Optional[PersistentVLMRuntimeCache] = None,
) -> Tuple[str, str, Optional[str]]:
    raw_answer, error = query_vlm_text(
        image_path=image_path,
        question=str(args.scene_vlm_question),
        vlm_backend=str(args.scene_vlm_backend),
        vlm_model_id=str(args.scene_vlm_model_id),
        vlm_device_raw=str(args.scene_vlm_device),
        max_new_tokens=int(args.scene_vlm_max_new_tokens),
        enable_thinking=bool(args.scene_vlm_thinking),
        do_sample=bool(args.scene_vlm_do_sample),
        classifier_name=str(getattr(args, "classifier_name", "")),
        runtime_cache=runtime_cache,
    )
    if error is not None:
        return fallback_word, "", error
    candidate_word = normalize_slot_value(raw_answer, fallback_word, slot_kind)
    return candidate_word, raw_answer, None


def run_render_subprocess(
    *,
    args: argparse.Namespace,
    prompts: Sequence[str],
    output_path: str,
    has_input_image: bool,
) -> None:
    gcg_script = Path(__file__).resolve().parent / "third_party" / "gcg_flux_edit.py"
    if not gcg_script.is_file():
        raise FileNotFoundError(f"missing render helper: {gcg_script}")
    prompt_list = [str(p).strip() for p in prompts if str(p).strip()]
    if len(prompt_list) == 0:
        raise ValueError("run_render_subprocess requires at least one prompt.")

    render_mode = "edit" if has_input_image else "infer"
    cmd = [
        sys.executable,
        str(gcg_script),
        "--mode",
        render_mode,
        "--classifier_mode",
        str(args.classifier_mode),
        "--model_path",
        str(args.model_path),
        "--hf_token",
        str(args.hf_token),
        "--output_path",
        str(output_path),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_inference_steps",
        str(int(args.num_inference_steps)),
        "--max_sequence_length",
        str(int(args.max_sequence_length)),
        "--guidance_scale",
        str(float(args.guidance_scale)),
        "--latent_nudging_scalar",
        str(float(args.latent_nudging_scalar)),
        "--prompts",
        *prompt_list,
    ]
    if getattr(args, "classifier_label", None) is not None:
        cmd.extend(["--classifier_label", str(int(args.classifier_label))])
    if has_input_image:
        cmd.extend(["--input_img_path", str(args.input_img_path)])
    if bool(args.cpu_offload):
        cmd.append("--cpu_offload")
    if bool(args.gradient_checkpointing):
        cmd.append("--gradient_checkpointing")

    subprocess.run(cmd, check=True)


def sanitize_filename_component(raw: object, *, fallback: str, max_len: int = 64) -> str:
    token = str(raw or "").strip().lower()
    token = token.encode("ascii", errors="ignore").decode("ascii")
    token = re.sub(r"[^a-z0-9._-]+", "_", token)
    token = token.strip("._-")
    if len(token) > int(max_len):
        token = token[: int(max_len)].rstrip("._-")
    return token or str(fallback)


def render_prompt_pair(
    *,
    args: argparse.Namespace,
    inversion_prompt: str,
    editable_prompt: str,
    output_path: Path,
    has_input_image: bool,
    render_session: Optional[PersistentFluxRenderSession] = None,
) -> None:
    use_source_first_strip = bool(
        is_flux2_klein_model_path(getattr(args, "model_path", None))
        and bool(has_input_image)
        and args.input_img_path is not None
        and str(args.input_img_path).strip()
    )
    if use_source_first_strip:
        with tempfile.TemporaryDirectory(prefix="flux2_prompt_pair_") as tmpdir:
            generated_only_path = Path(tmpdir) / "generated_only.png"
            if render_session is not None:
                render_session.render(
                    prompts=[str(editable_prompt)],
                    output_path=str(generated_only_path),
                )
            else:
                run_render_subprocess(
                    args=args,
                    prompts=[str(editable_prompt)],
                    output_path=str(generated_only_path),
                    has_input_image=has_input_image,
                )
            save_source_prefixed_generated_strip(
                source_image_path=Path(str(args.input_img_path)),
                generated_strip_path=generated_only_path,
                output_path=output_path,
                num_generated_prompts=1,
            )
        return

    prompts = [str(inversion_prompt), str(editable_prompt)]
    if render_session is not None:
        render_session.render(
            prompts=prompts,
            output_path=str(output_path),
        )
        return
    run_render_subprocess(
        args=args,
        prompts=prompts,
        output_path=str(output_path),
        has_input_image=has_input_image,
    )


def save_source_vs_generated_comparison(
    *,
    source_image_path: Path,
    generated_strip_path: Path,
    output_path: Path,
    num_prompts: int = 2,
    generated_index: int = 1,
) -> None:
    if int(num_prompts) < 1:
        raise ValueError("num_prompts must be >= 1")
    tiles = split_prompt_strip(generated_strip_path, num_prompts=int(num_prompts))
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles decoded from generated strip: {generated_strip_path}")

    idx = int(generated_index)
    if idx < 0:
        idx += len(tiles)
    if idx < 0 or idx >= len(tiles):
        raise IndexError(
            f"generated_index out of range: {generated_index} for tile_count={len(tiles)}"
        )

    generated_img = tiles[idx].convert("RGB")
    with Image.open(source_image_path) as source_img_opened:
        source_img = source_img_opened.convert("RGB")
    if source_img.size != generated_img.size:
        source_img = source_img.resize(generated_img.size, Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = Image.fromarray(np.hstack([np.array(source_img), np.array(generated_img)]))
    comparison.save(output_path)


def save_source_prefixed_generated_strip(
    *,
    source_image_path: Path,
    generated_strip_path: Path,
    output_path: Path,
    num_generated_prompts: int,
) -> None:
    if int(num_generated_prompts) < 1:
        raise ValueError("num_generated_prompts must be >= 1")
    tiles = split_prompt_strip(generated_strip_path, num_prompts=int(num_generated_prompts))
    if len(tiles) == 0:
        raise RuntimeError(f"No tiles decoded from generated strip: {generated_strip_path}")

    generated_tiles = [tile.convert("RGB") for tile in tiles]
    reference_size = generated_tiles[0].size
    with Image.open(source_image_path) as source_img_opened:
        source_img = source_img_opened.convert("RGB")
    if source_img.size != reference_size:
        source_img = source_img.resize(reference_size, Image.LANCZOS)

    strip_tiles = [np.array(source_img)] + [np.array(tile) for tile in generated_tiles]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.hstack(strip_tiles)).save(output_path)


def save_prompt_pair_from_generated_strip(
    *,
    generated_strip_path: Path,
    output_path: Path,
    num_prompts: int = 2,
    generated_index: int = 1,
) -> None:
    if int(num_prompts) < 2:
        raise ValueError("num_prompts must be >= 2")
    tiles = split_prompt_strip(generated_strip_path, num_prompts=int(num_prompts))
    if len(tiles) < 2:
        raise RuntimeError(f"Expected at least 2 tiles in strip: {generated_strip_path}")

    idx = int(generated_index)
    if idx < 0:
        idx += len(tiles)
    if idx <= 0 or idx >= len(tiles):
        raise IndexError(
            f"generated_index out of range: {generated_index} for tile_count={len(tiles)}"
        )

    inversion_img = tiles[0].convert("RGB")
    generated_img = tiles[idx].convert("RGB")
    if inversion_img.size != generated_img.size:
        inversion_img = inversion_img.resize(generated_img.size, Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair = Image.fromarray(np.hstack([np.array(inversion_img), np.array(generated_img)]))
    pair.save(output_path)


def save_selected_candidate_from_generated_strip(
    *,
    generated_strip_path: Path,
    output_path: Path,
    num_prompts: int = 2,
    generated_index: int = 1,
) -> None:
    if int(num_prompts) < 2:
        raise ValueError("num_prompts must be >= 2")
    tiles = split_prompt_strip(generated_strip_path, num_prompts=int(num_prompts))
    if len(tiles) < 2:
        raise RuntimeError(f"Expected at least 2 tiles in strip: {generated_strip_path}")

    idx = int(generated_index)
    if idx < 0:
        idx += len(tiles)
    if idx <= 0 or idx >= len(tiles):
        raise IndexError(
            f"generated_index out of range: {generated_index} for tile_count={len(tiles)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles[idx].convert("RGB").save(output_path)


def save_generated_tile_from_generated_strip(
    *,
    generated_strip_path: Path,
    output_path: Path,
    num_prompts: int = 1,
    generated_index: int = 0,
) -> None:
    if int(num_prompts) < 1:
        raise ValueError("num_prompts must be >= 1")
    tiles = split_prompt_strip(generated_strip_path, num_prompts=int(num_prompts))
    if len(tiles) < 1:
        raise RuntimeError(f"Expected at least 1 tile in strip: {generated_strip_path}")

    idx = int(generated_index)
    if idx < 0:
        idx += len(tiles)
    if idx < 0 or idx >= len(tiles):
        raise IndexError(
            f"generated_index out of range: {generated_index} for tile_count={len(tiles)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles[idx].convert("RGB").save(output_path)


def image_tensor_01_to_pil(image_01: torch.Tensor) -> Image.Image:
    tensor = image_01
    if tensor.ndim == 4:
        if int(tensor.shape[0]) < 1:
            raise ValueError("image tensor batch is empty")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) not in {1, 3}:
        raise ValueError(f"expected 1 or 3 channels, got shape={tuple(tensor.shape)}")
    chw = tensor.detach().to(device="cpu", dtype=torch.float32)
    if int(chw.shape[0]) == 1:
        chw = chw.repeat(3, 1, 1)
    chw = torch.clamp(chw, 0.0, 1.0)
    hwc = (chw.permute(1, 2, 0).contiguous().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


def save_blackbox_step_image(
    *,
    run_dir: Path,
    human_step: int,
    phase: str,
    replacement_word: str,
    inversion_prompt: str,
    candidate_prompt: str,
    args: argparse.Namespace,
    has_input_image: bool,
    render_session: Optional[PersistentFluxRenderSession] = None,
    existing_generated_strip_path: Optional[Path] = None,
    existing_generated_index: Optional[int] = None,
    existing_num_prompts: Optional[int] = None,
    precomputed_selected_image_path: Optional[Path] = None,
    precomputed_selected_image: Optional[Image.Image] = None,
) -> str:
    def _resolve_cwor_fallback_prompt(
        *,
        raw_candidate_prompt: str,
        cwor_snapshot: Optional[Dict[str, object]],
    ) -> str:
        resolved_prompt = str(raw_candidate_prompt or "").strip()
        if not resolved_prompt:
            return resolved_prompt
        if not any(token in resolved_prompt.lower() for token in ("<cwor>", "{cwor}")):
            return resolved_prompt
        if not isinstance(cwor_snapshot, dict):
            return resolved_prompt

        anchor_reference_prompt = ""
        fallback_reference_prompt = ""
        strategy_components = cwor_snapshot.get("strategy_components")
        if isinstance(strategy_components, list):
            for component in strategy_components:
                if not isinstance(component, dict):
                    continue
                reference_prompt = str(component.get("reference_prompt", "") or "").strip()
                if not reference_prompt:
                    continue
                if any(token in reference_prompt.lower() for token in ("<cwor>", "{cwor}")):
                    continue
                if not fallback_reference_prompt:
                    fallback_reference_prompt = reference_prompt
                if bool(component.get("merge_anchor", False)):
                    anchor_reference_prompt = reference_prompt
                    break

        replacement_prompt = anchor_reference_prompt or fallback_reference_prompt
        if not replacement_prompt:
            reference_prompt = str(cwor_snapshot.get("reference_prompt", "") or "").strip()
            if reference_prompt and not any(
                token in reference_prompt.lower() for token in ("<cwor>", "{cwor}")
            ):
                replacement_prompt = reference_prompt
        if not replacement_prompt:
            return resolved_prompt

        for placeholder in ("<CWOR>", "<cwor>", "{CWOR}", "{cwor}"):
            resolved_prompt = resolved_prompt.replace(placeholder, replacement_prompt)
        return resolved_prompt

    step_images_dir = run_dir / "step_images"
    step_images_dir.mkdir(parents=True, exist_ok=True)

    phase_token = sanitize_filename_component(phase, fallback="step", max_len=24)
    word_token = sanitize_filename_component(replacement_word, fallback="word", max_len=80)
    image_path = step_images_dir / f"step_{int(human_step):03d}_{phase_token}_{word_token}.png"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    selected_image_path = images_dir / f"step_{int(human_step):03d}_{phase_token}_{word_token}_selected.png"
    reusable_strip_path: Optional[Path] = None
    reusable_generated_index: Optional[int] = None
    reusable_num_prompts: Optional[int] = None
    if (
        existing_generated_strip_path is not None
        and existing_generated_strip_path.is_file()
        and existing_generated_index is not None
        and existing_num_prompts is not None
        and int(existing_num_prompts) >= 2
        and 0 <= int(existing_generated_index) < int(existing_num_prompts)
    ):
        reusable_strip_path = existing_generated_strip_path
        reusable_generated_index = int(existing_generated_index)
        reusable_num_prompts = int(existing_num_prompts)

    use_source_comparison = bool(
        args.classifier_mode == "black-box"
        and has_input_image
        and args.input_img_path is not None
        and str(args.input_img_path).strip()
    )
    generated_img: Optional[Image.Image] = None
    if precomputed_selected_image is not None:
        generated_img = precomputed_selected_image.convert("RGB")
    elif precomputed_selected_image_path is not None and Path(precomputed_selected_image_path).is_file():
        with Image.open(precomputed_selected_image_path) as selected_opened:
            generated_img = selected_opened.convert("RGB")

    if generated_img is not None:
        selected_image_path.parent.mkdir(parents=True, exist_ok=True)
        generated_img.save(selected_image_path)

        if use_source_comparison:
            source_image_path = Path(str(args.input_img_path))
            with Image.open(source_image_path) as source_img_opened:
                source_img = source_img_opened.convert("RGB")
            if source_img.size != generated_img.size:
                source_img = source_img.resize(generated_img.size, Image.LANCZOS)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            comparison = Image.fromarray(np.hstack([np.array(source_img), np.array(generated_img)]))
            comparison.save(image_path)
        else:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            generated_img.save(image_path)
        return relpath_from_run_dir(run_dir, image_path)

    is_cwor_placeholder = str(replacement_word).strip().lower() == "<cwor>"
    cwor_snapshot: Optional[Dict[str, object]] = None
    fallback_candidate_prompt = str(candidate_prompt)
    if (
        is_cwor_placeholder
        and reusable_strip_path is None
        and render_session is not None
    ):
        try:
            cwor_snapshot = render_session.get_cwor_embedding_snapshot()
        except Exception:
            cwor_snapshot = None
        fallback_candidate_prompt = _resolve_cwor_fallback_prompt(
            raw_candidate_prompt=str(candidate_prompt),
            cwor_snapshot=cwor_snapshot,
        )
        if isinstance(cwor_snapshot, dict):
            raw_t5_embeds = cwor_snapshot.get("t5_prompt_embeds")
            raw_clip_embeds = cwor_snapshot.get("clip_pooled_prompt_embeds")
            if torch.is_tensor(raw_t5_embeds) and torch.is_tensor(raw_clip_embeds):
                try:
                    ref_prompt = str(cwor_snapshot.get("reference_prompt", "")).strip()
                    if len(ref_prompt) == 0:
                        ref_prompt = str(candidate_prompt)
                    if len(ref_prompt) == 0:
                        ref_prompt = str(inversion_prompt)
                    ref_t5, ref_clip = render_session._encode_prompt_embeds(ref_prompt)
                    prompt_embeds = raw_t5_embeds.to(device=ref_t5.device, dtype=ref_t5.dtype)
                    pooled_prompt_embeds = raw_clip_embeds.to(
                        device=ref_clip.device, dtype=ref_clip.dtype
                    )
                    cwor_image_01 = render_session._render_candidate_tensor_from_embeds(
                        inversion_prompt=str(inversion_prompt),
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                    )
                    generated_img = image_tensor_01_to_pil(cwor_image_01).convert("RGB")
                    selected_image_path.parent.mkdir(parents=True, exist_ok=True)
                    generated_img.save(selected_image_path)

                    if use_source_comparison:
                        source_image_path = Path(str(args.input_img_path))
                        with Image.open(source_image_path) as source_img_opened:
                            source_img = source_img_opened.convert("RGB")
                        if source_img.size != generated_img.size:
                            source_img = source_img.resize(generated_img.size, Image.LANCZOS)
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        comparison = Image.fromarray(np.hstack([np.array(source_img), np.array(generated_img)]))
                        comparison.save(image_path)
                    else:
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        generated_img.save(image_path)
                    return relpath_from_run_dir(run_dir, image_path)
                except Exception as exc:
                    print(
                        "WARNING: failed to save CWOR step image from embedding snapshot; "
                        f"falling back to prompt rerender ({type(exc).__name__}: {exc})"
                    )

    if use_source_comparison:
        source_image_path = Path(str(args.input_img_path))
        if (
            reusable_strip_path is not None
            and reusable_generated_index is not None
            and reusable_num_prompts is not None
        ):
            try:
                save_source_vs_generated_comparison(
                    source_image_path=source_image_path,
                    generated_strip_path=reusable_strip_path,
                    output_path=image_path,
                    num_prompts=reusable_num_prompts,
                    generated_index=reusable_generated_index,
                )
                try:
                    save_selected_candidate_from_generated_strip(
                        generated_strip_path=reusable_strip_path,
                        output_path=selected_image_path,
                        num_prompts=reusable_num_prompts,
                        generated_index=reusable_generated_index,
                    )
                except Exception as exc:
                    try:
                        save_generated_tile_from_generated_strip(
                            generated_strip_path=reusable_strip_path,
                            output_path=selected_image_path,
                            num_prompts=reusable_num_prompts,
                            generated_index=reusable_generated_index,
                        )
                    except Exception as fallback_exc:
                        print(
                            "WARNING: failed to save selected candidate image from reused strip; "
                            "continuing without selected image "
                            f"({type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc})"
                        )
                return relpath_from_run_dir(run_dir, image_path)
            except Exception as exc:
                print(
                    "WARNING: failed to reuse candidate strip for source-vs-step comparison; "
                    f"falling back to rerender ({type(exc).__name__}: {exc})"
                )
        with tempfile.TemporaryDirectory(prefix="vlm_attack_step_") as tmpdir:
            generated_strip_path = Path(tmpdir) / "generated_strip.png"
            render_prompt_pair(
                args=args,
                inversion_prompt=inversion_prompt,
                editable_prompt=fallback_candidate_prompt,
                output_path=generated_strip_path,
                has_input_image=has_input_image,
                render_session=render_session,
            )
            try:
                save_source_vs_generated_comparison(
                    source_image_path=source_image_path,
                    generated_strip_path=generated_strip_path,
                    output_path=image_path,
                    num_prompts=2,
                    generated_index=1,
                )
            except Exception as exc:
                print(
                    "WARNING: failed to compose source-vs-step comparison; "
                    f"falling back to generated strip ({type(exc).__name__}: {exc})"
                )
                with Image.open(generated_strip_path) as fallback_img:
                    fallback_img.convert("RGB").save(image_path)
            try:
                save_selected_candidate_from_generated_strip(
                    generated_strip_path=generated_strip_path,
                    output_path=selected_image_path,
                    num_prompts=2,
                    generated_index=1,
                )
            except Exception as exc:
                try:
                    save_generated_tile_from_generated_strip(
                        generated_strip_path=generated_strip_path,
                        output_path=selected_image_path,
                        num_prompts=2,
                        generated_index=1,
                    )
                except Exception as fallback_exc:
                    print(
                        "WARNING: failed to save selected candidate image from rerendered strip; "
                        "continuing without selected image "
                        f"({type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc})"
                    )
    else:
        if (
            reusable_strip_path is not None
            and reusable_generated_index is not None
            and reusable_num_prompts is not None
        ):
            try:
                save_prompt_pair_from_generated_strip(
                    generated_strip_path=reusable_strip_path,
                    output_path=image_path,
                    num_prompts=reusable_num_prompts,
                    generated_index=reusable_generated_index,
                )
                try:
                    save_selected_candidate_from_generated_strip(
                        generated_strip_path=reusable_strip_path,
                        output_path=selected_image_path,
                        num_prompts=reusable_num_prompts,
                        generated_index=reusable_generated_index,
                    )
                except Exception as exc:
                    try:
                        save_generated_tile_from_generated_strip(
                            generated_strip_path=reusable_strip_path,
                            output_path=selected_image_path,
                            num_prompts=reusable_num_prompts,
                            generated_index=reusable_generated_index,
                        )
                    except Exception as fallback_exc:
                        print(
                            "WARNING: failed to save selected candidate image from reused strip; "
                            "continuing without selected image "
                            f"({type(exc).__name__}: {exc}; fallback {type(fallback_exc).__name__}: {fallback_exc})"
                        )
                return relpath_from_run_dir(run_dir, image_path)
            except Exception as exc:
                print(
                    "WARNING: failed to reuse candidate strip for step image; "
                    f"falling back to rerender ({type(exc).__name__}: {exc})"
                )
        render_prompt_pair(
            args=args,
            inversion_prompt=inversion_prompt,
            editable_prompt=fallback_candidate_prompt,
            output_path=image_path,
            has_input_image=has_input_image,
            render_session=render_session,
        )
        try:
            save_selected_candidate_from_generated_strip(
                generated_strip_path=image_path,
                output_path=selected_image_path,
                num_prompts=2,
                generated_index=1,
            )
        except Exception as exc:
            print(
                "WARNING: failed to extract selected candidate image from step pair; "
                f"continuing without selected image ({type(exc).__name__}: {exc})"
            )
    return relpath_from_run_dir(run_dir, image_path)


def image_to_tensor_01(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32))
    # HWC -> CHW, [0,1]
    arr = arr.permute(2, 0, 1).contiguous() / 255.0
    return arr.unsqueeze(0)


def classifier_input_image(image: Image.Image, classifier) -> Image.Image:
    """Materialize the RGB image tensor actually presented to the victim classifier."""

    input_size = 224
    for attr_name in ("input_res", "input_size"):
        try:
            value = int(getattr(classifier, attr_name, 0))
        except Exception:
            continue
        if value > 0:
            input_size = value
            break
    tensor = image_to_tensor_01(image)
    if tuple(tensor.shape[-2:]) != (input_size, input_size):
        tensor = F.interpolate(
            tensor,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return image_tensor_01_to_pil(tensor).convert("RGB")


def split_prompt_strip(image_path: Path, num_prompts: int) -> List[Image.Image]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        if int(num_prompts) <= 1:
            return [rgb.copy()]
        base = width // int(num_prompts)
        tiles: List[Image.Image] = []
        for idx in range(int(num_prompts)):
            left = int(idx * base)
            right = int(width) if idx == int(num_prompts) - 1 else int((idx + 1) * base)
            tiles.append(rgb.crop((left, 0, right, height)))
        return tiles


def evaluate_prompt_candidates(
    *,
    args: argparse.Namespace,
    classifier: TorchvisionClassifier,
    inversion_prompt: str,
    cwor_base_prompt: Optional[str] = None,
    candidate_words: Sequence[str],
    candidate_prompts: Sequence[str],
    has_input_image: bool,
    cwor_base_confidence: Optional[float] = None,
    cwor_feedback_candidates: Optional[Sequence[Dict[str, object]]] = None,
    cwor_require_feedback: bool = False,
    cwor_aggregate_feedback: bool = False,
    render_session: Optional[PersistentFluxRenderSession] = None,
    mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    render_output_path: Optional[Path] = None,
    capture_classifier_tile_image: bool = False,
    cwor_only: bool = False,
    enable_cwor: Optional[bool] = None,
    cwor_step_index: int = 1,
) -> Tuple[List[Dict[str, object]], Optional[str], Optional[Tuple[Path, int]]]:
    candidate_words = [str(item) for item in candidate_words]
    candidate_prompts = [str(item) for item in candidate_prompts]

    skip_initial_render = bool(getattr(args, "gcg_skip_initial_render", False))
    include_inversion_prompt = not (
        skip_initial_render
        and str(getattr(args, "classifier_mode", "")).strip().lower() == "black-box"
        and bool(has_input_image)
    )
    candidate_base_index = 1 if include_inversion_prompt else 0
    cwor_enabled = bool(getattr(args, "cwor_enable", False)) if enable_cwor is None else bool(enable_cwor)
    cwor_mode = str(getattr(args, "cwor_mode", "untargeted") or "untargeted").strip().lower()
    if cwor_mode not in {"untargeted", "target"}:
        cwor_mode = "untargeted"
    cwor_embed_inject_mode = normalize_cwor_embed_inject_mode(
        getattr(args, "cwor_embed_inject_mode", "both")
    )
    cwor_feedback_merge_mode = normalize_cwor_feedback_merge_mode(
        getattr(args, "cwor_feedback_merge_mode", "accumulate")
    )
    cwor_target_label = getattr(args, "cwor_target_label", None)
    if cwor_target_label is not None:
        try:
            cwor_target_label = int(cwor_target_label)
        except Exception:
            cwor_target_label = None
    feedback_candidates: Optional[List[Dict[str, object]]] = None
    if cwor_feedback_candidates is not None:
        feedback_candidates = [dict(item) for item in cwor_feedback_candidates]

    def _append_error(base_error: Optional[str], extra_error: Optional[str]) -> Optional[str]:
        if not extra_error:
            return base_error
        if not base_error:
            return extra_error
        return f"{base_error} | {extra_error}"

    def _classifier_input_size() -> int:
        for attr_name in ("input_res", "input_size"):
            raw_size = getattr(classifier, attr_name, None)
            try:
                size = int(raw_size)
            except Exception:
                continue
            if size > 0:
                return size
        return 224

    def _capture_classifier_tile(tile: Image.Image) -> Optional[Image.Image]:
        if not bool(capture_classifier_tile_image):
            return None
        try:
            size = int(_classifier_input_size())
            image_01 = image_to_tensor_01(tile)
            if int(image_01.shape[-2]) != size or int(image_01.shape[-1]) != size:
                image_01 = F.interpolate(
                    image_01,
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            return image_tensor_01_to_pil(image_01).copy()
        except Exception:
            return None

    def _evaluate_from_output(
        tmp_output: Path,
        *,
        prompts: Sequence[str],
    ) -> Tuple[List[Dict[str, object]], Optional[str]]:
        if not tmp_output.is_file():
            return [], "render_output_missing"

        tiles = split_prompt_strip(tmp_output, num_prompts=len(prompts))
        if len(tiles) != len(prompts):
            return [], f"tile_count_mismatch:{len(tiles)}!={len(prompts)}"

        results: List[Dict[str, object]] = []
        for idx in range(len(candidate_prompts)):
            tile = tiles[idx + candidate_base_index]
            image_01 = image_to_tensor_01(tile).to(device=str(args.device))
            with torch.no_grad():
                objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
            classifier_image = _capture_classifier_tile(tile)
            result = {
                "candidate_word": str(candidate_words[idx]),
                "candidate_prompt": str(candidate_prompts[idx]),
                "candidate_objective": float(objective),
                "pred_idx": stats.get("pred_idx"),
                "pred_conf": stats.get("pred_conf"),
                "pred_logit": stats.get("pred_logit"),
                "target_conf": stats.get("target_conf"),
                "target_logit": stats.get("target_logit"),
                "target_label_conf": stats.get("target_label_conf"),
                "target_label_logit": stats.get("target_label_logit"),
                "ce": stats.get("ce"),
                "candidate_variant": "prompt",
                "candidate_strip_index": int(idx + candidate_base_index),
                "candidate_selected_image": tile.copy(),
                "candidate_selected_image_width": int(tile.size[0]),
                "candidate_selected_image_height": int(tile.size[1]),
                "candidate_selected_image_source": "raw_tile",
            }
            if classifier_image is not None:
                result["candidate_classifier_image"] = classifier_image
                result["candidate_classifier_image_size"] = int(_classifier_input_size())
            results.append(
                result
            )
        return results, None

    def _augment_with_cwor(
        *,
        base_results: List[Dict[str, object]],
        base_error: Optional[str],
    ) -> Tuple[List[Dict[str, object]], Optional[str]]:
        if not cwor_enabled:
            return base_results, base_error
        if render_session is None:
            return base_results, base_error
        if cwor_base_confidence is None:
            return base_results, base_error
        if bool(cwor_require_feedback):
            if feedback_candidates is None or len(feedback_candidates) == 0:
                return base_results, base_error
            cwor_seed_candidates: Sequence[Dict[str, object]] = feedback_candidates
        else:
            if feedback_candidates is not None and len(feedback_candidates) > 0:
                cwor_seed_candidates = feedback_candidates
            elif len(base_results) > 0:
                cwor_seed_candidates = base_results
            else:
                return base_results, base_error
        cwor_results, cwor_error = render_session.evaluate_cwor_candidates(
            inversion_prompt=str(inversion_prompt),
            cwor_base_prompt=None if cwor_base_prompt is None else str(cwor_base_prompt),
            base_candidates=cwor_seed_candidates,
            classifier=classifier,
            base_confidence=cwor_base_confidence,
            cwor_mode=cwor_mode,
            cwor_embed_inject_mode=cwor_embed_inject_mode,
            cwor_target_label=cwor_target_label,
            aggregate_feedback=bool(cwor_aggregate_feedback),
            cwor_feedback_merge_mode=cwor_feedback_merge_mode,
            cwor_step_index=int(cwor_step_index),
        )
        merged_results = list(base_results)
        if len(cwor_results) > 0:
            merged_results.extend(cwor_results)
        if cwor_error and len(cwor_results) == 0:
            return merged_results, base_error
        return merged_results, _append_error(base_error, cwor_error)

    if bool(cwor_only):
        results, error = _augment_with_cwor(base_results=[], base_error=None)
        return results, error, None

    if len(candidate_prompts) == 0:
        return [], "no_candidates", None

    prompts: List[str] = []
    if include_inversion_prompt:
        prompts.append(str(inversion_prompt))
    prompts.extend([str(p) for p in candidate_prompts])
    use_source_first_tile = bool(
        include_inversion_prompt
        and is_flux2_klein_model_path(getattr(args, "model_path", None))
        and bool(has_input_image)
        and args.input_img_path is not None
        and str(args.input_img_path).strip()
    )

    def _render_prompts_to_output(
        tmp_output: Path,
        *,
        prompts_for_render: Sequence[str],
    ) -> None:
        if render_session is not None:
            render_session.render(
                prompts=prompts_for_render,
                output_path=str(tmp_output),
                mixed_initial_edit_cache=mixed_initial_edit_cache,
            )
            return
        run_render_subprocess(
            args=args,
            prompts=prompts_for_render,
            output_path=str(tmp_output),
            has_input_image=has_input_image,
        )

    if render_output_path is not None:
        tmp_output = Path(render_output_path)
        tmp_output.parent.mkdir(parents=True, exist_ok=True)
        if use_source_first_tile:
            with tempfile.TemporaryDirectory(prefix="flux2_candidate_strip_") as tmpdir:
                generated_only_path = Path(tmpdir) / "generated_only.png"
                _render_prompts_to_output(
                    generated_only_path,
                    prompts_for_render=candidate_prompts,
                )
                save_source_prefixed_generated_strip(
                    source_image_path=Path(str(args.input_img_path)),
                    generated_strip_path=generated_only_path,
                    output_path=tmp_output,
                    num_generated_prompts=len(candidate_prompts),
                )
        else:
            _render_prompts_to_output(tmp_output, prompts_for_render=prompts)
        results, base_error = _evaluate_from_output(tmp_output, prompts=prompts)
        strip_meta = (tmp_output, len(prompts)) if base_error is None else None
        error = base_error
        if base_error is None:
            results, error = _augment_with_cwor(base_results=results, base_error=base_error)
        return results, error, strip_meta

    with tempfile.TemporaryDirectory(prefix="vlm_attack_eval_") as tmpdir:
        tmp_output = Path(tmpdir) / "candidate_strip.png"
        if use_source_first_tile:
            generated_only_path = Path(tmpdir) / "generated_only.png"
            _render_prompts_to_output(
                generated_only_path,
                prompts_for_render=candidate_prompts,
            )
            save_source_prefixed_generated_strip(
                source_image_path=Path(str(args.input_img_path)),
                generated_strip_path=generated_only_path,
                output_path=tmp_output,
                num_generated_prompts=len(candidate_prompts),
            )
        else:
            _render_prompts_to_output(tmp_output, prompts_for_render=prompts)
        results, error = _evaluate_from_output(tmp_output, prompts=prompts)
        if error is None:
            results, error = _augment_with_cwor(base_results=results, base_error=error)
        return results, error, None


def evaluate_attack_candidates(
    *,
    args: argparse.Namespace,
    classifier: TorchvisionClassifier,
    candidate_words: Sequence[str],
    candidate_prompts: Sequence[str],
    has_input_image: bool,
    render_session: Optional[PersistentFluxRenderSession] = None,
    mixed_initial_edit_cache: Optional[Dict[str, object]] = None,
    render_output_path: Optional[Path] = None,
    capture_classifier_tile_image: bool = True,
    **_unused_legacy_options,
) -> Tuple[List[Dict[str, object]], Optional[str], Optional[Tuple[Path, int]]]:
    """Render and score edit candidates only; no inversion or source-image tile is rendered."""

    del _unused_legacy_options
    words = [str(item) for item in candidate_words]
    prompts = [str(item) for item in candidate_prompts]
    if len(prompts) == 0:
        return [], "no_candidates", None
    if len(words) != len(prompts):
        return [], f"candidate_count_mismatch:{len(words)}!={len(prompts)}", None

    def _render(output: Path) -> None:
        if render_session is not None:
            render_session.render(
                prompts=prompts,
                output_path=str(output),
                mixed_initial_edit_cache=mixed_initial_edit_cache,
            )
            return
        run_render_subprocess(
            args=args,
            prompts=prompts,
            output_path=str(output),
            has_input_image=has_input_image,
        )

    prompt_query_count = 0
    if render_session is not None:
        setattr(render_session, "last_prompt_query_count", 0)

    def _score(output: Path) -> Tuple[List[Dict[str, object]], Optional[str]]:
        nonlocal prompt_query_count
        if not output.is_file():
            return [], "render_output_missing"
        tiles = split_prompt_strip(output, num_prompts=len(prompts))
        if len(tiles) != len(prompts):
            return [], f"tile_count_mismatch:{len(tiles)}!={len(prompts)}"

        results: List[Dict[str, object]] = []
        for index, tile in enumerate(tiles):
            try:
                image_01 = image_to_tensor_01(tile).to(device=str(args.device))
                prompt_query_count += 1
                if render_session is not None:
                    setattr(
                        render_session,
                        "last_prompt_query_count",
                        int(prompt_query_count),
                    )
                with torch.no_grad():
                    objective, stats = classifier.objective_and_stats(image_01, target_label=None)
            except Exception as exc:
                return results, f"candidate_{int(index)}:{type(exc).__name__}:{exc}"
            result: Dict[str, object] = {
                "candidate_word": words[index],
                "candidate_prompt": prompts[index],
                "candidate_objective": float(objective),
                "pred_idx": stats.get("pred_idx"),
                "pred_conf": stats.get("pred_conf"),
                "pred_logit": stats.get("pred_logit"),
                "target_conf": stats.get("target_conf"),
                "target_logit": stats.get("target_logit"),
                "target_label_conf": stats.get("target_label_conf"),
                "target_label_logit": stats.get("target_label_logit"),
                "ce": stats.get("ce"),
                "candidate_variant": "prompt",
                "candidate_strip_index": int(index),
                "candidate_selected_image": tile.copy(),
                "candidate_selected_image_width": int(tile.width),
                "candidate_selected_image_height": int(tile.height),
                "candidate_selected_image_source": "raw_tile",
            }
            results.append(result)
            if bool(capture_classifier_tile_image):
                try:
                    evaluated_image = classifier_input_image(tile, classifier)
                    result["candidate_classifier_image"] = evaluated_image.copy()
                    result["candidate_classifier_image_size"] = int(evaluated_image.size[0])
                except Exception as exc:
                    return results, f"candidate_{int(index)}_capture:{type(exc).__name__}:{exc}"
        return results, None

    if render_output_path is not None:
        output = Path(render_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _render(output)
        results, error = _score(output)
        metadata = (output, len(prompts)) if error is None else None
        return results, error, metadata

    with tempfile.TemporaryDirectory(prefix="vlm_attack_eval_") as tmpdir:
        output = Path(tmpdir) / "candidate_strip.png"
        _render(output)
        results, error = _score(output)
        return results, error, None


def evaluate_source_image_baseline(
    *,
    args: argparse.Namespace,
    classifier: TorchvisionClassifier,
    source_image_path: Path,
    cwor_target_label: Optional[int] = None,
) -> Tuple[float, Dict[str, object], Optional[str]]:
    if not source_image_path.is_file():
        return 0.0, {}, f"source_image_missing:{source_image_path}"
    try:
        with Image.open(source_image_path) as source_img:
            image_01 = image_to_tensor_01(source_img).to(device=str(args.device))
        with torch.no_grad():
            objective, stats = classifier.objective_and_stats(image_01, target_label=cwor_target_label)
        return float(objective), stats, None
    except Exception as exc:
        return 0.0, {}, f"source_image_eval_failed:{type(exc).__name__}:{exc}"


def compute_victim_query_count(history: Sequence[Dict[str, object]]) -> int:
    total = 0
    for item in history:
        raw_count = item.get("candidate_count", 0)
        try:
            count = int(raw_count)
        except Exception:
            count = 0
        if count > 0:
            total += count
    return int(total)


def write_report(
    *,
    args: argparse.Namespace,
    unknown_args: List[str],
    original_prompt: str,
    optimized_prompt: str,
    best_objective: float,
    history: List[Dict[str, object]],
    early_stop_event: Optional[Dict[str, object]],
    final_selected_image_path: Optional[str] = None,
) -> None:
    args_payload: Dict[str, object] = dict(vars(args))
    args_payload["mode"] = "vlm_attack_black_box"
    if unknown_args:
        args_payload["ignored_cli_args"] = unknown_args

    victim_query_count = compute_victim_query_count(history)
    payload = {
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt,
        "best_objective": float(best_objective),
        "gcg_word": str(args.gcg_word),
        "gcg_occurrence": int(args.gcg_occurrence),
        "attack_success_rule": attack_success_rule(str(args.classifier_objective)),
        "victim_query_count": int(victim_query_count),
        "early_stop": None if early_stop_event is None else dict(early_stop_event),
        "history": history,
        "args": args_payload,
    }
    if final_selected_image_path:
        payload["final_selected_image_path"] = str(final_selected_image_path)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relpath_from_run_dir(run_dir: Path, path: Path) -> str:
    run_dir_abs = run_dir.resolve()
    path_abs = path.resolve()
    try:
        return str(path_abs.relative_to(run_dir_abs))
    except Exception:
        return str(path_abs)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def should_save_intermediate_step(*, step_idx: int, interval: int) -> bool:
    every = max(1, int(interval))
    human_step = int(step_idx) + 1
    # Save the first step for traceability, then save every N steps.
    return bool(human_step == 1 or (human_step % every) == 0)


def save_blackbox_prompt_artifacts(
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
    def _json_safe_candidate(item: Dict[str, object]) -> Dict[str, object]:
        return {
            key: value
            for key, value in dict(item).items()
            if key
            not in {
                "candidate_classifier_image",
                "candidate_selected_image",
                "candidate_precomputed_selected_image_path",
            }
        }

    artifact_dir = run_dir / "prompt_artifacts"
    human_step = int(step_idx) + 1
    stem = "scene_vocab" if str(candidate_source) == "gemma_scene_vocab" else "vlm_query"
    prompt_text_path = artifact_dir / f"step_{human_step:03d}_{stem}_prompt.txt"
    response_json_path = artifact_dir / f"step_{human_step:03d}_{stem}_response.json"
    raw_answer_text_path = artifact_dir / f"step_{human_step:03d}_{stem}_raw_answer.txt"

    raw_answer_text = str(raw_answer or "").replace("\r\n", "\n").replace("\r", "\n")

    prompt_text_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_text_path.write_text(str(prompt_text or ""), encoding="utf-8")
    raw_answer_text_path.write_text(raw_answer_text, encoding="utf-8")

    response_payload: Dict[str, object] = {
        "step_idx": int(step_idx),
        "step": human_step,
        "requested_candidate_source": str(candidate_source),
        "used_candidate_source": str(candidate_source),
        "fallback_to_grad": False,
        "error": None,
        "feedback_used": list(feedback_used),
        "raw_answer": raw_answer_text,
        "raw_answer_lines": raw_answer_text.split("\n"),
        "raw_answer_text_path": relpath_from_run_dir(run_dir, raw_answer_text_path),
        "generated_words": list(generated_words),
        "filtered_words": list(filtered_words),
        "selected_words": [str(item.get("candidate_word", "")) for item in scored_candidates],
        "scored_candidates": [_json_safe_candidate(item) for item in scored_candidates],
    }
    if vlm_error is not None:
        response_payload["error"] = str(vlm_error)
    if score_error is not None:
        response_payload["score_error"] = str(score_error)
    write_json(response_json_path, response_payload)

    return {
        "prompt_text_path": relpath_from_run_dir(run_dir, prompt_text_path),
        "response_json_path": relpath_from_run_dir(run_dir, response_json_path),
    }


def main() -> int:
    args, unknown_args = parse_args()
    args.classifier_mode = normalize_classifier_mode(args.classifier_mode)
    args.gcg_candidate_source = normalize_candidate_source(args.gcg_candidate_source)
    _set_process_title_from_args(args)

    if int(args.scene_vlm_max_new_tokens) < 1:
        raise ValueError("--scene_vlm_max_new_tokens must be >= 1")
    if int(args.gcg_steps) < 1:
        raise ValueError("--gcg_steps must be >= 1")
    if int(args.gcg_scene_vocab_size) < 1:
        raise ValueError("--gcg_scene_vocab_size must be >= 1")
    if int(args.gcg_scene_vocab_prompts_per_strategy) < 0:
        raise ValueError("--gcg_scene_vocab_prompts_per_strategy must be >= 0")
    resolve_scene_vocab_strategy_specs(getattr(args, "gcg_scene_vocab_enabled_strategies", "all"))
    if int(args.gcg_slot_candidate_max_words) < 1:
        raise ValueError("--gcg_slot_candidate_max_words must be >= 1")
    if int(args.gcg_scene_feedback_limit) < 1:
        raise ValueError("--gcg_scene_feedback_limit must be >= 1")
    if int(args.gcg_save_intermediate_interval) < 1:
        raise ValueError("--gcg_save_intermediate_interval must be >= 1")
    if int(args.gcg_scene_llm_max_new_tokens) < 1:
        raise ValueError("--gcg_scene_llm_max_new_tokens must be >= 1")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(str(args.report_path)).expanduser().resolve().parent

    editable_prompt = get_editable_prompt(args)
    inversion_prompt = get_inversion_prompt(args, editable_prompt)
    has_class_placeholder = contains_class_placeholder(editable_prompt) or contains_class_placeholder(
        inversion_prompt
    )
    resolved_class_name = str(args.class_name or "").strip()
    if has_class_placeholder and not resolved_class_name:
        try:
            resolved_class_name = infer_imagenet_class_name(
                model_name=str(args.classifier_name),
                label_idx=int(args.classifier_label),
            )
            print(
                "[VLM attack] auto-filled class placeholder from classifier label: "
                f"class_name='{resolved_class_name}'"
            )
        except Exception as exc:
            print(
                "WARNING: failed to auto-fill class placeholder; "
                f"keeping prompt as-is ({type(exc).__name__}: {exc})"
            )
    if resolved_class_name:
        editable_prompt = apply_class_placeholder(editable_prompt, resolved_class_name)
        inversion_prompt = apply_class_placeholder(inversion_prompt, resolved_class_name)
        args.class_name = resolved_class_name
    current_prompt = editable_prompt
    current_word = str(args.gcg_word or "").strip()
    slot_kind = infer_slot_kind(editable_prompt, args.scene_vlm_question)
    default_fallback = "outdoor" if slot_kind == "scene" else "object"
    fallback_word = str(args.scene_fallback or "").strip().lower() or current_word or default_fallback

    has_input_image = False
    if args.input_img_path is not None and str(args.input_img_path).strip():
        input_path = Path(str(args.input_img_path))
        if not input_path.is_file():
            raise FileNotFoundError(f"--input_img_path not found: {input_path}")
        has_input_image = True

    if args.classifier_label is None:
        raise ValueError("--classifier_label is required for CE score-based black-box mode.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is not available.")

    classifier = TorchvisionClassifier(
        model_name=str(args.classifier_name),
        device=str(args.device),
        checkpoint_path=args.classifier_ckpt,
        num_classes=args.classifier_num_classes,
        input_size=int(args.classifier_input_size),
        objective_mode=str(args.classifier_objective),
        label=int(args.classifier_label),
    )
    wandb_run, wandb_enabled = init_wandb_run(args)
    vlm_runtime_cache = PersistentVLMRuntimeCache()
    render_session: Optional[PersistentFluxRenderSession] = None
    try:
        try:
            render_session = PersistentFluxRenderSession(args=args, has_input_image=has_input_image)
        except Exception as exc:
            render_session = None
            print(
                "WARNING: failed to initialize persistent FLUX render session; "
                f"falling back to subprocess rendering ({type(exc).__name__}: {exc})"
            )

        baseline_source = "rendered_prompt"
        baseline_strip_meta: Optional[Tuple[Path, int]] = None
        candidate_strip_dir = run_dir / "candidate_strips"
        if args.classifier_mode == "black-box" and has_input_image:
            source_objective, source_stats, source_error = evaluate_source_image_baseline(
                args=args,
                classifier=classifier,
                source_image_path=Path(str(args.input_img_path)),
            )
            if source_error is not None:
                raise RuntimeError(f"Failed to evaluate source-image baseline objective: {source_error}")
            best_objective = float(source_objective)
            best_pred_idx = source_stats.get("pred_idx")
            best_pred_conf = source_stats.get("pred_conf")
            best_target_conf = source_stats.get("target_conf")
            best_pred_logit = source_stats.get("pred_logit", source_stats.get("pred_conf"))
            best_target_logit = source_stats.get("target_logit", source_stats.get("target_conf"))
            best_ce = source_stats.get("ce")
            baseline_source = "source_image"
        else:
            baseline_render_output_path: Optional[Path] = None
            if bool(args.gcg_save_intermediate):
                baseline_render_output_path = candidate_strip_dir / "step_000_candidates.png"
            baseline_results, baseline_error, baseline_strip_meta = evaluate_prompt_candidates(
                args=args,
                classifier=classifier,
                inversion_prompt=inversion_prompt,
                candidate_words=[current_word],
                candidate_prompts=[current_prompt],
                has_input_image=has_input_image,
                render_session=render_session,
                render_output_path=baseline_render_output_path,
                cwor_step_index=1,
            )
            if baseline_error is not None or len(baseline_results) == 0:
                raise RuntimeError(f"Failed to evaluate baseline CE objective: {baseline_error}")

            baseline = baseline_results[0]
            best_objective = float(baseline["candidate_objective"])
            best_pred_idx = baseline.get("pred_idx")
            best_pred_conf = baseline.get("pred_conf")
            best_target_conf = baseline.get("target_conf")
            best_pred_logit = baseline.get("pred_logit", baseline.get("pred_conf"))
            best_target_logit = baseline.get("target_logit", baseline.get("target_conf"))
            best_ce = baseline.get("ce")

        if wandb_enabled and wandb_run is not None:
            baseline_payload: Dict[str, object] = {
                "step": 0,
                "objective/candidate": float(best_objective),
                "objective/best": float(best_objective),
                "loss/cls": float(best_objective),
                "loss/adv": float(best_objective),
                "loss/opt": float(-best_objective),
                "candidate/source": str(baseline_source),
                "candidate/count": 1,
                "flags/accepted_candidate": 1,
                "pred/pred_idx": -1 if best_pred_idx is None else int(best_pred_idx),
            }
            if best_pred_logit is not None:
                baseline_payload["logit/pred"] = float(best_pred_logit)
            if best_target_logit is not None:
                baseline_payload["logit/target"] = float(best_target_logit)
            if best_pred_conf is not None:
                baseline_payload["conf/pred"] = float(best_pred_conf)
            if best_target_conf is not None:
                baseline_payload["conf/target"] = float(best_target_conf)
            if best_ce is not None:
                baseline_payload["ce"] = float(best_ce)
            wandb_enabled = log_wandb_payload(
                run=wandb_run,
                args=args,
                payload=baseline_payload,
                step=0,
                honor_log_every=False,
            )
        initial_pred_idx = best_pred_idx
        initial_pred_conf = best_pred_conf
        initial_pred_logit = best_pred_logit
        initial_target_logit = best_target_logit
        initial_ce = best_ce

        step_prompts_path = run_dir / "step_prompts.json"
        history: List[Dict[str, object]] = []
        previous_scene_feedback: List[Dict[str, object]] = []
        prompt_trace_steps: List[Dict[str, object]] = []
        early_stop_event: Optional[Dict[str, object]] = None
        attack_success_rule_text = attack_success_rule(str(args.classifier_objective))
        prompt_trace_doc: Dict[str, object] = {
            "gcg_word": str(args.gcg_word),
            "gcg_occurrence": int(args.gcg_occurrence),
            "inversion_prompt": str(inversion_prompt),
            "initial_prompt": str(editable_prompt),
            "steps": prompt_trace_steps,
        }
        init_trace_step: Dict[str, object] = {
            "step": 0,
            "kind": "init_prompt",
            "prompt": str(current_prompt),
            "objective": float(best_objective),
            "pred_idx": best_pred_idx,
            "pred_conf": best_pred_conf,
            "target_conf": best_target_conf,
            "pred_logit": best_pred_logit,
            "target_logit": best_target_logit,
            "ce": best_ce,
            "requested_candidate_source": str(args.gcg_candidate_source),
            "used_candidate_source": "baseline",
            "baseline_source": str(baseline_source),
        }
        skip_init_render = bool(
            args.classifier_mode == "black-box"
            and has_input_image
            and baseline_source == "source_image"
        )
        if bool(args.gcg_save_intermediate) and not skip_init_render:
            init_reusable_strip_path: Optional[Path] = None
            init_reusable_num_prompts: Optional[int] = None
            init_reusable_generated_index: Optional[int] = None
            if baseline_strip_meta is not None:
                init_reusable_strip_path, init_reusable_num_prompts = baseline_strip_meta
                init_reusable_generated_index = 1
            init_image_path = save_blackbox_step_image(
                run_dir=run_dir,
                human_step=0,
                phase="init",
                replacement_word=current_word or fallback_word,
                inversion_prompt=inversion_prompt,
                candidate_prompt=current_prompt,
                args=args,
                has_input_image=has_input_image,
                render_session=render_session,
                existing_generated_strip_path=init_reusable_strip_path,
                existing_generated_index=init_reusable_generated_index,
                existing_num_prompts=init_reusable_num_prompts,
            )
            init_trace_step["step_image_path"] = init_image_path
        elif bool(args.gcg_save_intermediate) and skip_init_render:
            init_trace_step["step_image_skipped"] = "black_box_source_baseline"
        prompt_trace_steps.append(init_trace_step)
        write_json(step_prompts_path, prompt_trace_doc)

        for step in range(int(args.gcg_steps)):
            save_step_intermediate = bool(args.gcg_save_intermediate) and should_save_intermediate_step(
                step_idx=int(step),
                interval=int(args.gcg_save_intermediate_interval),
            )
            candidate_source = str(args.gcg_candidate_source)
            feedback_for_generation = rank_scene_vocab_feedback_entries(
                feedback_entries=previous_scene_feedback,
                limit=int(args.gcg_scene_feedback_limit),
            )
            generated_words: List[str] = []
            generation_prompt = ""
            raw_answer = ""
            vlm_error: Optional[str] = None

            if has_input_image:
                if candidate_source == "gemma_scene_vocab":
                    generated_words, raw_answer, generation_prompt, vlm_error = generate_scene_vocab_words(
                        args=args,
                        step_idx=step,
                        current_prompt=current_prompt,
                        current_word=current_word,
                        slot_kind=slot_kind,
                        best_objective=best_objective,
                        previous_feedback=feedback_for_generation,
                        reference_image_path=Path(str(args.input_img_path)),
                        fallback_word=fallback_word,
                        runtime_cache=vlm_runtime_cache,
                    )
                if candidate_source != "gemma_scene_vocab" or len(generated_words) == 0:
                    candidate_word, raw_answer_simple, vlm_error_simple = query_vlm_word(
                        image_path=Path(str(args.input_img_path)),
                        args=args,
                        slot_kind=slot_kind,
                        fallback_word=fallback_word,
                        runtime_cache=vlm_runtime_cache,
                    )
                    if len(generated_words) == 0:
                        generated_words = [candidate_word]
                    if not raw_answer:
                        raw_answer = raw_answer_simple
                    if vlm_error is None:
                        vlm_error = vlm_error_simple
            else:
                generated_words = [fallback_word]
                raw_answer = ""
                vlm_error = "no_input_image"

            candidate_words_for_eval: List[str] = []
            candidate_prompts_for_eval: List[str] = []
            max_eval_candidates = max(1, int(args.gcg_batch_size))
            for word in generated_words:
                candidate_prompt_try, replaced_try = replace_nth_word(
                    current_prompt,
                    current_word=current_word,
                    replacement=word,
                    occurrence=int(args.gcg_occurrence),
                )
                if replaced_try and candidate_prompt_try != current_prompt:
                    candidate_words_for_eval.append(str(word))
                    candidate_prompts_for_eval.append(candidate_prompt_try)
                    if len(candidate_prompts_for_eval) >= max_eval_candidates:
                        break

            step_render_output_path: Optional[Path] = None
            if save_step_intermediate:
                step_render_output_path = candidate_strip_dir / f"step_{int(step) + 1:03d}_candidates.png"

            scored_candidates, score_error, score_strip_meta = evaluate_prompt_candidates(
                args=args,
                classifier=classifier,
                inversion_prompt=inversion_prompt,
                candidate_words=candidate_words_for_eval,
                candidate_prompts=candidate_prompts_for_eval,
                has_input_image=has_input_image,
                render_session=render_session,
                render_output_path=step_render_output_path,
                cwor_step_index=int(step) + 1,
            )

            candidate_word = fallback_word
            candidate_prompt = current_prompt
            candidate_objective = float(best_objective)
            candidate_pred_idx = None
            candidate_pred_conf = None
            candidate_target_conf = None
            candidate_pred_logit = None
            candidate_target_logit = None
            candidate_ce = None
            best_candidate_idx: Optional[int] = None
            if len(scored_candidates) > 0:
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
                candidate_pred_logit = best_candidate.get("pred_logit", best_candidate.get("pred_conf"))
                candidate_target_logit = best_candidate.get("target_logit", best_candidate.get("target_conf"))
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
            force_save_cwor_success_image = bool(
                candidate_attack_success is True and str(candidate_word) == "<CWOR>"
            )
            early_stop_triggered = bool(
                bool(args.gcg_early_stop_on_attack_success)
                and candidate_attack_success is True
            )
            # Mark non-accepted only when prediction changed and objective failed to improve.
            accepted = bool(objective_improved or not prediction_changed or early_stop_triggered)
            if objective_improved or early_stop_triggered:
                best_objective = float(candidate_objective)
                current_prompt = candidate_prompt
                current_word = candidate_word
                best_pred_idx = candidate_pred_idx
                best_pred_conf = candidate_pred_conf
                best_target_conf = candidate_target_conf
                best_pred_logit = candidate_pred_logit
                best_target_logit = candidate_target_logit
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
                "candidate_count": int(len(candidate_prompts_for_eval)),
                "scored_candidates": scored_candidates,
                "pred_idx": candidate_pred_idx,
                "pred_conf": candidate_pred_conf,
                "target_conf": candidate_target_conf,
                "pred_logit": candidate_pred_logit,
                "target_logit": candidate_target_logit,
                "ce": candidate_ce,
                "attack_success": candidate_attack_success,
                "attack_success_rule": attack_success_rule_text,
                "early_stop_triggered": bool(early_stop_triggered),
            }
            if generation_prompt:
                entry["scene_vocab_generation_prompt"] = generation_prompt
            if feedback_for_generation:
                entry["scene_vocab_feedback_used"] = feedback_for_generation
            if vlm_error:
                entry["vlm_error"] = str(vlm_error)
            if score_error:
                entry["score_error"] = str(score_error)
            if early_stop_triggered:
                entry["early_stop_reason"] = "attack_success"

            prompt_text_for_artifact = generation_prompt if generation_prompt else str(args.scene_vlm_question)
            prompt_artifact_paths = save_blackbox_prompt_artifacts(
                run_dir=run_dir,
                step_idx=step,
                candidate_source=candidate_source,
                prompt_text=prompt_text_for_artifact,
                raw_answer=str(raw_answer),
                feedback_used=feedback_for_generation,
                generated_words=generated_words,
                filtered_words=candidate_words_for_eval,
                scored_candidates=scored_candidates,
                vlm_error=vlm_error,
                score_error=score_error,
            )
            if candidate_source == "gemma_scene_vocab":
                entry["scene_vocab_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                entry["scene_vocab_response_json_path"] = prompt_artifact_paths.get("response_json_path")
            else:
                entry["vlm_query_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                entry["vlm_query_response_json_path"] = prompt_artifact_paths.get("response_json_path")
            save_step_image_this_step = bool(save_step_intermediate or force_save_cwor_success_image)
            if save_step_image_this_step:
                step_phase = "accepted" if accepted else "rejected"
                reusable_strip_path: Optional[Path] = None
                reusable_num_prompts: Optional[int] = None
                reusable_generated_index: Optional[int] = None
                if score_strip_meta is not None and best_candidate_idx is not None:
                    reusable_strip_path, reusable_num_prompts = score_strip_meta
                    reusable_generated_index = int(best_candidate_idx) + 1
                entry["step_image_path"] = save_blackbox_step_image(
                    run_dir=run_dir,
                    human_step=int(step) + 1,
                    phase=step_phase,
                    replacement_word=candidate_word,
                    inversion_prompt=inversion_prompt,
                    candidate_prompt=candidate_prompt,
                    args=args,
                    has_input_image=has_input_image,
                    render_session=render_session,
                    existing_generated_strip_path=reusable_strip_path,
                    existing_generated_index=reusable_generated_index,
                    existing_num_prompts=reusable_num_prompts,
                )
                if force_save_cwor_success_image and not save_step_intermediate:
                    entry["step_image_reason"] = "forced_cwor_success"

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
                "candidate_count": int(len(candidate_prompts_for_eval)),
                "pred_idx": candidate_pred_idx,
                "pred_conf": candidate_pred_conf,
                "target_conf": candidate_target_conf,
                "pred_logit": candidate_pred_logit,
                "target_logit": candidate_target_logit,
                "ce": candidate_ce,
                "attack_success": candidate_attack_success,
                "attack_success_rule": attack_success_rule_text,
                "early_stop_triggered": bool(early_stop_triggered),
            }
            if candidate_source == "gemma_scene_vocab":
                trace_step["scene_vocab_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                trace_step["scene_vocab_response_json_path"] = prompt_artifact_paths.get("response_json_path")
            else:
                trace_step["vlm_query_prompt_text_path"] = prompt_artifact_paths.get("prompt_text_path")
                trace_step["vlm_query_response_json_path"] = prompt_artifact_paths.get("response_json_path")
            if "step_image_path" in entry:
                trace_step["step_image_path"] = entry["step_image_path"]
            if "step_image_reason" in entry:
                trace_step["step_image_reason"] = entry["step_image_reason"]
            if vlm_error is not None:
                trace_step["vlm_error"] = str(vlm_error)
            if score_error is not None:
                trace_step["score_error"] = str(score_error)
            prompt_trace_steps.append(trace_step)
            write_json(step_prompts_path, prompt_trace_doc)

            if wandb_enabled and wandb_run is not None:
                wandb_payload: Dict[str, object] = {
                    "step": int(step) + 1,
                    "objective/candidate": float(candidate_objective),
                    "objective/best": float(best_objective),
                    "loss/cls": float(candidate_objective),
                    "loss/adv": float(candidate_objective),
                    "loss/opt": float(-candidate_objective),
                    "candidate/source": str(candidate_source),
                    "candidate/count": int(len(candidate_prompts_for_eval)),
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
                if candidate_pred_conf is not None:
                    wandb_payload["conf/pred"] = float(candidate_pred_conf)
                if candidate_target_conf is not None:
                    wandb_payload["conf/target"] = float(candidate_target_conf)
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
                    "classifier_label": None if args.classifier_label is None else int(args.classifier_label),
                    "attack_success_rule": attack_success_rule_text,
                }
                print(f"[VLM attack] early stop triggered at step {int(step) + 1}: attack_success=True")
                break

            if not accepted:
                break

        final_selected_image_relpath: Optional[str] = None
        final_selected_image_path = run_dir / "images" / "final_selected.png"
        if has_input_image and args.input_img_path is not None and str(args.input_img_path).strip():
            with tempfile.TemporaryDirectory(prefix="vlm_attack_final_") as tmpdir:
                generated_strip_path = Path(tmpdir) / "generated_strip.png"
                render_prompt_pair(
                    args=args,
                    inversion_prompt=inversion_prompt,
                    editable_prompt=current_prompt,
                    output_path=generated_strip_path,
                    has_input_image=has_input_image,
                    render_session=render_session,
                )
                try:
                    save_source_vs_generated_comparison(
                        source_image_path=Path(str(args.input_img_path)),
                        generated_strip_path=generated_strip_path,
                        output_path=output_path,
                        num_prompts=2,
                        generated_index=1,
                    )
                except Exception as exc:
                    print(
                        "WARNING: failed to compose source-vs-final comparison; "
                        f"falling back to generated strip ({type(exc).__name__}: {exc})"
                    )
                    with Image.open(generated_strip_path) as fallback_img:
                        fallback_img.convert("RGB").save(output_path)
                try:
                    save_selected_candidate_from_generated_strip(
                        generated_strip_path=generated_strip_path,
                        output_path=final_selected_image_path,
                        num_prompts=2,
                        generated_index=1,
                    )
                    final_selected_image_relpath = relpath_from_run_dir(run_dir, final_selected_image_path)
                except Exception as exc:
                    print(
                        "WARNING: failed to save final selected generated image; "
                        f"continuing without it ({type(exc).__name__}: {exc})"
                    )
        else:
            render_prompt_pair(
                args=args,
                inversion_prompt=inversion_prompt,
                editable_prompt=current_prompt,
                output_path=output_path,
                has_input_image=has_input_image,
                render_session=render_session,
            )
            try:
                save_selected_candidate_from_generated_strip(
                    generated_strip_path=output_path,
                    output_path=final_selected_image_path,
                    num_prompts=2,
                    generated_index=1,
                )
                final_selected_image_relpath = relpath_from_run_dir(run_dir, final_selected_image_path)
            except Exception as exc:
                print(
                    "WARNING: failed to save final selected generated image; "
                    f"continuing without it ({type(exc).__name__}: {exc})"
                )
        if wandb_enabled and wandb_run is not None:
            wandb_enabled = log_wandb_final_image(
                run=wandb_run,
                args=args,
                image_path=output_path,
                step=int(len(history)),
            )
        write_report(
            args=args,
            unknown_args=unknown_args,
            original_prompt=editable_prompt,
            optimized_prompt=current_prompt,
            best_objective=best_objective,
            history=history,
            early_stop_event=early_stop_event,
            final_selected_image_path=final_selected_image_relpath,
        )

        accepted_steps = sum(1 for item in history if bool(item.get("accepted", False)))
        victim_query_count = compute_victim_query_count(history)
        final_attack_success = compute_attack_success(
            pred_idx=best_pred_idx,
            classifier_label=args.classifier_label,
            objective_mode=str(args.classifier_objective),
        )
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
                "classifier_mode": str(args.classifier_mode),
                "candidate_source": str(args.gcg_candidate_source),
                "original_prompt": str(editable_prompt),
                "optimized_prompt": str(current_prompt),
            },
        )

        print(f"[VLM attack] classifier_mode={args.classifier_mode}")
        print(f"[VLM attack] original prompt:  {editable_prompt}")
        print(f"[VLM attack] optimized prompt: {current_prompt}")
        print(f"[VLM attack] best objective:   {best_objective:.6f}")
        print(
            f"[VLM attack] final stats: pred_idx={best_pred_idx}, pred_logit={best_pred_logit}, "
            f"target_logit={best_target_logit}, ce={best_ce}"
        )
        if early_stop_event is not None:
            print(
                "[VLM attack] early stop: "
                f"triggered={early_stop_event.get('triggered')} "
                f"reason={early_stop_event.get('reason')} "
                f"step={early_stop_event.get('step')}"
            )
        print(f"[VLM attack] saved image:      {args.output_path}")
        if final_selected_image_relpath is not None:
            print(f"[VLM attack] saved final selected image: {final_selected_image_relpath}")
        print(f"[VLM attack] saved report:     {args.report_path}")
        return 0
    finally:
        if render_session is not None:
            render_session.close()
        vlm_runtime_cache.close()


# Deliberately no executable entry point. The supported surface is vlm_attack.py.

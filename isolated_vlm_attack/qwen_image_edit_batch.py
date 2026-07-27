"""Experimental single-GPU batching for QwenImageEditPlusPipeline.

The upstream pipeline currently rejects prompt batches even though its latent
and denoising internals carry a batch dimension.  This module keeps the
upstream package untouched and batches only the part that is safe to share:

* each prompt is encoded independently with the same reference image;
* the reference image is VAE-encoded once and expanded by ``prepare_latents``;
* prompt embeddings, noise latents, transformer denoising, and VAE decoding
  are executed as one batch.

The implementation intentionally depends on the pipeline's existing helper
methods.  Unsupported diffusers versions fail early so the caller can fall
back to the standard sequential path.
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import inspect
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


_REQUIRED_PIPELINE_ATTRIBUTES = (
    "encode_prompt",
    "image_processor",
    "latent_channels",
    "maybe_free_model_hooks",
    "prepare_latents",
    "scheduler",
    "transformer",
    "vae",
    "vae_scale_factor",
    "_unpack_latents",
)


def _require_compatible_pipeline(pipe) -> None:
    missing = [name for name in _REQUIRED_PIPELINE_ATTRIBUTES if not hasattr(pipe, name)]
    if missing:
        raise RuntimeError(
            "Qwen batch rendering requires QwenImageEditPlusPipeline internals; "
            f"missing={','.join(missing)}"
        )


def _pipeline_helper(pipe, name: str):
    try:
        module = importlib.import_module(pipe.__class__.__module__)
    except Exception:
        return None
    helper = getattr(module, name, None)
    return helper if callable(helper) else None


def _calculate_dimensions(pipe, target_area: int, ratio: float) -> Tuple[int, int]:
    helper = _pipeline_helper(pipe, "calculate_dimensions")
    if helper is not None:
        width, height = helper(target_area, ratio)
        return int(width), int(height)

    width = np.sqrt(float(target_area) * float(ratio))
    height = width / float(ratio)
    return int(round(width / 32) * 32), int(round(height / 32) * 32)


def _calculate_shift(pipe, image_seq_len: int) -> float:
    config = pipe.scheduler.config
    get_value = config.get if hasattr(config, "get") else lambda key, default: getattr(config, key, default)
    kwargs = {
        "base_seq_len": int(get_value("base_image_seq_len", 256)),
        "max_seq_len": int(get_value("max_image_seq_len", 4096)),
        "base_shift": float(get_value("base_shift", 0.5)),
        "max_shift": float(get_value("max_shift", 1.15)),
    }
    helper = _pipeline_helper(pipe, "calculate_shift")
    if helper is not None:
        return float(helper(int(image_seq_len), **kwargs))

    slope = (kwargs["max_shift"] - kwargs["base_shift"]) / (
        kwargs["max_seq_len"] - kwargs["base_seq_len"]
    )
    intercept = kwargs["base_shift"] - slope * kwargs["base_seq_len"]
    return float(int(image_seq_len) * slope + intercept)


def _retrieve_timesteps(
    pipe,
    *,
    num_inference_steps: int,
    device: torch.device,
    sigmas: Sequence[float],
    mu: float,
) -> Tuple[torch.Tensor, int]:
    helper = _pipeline_helper(pipe, "retrieve_timesteps")
    if helper is not None:
        timesteps, effective_steps = helper(
            pipe.scheduler,
            int(num_inference_steps),
            device,
            sigmas=list(sigmas),
            mu=float(mu),
        )
        return timesteps, int(effective_steps)

    set_parameters = inspect.signature(pipe.scheduler.set_timesteps).parameters
    kwargs = {}
    if "device" in set_parameters:
        kwargs["device"] = device
    if "sigmas" in set_parameters:
        kwargs["sigmas"] = list(sigmas)
    if "mu" in set_parameters:
        kwargs["mu"] = float(mu)
    if "sigmas" not in kwargs:
        kwargs["num_inference_steps"] = int(num_inference_steps)
    pipe.scheduler.set_timesteps(**kwargs)
    return pipe.scheduler.timesteps, int(len(pipe.scheduler.timesteps))


def _encode_prompt(
    pipe,
    *,
    prompt: str,
    condition_images,
    device: torch.device,
    max_sequence_length: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    kwargs = {
        "image": condition_images,
        "prompt": str(prompt),
        "device": device,
        "num_images_per_prompt": 1,
        "max_sequence_length": int(max_sequence_length),
    }
    parameters = inspect.signature(pipe.encode_prompt).parameters
    output = pipe.encode_prompt(**{key: value for key, value in kwargs.items() if key in parameters})
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise RuntimeError("Qwen encode_prompt returned an unsupported result")
    embeddings, mask = output
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
        raise RuntimeError("Qwen encode_prompt did not return rank-3 prompt embeddings")
    if int(embeddings.shape[0]) != 1:
        raise RuntimeError(
            "Qwen batch rendering encodes one prompt at a time; "
            f"received embedding batch={int(embeddings.shape[0])}"
        )
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 2):
        raise RuntimeError("Qwen encode_prompt returned an invalid attention mask")
    return embeddings, mask


def _pad_prompt_batches(
    encoded: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not encoded:
        raise ValueError("at least one encoded prompt is required")
    max_length = max(int(embeddings.shape[1]) for embeddings, _ in encoded)
    padded_embeddings: List[torch.Tensor] = []
    padded_masks: List[torch.Tensor] = []
    for embeddings, mask in encoded:
        current_length = int(embeddings.shape[1])
        if mask is None:
            mask = torch.ones(
                (int(embeddings.shape[0]), current_length),
                dtype=torch.long,
                device=embeddings.device,
            )
        if current_length < max_length:
            embedding_pad = embeddings.new_zeros(
                (int(embeddings.shape[0]), max_length - current_length, int(embeddings.shape[2]))
            )
            mask_pad = mask.new_zeros((int(mask.shape[0]), max_length - current_length))
            embeddings = torch.cat([embeddings, embedding_pad], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
        padded_embeddings.append(embeddings)
        padded_masks.append(mask)

    combined_embeddings = torch.cat(padded_embeddings, dim=0)
    combined_mask = torch.cat(padded_masks, dim=0)
    if bool(combined_mask.bool().all()):
        return combined_embeddings, None
    return combined_embeddings, combined_mask


def _expand_encoded_prompt(
    encoded: Tuple[torch.Tensor, Optional[torch.Tensor]],
    batch_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    embeddings, mask = encoded
    embeddings = embeddings.repeat(int(batch_size), 1, 1)
    if mask is not None:
        mask = mask.repeat(int(batch_size), 1)
    return embeddings, mask


def _transformer_cache_context(transformer, name: str):
    cache_context = getattr(transformer, "cache_context", None)
    if callable(cache_context):
        return cache_context(name)
    return nullcontext()


def _progress_context(pipe, total: int):
    progress_bar = getattr(pipe, "progress_bar", None)
    if callable(progress_bar):
        return progress_bar(total=int(total))
    return nullcontext(None)


def _transformer_forward(transformer, **kwargs):
    forward = getattr(transformer, "forward", None)
    if callable(forward):
        parameters = inspect.signature(forward).parameters
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    output = transformer(**kwargs)
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise RuntimeError("Qwen transformer returned an unsupported result")


@torch.inference_mode()
def render_qwen_image_edit_batch(
    *,
    pipe,
    prompts: Sequence[str],
    image: Image.Image,
    generators: Sequence[torch.Generator],
    true_cfg_scale: float,
    negative_prompt: str | Sequence[str] | None,
    num_inference_steps: int,
    max_sequence_length: int,
    guidance_scale: Optional[float],
) -> List[Image.Image]:
    """Render distinct prompts together in one Qwen denoising batch."""

    _require_compatible_pipeline(pipe)
    prompt_list = [str(prompt) for prompt in prompts]
    batch_size = len(prompt_list)
    if batch_size < 2:
        raise ValueError("experimental Qwen batch rendering requires at least two prompts")
    if len(generators) != batch_size:
        raise ValueError(
            f"generator count {len(generators)} does not match prompt count {batch_size}"
        )
    if not isinstance(image, Image.Image):
        raise TypeError("Qwen batch rendering currently requires one PIL reference image")

    device = pipe._execution_device
    image_width, image_height = image.size
    width, height = _calculate_dimensions(
        pipe,
        1024 * 1024,
        float(image_width) / float(image_height),
    )
    multiple_of = int(pipe.vae_scale_factor) * 2
    width = int(width) // multiple_of * multiple_of
    height = int(height) // multiple_of * multiple_of

    condition_width, condition_height = _calculate_dimensions(
        pipe,
        384 * 384,
        float(image_width) / float(image_height),
    )
    vae_width, vae_height = _calculate_dimensions(
        pipe,
        1024 * 1024,
        float(image_width) / float(image_height),
    )
    condition_images = [
        pipe.image_processor.resize(image, int(condition_height), int(condition_width))
    ]
    vae_images = [
        pipe.image_processor.preprocess(image, int(vae_height), int(vae_width)).unsqueeze(2)
    ]

    positive_encoded = [
        _encode_prompt(
            pipe,
            prompt=prompt,
            condition_images=condition_images,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        for prompt in prompt_list
    ]
    prompt_embeds, prompt_embeds_mask = _pad_prompt_batches(positive_encoded)

    has_negative_prompt = negative_prompt is not None
    do_true_cfg = float(true_cfg_scale) > 1.0 and has_negative_prompt
    negative_prompt_embeds = None
    negative_prompt_embeds_mask = None
    if do_true_cfg:
        if isinstance(negative_prompt, str):
            negative_encoded = _encode_prompt(
                pipe,
                prompt=negative_prompt,
                condition_images=condition_images,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds, negative_prompt_embeds_mask = _expand_encoded_prompt(
                negative_encoded,
                batch_size,
            )
        else:
            negative_prompts = [str(item) for item in negative_prompt or []]
            if len(negative_prompts) != batch_size:
                raise ValueError(
                    "negative prompt count must be one shared string or match the prompt batch"
                )
            negative_prompt_embeds, negative_prompt_embeds_mask = _pad_prompt_batches(
                [
                    _encode_prompt(
                        pipe,
                        prompt=prompt,
                        condition_images=condition_images,
                        device=device,
                        max_sequence_length=max_sequence_length,
                    )
                    for prompt in negative_prompts
                ]
            )

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = {}
    pipe._current_timestep = None
    pipe._interrupt = False

    num_channels_latents = int(pipe.transformer.config.in_channels) // 4
    latents, image_latents = pipe.prepare_latents(
        vae_images,
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        list(generators),
        None,
    )
    img_shapes = [
        [
            (1, height // int(pipe.vae_scale_factor) // 2, width // int(pipe.vae_scale_factor) // 2),
            (1, vae_height // int(pipe.vae_scale_factor) // 2, vae_width // int(pipe.vae_scale_factor) // 2),
        ]
        for _ in range(batch_size)
    ]

    sigmas = np.linspace(
        1.0,
        1.0 / int(num_inference_steps),
        int(num_inference_steps),
    ).tolist()
    mu = _calculate_shift(pipe, int(latents.shape[1]))
    timesteps, effective_steps = _retrieve_timesteps(
        pipe,
        num_inference_steps=int(num_inference_steps),
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    guidance = None
    if bool(getattr(pipe.transformer.config, "guidance_embeds", False)):
        if guidance_scale is None:
            raise ValueError("guidance_scale is required by this Qwen transformer")
        guidance = torch.full(
            (batch_size,),
            float(guidance_scale),
            device=device,
            dtype=torch.float32,
        )

    set_begin_index = getattr(pipe.scheduler, "set_begin_index", None)
    if callable(set_begin_index):
        set_begin_index(0)
    scheduler_order = int(getattr(pipe.scheduler, "order", 1))
    warmup_steps = max(len(timesteps) - int(effective_steps) * scheduler_order, 0)

    with _progress_context(pipe, effective_steps) as progress_bar:
        for step_index, timestep_value in enumerate(timesteps):
            pipe._current_timestep = timestep_value
            latent_model_input = (
                torch.cat([latents, image_latents], dim=1)
                if image_latents is not None
                else latents
            )
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            transformer_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "attention_kwargs": pipe._attention_kwargs,
                "return_dict": False,
            }
            with _transformer_cache_context(pipe.transformer, "cond"):
                noise_pred = _transformer_forward(pipe.transformer, **transformer_kwargs)
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                transformer_kwargs["encoder_hidden_states_mask"] = negative_prompt_embeds_mask
                transformer_kwargs["encoder_hidden_states"] = negative_prompt_embeds
                with _transformer_cache_context(pipe.transformer, "uncond"):
                    negative_noise_pred = _transformer_forward(
                        pipe.transformer,
                        **transformer_kwargs,
                    )
                negative_noise_pred = negative_noise_pred[:, : latents.size(1)]
                combined_pred = negative_noise_pred + float(true_cfg_scale) * (
                    noise_pred - negative_noise_pred
                )
                conditional_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                combined_norm = torch.norm(combined_pred, dim=-1, keepdim=True)
                noise_pred = combined_pred * (
                    conditional_norm / combined_norm.clamp_min(torch.finfo(combined_norm.dtype).eps)
                )

            previous_dtype = latents.dtype
            latents = pipe.scheduler.step(
                noise_pred,
                timestep_value,
                latents,
                return_dict=False,
            )[0]
            if latents.dtype != previous_dtype:
                latents = latents.to(previous_dtype)
            if (
                progress_bar is not None
                and (
                    step_index == len(timesteps) - 1
                    or (
                        (step_index + 1) > warmup_steps
                        and (step_index + 1) % scheduler_order == 0
                    )
                )
            ):
                progress_bar.update()

    pipe._current_timestep = None
    latents = pipe._unpack_latents(latents, height, width, int(pipe.vae_scale_factor))
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, int(pipe.vae.config.z_dim), 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(pipe.vae.config.latents_std)
        .view(1, int(pipe.vae.config.z_dim), 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    output_images = pipe.image_processor.postprocess(decoded, output_type="pil")
    pipe.maybe_free_model_hooks()

    images = [item.convert("RGB").copy() for item in output_images if isinstance(item, Image.Image)]
    if len(images) != batch_size:
        raise RuntimeError(
            f"Qwen batch renderer returned {len(images)} images for {batch_size} prompts"
        )
    return images


__all__ = (
    "_pad_prompt_batches",
    "render_qwen_image_edit_batch",
)

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image
import torch
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm


_PACK = "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)"
_UNPACK = "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)"


def _to_spatial(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return rearrange(x, _PACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def _to_packed(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return rearrange(x, _UNPACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def _pack_samples(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [B, N, D] tensor, got shape={tuple(x.shape)}")
    return x.reshape(1, x.shape[0] * x.shape[1], x.shape[2])


def _repeat_rotary(rotary_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
    if rotary_emb.dim() != 4 or rotary_emb.shape[0] != 1 or rotary_emb.shape[1] != 1:
        raise ValueError(f"unexpected rotary shape: {tuple(rotary_emb.shape)}")
    return rotary_emb.repeat(1, 1, int(batch_size), 1)


def _decode_batch_to_images(vae, latents: torch.Tensor) -> List[Image.Image]:
    latents = latents.to(vae.dtype)
    z_dim = vae.config.z_dim
    mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, z_dim, 1, 1, 1)
    decoded = vae.decode(latents * std + mean, return_dict=False)[0]
    processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    clips = processor.postprocess_video(decoded, output_type="np")

    if isinstance(clips, np.ndarray):
        clip_list = list(clips) if clips.ndim == 5 else [clips]
    else:
        clip_list = list(clips)

    images: List[Image.Image] = []
    for clip in clip_list:
        frame = clip[0] if getattr(clip, "ndim", 0) == 4 else clip
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        images.append(Image.fromarray(frame).convert("RGB"))
    return images


@torch.no_grad()
def _sample_i2i_v2v_batch(
    *,
    renderer_model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    neg_ids: torch.Tensor,
    neg_mask: torch.Tensor,
    image_vae_latents: torch.Tensor,
    num_frames: int,
    width: int,
    height: int,
    num_inference_steps: int,
    omega_TI: float,
    omega_scale: float,
    flow_shift: float,
    seed: int,
    device: torch.device,
    keep_models_on_gpu: bool,
) -> torch.Tensor:
    prompt_count = int(prompt_ids.shape[0])
    if prompt_count <= 0:
        raise ValueError("prompt batch is empty")
    if image_vae_latents.shape[0] != 1 or image_vae_latents.shape[2] != 1:
        raise ValueError(
            "isolated Bernini batch path currently supports one shared i2i source image only"
        )

    renderer_model.t5_text_encoder = renderer_model.t5_text_encoder.to(device)
    prompt_embeds = renderer_model.encode_prompt(prompt_ids.to(device), prompt_mask.to(device))
    uncond_embeds = renderer_model.encode_prompt(neg_ids.to(device), neg_mask.to(device))
    if not bool(keep_models_on_gpu):
        renderer_model.t5_text_encoder = renderer_model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

    diffusion = renderer_model.diff_dec
    if bool(getattr(diffusion, "use_unipc", False)):
        diffusion.scheduler.set_timesteps(int(num_inference_steps))
    else:
        diffusion.scheduler.set_timesteps(int(num_inference_steps), shift=float(flow_shift))

    num_frames = int(num_frames) // diffusion.vae_scale_factor_temporal * diffusion.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)
    timesteps = diffusion.scheduler.timesteps.to(device)
    boundary_timestep = diffusion.switch_dit_boundary * diffusion.scheduler.num_train_timesteps

    base_transformer = diffusion.transformer if diffusion.transformer is not None else diffusion.transformer_2
    if base_transformer is None:
        raise RuntimeError("Bernini transformer is not initialized")
    num_channels_latents = int(base_transformer.config.in_channels)
    num_latent_frames = (num_frames - 1) // diffusion.vae_scale_factor_temporal + 1
    shape = (
        prompt_count,
        num_channels_latents,
        num_latent_frames,
        int(height) // diffusion.vae_scale_factor_spatial,
        int(width) // diffusion.vae_scale_factor_spatial,
    )

    # Sequential Bernini calls reuse the same seed for every candidate. Repeat
    # one noise sample so batched candidates stay comparable to that behavior.
    noise_shape = (1, *shape[1:])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = randn_tensor(noise_shape, device=device, dtype=torch.float32, generator=gen)
    noise = noise.repeat(prompt_count, 1, 1, 1, 1)
    noisy_vae_latent = rearrange(
        noise,
        "b c t (h ph) (w pw) -> b (t h w) (ph pw c)",
        ph=2,
        pw=2,
    ).to(device)

    if diffusion.transformer is not None:
        diffusion.transformer.to(device)
    if diffusion.transformer_2 is not None:
        diffusion.transformer_2.to(device if bool(keep_models_on_gpu) else "cpu")
    if not bool(keep_models_on_gpu):
        torch.cuda.empty_cache()
    switched = False

    image_vae_latents = image_vae_latents.repeat(prompt_count, 1, 1, 1, 1)

    for t in tqdm(timesteps):
        model_id = "transformer_1" if t >= boundary_timestep else "transformer_2"
        cond_text = prompt_embeds
        uncond_text = uncond_embeds

        if t < boundary_timestep and not switched and diffusion.transformer_2 is not None:
            if diffusion.transformer is not None and not bool(keep_models_on_gpu):
                diffusion.transformer.to("cpu")
                torch.cuda.empty_cache()
            diffusion.transformer_2.to(device)
            switched = True
            omega_TI *= omega_scale

        cur_transformer = diffusion.transformer_2 if switched else diffusion.transformer
        if cur_transformer is None:
            raise RuntimeError("Bernini active transformer is not initialized")

        image_tokens, image_rotary = cur_transformer.patch_vae_latent(
            image_vae_latents.to(dtype=cur_transformer.dtype),
            source_id=1,
        )
        unpacked_noisy = _to_spatial(noisy_vae_latent, shape).to(cur_transformer.dtype)
        noisy_tokens, noisy_rotary = cur_transformer.patch_vae_latent(unpacked_noisy, source_id=0)

        image_len = int(image_tokens.shape[1])
        noisy_len = int(noisy_tokens.shape[1])
        total_len = image_len + noisy_len
        vi_tokens = torch.cat([image_tokens, noisy_tokens], dim=1).to(cur_transformer.dtype)
        vi_inp = _pack_samples(vi_tokens)
        vi_rot = _repeat_rotary(torch.cat([image_rotary, noisy_rotary], dim=2), prompt_count)
        target_mask = torch.cat(
            [
                torch.zeros(image_len, device=device, dtype=torch.bool),
                torch.ones(noisy_len, device=device, dtype=torch.bool),
            ],
            dim=0,
        ).repeat(prompt_count)
        batch_vae_seqlen = [total_len] * prompt_count
        batch_text_seqlen = [int(cond_text.shape[1])] * prompt_count
        timestep = t.expand(prompt_count)

        def _fwd(text_embeds: torch.Tensor) -> torch.Tensor:
            text_packed = _pack_samples(text_embeds)
            pred = diffusion.shared_step(
                model_id=model_id,
                noisy_latents=vi_inp,
                timesteps=timestep,
                cond_embeds=text_packed,
                rotary_embs=vi_rot,
                batch_vae_seqlen=batch_vae_seqlen,
                batch_text_seqlen=batch_text_seqlen,
            )
            return pred[:, target_mask, :].reshape(prompt_count, noisy_len, -1)

        eps_uncond = _fwd(uncond_text)
        eps_cond = _fwd(cond_text)
        noise_pred = eps_uncond + float(omega_TI) * (eps_cond - eps_uncond)

        step_out = diffusion.scheduler.step(noise_pred, t, noisy_vae_latent, return_dict=False)
        noisy_vae_latent = step_out[0] if isinstance(step_out, tuple) else step_out

    return _to_spatial(noisy_vae_latent, shape)


@torch.no_grad()
def render_bernini_i2i_v2v_batch(
    *,
    pipeline,
    prompts: Sequence[str],
    image_path: str,
    system_prompt: str,
    generation_kwargs: dict,
) -> List[Image.Image]:
    from bernini.data_utils import make_divisible, preprocess_image
    from bernini.pipeline import _prompt_clean, _vae_encode

    prompt_list = [str(item).strip() for item in prompts if str(item).strip()]
    if not prompt_list:
        raise ValueError("prompt batch is empty")

    kwargs = dict(generation_kwargs)
    if str(kwargs.get("guidance_mode", "v2v") or "v2v").strip().lower() != "v2v":
        raise ValueError("isolated Bernini batch path supports guidance_mode='v2v' only")
    if int(kwargs.get("num_frames", 1)) != 1:
        raise ValueError("isolated Bernini batch path supports num_frames=1 only")

    device = pipeline.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    cleaned_prompts = [str(system_prompt or "") + _prompt_clean(prompt) for prompt in prompt_list]
    neg_prompt = str(kwargs.get("neg_prompt", "") or "")
    neg_prompts = [neg_prompt] * len(cleaned_prompts)

    prompt_tokens = pipeline.tokenizer(
        cleaned_prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    neg_tokens = pipeline.tokenizer(
        neg_prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    keep_models_on_gpu = bool(kwargs.get("keep_models_on_gpu", False))
    if keep_models_on_gpu:
        pipeline.model.to(device)
    pipeline.vae.to(device)
    image_tensor = preprocess_image(
        image_path,
        max_image_size=int(kwargs.get("max_image_size", 848)),
        device=device,
    )
    height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    image_vae_latents = _vae_encode(pipeline.vae, image_tensor)
    if not keep_models_on_gpu:
        pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

    height = make_divisible(height, 16)
    width = make_divisible(width, 16)
    latents = _sample_i2i_v2v_batch(
        renderer_model=pipeline.model,
        prompt_ids=prompt_tokens.input_ids,
        prompt_mask=prompt_tokens.attention_mask,
        neg_ids=neg_tokens.input_ids,
        neg_mask=neg_tokens.attention_mask,
        image_vae_latents=image_vae_latents,
        num_frames=int(kwargs.get("num_frames", 1)),
        width=width,
        height=height,
        num_inference_steps=int(kwargs.get("num_inference_steps", 40)),
        omega_TI=float(kwargs.get("omega_TI", 4.0)),
        omega_scale=float(kwargs.get("omega_scale", 0.8)),
        flow_shift=float(kwargs.get("flow_shift", 5.0)),
        seed=int(kwargs.get("seed", 42)),
        device=device,
        keep_models_on_gpu=keep_models_on_gpu,
    )
    if not keep_models_on_gpu:
        pipeline.model.to("cpu")
        torch.cuda.empty_cache()

    pipeline.vae.to(device)
    images = _decode_batch_to_images(pipeline.vae, latents)
    if not keep_models_on_gpu:
        pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

    if len(images) != len(prompt_list):
        raise RuntimeError(f"Bernini batch decode returned {len(images)} images for {len(prompt_list)} prompts")
    return images

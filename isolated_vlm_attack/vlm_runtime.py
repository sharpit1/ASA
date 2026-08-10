from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple


QWEN3_VL_4B_INSTRUCT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
INTERNVL3_5_4B_MODEL_ID = "OpenGVLab/InternVL3_5-4B"
INTERNVL3_5_4B_INSTRUCT_MODEL_ID = "OpenGVLab/InternVL3_5-4B-Instruct"
SUPPORTED_STRATEGY_MLLM_MODES = (
    "configured",
    "qwen3_vl_4b_instruct",
    "internvl3_5_4b",
    "internvl3_5_4b_instruct",
)


def normalize_strategy_mllm_mode(raw: object) -> str:
    token = (
        str(raw or "configured")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(".", "_")
    )
    aliases = {
        "": "configured",
        "default": "configured",
        "shared": "configured",
        "configured": "configured",
        "qwen3_vl_4b": "qwen3_vl_4b_instruct",
        "qwen3_vl_4b_instruct": "qwen3_vl_4b_instruct",
        "qwen/qwen3_vl_4b_instruct": "qwen3_vl_4b_instruct",
        "internvl3_5_4b": "internvl3_5_4b",
        "opengvlab/internvl3_5_4b": "internvl3_5_4b",
        "internvl3_5_4b_instruct": "internvl3_5_4b_instruct",
        "opengvlab/internvl3_5_4b_instruct": "internvl3_5_4b_instruct",
    }
    mode = aliases.get(token)
    if mode is None:
        raise ValueError(
            "strategy_mllm_mode must be one of: "
            + ", ".join(SUPPORTED_STRATEGY_MLLM_MODES)
        )
    return mode


def resolve_strategy_mllm_runtime(
    *,
    mode: object,
    configured_backend: str,
    configured_model_id: str,
    configured_thinking: bool,
    configured_do_sample: bool,
) -> Tuple[str, str, str, bool, bool]:
    """Resolve the strategy/verifier MLLM without changing the scene VLM."""

    mode_norm = normalize_strategy_mllm_mode(mode)
    if mode_norm == "qwen3_vl_4b_instruct":
        return (
            mode_norm,
            "qwen",
            QWEN3_VL_4B_INSTRUCT_MODEL_ID,
            False,
            bool(configured_do_sample),
        )
    if mode_norm == "internvl3_5_4b":
        return (
            mode_norm,
            "internvl",
            INTERNVL3_5_4B_MODEL_ID,
            False,
            bool(configured_do_sample),
        )
    if mode_norm == "internvl3_5_4b_instruct":
        return (
            mode_norm,
            "internvl",
            INTERNVL3_5_4B_INSTRUCT_MODEL_ID,
            False,
            bool(configured_do_sample),
        )
    return (
        mode_norm,
        infer_vlm_backend(configured_backend, configured_model_id, allow_blip=True),
        str(configured_model_id),
        bool(configured_thinking),
        bool(configured_do_sample),
    )


def infer_vlm_backend(
    vlm_backend: str,
    model_id: str,
    *,
    allow_blip: bool = False,
) -> str:
    backend = str(vlm_backend or "").strip().lower()
    supported = {"llava", "qwen", "internvl", "gemma3", "gemma4"}
    if allow_blip:
        supported.add("blip")

    if backend in supported:
        return backend

    if backend in {"gemma-4", "gemma_4"}:
        return "gemma4"
    if backend in {"gemma-3", "gemma_3", "gemma"}:
        return "gemma3"

    # Keep auto deterministic and backward-compatible with previous defaults.
    if backend in {"", "auto"}:
        return "llava"

    model_id_lower = str(model_id or "").strip().lower()
    for candidate in (
        "qwen",
        "internvl",
        "llava",
        "gemma4",
        "gemma-4",
        "gemma3",
        "gemma-3",
        "gemma",
        "blip",
    ):
        if candidate == "blip" and not allow_blip:
            continue
        if candidate in model_id_lower:
            if candidate in {"gemma4", "gemma-4"}:
                return "gemma4"
            if candidate in {"gemma3", "gemma-3", "gemma"}:
                return "gemma3"
            return candidate

    return "llava"


def _hf_pipeline_device_arg(vlm_device) -> int:
    if getattr(vlm_device, "type", "") == "cuda":
        idx = getattr(vlm_device, "index", None)
        return 0 if idx is None else int(idx)
    return -1


def _extract_generated_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        for item in reversed(payload):
            text = _extract_generated_text(item)
            if text:
                return text
        return ""
    if isinstance(payload, dict):
        role = str(payload.get("role", "")).strip().lower()
        if role == "assistant":
            assistant_text = _extract_generated_text(payload.get("content"))
            if assistant_text:
                return assistant_text
        for key in ("generated_text", "content", "text", "answer"):
            if key in payload:
                text = _extract_generated_text(payload.get(key))
                if text:
                    return text
        return ""
    return str(payload).strip()


def _strip_gemma_thought_blocks(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""

    cleaned = text
    # Gemma thinking-enabled formats may wrap an internal thought channel before the final answer.
    for pattern in (
        r"<\|channel\|>\s*thought\s*\n.*?<\|/?channel\|>",
        r"<\|channel\|>\s*thought\s*\n.*?<channel\|>",
    ):
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    cleaned = re.sub(r"^(?:<\|/?channel\|>|<channel\|>)+", "", cleaned).strip()
    if cleaned:
        return cleaned

    for token in ("<|/channel|>", "<|channel|>", "<channel|>"):
        idx = text.rfind(token)
        if idx >= 0:
            suffix = text[idx + len(token) :].strip()
            if suffix:
                return suffix
    return text


def _ask_with_blip(
    image,
    question: str,
    model,
    processor,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    do_sample: bool = False,
) -> str:
    del enable_thinking
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def _ask_with_llava(
    image,
    question: str,
    model,
    processor,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    do_sample: bool = False,
) -> str:
    del enable_thinking
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )
    answer_ids = output_ids[:, input_len:]
    return processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()


def _ask_with_multimodal_chat_pipeline(
    image,
    question: str,
    model,
    processor,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    do_sample: bool = False,
) -> str:
    del processor, device
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    output = None
    last_type_error: Optional[Exception] = None
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    call_variants = []
    if bool(enable_thinking):
        call_variants.extend(
            (
                {
                    "processor_kwargs": {"enable_thinking": True},
                    "generate_kwargs": generation_kwargs,
                },
                {"generate_kwargs": generation_kwargs},
            )
        )
    else:
        call_variants.append({"generate_kwargs": generation_kwargs})
    for extra_kwargs in call_variants:
        try:
            output = model(
                text=messages,
                **extra_kwargs,
            )
            break
        except TypeError as exc:
            last_type_error = exc
    if output is None and last_type_error is not None:
        raise last_type_error

    answer = _extract_generated_text(output)
    if not answer:
        raise RuntimeError("Image-text-to-text chat pipeline returned an empty answer.")
    return answer.strip()


# Backward-compatible internal alias retained for existing imports/tests.
_ask_with_qwen_pipeline = _ask_with_multimodal_chat_pipeline


class _InternVLChatRuntime:
    __slots__ = ("model", "tokenizer", "dtype", "device")

    def __init__(self, *, model, tokenizer, dtype, device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device


def _internvl_dynamic_preprocess(
    image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(int(min_num), int(max_num) + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if int(min_num) <= i * j <= int(max_num)
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])

    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    image_area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if image_area > 0.5 * target_area:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))
    if len(processed_images) != blocks:
        raise RuntimeError("InternVL dynamic image tiling produced an invalid block count.")
    if bool(use_thumbnail) and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _prepare_internvl_pixel_values(
    image,
    *,
    dtype,
    device,
    input_size: int = 448,
    max_num: int = 12,
):
    try:
        import torch
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode
    except Exception as exc:
        raise RuntimeError(
            "InternVL custom chat runtime requires torch and torchvision."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda value: (
                    value.convert("RGB") if value.mode != "RGB" else value
                )
            ),
            transforms.Resize(
                (int(input_size), int(input_size)),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    tiles = _internvl_dynamic_preprocess(
        image.convert("RGB"),
        image_size=int(input_size),
        max_num=int(max_num),
        use_thumbnail=True,
    )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values.to(device=device, dtype=dtype)


def _ask_with_internvl_chat(
    image,
    question: str,
    model,
    processor,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    do_sample: bool = False,
) -> str:
    del processor, device, enable_thinking
    if not isinstance(model, _InternVLChatRuntime):
        raise TypeError("InternVL chat runtime adapter is required.")

    pixel_values = _prepare_internvl_pixel_values(
        image,
        dtype=model.dtype,
        device=model.device,
    )
    question_text = str(question or "").strip()
    if "<image>" not in question_text:
        question_text = f"<image>\n{question_text}"
    generation_config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    answer = model.model.chat(
        model.tokenizer,
        pixel_values,
        question_text,
        generation_config,
    )
    if isinstance(answer, tuple):
        answer = answer[0]
    answer = str(answer or "").strip()
    if not answer:
        raise RuntimeError("InternVL custom chat runtime returned an empty answer.")
    return answer


def _ask_with_gemma3(
    image,
    question: str,
    model,
    processor,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    do_sample: bool = False,
) -> str:
    messages = []
    if bool(enable_thinking):
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": "<|think|>"}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
    )
    answer_ids = output_ids[:, input_len:]
    answer = processor.batch_decode(answer_ids, skip_special_tokens=True)[0]
    answer = _strip_gemma_thought_blocks(answer)
    if not answer:
        raise RuntimeError("Gemma3 image-text generation returned an empty answer.")
    return answer


def load_vlm_runtime(
    backend: str,
    model_id: str,
    vlm_dtype,
    vlm_device,
    *,
    allow_blip: bool = False,
) -> Tuple[Any, Any, Callable[..., str], bool]:
    backend_norm = infer_vlm_backend(backend, model_id, allow_blip=allow_blip)

    if backend_norm == "blip":
        if not allow_blip:
            raise ValueError("BLIP backend is not enabled in this runtime.")
        try:
            from transformers import BlipForQuestionAnswering, BlipProcessor
        except Exception as exc:
            raise RuntimeError(
                "BLIP backend requested, but BlipForQuestionAnswering/BlipProcessor are unavailable. "
                "Please upgrade transformers."
            ) from exc

        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForQuestionAnswering.from_pretrained(
            model_id,
            torch_dtype=vlm_dtype,
        )
        return model, processor, _ask_with_blip, False

    if backend_norm == "llava":
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "LLaVA backend requested, but LlavaForConditionalGeneration/AutoProcessor are unavailable. "
                "Please upgrade transformers."
            ) from exc

        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=vlm_dtype,
        )
        return model, processor, _ask_with_llava, False

    if backend_norm in {"gemma3", "gemma4"}:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Gemma backend requested, but AutoModelForImageTextToText/AutoProcessor are unavailable. "
                "Please upgrade transformers."
            ) from exc

        processor = AutoProcessor.from_pretrained(model_id)
        gemma_dtype = torch.bfloat16 if getattr(vlm_device, "type", "") == "cuda" else torch.float32
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=gemma_dtype,
            )
        except TypeError:
            # Backward compatibility for transformers versions that still use torch_dtype.
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=gemma_dtype,
            )
        return model, processor, _ask_with_gemma3, False

    if backend_norm == "internvl":
        try:
            from transformers import AutoModel, AutoTokenizer
            from transformers.modeling_utils import PreTrainedModel
        except Exception as exc:
            raise RuntimeError(
                "InternVL backend requested, but AutoModel/AutoTokenizer are unavailable. "
                "Please upgrade transformers."
            ) from exc

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        compat_missing = object()
        previous_tied_keys = getattr(
            PreTrainedModel,
            "all_tied_weights_keys",
            compat_missing,
        )
        if previous_tied_keys is compat_missing:
            # InternVL's custom outer model targets Transformers 4.x and does
            # not call the Transformers 5.x post-init that creates this map.
            # Its outer wrapper declares no tied weights, so an empty temporary
            # map preserves the 4.x loading contract without changing weights.
            PreTrainedModel.all_tied_weights_keys = {}
        try:
            try:
                raw_model = AutoModel.from_pretrained(
                    model_id,
                    dtype=vlm_dtype,
                    **model_kwargs,
                )
            except TypeError:
                raw_model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=vlm_dtype,
                    **model_kwargs,
                )
        finally:
            if previous_tied_keys is compat_missing:
                delattr(PreTrainedModel, "all_tied_weights_keys")
        raw_model = raw_model.eval().to(vlm_device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        runtime = _InternVLChatRuntime(
            model=raw_model,
            tokenizer=tokenizer,
            dtype=vlm_dtype,
            device=vlm_device,
        )
        return runtime, None, _ask_with_internvl_chat, True

    if backend_norm == "qwen":
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                f"{backend_norm} backend requested, but transformers.pipeline is unavailable. "
                "Please upgrade transformers."
            ) from exc

        pipeline_kwargs = {
            "model": model_id,
            "device": _hf_pipeline_device_arg(vlm_device),
        }
        try:
            model = pipeline(
                "image-text-to-text",
                dtype=vlm_dtype,
                **pipeline_kwargs,
            )
        except TypeError:
            # Backward compatibility for transformers versions that still use torch_dtype.
            model = pipeline(
                "image-text-to-text",
                torch_dtype=vlm_dtype,
                **pipeline_kwargs,
            )
        return model, None, _ask_with_multimodal_chat_pipeline, True

    raise ValueError(f"Unsupported VLM backend: {backend}")

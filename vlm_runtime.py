from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple
import sys


def infer_vlm_backend(
    vlm_backend: str,
    model_id: str,
    *,
    allow_blip: bool = False,
) -> str:
    backend = str(vlm_backend or "").strip().lower()
    supported = {"llava", "qwen", "gemma3", "gemma4"}
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
    for candidate in ("qwen", "llava", "gemma4", "gemma-4", "gemma3", "gemma-3", "gemma", "blip"):
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


def _ask_with_qwen_pipeline(
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
    for extra_kwargs in (
        {"enable_thinking": bool(enable_thinking), "do_sample": bool(do_sample)},
        {"do_sample": bool(do_sample)},
        {"enable_thinking": bool(enable_thinking)},
        {},
    ):
        try:
            output = model(
                text=messages,
                max_new_tokens=int(max_new_tokens),
                **extra_kwargs,
            )
            break
        except TypeError as exc:
            last_type_error = exc
    if output is None and last_type_error is not None:
        raise last_type_error

    answer = _extract_generated_text(output)
    if not answer:
        raise RuntimeError("Qwen image-text-to-text pipeline returned an empty answer.")
    return answer.strip()


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

    if backend_norm == "qwen":
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "Qwen backend requested, but transformers.pipeline is unavailable. "
                "Please upgrade transformers."
            ) from exc

        model = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=vlm_dtype,
            device=_hf_pipeline_device_arg(vlm_device),
        )
        return model, None, _ask_with_qwen_pipeline, True

    raise ValueError(f"Unsupported VLM backend: {backend}")

"""Exact generator registry for the isolated robustness-attack pipeline."""

from typing import Optional


FLUX2_KLEIN_MODEL_IDS = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-4b-kv",
    "black-forest-labs/FLUX.2-klein-9b-kv",
)
QWEN_IMAGE_EDIT_MODEL_IDS = ("Qwen/Qwen-Image-Edit-2511",)
BERNINI_MODEL_ID = "bernini"


def _normalized_model_id(model_path: object) -> str:
    return str(model_path or "").strip().casefold()


def validate_generator_model(
    model_path: object,
    *,
    expected_family: Optional[str] = None,
) -> str:
    """Return the registered family or reject an unlisted model identifier."""

    token = _normalized_model_id(model_path)
    flux_ids = {_normalized_model_id(value) for value in FLUX2_KLEIN_MODEL_IDS}
    qwen_ids = {_normalized_model_id(value) for value in QWEN_IMAGE_EDIT_MODEL_IDS}
    if token in flux_ids:
        family = "flux2-klein"
    elif token in qwen_ids:
        family = "qwen-image-edit"
    elif token == BERNINI_MODEL_ID:
        family = "bernini"
    else:
        allowed = ", ".join(
            [*FLUX2_KLEIN_MODEL_IDS, *QWEN_IMAGE_EDIT_MODEL_IDS, BERNINI_MODEL_ID]
        )
        raise ValueError(f"unsupported generator model_path={model_path!r}; allowed: {allowed}")

    if expected_family is not None and family != str(expected_family):
        raise ValueError(
            f"generator family mismatch: expected {expected_family}, got {family} ({model_path!r})"
        )
    return family

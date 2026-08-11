"""FireRed-Image-Edit runtime built on the shared Qwen Edit pipeline adapter."""

from qwen2_blackbox_runtime import (
    Qwen2AttackRuntimeAdapter,
    QwenImageEditRenderSession,
)


FIRERED_IMAGE_EDIT_MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.1"


class FireRedImageEditRenderSession(QwenImageEditRenderSession):
    MODEL_ID = FIRERED_IMAGE_EDIT_MODEL_ID
    MODEL_FAMILY = "firered-image-edit"
    DISPLAY_NAME = "FireRed Image Edit 1.1"
    OPTION_PREFIX = "firered"
    ERROR_PREFIX = "firered"
    # FireRed's official inference path uses true_cfg_scale and does not pass
    # the guidance-distillation argument.
    PASS_GUIDANCE_SCALE = False


class FireRedAttackRuntimeAdapter(Qwen2AttackRuntimeAdapter):
    MODEL_ID = FIRERED_IMAGE_EDIT_MODEL_ID
    MODEL_FAMILY = "firered-image-edit"
    DISPLAY_NAME = "FireRed Image Edit 1.1"
    OPTION_PREFIX = "firered"
    ERROR_PREFIX = "firered"
    SESSION_CLASS = FireRedImageEditRenderSession


__all__ = (
    "FIRERED_IMAGE_EDIT_MODEL_ID",
    "FireRedAttackRuntimeAdapter",
    "FireRedImageEditRenderSession",
)

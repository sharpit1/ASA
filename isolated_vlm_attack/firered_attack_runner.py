"""FireRed-Image-Edit-1.1 black-box attack runner."""

import argparse
from typing import Sequence

from firered_blackbox_runtime import (
    FIRERED_IMAGE_EDIT_MODEL_ID,
    FireRedAttackRuntimeAdapter,
)
from qwen2_attack_runner import (
    ImageEditRunnerProfile,
    apply_image_edit_runtime_args,
    build_core_cli,
    build_image_edit_parser,
    configure_image_edit_attack_mode,
    normalize_image_edit_attack_mode,
    parse_image_edit_core_args,
    run_image_edit_attack,
)


FIRERED_RUNNER_PROFILE = ImageEditRunnerProfile(
    display_name="FireRed Image Edit 1.1",
    model_id=FIRERED_IMAGE_EDIT_MODEL_ID,
    model_family="firered-image-edit",
    option_prefix="firered",
    process_title_backend="fireredEdit",
    run_name_prefix="firered_run",
    runner_script="firered_attack_runner.py",
    log_name="firered_runner",
    temp_prefix="firered_attack_input_",
    runtime_factory=FireRedAttackRuntimeAdapter,
    default_manual_seed=49,
    default_num_inference_steps=40,
    default_height=1024,
    default_width=1024,
)


def build_parser() -> argparse.ArgumentParser:
    return build_image_edit_parser(FIRERED_RUNNER_PROFILE)


def normalize_firered_attack_mode(raw: object) -> str:
    return normalize_image_edit_attack_mode(raw, profile=FIRERED_RUNNER_PROFILE)


def configure_firered_attack_mode(cfg: argparse.Namespace) -> None:
    configure_image_edit_attack_mode(cfg, profile=FIRERED_RUNNER_PROFILE)


def apply_firered_runtime_args(
    core_args: argparse.Namespace,
    cfg: argparse.Namespace,
) -> None:
    apply_image_edit_runtime_args(core_args, cfg, profile=FIRERED_RUNNER_PROFILE)


def parse_firered_core_args(
    *,
    cfg: argparse.Namespace,
    passthrough_core_args: Sequence[str],
    core_cli: Sequence[str],
):
    return parse_image_edit_core_args(
        cfg=cfg,
        passthrough_core_args=passthrough_core_args,
        core_cli=core_cli,
        profile=FIRERED_RUNNER_PROFILE,
    )


def main() -> int:
    return run_image_edit_attack(FIRERED_RUNNER_PROFILE)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FIRERED_RUNNER_PROFILE",
    "apply_firered_runtime_args",
    "build_core_cli",
    "build_parser",
    "configure_firered_attack_mode",
    "main",
    "normalize_firered_attack_mode",
    "parse_firered_core_args",
)

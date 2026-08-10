import argparse
import json
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

from PIL import Image

from attack_model_registry import validate_generator_model
from qwen2_blackbox_runtime import Qwen2AttackRuntimeAdapter
from vlm_attack_blackbox_core import (
    parse_core_args,
    run_blackbox_attack_core,
    set_process_title_from_args,
)
from attack_runner_common import (
    VictimModelAdapter,
    apply_manual_seed,
    build_core_cli as base_build_core_cli,
    build_parser as base_build_parser,
    configure_attack_mode,
    collect_sensitive_values,
    has_valid_saved_attack_image,
    iter_nips_metadata_batches,
    load_nips_ground_truth,
    load_preserved_attack_report,
    parse_bool_flag,
    prepare_model_input,
    redact_transient_paths,
    resolve_run_root,
    resolve_sample_indices,
    resolve_optional_hf_token,
    sample_clean_correct_indices,
    select_clean_correct_indices,
    validate_passthrough_core_args,
)


class ImageSaveTimer:
    def __init__(self) -> None:
        self.elapsed_seconds = 0.0
        self._original_save = None

    def start(self) -> None:
        if self._original_save is not None:
            return
        original_save = Image.Image.save
        self._original_save = original_save
        timer = self

        def timed_save(image, fp, format=None, **params):
            started_at = time.perf_counter()
            try:
                return original_save(image, fp, format=format, **params)
            finally:
                timer.elapsed_seconds += time.perf_counter() - started_at

        Image.Image.save = timed_save

    def stop(self) -> None:
        if self._original_save is None:
            return
        Image.Image.save = self._original_save
        self._original_save = None


def elapsed_excluding_image_saves(start_time: float, image_save_timer: ImageSaveTimer) -> float:
    elapsed = time.perf_counter() - start_time
    return max(0.0, elapsed - image_save_timer.elapsed_seconds)


def add_total_elapsed_to_report(
    report_path: Path,
    total_elapsed_seconds: float,
    image_save_elapsed_seconds: float,
) -> None:
    try:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["total_elapsed_seconds"] = round(float(total_elapsed_seconds), 3)
        payload["image_save_elapsed_seconds"] = round(float(image_save_elapsed_seconds), 3)
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(
            "[qwen2_runner] WARNING: failed to add timing fields to report "
            f"{report_path}: {type(exc).__name__}: {exc}"
        )


def _collect_report_runtime_errors(value, path="$"):
    error_keys = {
        "vlm_error",
        "naturalness_error",
        "attack_success_naturalness_error",
    }
    allowed_sentinel = "skipped_text_eval_due_to_cwor_success"
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in error_keys:
                text = str(item or "").strip()
                if text and text != allowed_sentinel:
                    found.append({"path": item_path, "error": text})
            found.extend(_collect_report_runtime_errors(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(
                _collect_report_runtime_errors(item, f"{path}[{index}]")
            )
    return found


def load_resumable_report(
    report_path: Path,
    *,
    cfg,
    sample_index: int,
    image_id: str,
    true_label: int,
    target_label: int,
    sample_dir: Path,
):
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") == "failed":
        return None
    if payload.get("error") or payload.get("traceback"):
        return None
    if not isinstance(payload.get("final_attack_success"), bool):
        return None
    if _collect_report_runtime_errors(payload):
        return None

    report_args = payload.get("args")
    if not isinstance(report_args, dict):
        return None
    expected_args = {
        "model_path": str(cfg.model_path),
        "attack_mode": str(cfg.attack_mode),
        "strategy_mllm_mode": str(cfg.strategy_mllm_mode),
        "gcg_scene_vocab_enabled_strategies": str(
            cfg.gcg_scene_vocab_enabled_strategies
        ),
        "gcg_scene_vocab_prompts_per_strategy": int(
            cfg.gcg_scene_vocab_prompts_per_strategy
        ),
        "gcg_scene_llm_do_sample": bool(cfg.gcg_scene_llm_do_sample),
        "gcg_eval_naturalness_on_attack_success": bool(
            cfg.gcg_eval_naturalness_on_attack_success
        ),
        "gcg_eval_naturalness_llm_thinking": bool(
            cfg.gcg_eval_naturalness_llm_thinking
        ),
        "seed": int(cfg.manual_seed),
    }
    if any(report_args.get(key) != expected for key, expected in expected_args.items()):
        return None

    strategy = payload.get("strategy_generator")
    naturalness = payload.get("naturalness_verifier")
    if not isinstance(strategy, dict) or not isinstance(naturalness, dict):
        return None
    if (
        strategy.get("mode") != str(cfg.strategy_mllm_mode)
        or strategy.get("do_sample") is not bool(cfg.gcg_scene_llm_do_sample)
        or naturalness.get("mode") != str(cfg.strategy_mllm_mode)
        or naturalness.get("enabled") is not True
        or naturalness.get("do_sample") is not bool(cfg.scene_vlm_do_sample)
    ):
        return None

    strategy_steps = 0
    expected_slot_count = (
        0
        if str(cfg.gcg_scene_vocab_enabled_strategies).strip().lower() == "none"
        else 3
    )
    for entry in payload.get("history", []):
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("raw_vlm_answer") or "").strip():
            continue
        strategy_steps += 1
        if int(entry.get("strategy_query_accounting_version", 0) or 0) != 3:
            return None
        if int(entry.get("strategy_slot_count", 0) or 0) != expected_slot_count:
            return None
        duplicate_count = int(
            entry.get("strategy_duplicate_skipped_count", 0) or 0
        )
        duplicate_queries = int(
            entry.get("strategy_duplicate_query_count", 0) or 0
        )
        if duplicate_queries != duplicate_count:
            return None
    if strategy_steps == 0:
        return None

    result = dict(payload)
    result.update(
        {
            "sample_index": int(sample_index),
            "image_id": str(image_id),
            "true_label": int(true_label),
            "target_label": int(target_label),
            "sample_dir": str(sample_dir),
            "resumed_from_existing_report": True,
        }
    )
    return result


def build_resume_validation_config(cfg, passthrough_core_args):
    core_args, core_unknown = parse_core_args(
        ["--model_path", str(cfg.model_path), *list(passthrough_core_args)]
    )
    if core_unknown:
        raise ValueError(
            "unsupported passthrough args while building resume validation "
            f"config: {core_unknown}"
        )
    return SimpleNamespace(
        model_path=str(cfg.model_path),
        attack_mode=str(cfg.attack_mode),
        strategy_mllm_mode=str(cfg.strategy_mllm_mode),
        gcg_scene_vocab_enabled_strategies=str(
            cfg.gcg_scene_vocab_enabled_strategies
        ),
        gcg_scene_vocab_prompts_per_strategy=int(
            cfg.gcg_scene_vocab_prompts_per_strategy
        ),
        gcg_scene_llm_do_sample=bool(core_args.gcg_scene_llm_do_sample),
        gcg_eval_naturalness_on_attack_success=bool(
            core_args.gcg_eval_naturalness_on_attack_success
        ),
        gcg_eval_naturalness_llm_thinking=bool(
            core_args.gcg_eval_naturalness_llm_thinking
        ),
        scene_vlm_do_sample=bool(core_args.scene_vlm_do_sample),
        manual_seed=int(cfg.manual_seed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_build_parser()
    parser.description = "Qwen Image Edit 2511 black-box attack runner."
    parser.set_defaults(gcg_candidate_source="gemma_scene_vocab")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--cpu_offload", type=parse_bool_flag, default=False)
    parser.add_argument("--qwen_true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--qwen_negative_prompt", type=str, default=" ")
    parser.add_argument("--qwen_num_images_per_prompt", type=int, choices=[1], default=1)
    parser.add_argument(
        "--qwen_batch_size",
        type=int,
        default=1,
        help="Experimental single-GPU prompt batch size; 1 keeps upstream sequential inference.",
    )
    parser.add_argument(
        "--qwen_batch_fallback",
        type=parse_bool_flag,
        default=True,
        help="Fall back to sequential Qwen rendering if experimental batching fails.",
    )
    parser.add_argument(
        "--resume_existing_reports",
        type=parse_bool_flag,
        default=False,
    )
    return parser


def normalize_qwen_attack_mode(raw: object) -> str:
    token = str(raw or "vlm").strip().lower()
    if token not in {"vlm", "and"}:
        raise ValueError("Qwen Image Edit runner supports attack_mode=vlm or and.")
    return token


def configure_qwen_attack_mode(cfg: argparse.Namespace) -> None:
    cfg.attack_mode = normalize_qwen_attack_mode(getattr(cfg, "attack_mode", "vlm"))
    configure_attack_mode(cfg)
    cfg.qwen_attack_mode = cfg.attack_mode


def apply_qwen_runtime_args(core_args: argparse.Namespace, cfg: argparse.Namespace) -> None:
    core_args.cpu_offload = bool(getattr(cfg, "cpu_offload", False))
    core_args.qwen_true_cfg_scale = float(getattr(cfg, "qwen_true_cfg_scale", 4.0))
    core_args.qwen_negative_prompt = str(getattr(cfg, "qwen_negative_prompt", " ") or " ")
    core_args.qwen_num_images_per_prompt = int(getattr(cfg, "qwen_num_images_per_prompt", 1))
    core_args.qwen_batch_size = int(getattr(cfg, "qwen_batch_size", 1))
    core_args.qwen_batch_fallback = bool(getattr(cfg, "qwen_batch_fallback", True))
    core_args.gcg_skip_initial_render = True

    qwen_attack_mode = str(getattr(cfg, "qwen_attack_mode", "vlm") or "vlm").strip().lower()
    core_args.qwen_attack_mode = qwen_attack_mode
    core_args.qwen_edit_prompt_mode = True
    core_args.qwen_strategy_and_enable = bool(qwen_attack_mode == "and")
    core_args.attack_mode = qwen_attack_mode


def build_core_cli(
    *,
    cfg,
    hf_token: str,
    sample_image_path: Path,
    output_path: Path,
    report_path: Path,
    classifier_label: int,
    sample_target_label,
    sample_run_name: str,
) -> List[str]:
    cli = base_build_core_cli(
        cfg=cfg,
        hf_token=hf_token,
        sample_image_path=sample_image_path,
        output_path=output_path,
        report_path=report_path,
        classifier_label=classifier_label,
        sample_target_label=sample_target_label,
        sample_run_name=sample_run_name,
    )
    cli.extend(
        [
            "--model_path",
            str(cfg.model_path),
            "--max_sequence_length",
            str(int(cfg.max_sequence_length)),
            "--guidance_scale",
            str(float(cfg.guidance_scale)),
        ]
    )
    return cli


def parse_qwen_core_args(
    *,
    cfg: argparse.Namespace,
    passthrough_core_args: Sequence[str],
    core_cli: Sequence[str],
):
    core_args, core_unknown = parse_core_args([*list(core_cli), *list(passthrough_core_args)])
    apply_qwen_runtime_args(core_args, cfg)
    return core_args, core_unknown


def main() -> int:
    parser = build_parser()
    cfg, passthrough_core_args = parser.parse_known_args()
    validate_passthrough_core_args(passthrough_core_args)
    configure_qwen_attack_mode(cfg)
    resume_validation_cfg = (
        build_resume_validation_config(cfg, passthrough_core_args)
        if bool(cfg.resume_existing_reports)
        else None
    )

    validate_generator_model(cfg.model_path, expected_family="qwen-image-edit")

    cfg.classifier_mode = "black-box"
    cfg.classifier_name = str(cfg.victim_model)
    cfg.process_title_backend = "qwenEdit"
    set_process_title_from_args(cfg)

    if cfg.manual_seed is not None:
        apply_manual_seed(int(cfg.manual_seed))

    hf_token = resolve_optional_hf_token(cfg.hf_token)
    sensitive_values = collect_sensitive_values(hf_token, cfg.wandb_api_key)
    run_name = str(cfg.run_name or "").strip() or datetime.now().strftime("qwen2_run_%Y%m%d_%H%M%S")
    cfg.run_name = run_name
    if int(cfg.height) <= 0:
        cfg.height = int(cfg.image_size)
    if int(cfg.width) <= 0:
        cfg.width = int(cfg.image_size)
    if int(cfg.max_sequence_length) < 1:
        raise ValueError("--max_sequence_length must be >= 1")
    if int(cfg.qwen_num_images_per_prompt) != 1:
        raise ValueError("--qwen_num_images_per_prompt is fixed at 1")
    if int(cfg.qwen_batch_size) < 1:
        raise ValueError("--qwen_batch_size must be >= 1")
    dataset_root = Path(str(cfg.dataset_root)).expanduser().resolve()
    images_csv = dataset_root / "images.csv"
    images_dir = dataset_root / "images"
    if not images_csv.is_file():
        raise FileNotFoundError(f"dataset csv not found: {images_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"dataset images dir not found: {images_dir}")

    batchsize = max(1, int(cfg.batchsize))
    image_ids, true_labels, target_labels = load_nips_ground_truth(dataset_root)
    data_loader = iter_nips_metadata_batches(
        image_ids,
        true_labels,
        target_labels,
        batch_size=batchsize,
    )
    total_available = min(1000, len(image_ids))
    start_index = max(0, int(cfg.start_index))
    if start_index >= total_available:
        raise ValueError(f"start_index {start_index} is out of range (available={total_available})")

    if cfg.end_index is not None:
        requested_end_index = int(cfg.end_index)
        if requested_end_index < start_index:
            raise ValueError(
                f"end_index {requested_end_index} must be >= start_index {start_index}"
            )
        end_index = min(total_available, requested_end_index)
    else:
        max_samples = int(cfg.max_samples)
        end_index = total_available if max_samples <= 0 else min(total_available, start_index + max_samples)

    sample_indices = resolve_sample_indices(cfg.sample_indices, cfg.sample_indices_file)
    sample_index_set = set(sample_indices) if sample_indices else None
    if sample_indices:
        out_of_range = [idx for idx in sample_indices if idx >= total_available]
        if out_of_range:
            raise ValueError(
                f"sample_indices contains out-of-range values: {out_of_range[:10]} "
                f"(available={total_available})"
            )
        selected_in_range = [idx for idx in sample_indices if start_index <= idx < end_index]
        if not selected_in_range:
            raise ValueError(
                "sample_indices selected no samples inside "
                f"start_index={start_index}, end_index={end_index}"
            )
        print(f"[qwen2_runner] sample_indices selected={len(selected_in_range)} file={cfg.sample_indices_file or ''}")

    run_root = resolve_run_root(cfg, run_name)
    run_root.mkdir(parents=True, exist_ok=True)

    victim = VictimModelAdapter(
        model_name=str(cfg.victim_model),
        device=str(cfg.device),
        objective_mode=str(cfg.classifier_objective),
    )
    candidate_indices = [
        idx
        for idx in range(start_index, end_index)
        if sample_index_set is None or idx in sample_index_set
    ]
    attack_indices = set(candidate_indices)
    clean_correct_pool_indices: set[int] = set()
    clean_filter_results: List[dict] = []
    clean_correct_sample_size = int(cfg.clean_correct_sample_size)
    clean_correct_sample_seed = cfg.clean_correct_sample_seed
    if clean_correct_sample_size < 0:
        raise ValueError("--clean_correct_sample_size must be >= 0")
    if clean_correct_sample_size > 0 and not bool(cfg.attack_only_clean_correct):
        raise ValueError(
            "--clean_correct_sample_size requires --attack_only_clean_correct true"
        )
    if clean_correct_sample_size > 0 and clean_correct_sample_seed is None:
        raise ValueError(
            "--clean_correct_sample_seed is required when "
            "--clean_correct_sample_size is positive"
        )
    if clean_correct_sample_size == 0 and clean_correct_sample_seed is not None:
        raise ValueError(
            "--clean_correct_sample_seed requires a positive "
            "--clean_correct_sample_size"
        )
    if bool(cfg.attack_only_clean_correct):
        attack_indices, clean_filter_results = select_clean_correct_indices(
            victim=victim,
            images_dir=images_dir,
            image_ids=image_ids,
            true_labels=true_labels,
            candidate_indices=candidate_indices,
            batch_size=batchsize,
        )
        clean_correct_pool_indices = set(attack_indices)
        if clean_correct_sample_size > 0:
            attack_indices = sample_clean_correct_indices(
                clean_correct_pool_indices,
                sample_size=clean_correct_sample_size,
                seed=int(clean_correct_sample_seed),
            )
        clean_filter_error_count = sum(
            item.get("status") == "error" for item in clean_filter_results
        )
        print(
            "[qwen2_runner] clean filter "
            f"examined={len(clean_filter_results)} "
            f"selected={len(clean_correct_pool_indices)} "
            f"skipped={len(clean_filter_results) - len(clean_correct_pool_indices)} "
            f"errors={clean_filter_error_count}"
        )
        if clean_correct_sample_size > 0:
            print(
                "[qwen2_runner] clean-correct sample "
                f"pool={len(clean_correct_pool_indices)} "
                f"selected={len(attack_indices)} "
                f"seed={int(clean_correct_sample_seed)}"
            )

    results: List[dict] = []
    success_count = 0
    fail_count = 0
    attack_success_count = 0
    attack_failure_count = 0
    attack_unknown_count = 0
    runtime = Qwen2AttackRuntimeAdapter()
    try:
        stop_requested = False
        for batch_idx, (image_id_batch, label_ori_batch, label_tar_batch) in enumerate(data_loader):
            base_idx = batch_idx * batchsize
            batch_len = len(label_ori_batch)
            for in_batch_idx in range(batch_len):
                idx = base_idx + in_batch_idx
                if idx < start_index:
                    continue
                if idx >= end_index:
                    stop_requested = True
                    break
                if sample_index_set is not None and idx not in sample_index_set:
                    continue
                if idx not in attack_indices:
                    continue

                image_id = str(image_id_batch[in_batch_idx])
                true_label = int(label_ori_batch[in_batch_idx])
                target_label = int(label_tar_batch[in_batch_idx])
                src_image_path = images_dir / f"{image_id}.png"
                sample_dir = run_root / f"sample_{idx:04d}"
                output_path = sample_dir / "images" / "attack_success.png"
                report_path = sample_dir / "report.json"
                sample_run_name = f"{run_name}_{image_id}"
                sample_start_time = time.perf_counter()
                image_save_timer = ImageSaveTimer()

                if bool(cfg.resume_existing_reports):
                    resumed_result = load_resumable_report(
                        report_path,
                        cfg=resume_validation_cfg,
                        sample_index=idx,
                        image_id=image_id,
                        true_label=true_label,
                        target_label=target_label,
                        sample_dir=sample_dir,
                    )
                    if resumed_result is not None:
                        results.append(resumed_result)
                        success_count += 1
                        if resumed_result.get("final_attack_success") is True:
                            attack_success_count += 1
                        else:
                            attack_failure_count += 1
                        print(
                            f"[qwen2_runner] resume sample {idx:04d} "
                            "from verified existing report"
                        )
                        continue

                print(f"[qwen2_runner] sample {idx:04d} image_id={image_id} label={true_label}")
                sample_dir.mkdir(parents=True, exist_ok=True)
                if output_path.is_file():
                    output_path.unlink()

                if not src_image_path.is_file():
                    fail_count += 1
                    total_elapsed_seconds = elapsed_excluding_image_saves(sample_start_time, image_save_timer)
                    err_payload = {
                        "status": "failed",
                        "sample_index": int(idx),
                        "image_id": image_id,
                        "true_label": int(true_label),
                        "error": f"missing_source_image:{src_image_path}",
                        "total_elapsed_seconds": round(float(total_elapsed_seconds), 3),
                        "image_save_elapsed_seconds": round(float(image_save_timer.elapsed_seconds), 3),
                    }
                    report_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    results.append(err_payload)
                    continue

                transient_input_paths: List[object] = []
                try:
                    image_save_timer.start()
                    victim.set_label(true_label)
                    with tempfile.TemporaryDirectory(prefix="qwen2_attack_input_") as temp_dir:
                        qwen_input_path = Path(temp_dir) / "condition.png"
                        transient_input_paths = [temp_dir, qwen_input_path]
                        prepare_model_input(
                            src_image_path,
                            qwen_input_path,
                            height=int(cfg.height),
                            width=int(cfg.width),
                        )
                        core_cli = build_core_cli(
                            cfg=cfg,
                            hf_token=hf_token,
                            sample_image_path=qwen_input_path,
                            output_path=output_path,
                            report_path=report_path,
                            classifier_label=true_label,
                            sample_target_label=target_label,
                            sample_run_name=sample_run_name,
                        )
                        core_args, core_unknown = parse_qwen_core_args(
                            cfg=cfg,
                            passthrough_core_args=passthrough_core_args,
                            core_cli=core_cli,
                        )
                        runtime.setup(
                            args=core_args,
                            has_input_image=bool(core_args.input_img_path and str(core_args.input_img_path).strip()),
                        )
                        result = run_blackbox_attack_core(
                            args=core_args,
                            unknown_args=core_unknown,
                            classifier=victim,
                            runtime=runtime,
                            manage_runtime=False,
                        )
                    total_elapsed_seconds = elapsed_excluding_image_saves(sample_start_time, image_save_timer)
                    result["total_elapsed_seconds"] = round(float(total_elapsed_seconds), 3)
                    result["image_save_elapsed_seconds"] = round(float(image_save_timer.elapsed_seconds), 3)
                    add_total_elapsed_to_report(
                        report_path,
                        total_elapsed_seconds,
                        image_save_timer.elapsed_seconds,
                    )
                    result["sample_index"] = int(idx)
                    result["image_id"] = image_id
                    result["true_label"] = int(true_label)
                    result["target_label"] = int(target_label)
                    result["sample_dir"] = str(sample_dir)
                    results.append(result)
                    success_count += 1
                    if result.get("final_attack_success") is True:
                        attack_success_count += 1
                    elif result.get("final_attack_success") is False:
                        attack_failure_count += 1
                    else:
                        attack_unknown_count += 1
                except Exception as exc:
                    fail_count += 1
                    total_elapsed_seconds = elapsed_excluding_image_saves(sample_start_time, image_save_timer)
                    safe_error = redact_transient_paths(
                        exc,
                        transient_input_paths,
                        sensitive_values=sensitive_values,
                    )
                    safe_traceback = redact_transient_paths(
                        traceback.format_exc(),
                        transient_input_paths,
                        sensitive_values=sensitive_values,
                    )
                    preserved_success = has_valid_saved_attack_image(output_path)
                    if preserved_success:
                        attack_success_count += 1
                    fail_payload = {
                        "status": "failed",
                        "sample_index": int(idx),
                        "image_id": image_id,
                        "true_label": int(true_label),
                        "target_label": int(target_label),
                        "sample_dir": str(sample_dir),
                        "total_elapsed_seconds": round(float(total_elapsed_seconds), 3),
                        "image_save_elapsed_seconds": round(float(image_save_timer.elapsed_seconds), 3),
                        "error_type": type(exc).__name__,
                        "error": safe_error,
                        "traceback": safe_traceback,
                    }
                    if preserved_success:
                        fail_payload["attack_success_before_failure"] = True
                        fail_payload["attack_success_image_path"] = "images/attack_success.png"
                        partial_report = load_preserved_attack_report(report_path)
                        if partial_report is not None:
                            fail_payload["partial_core_report"] = partial_report
                            for key in (
                                "attack_success_query_count",
                                "attack_success_candidate",
                                "attack_success_float32_path",
                                "attack_success_float32_error",
                            ):
                                if key in partial_report:
                                    fail_payload[key] = partial_report[key]
                    report_path.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    results.append(fail_payload)
                    print(f"[qwen2_runner] sample {idx:04d} failed: {type(exc).__name__}: {safe_error}")
                finally:
                    image_save_timer.stop()
            if stop_requested:
                break
    finally:
        runtime.close()

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_name": str(cfg.dataset_name),
        "victim_model": str(cfg.victim_model),
        "runner": "qwen2_attack_runner.py",
        "model_path": str(cfg.model_path),
        "qwen_attack_mode": str(getattr(cfg, "qwen_attack_mode", "vlm")),
        "qwen_batch_size": int(getattr(cfg, "qwen_batch_size", 1)),
        "qwen_batch_fallback": bool(getattr(cfg, "qwen_batch_fallback", True)),
        "run_name": run_name,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "sample_indices_file": str(cfg.sample_indices_file or ""),
        "sample_indices": sample_indices,
        "attack_only_clean_correct": bool(cfg.attack_only_clean_correct),
        "clean_correct_sample_size": int(clean_correct_sample_size),
        "clean_correct_sample_seed": (
            int(clean_correct_sample_seed)
            if clean_correct_sample_seed is not None
            else None
        ),
        "clean_correct_pool_indices": sorted(clean_correct_pool_indices),
        "clean_correct_selected_indices": (
            sorted(attack_indices) if cfg.attack_only_clean_correct else []
        ),
        "clean_correct_sampled_count": (
            int(len(attack_indices)) if cfg.attack_only_clean_correct else 0
        ),
        "clean_filter_examined_count": int(len(clean_filter_results)),
        "clean_filter_passed_count": int(len(clean_correct_pool_indices))
        if cfg.attack_only_clean_correct
        else 0,
        "clean_filter_skipped_count": int(
            len(clean_filter_results) - len(clean_correct_pool_indices)
        )
        if cfg.attack_only_clean_correct
        else 0,
        "clean_filter_error_count": int(
            sum(item.get("status") == "error" for item in clean_filter_results)
        ),
        "clean_filter_results": clean_filter_results,
        "total_processed": int(len(results)),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "attack_success_count": int(attack_success_count),
        "attack_failure_count": int(attack_failure_count),
        "attack_unknown_count": int(attack_unknown_count),
        "run_root": str(run_root),
        "results": results,
    }
    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("[qwen2_runner] completed")
    print(f"[qwen2_runner] run_root={run_root}")
    print(f"[qwen2_runner] summary={summary_path}")
    print(f"[qwen2_runner] success={success_count} fail={fail_count}")
    return 0 if fail_count == 0 and summary["clean_filter_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

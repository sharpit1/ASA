import argparse
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from attack_model_registry import validate_generator_model
from bernini_blackbox_runtime import BerniniAttackRuntimeAdapter
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
    resolve_optional_hf_token,
    resolve_run_root,
    resolve_sample_indices,
    select_clean_correct_indices,
    validate_passthrough_core_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = base_build_parser()
    parser.description = "Bernini black-box attack runner."

    # Consume legacy model_path configs so they do not leak into the core as
    # Flux2 model paths. Bernini uses --bernini_config instead.
    parser.add_argument("--model_path", type=str, default="bernini")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--cpu_offload", type=parse_bool_flag, default=False)

    parser.add_argument("--bernini_root", type=str, default="third_party/bernini")
    parser.add_argument(
        "--bernini_config",
        type=str,
        default="third_party/bernini/Bernini-R-Diffusers",
    )
    parser.add_argument("--bernini_high_noise_ckpt", type=str, default="")
    parser.add_argument("--bernini_low_noise_ckpt", type=str, default="")
    parser.add_argument("--bernini_task_type", type=str, default="i2i")
    parser.add_argument("--bernini_guidance_mode", type=str, default="v2v")
    parser.add_argument("--bernini_num_frames", type=int, default=1)
    parser.add_argument("--bernini_max_image_size", type=int, default=848)
    parser.add_argument("--bernini_use_unipc", type=parse_bool_flag, default=True)
    parser.add_argument("--bernini_use_src_tgt_id", type=parse_bool_flag, default=True)
    parser.add_argument("--bernini_neg_prompt", type=str, default="")
    parser.add_argument("--bernini_system_prompt", type=str, default="")
    parser.add_argument("--bernini_omega_V", type=float, default=1.25)
    parser.add_argument("--bernini_omega_I", type=float, default=4.5)
    parser.add_argument("--bernini_omega_TI", type=float, default=4.0)
    parser.add_argument("--bernini_omega_scale", type=float, default=0.8)
    parser.add_argument("--bernini_flow_shift", type=float, default=5.0)
    parser.add_argument("--bernini_fps", type=int, default=16)
    parser.add_argument("--bernini_eta", type=float, default=0.5)
    parser.add_argument("--bernini_norm_threshold", type=float, nargs="+", default=[50.0, 50.0, 50.0])
    parser.add_argument("--bernini_momentum", type=float, default=0.0)
    parser.add_argument("--bernini_batch_render", type=parse_bool_flag, default=True)
    parser.add_argument("--bernini_batch_fallback", type=parse_bool_flag, default=True)
    parser.add_argument("--bernini_batch_size", type=int, default=0)
    return parser


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
            "bernini",
            "--max_sequence_length",
            str(int(cfg.max_sequence_length)),
            "--guidance_scale",
            str(float(cfg.guidance_scale)),
        ]
    )
    return cli


def apply_bernini_runtime_args(core_args: argparse.Namespace, cfg: argparse.Namespace) -> None:
    core_args.model_path = "bernini"
    core_args.cpu_offload = bool(getattr(cfg, "cpu_offload", False))
    core_args.bernini_edit_prompt_mode = True
    core_args.bernini_root = str(getattr(cfg, "bernini_root", "third_party/bernini"))
    core_args.bernini_config = str(
        getattr(cfg, "bernini_config", "third_party/bernini/Bernini-R-Diffusers")
    )
    core_args.bernini_high_noise_ckpt = str(getattr(cfg, "bernini_high_noise_ckpt", "") or "")
    core_args.bernini_low_noise_ckpt = str(getattr(cfg, "bernini_low_noise_ckpt", "") or "")
    core_args.bernini_task_type = str(getattr(cfg, "bernini_task_type", "i2i") or "i2i")
    core_args.bernini_guidance_mode = str(getattr(cfg, "bernini_guidance_mode", "v2v") or "v2v")
    core_args.bernini_num_frames = int(getattr(cfg, "bernini_num_frames", 1))
    core_args.bernini_max_image_size = int(getattr(cfg, "bernini_max_image_size", 848))
    core_args.bernini_use_unipc = bool(getattr(cfg, "bernini_use_unipc", True))
    core_args.bernini_use_src_tgt_id = bool(getattr(cfg, "bernini_use_src_tgt_id", True))
    core_args.bernini_neg_prompt = str(getattr(cfg, "bernini_neg_prompt", "") or "")
    core_args.bernini_system_prompt = str(getattr(cfg, "bernini_system_prompt", "") or "")
    core_args.bernini_omega_V = float(getattr(cfg, "bernini_omega_V", 1.25))
    core_args.bernini_omega_I = float(getattr(cfg, "bernini_omega_I", 4.5))
    core_args.bernini_omega_TI = float(getattr(cfg, "bernini_omega_TI", 4.0))
    core_args.bernini_omega_scale = float(getattr(cfg, "bernini_omega_scale", 0.8))
    core_args.bernini_flow_shift = float(getattr(cfg, "bernini_flow_shift", 5.0))
    core_args.bernini_fps = int(getattr(cfg, "bernini_fps", 16))
    core_args.bernini_eta = float(getattr(cfg, "bernini_eta", 0.5))
    core_args.bernini_norm_threshold = list(getattr(cfg, "bernini_norm_threshold", [50.0, 50.0, 50.0]))
    core_args.bernini_momentum = float(getattr(cfg, "bernini_momentum", 0.0))
    core_args.bernini_batch_render = parse_bool_flag(getattr(cfg, "bernini_batch_render", True))
    core_args.bernini_batch_fallback = parse_bool_flag(getattr(cfg, "bernini_batch_fallback", True))
    core_args.bernini_batch_size = int(getattr(cfg, "bernini_batch_size", 0) or 0)
    if core_args.bernini_num_frames != 1:
        raise ValueError("--bernini_num_frames must be 1 for image attack evaluation.")


def parse_bernini_core_args(
    *,
    cfg: argparse.Namespace,
    passthrough_core_args: Sequence[str],
    core_cli: Sequence[str],
):
    core_args, core_unknown = parse_core_args([*list(core_cli), *list(passthrough_core_args)])
    apply_bernini_runtime_args(core_args, cfg)
    return core_args, core_unknown


def main() -> int:
    parser = build_parser()
    cfg, passthrough_core_args = parser.parse_known_args()
    validate_passthrough_core_args(passthrough_core_args)
    configure_attack_mode(cfg)
    validate_generator_model(cfg.model_path, expected_family="bernini")

    cfg.classifier_mode = "black-box"
    cfg.classifier_name = str(cfg.victim_model)
    cfg.process_title_backend = "bernini"
    set_process_title_from_args(cfg)

    if cfg.manual_seed is not None:
        apply_manual_seed(int(cfg.manual_seed))

    hf_token = resolve_optional_hf_token(cfg.hf_token)
    sensitive_values = collect_sensitive_values(hf_token, cfg.wandb_api_key)
    run_name = str(cfg.run_name or "").strip()
    if not run_name:
        run_name = datetime.now().strftime("bernini_run_%Y%m%d_%H%M%S")
    cfg.run_name = run_name
    if int(cfg.height) <= 0:
        cfg.height = int(cfg.image_size)
    if int(cfg.width) <= 0:
        cfg.width = int(cfg.image_size)
    if int(cfg.max_sequence_length) < 1:
        raise ValueError("--max_sequence_length must be >= 1")
    if int(cfg.bernini_num_frames) != 1:
        raise ValueError("--bernini_num_frames must be 1 for image attack evaluation.")
    if int(cfg.bernini_max_image_size) < 1:
        raise ValueError("--bernini_max_image_size must be >= 1")
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
        print(f"[bernini_runner] sample_indices selected={len(selected_in_range)} file={cfg.sample_indices_file or ''}")

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
    clean_filter_results: List[dict] = []
    if bool(cfg.attack_only_clean_correct):
        attack_indices, clean_filter_results = select_clean_correct_indices(
            victim=victim,
            images_dir=images_dir,
            image_ids=image_ids,
            true_labels=true_labels,
            candidate_indices=candidate_indices,
            batch_size=batchsize,
        )
        clean_filter_error_count = sum(
            item.get("status") == "error" for item in clean_filter_results
        )
        print(
            "[bernini_runner] clean filter "
            f"examined={len(clean_filter_results)} selected={len(attack_indices)} "
            f"skipped={len(clean_filter_results) - len(attack_indices)} "
            f"errors={clean_filter_error_count}"
        )

    results: List[dict] = []
    success_count = 0
    fail_count = 0
    attack_success_count = 0
    attack_failure_count = 0
    attack_unknown_count = 0
    runtime = BerniniAttackRuntimeAdapter()
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

                print(f"[bernini_runner] sample {idx:04d} image_id={image_id} label={true_label}")
                sample_dir.mkdir(parents=True, exist_ok=True)
                if output_path.is_file():
                    output_path.unlink()

                if not src_image_path.is_file():
                    fail_count += 1
                    err_payload = {
                        "status": "failed",
                        "sample_index": int(idx),
                        "image_id": image_id,
                        "true_label": int(true_label),
                        "error": f"missing_source_image:{src_image_path}",
                    }
                    report_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    results.append(err_payload)
                    continue

                transient_input_paths: List[object] = []
                try:
                    victim.set_label(true_label)
                    with tempfile.TemporaryDirectory(prefix="bernini_attack_input_") as temp_dir:
                        bernini_input_path = Path(temp_dir) / "condition.png"
                        transient_input_paths = [temp_dir, bernini_input_path]
                        prepare_model_input(
                            src_image_path,
                            bernini_input_path,
                            height=int(cfg.height),
                            width=int(cfg.width),
                        )
                        core_cli = build_core_cli(
                            cfg=cfg,
                            hf_token=hf_token,
                            sample_image_path=bernini_input_path,
                            output_path=output_path,
                            report_path=report_path,
                            classifier_label=true_label,
                            sample_target_label=target_label,
                            sample_run_name=sample_run_name,
                        )
                        core_args, core_unknown = parse_bernini_core_args(
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
                    print(f"[bernini_runner] sample {idx:04d} failed: {type(exc).__name__}: {safe_error}")
            if stop_requested:
                break
    finally:
        runtime.close()

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_name": str(cfg.dataset_name),
        "victim_model": str(cfg.victim_model),
        "runner": "bernini_attack_runner.py",
        "bernini_config": str(cfg.bernini_config),
        "bernini_task_type": str(cfg.bernini_task_type),
        "bernini_guidance_mode": str(cfg.bernini_guidance_mode),
        "run_name": run_name,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "sample_indices_file": str(cfg.sample_indices_file or ""),
        "sample_indices": sample_indices,
        "attack_only_clean_correct": bool(cfg.attack_only_clean_correct),
        "clean_filter_examined_count": int(len(clean_filter_results)),
        "clean_filter_passed_count": int(len(attack_indices))
        if cfg.attack_only_clean_correct
        else 0,
        "clean_filter_skipped_count": int(
            len(clean_filter_results) - len(attack_indices)
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

    print("[bernini_runner] completed")
    print(f"[bernini_runner] run_root={run_root}")
    print(f"[bernini_runner] summary={summary_path}")
    print(f"[bernini_runner] success={success_count} fail={fail_count}")
    return 0 if fail_count == 0 and summary["clean_filter_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

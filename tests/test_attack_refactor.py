from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
import yaml
from PIL import Image

from attack_model_registry import (
    FLUX2_KLEIN_MODEL_IDS,
    QWEN_IMAGE_EDIT_MODEL_IDS,
    validate_generator_model,
)
from attack_runner_common import (
    VictimModelAdapter,
    configure_attack_mode,
    load_preserved_attack_report,
    normalize_attack_mode as normalize_runner_attack_mode,
)
from vlm_attack_blackbox_core import (
    normalize_attack_mode as normalize_core_attack_mode,
    parse_core_args,
    run_blackbox_attack_core,
    save_evaluated_attack_image,
    write_report,
)


class AttackModeRefactorTests(unittest.TestCase):
    def test_all_supported_presets_use_only_registered_modes_and_models(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "configs"
        patterns = (
            "flux2_vlm_attack_nips*.yaml",
            "flux2_and_attack_nips*.yaml",
            "qwen_edit_vlm_*.yaml",
            "qwen_edit_and_*.yaml",
            "bernini_vlm_attack_nips*.yaml",
            "bernini_and_attack_nips*.yaml",
        )
        paths = sorted({path for pattern in patterns for path in config_root.glob(pattern)})
        self.assertEqual(len(paths), 32)

        forbidden_exact = {
            "inversion_prompt",
            "run_mode",
            "fixed_prompt",
            "latent_nudging_scalar",
            "gcg_early_stop_on_cwor_success_only",
        }
        for path in paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                run_args = config.get("run_args", {}) or {}
                self.assertIn(run_args.get("attack_mode"), {"vlm", "and"})
                for key in [*config.keys(), *run_args.keys()]:
                    key_text = str(key)
                    self.assertNotIn(key_text, forbidden_exact)
                    self.assertFalse(key_text.startswith("cwor_"))
                    self.assertFalse(key_text.startswith("flux2_strategy_cwor_"))

                if path.name.startswith("flux2_"):
                    self.assertEqual(config.get("script_path"), "flux2_attack_runner.py")
                    validate_generator_model(
                        run_args.get("model_path"),
                        expected_family="flux2-klein",
                    )
                elif path.name.startswith("qwen_"):
                    self.assertEqual(config.get("runner_variant"), "qwen2")
                    validate_generator_model(
                        run_args.get("model_path"),
                        expected_family="qwen-image-edit",
                    )
                else:
                    self.assertEqual(config.get("script_path"), "bernini_attack_runner.py")

    def test_only_vlm_and_and_modes_are_accepted(self) -> None:
        normalizers = (normalize_runner_attack_mode, normalize_core_attack_mode)
        for normalize in normalizers:
            with self.subTest(normalizer=normalize.__module__, supported="vlm"):
                self.assertEqual(normalize("vlm"), "vlm")
            with self.subTest(normalizer=normalize.__module__, supported="AND"):
                self.assertEqual(normalize("AND"), "and")

            for unsupported in ("ic", "cwor", "gs", "legacy", "vlm_and"):
                with self.subTest(normalizer=normalize.__module__, unsupported=unsupported):
                    with self.assertRaises(ValueError):
                        normalize(unsupported)

    def test_configure_attack_mode_enforces_and_invariants(self) -> None:
        cfg = argparse.Namespace(
            attack_mode="AND",
            cwor_enable=False,
            cwor_reference_mode="best_candidate",
            cwor_mode="target",
            cwor_target_label=17,
            cwor_embed_inject_mode="clip",
            cwor_feedback_merge_mode="step_prompt_weighted",
            cwor_accumulate_secondary_ortho=True,
            cwor_accumulate_update_if_improved_only=True,
            cwor_accumulate_delta_use_basis_logit_without_secondary_ortho=True,
            cwor_step_prompt_candidate_ortho=True,
            cwor_step_prompt_flip_alpha_on_regression=True,
            cwor_embed_subtract_scale_by_step=True,
            flux2_strategy_cwor_merge_mode="weighted",
            gcg_candidate_source="single_vlm",
            gcg_scene_vocab_prompts_per_strategy=0,
        )

        configure_attack_mode(cfg)

        self.assertEqual(cfg.attack_mode, "and")
        self.assertTrue(cfg.cwor_enable)
        self.assertEqual(cfg.cwor_reference_mode, "base_prompt")
        self.assertEqual(cfg.cwor_mode, "untargeted")
        self.assertIsNone(cfg.cwor_target_label)
        self.assertEqual(cfg.cwor_embed_inject_mode, "both")
        self.assertEqual(cfg.cwor_feedback_merge_mode, "accumulate")
        self.assertFalse(cfg.cwor_accumulate_secondary_ortho)
        self.assertFalse(cfg.cwor_accumulate_update_if_improved_only)
        self.assertFalse(cfg.cwor_accumulate_delta_use_basis_logit_without_secondary_ortho)
        self.assertFalse(cfg.cwor_step_prompt_candidate_ortho)
        self.assertFalse(cfg.cwor_step_prompt_flip_alpha_on_regression)
        self.assertFalse(cfg.cwor_embed_subtract_scale_by_step)
        self.assertEqual(cfg.flux2_strategy_cwor_merge_mode, "and")
        self.assertEqual(cfg.gcg_candidate_source, "gemma_scene_vocab")
        self.assertGreaterEqual(cfg.gcg_scene_vocab_prompts_per_strategy, 1)

    def test_exact_generator_registry_rejects_substring_lookalikes(self) -> None:
        for model_id in FLUX2_KLEIN_MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    validate_generator_model(model_id, expected_family="flux2-klein"),
                    "flux2-klein",
                )
        for model_id in QWEN_IMAGE_EDIT_MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertEqual(
                    validate_generator_model(model_id, expected_family="qwen-image-edit"),
                    "qwen-image-edit",
                )
        self.assertEqual(
            validate_generator_model("bernini", expected_family="bernini"),
            "bernini",
        )

        for lookalike in (
            "attacker/flux2-klein-backdoor",
            "attacker/qwen-image-edit-backdoor",
            "bernini-v2",
        ):
            with self.subTest(lookalike=lookalike):
                with self.assertRaises(ValueError):
                    validate_generator_model(lookalike)

    def test_removed_internal_options_are_rejected_and_feedback_is_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "removed core option"):
            parse_core_args(
                [
                    "--model_path",
                    "bernini",
                    "--cwor_enable",
                    "1",
                ]
            )
        with self.assertRaisesRegex(ValueError, "removed core option"):
            parse_core_args(
                [
                    "--model_path",
                    "bernini",
                    "--latent_nudging_scalar",
                    "1.15",
                ]
            )

        args, unknown = parse_core_args(
            [
                "--model_path",
                "bernini",
                "--gcg_scene_vocab_feedback",
                "false",
            ]
        )
        self.assertFalse(args.gcg_scene_vocab_feedback)
        self.assertEqual(unknown, [])


class AttackImageSavingTests(unittest.TestCase):
    def test_victim_callback_receives_immediate_224_classifier_image(self) -> None:
        class FakePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[0.0, 5.0, -1.0]], dtype=np.float32)

        victim = object.__new__(VictimModelAdapter)
        victim.model_name = "resnet50"
        victim.device = torch.device("cpu")
        victim.objective_mode = "ce_max"
        victim.label = 0
        victim.input_res = 224
        victim.f_model = FakePredictor()
        victim._evaluation_callback = None

        callback_payloads = []
        victim.set_evaluation_callback(callback_payloads.append)
        _, stats = victim.objective_and_stats(torch.rand(1, 3, 333, 400))

        self.assertEqual(len(callback_payloads), 1)
        payload = callback_payloads[0]
        self.assertEqual(payload["pred_idx"], stats["pred_idx"])
        self.assertEqual(payload["candidate_classifier_image_size"], 224)
        self.assertIsInstance(payload["candidate_classifier_image"], Image.Image)
        self.assertEqual(payload["candidate_classifier_image"].size, (224, 224))

    def test_core_saves_success_before_runtime_returns(self) -> None:
        class SequencedPredictor:
            def __init__(self) -> None:
                self.query_count = 0

            def predict(self, image_batch, batch_size):
                del image_batch, batch_size
                self.query_count += 1
                if self.query_count == 1:
                    return np.asarray([[5.0, 0.0]], dtype=np.float32)
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        test_case = self

        class FakeRuntime:
            def __init__(self) -> None:
                self.output_seen_before_return = False
                self.closed = False

            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                self.closed = True

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(
                self,
                *,
                args,
                classifier,
                candidate_words,
                candidate_prompts,
                **_kwargs,
            ):
                objective, stats = classifier.objective_and_stats(
                    torch.full((1, 3, 320, 240), 0.25)
                )
                output_path = Path(args.output_path)
                test_case.assertTrue(output_path.is_file())
                with Image.open(output_path) as saved:
                    test_case.assertEqual(saved.size, (224, 224))
                self.output_seen_before_return = True
                return (
                    [
                        {
                            "candidate_word": candidate_words[0],
                            "candidate_prompt": candidate_prompts[0],
                            "candidate_objective": float(objective),
                            **stats,
                            "candidate_variant": "prompt",
                        }
                    ],
                    None,
                    None,
                )

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            output_path = root / "attack_success.png"
            report_path = root / "report.json"
            Image.new("RGB", (20, 20), color=(10, 20, 30)).save(input_path)

            args, unknown = parse_core_args(
                [
                    "--model_path",
                    "black-forest-labs/FLUX.2-klein-9B",
                    "--attack_mode",
                    "vlm",
                    "--input_img_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--report_path",
                    str(report_path),
                    "--prompt",
                    "an object in the background",
                    "--gcg_word",
                    "background",
                    "--gcg_steps",
                    "1",
                    "--gcg_batch_size",
                    "1",
                    "--max_victim_queries",
                    "1",
                    "--classifier_label",
                    "0",
                    "--gcg_candidate_source",
                    "vlm_query",
                    "--gcg_early_stop_on_attack_success",
                    "1",
                    "--saved_image_size",
                    "224",
                    "--device",
                    "cpu",
                ]
            )

            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = SequencedPredictor()
            victim._evaluation_callback = None
            runtime = FakeRuntime()

            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=victim,
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertTrue(runtime.output_seen_before_return)
            self.assertTrue(runtime.closed)
            self.assertTrue(result["final_attack_success"])
            self.assertEqual(result["attack_success_query_count"], 1)

    def test_success_state_survives_a_later_runtime_failure(self) -> None:
        class SequencedPredictor:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, image_batch, batch_size):
                del image_batch, batch_size
                self.calls += 1
                if self.calls == 1:
                    return np.asarray([[5.0, 0.0]], dtype=np.float32)
                if self.calls == 2:
                    raise RuntimeError("simulated victim failure")
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        class FailingAfterSuccessRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0

            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                return None

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(self, *, classifier, **_kwargs):
                self.last_prompt_query_count = 1
                try:
                    classifier.objective_and_stats(torch.full((1, 3, 224, 224), 0.5))
                except RuntimeError as exc:
                    return [], str(exc), None
                raise RuntimeError("artifact write failed after success")

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            output_path = root / "attack_success.png"
            report_path = root / "report.json"
            Image.new("RGB", (20, 20), "white").save(input_path)
            args, unknown = parse_core_args(
                [
                    "--model_path", "bernini",
                    "--input_img_path", str(input_path),
                    "--output_path", str(output_path),
                    "--report_path", str(report_path),
                    "--prompt", "an object in the background",
                    "--gcg_word", "background",
                    "--gcg_steps", "2",
                    "--gcg_batch_size", "1",
                    "--max_victim_queries", "2",
                    "--classifier_label", "0",
                    "--device", "cpu",
                ]
            )
            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = SequencedPredictor()
            victim._evaluation_callback = None

            with self.assertRaisesRegex(RuntimeError, "artifact write failed"):
                run_blackbox_attack_core(
                    args=args,
                    unknown_args=unknown,
                    classifier=victim,
                    runtime=FailingAfterSuccessRuntime(),
                    manage_runtime=True,
                )

            self.assertTrue(output_path.is_file())
            preserved = load_preserved_attack_report(report_path)
            self.assertIsNotNone(preserved)
            self.assertTrue(preserved["final_attack_success"])
            self.assertEqual(preserved["victim_query_count"], 2)
            self.assertEqual(preserved["attack_success_query_count"], 2)

    def test_classifier_image_is_preferred_and_saved_as_exactly_224_square(self) -> None:
        classifier_image = Image.new("RGB", (31, 19), color=(231, 17, 43))
        selected_image = Image.new("RGB", (640, 480), color=(5, 101, 203))
        candidate = {
            "candidate_classifier_image": classifier_image,
            "candidate_selected_image": selected_image,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "attack_success.png"
            returned_path = save_evaluated_attack_image(candidate, output_path)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as saved:
                saved_rgb = saved.convert("RGB")
                self.assertEqual(saved_rgb.size, (224, 224))
                self.assertEqual(saved_rgb.getpixel((112, 112)), (231, 17, 43))
                self.assertNotEqual(saved_rgb.getpixel((112, 112)), (5, 101, 203))

    def test_missing_evaluated_image_fails_without_calling_generator(self) -> None:
        generator = Mock(side_effect=AssertionError("generator must not be called"))
        candidate = {
            "candidate_classifier_image": None,
            "candidate_selected_image": None,
            "candidate_precomputed_selected_image_path": "",
            "generator": generator,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "attack_success.png"
            with self.assertRaisesRegex(ValueError, "no evaluated image payload"):
                save_evaluated_attack_image(candidate, output_path)

            generator.assert_not_called()
            self.assertFalse(output_path.exists())

    def test_baseline_misclassification_is_not_generated_attack_success(self) -> None:
        class BaselineMisclassifiedPredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        class EmptyRuntime:
            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                return None

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(self, **_kwargs):
                return [], "no_candidates", None

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            output_path = root / "attack_success.png"
            report_path = root / "report.json"
            Image.new("RGB", (20, 20), color=(10, 20, 30)).save(input_path)
            args, unknown = parse_core_args(
                [
                    "--model_path", "bernini",
                    "--attack_mode", "vlm",
                    "--input_img_path", str(input_path),
                    "--output_path", str(output_path),
                    "--report_path", str(report_path),
                    "--prompt", "an object in the background",
                    "--gcg_word", "background",
                    "--gcg_steps", "1",
                    "--gcg_batch_size", "1",
                    "--max_victim_queries", "1",
                    "--classifier_label", "0",
                    "--device", "cpu",
                ]
            )
            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = BaselineMisclassifiedPredictor()
            victim._evaluation_callback = None

            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=victim,
                runtime=EmptyRuntime(),
                manage_runtime=True,
            )

            self.assertFalse(result["final_attack_success"])
            self.assertFalse(output_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["history"][0]["accepted"])
            self.assertEqual(report["history"][0]["candidate_count"], 0)

    def test_early_stop_selects_the_first_successful_candidate(self) -> None:
        class CorrectBaselinePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[5.0, 0.0]], dtype=np.float32)

        class MixedResultRuntime:
            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                return None

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(self, **_kwargs):
                return (
                    [
                        {
                            "candidate_word": "success",
                            "candidate_prompt": "successful prompt",
                            "candidate_objective": 1.0,
                            "pred_idx": 1,
                            "candidate_variant": "prompt",
                            "candidate_classifier_image": Image.new("RGB", (224, 224), "red"),
                        },
                        {
                            "candidate_word": "failure",
                            "candidate_prompt": "higher objective but failed prompt",
                            "candidate_objective": 10.0,
                            "pred_idx": 0,
                            "candidate_variant": "prompt",
                            "candidate_classifier_image": Image.new("RGB", (224, 224), "blue"),
                        },
                    ],
                    None,
                    None,
                )

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            output_path = root / "attack_success.png"
            report_path = root / "report.json"
            Image.new("RGB", (20, 20), "white").save(input_path)
            args, unknown = parse_core_args(
                [
                    "--model_path", "bernini",
                    "--input_img_path", str(input_path),
                    "--output_path", str(output_path),
                    "--report_path", str(report_path),
                    "--prompt", "an object in the background",
                    "--gcg_word", "background",
                    "--gcg_steps", "2",
                    "--gcg_batch_size", "2",
                    "--max_victim_queries", "4",
                    "--classifier_label", "0",
                    "--classifier_objective", "ce_max",
                    "--gcg_early_stop_on_attack_success", "true",
                    "--device", "cpu",
                ]
            )
            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = CorrectBaselinePredictor()
            victim._evaluation_callback = None

            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=victim,
                runtime=MixedResultRuntime(),
                manage_runtime=True,
            )

            self.assertTrue(result["final_attack_success"])
            self.assertTrue(result["early_stop_triggered"])
            self.assertEqual(result["history_len"], 1)
            self.assertEqual(result["final_prompt"], "successful prompt")
            with Image.open(output_path) as saved:
                self.assertEqual(saved.convert("RGB").getpixel((112, 112)), (255, 0, 0))

    def test_failed_victim_attempt_consumes_query_budget(self) -> None:
        class CorrectBaselinePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[5.0, 0.0]], dtype=np.float32)

        class FailedQueryRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.evaluate_calls = 0

            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                return None

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(self, **_kwargs):
                self.evaluate_calls += 1
                self.last_prompt_query_count = 1
                return [], "victim_query_failed", None

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            Image.new("RGB", (20, 20), "white").save(input_path)
            args, unknown = parse_core_args(
                [
                    "--model_path", "bernini",
                    "--input_img_path", str(input_path),
                    "--output_path", str(root / "attack_success.png"),
                    "--report_path", str(root / "report.json"),
                    "--prompt", "an object in the background",
                    "--gcg_word", "background",
                    "--gcg_steps", "2",
                    "--gcg_batch_size", "1",
                    "--max_victim_queries", "1",
                    "--classifier_label", "0",
                    "--device", "cpu",
                ]
            )
            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = CorrectBaselinePredictor()
            victim._evaluation_callback = None
            runtime = FailedQueryRuntime()

            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=victim,
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertEqual(runtime.evaluate_calls, 1)
            self.assertEqual(result["history_len"], 1)
            self.assertEqual(result["victim_query_count"], 1)
            self.assertFalse(result["final_attack_success"])

    def test_success_query_count_includes_prior_failed_attempts(self) -> None:
        class FailThenSucceedPredictor:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, image_batch, batch_size):
                del image_batch, batch_size
                self.calls += 1
                if self.calls == 1:
                    return np.asarray([[5.0, 0.0]], dtype=np.float32)
                if self.calls == 2:
                    raise RuntimeError("simulated victim failure")
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        class QueryingRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0

            def setup(self, **_kwargs) -> None:
                return None

            def close(self) -> None:
                return None

            def reset_cwor_state(self) -> None:
                return None

            def query_vlm_word(self, **_kwargs):
                return "forest", "forest", None

            def evaluate_candidates(self, *, classifier, candidate_words, candidate_prompts, **_kwargs):
                self.last_prompt_query_count = 1
                try:
                    objective, stats = classifier.objective_and_stats(
                        torch.full((1, 3, 224, 224), 0.5)
                    )
                except RuntimeError as exc:
                    return [], str(exc), None
                return (
                    [
                        {
                            "candidate_word": candidate_words[0],
                            "candidate_prompt": candidate_prompts[0],
                            "candidate_objective": float(objective),
                            "candidate_strip_index": 0,
                            "candidate_variant": "prompt",
                            "candidate_classifier_image": Image.new("RGB", (224, 224), "gray"),
                            **stats,
                        }
                    ],
                    None,
                    None,
                )

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            Image.new("RGB", (20, 20), "white").save(input_path)
            args, unknown = parse_core_args(
                [
                    "--model_path", "bernini",
                    "--input_img_path", str(input_path),
                    "--output_path", str(root / "attack_success.png"),
                    "--report_path", str(root / "report.json"),
                    "--prompt", "an object in the background",
                    "--gcg_word", "background",
                    "--gcg_steps", "2",
                    "--gcg_batch_size", "1",
                    "--max_victim_queries", "2",
                    "--classifier_label", "0",
                    "--gcg_early_stop_on_attack_success", "true",
                    "--device", "cpu",
                ]
            )
            victim = object.__new__(VictimModelAdapter)
            victim.model_name = "resnet50"
            victim.device = torch.device("cpu")
            victim.objective_mode = "ce_max"
            victim.label = 0
            victim.input_res = 224
            victim.f_model = FailThenSucceedPredictor()
            victim._evaluation_callback = None

            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=victim,
                runtime=QueryingRuntime(),
                manage_runtime=True,
            )

            self.assertTrue(result["final_attack_success"])
            self.assertEqual(result["victim_query_count"], 2)
            self.assertEqual(result["attack_success_query_count"], 2)

    def test_report_redacts_temporary_paths_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory(prefix="attack_input_") as tmpdir:
            root = Path(tmpdir)
            input_path = root / "condition.png"
            report_path = root.parent / f"{root.name}_report.json"
            args = argparse.Namespace(
                input_img_path=str(input_path),
                report_path=str(report_path),
                hf_token="hf-secret-value",
                wandb_api_key="wandb-secret-value",
                wandb_api_key_file=str(root / "wandb.key"),
                gcg_word="background",
                gcg_occurrence=0,
                classifier_objective="ce_max",
            )
            write_report(
                args=args,
                unknown_args=[],
                original_prompt="original",
                optimized_prompt="optimized",
                best_objective=1.0,
                history=[
                    {
                        "vlm_error": (
                            f"failed to open {input_path}; "
                            "hf-secret-value; wandb-secret-value"
                        )
                    }
                ],
                early_stop_event=None,
                final_attack_success=False,
                attack_success_image_path=None,
                attack_success_image_error=None,
                attack_success_query_count=None,
                attack_success_candidate=None,
            )
            serialized = report_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("hf-secret-value", serialized)
            self.assertNotIn("wandb-secret-value", serialized)
            self.assertIn("<temporary_input>", serialized)
            self.assertIn("<redacted>", serialized)
            report_path.unlink()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ISOLATED_ROOT = Path(__file__).resolve().parents[1]
if str(ISOLATED_ROOT) in sys.path:
    sys.path.remove(str(ISOLATED_ROOT))
sys.path.insert(0, str(ISOLATED_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

import bernini_attack_runner as bernini_runner
import qwen2_attack_runner as qwen_runner
from attack_model_registry import (
    FLUX2_KLEIN_MODEL_IDS,
    QWEN_IMAGE_EDIT_MODEL_IDS,
    validate_generator_model,
)
from attack_runner_common import (
    OPENAI_CLIP_IMAGENET_MODEL_ID,
    SIGLIP2_IMAGENET_MODEL_ID,
    VictimModelAdapter,
    _OpenAIClipImageNetClassifier,
    _Siglip2ImageNetClassifier,
    _extract_clip_feature_tensor,
    _imagenet_zero_shot_prompt_groups,
    _is_openai_clip_imagenet_model,
    _is_siglip2_imagenet_model,
    _openai_clip_zero_shot_prompt_groups,
    build_core_cli,
    build_parser as build_runner_parser,
    configure_attack_mode,
    iter_nips_metadata_batches,
    load_nips_ground_truth,
    load_preserved_attack_report,
    normalize_attack_mode as normalize_runner_attack_mode,
    sample_clean_correct_indices,
    select_clean_correct_indices,
)
from flux2_blackbox_runtime import Flux2KleinRenderSession
from flux2_attack_runner import (
    build_resume_validation_config,
    load_resumable_report,
)
from qwen2_attack_runner import (
    apply_qwen_runtime_args,
    build_parser as build_qwen_runner_parser,
)
from qwen2_blackbox_runtime import QwenImageEditRenderSession
from qwen_image_edit_batch import (
    _pad_prompt_batches,
    render_qwen_image_edit_batch,
)
from vlm_attack_blackbox_core import (
    _resolve_prompt,
    infer_imagenet_class_name,
    merge_scene_vocab_feedback_history,
    normalize_attack_mode as normalize_core_attack_mode,
    parse_core_args,
    run_blackbox_attack_core,
    save_naturalness_comparison,
    save_evaluated_attack_float32,
    save_evaluated_attack_image,
    write_report,
)
from vlm_attack import (
    evaluate_attack_candidates,
    evaluate_attack_success_naturalness,
    generate_scene_vocab_words,
    image_to_tensor_01,
    parse_naturalness_eval_answer,
    save_blackbox_prompt_artifacts,
)
from vlm_runtime import (
    INTERNVL3_5_4B_MODEL_ID,
    INTERNVL3_5_4B_INSTRUCT_MODEL_ID,
    QWEN3_VL_4B_INSTRUCT_MODEL_ID,
    _InternVLChatRuntime,
    _ask_with_internvl_chat,
    _ask_with_qwen_pipeline,
    load_vlm_runtime,
    normalize_strategy_mllm_mode,
    resolve_strategy_mllm_runtime,
)


class AttackModeRefactorTests(unittest.TestCase):
    def test_clip_feature_output_compatibility_preserves_exact_tensor(self) -> None:
        features = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        )

        self.assertIs(
            _extract_clip_feature_tensor(features, feature_kind="text"),
            features,
        )
        self.assertIs(
            _extract_clip_feature_tensor(
                SimpleNamespace(pooler_output=features),
                feature_kind="text",
            ),
            features,
        )
        self.assertIs(
            _extract_clip_feature_tensor(
                SimpleNamespace(pooler_output=features),
                feature_kind="image",
            ),
            features,
        )

    def test_clip_feature_output_compatibility_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be rank 2"):
            _extract_clip_feature_tensor(
                SimpleNamespace(pooler_output=torch.ones(1, 2, 3)),
                feature_kind="text",
            )

    def test_openai_clip_uses_original_paper_imagenet_prompts(self) -> None:
        self.assertTrue(
            _is_openai_clip_imagenet_model(OPENAI_CLIP_IMAGENET_MODEL_ID)
        )
        self.assertTrue(_is_openai_clip_imagenet_model("clip-vit-b-16"))
        self.assertFalse(_is_openai_clip_imagenet_model("siglip2"))

        prompt_groups = _openai_clip_zero_shot_prompt_groups()
        self.assertEqual(len(prompt_groups), 1000)
        self.assertTrue(all(len(prompts) == 80 for prompts in prompt_groups))
        self.assertEqual(prompt_groups[0][0], "a bad photo of a tench.")
        self.assertEqual(
            prompt_groups[21][-1],
            "a tattoo of the kite (bird of prey).",
        )
        self.assertEqual(
            prompt_groups[-1][-1],
            "a tattoo of the toilet paper.",
        )

    def test_siglip2_victim_aliases_and_imagenet_prompt_order(self) -> None:
        self.assertTrue(_is_siglip2_imagenet_model(SIGLIP2_IMAGENET_MODEL_ID))
        self.assertTrue(_is_siglip2_imagenet_model("siglip2"))
        self.assertFalse(_is_siglip2_imagenet_model("resnet50"))

        prompt_groups = _imagenet_zero_shot_prompt_groups()
        self.assertEqual(len(prompt_groups), 1000)
        self.assertTrue(all(len(prompts) == 81 for prompts in prompt_groups))
        self.assertEqual(prompt_groups[0][0], "a bad photo of a tench")
        self.assertEqual(prompt_groups[0][-1], "tench")
        self.assertEqual(prompt_groups[7][-1], "rooster")
        self.assertEqual(prompt_groups[20][-1], "american dipper")
        self.assertEqual(prompt_groups[-1][-1], "toilet paper")

    def test_siglip2_predictor_returns_scaled_imagenet_scores(self) -> None:
        class FakeImageProcessor:
            def __init__(self):
                self.calls = []

            def __call__(self, *, images, do_rescale, return_tensors):
                self.calls.append(
                    {
                        "images": images,
                        "do_rescale": do_rescale,
                        "return_tensors": return_tensors,
                    }
                )
                means = torch.tensor(
                    [[float(image[..., 0].mean()), float(image[..., 1].mean())]
                     for image in images],
                    dtype=torch.float32,
                )
                return {"pixel_values": means}

        class FakeSiglip2Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor([2.0])))
                self.logit_bias = torch.nn.Parameter(torch.tensor([-0.5]))

            def get_image_features(self, *, pixel_values):
                return SimpleNamespace(pooler_output=pixel_values)

        predictor = object.__new__(_Siglip2ImageNetClassifier)
        torch.nn.Module.__init__(predictor)
        predictor.device = torch.device("cpu")
        image_processor = FakeImageProcessor()
        predictor.processor = SimpleNamespace(image_processor=image_processor)
        predictor.model = FakeSiglip2Model()
        predictor.register_buffer(
            "text_features",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            persistent=False,
        )

        images = np.zeros((2, 3, 2, 2), dtype=np.float32)
        images[0, 0] = 1.0
        images[1, 1] = 1.0
        scores = predictor.predict(images, batch_size=1)

        np.testing.assert_allclose(
            scores,
            np.array([[1.5, -0.5], [-0.5, 1.5]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(len(image_processor.calls), 2)
        self.assertTrue(
            all(call["do_rescale"] is False for call in image_processor.calls)
        )

    def test_siglip2_scores_keep_existing_softmax_confidence_contract(self) -> None:
        class FakePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                logits = np.zeros((1, 1000), dtype=np.float32)
                logits[0, 7] = 2.0
                return logits

        victim = object.__new__(VictimModelAdapter)
        victim.model_name = SIGLIP2_IMAGENET_MODEL_ID
        victim.device = torch.device("cpu")
        victim.objective_mode = "ce_max"
        victim.label = 7
        victim.input_res = 224
        victim.f_model = FakePredictor()
        victim._evaluation_callback = None
        victim._evaluation_attempt_count = 0
        victim._last_evaluation_payload = None

        objective, stats = victim.objective_and_stats(
            torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        )

        expected_confidence = float(np.exp(2.0) / (np.exp(2.0) + 999.0))
        self.assertEqual(stats["pred_idx"], 7)
        self.assertAlmostEqual(stats["pred_conf"], expected_confidence, places=7)
        self.assertAlmostEqual(stats["target_conf"], expected_confidence, places=7)
        self.assertAlmostEqual(objective, -np.log(expected_confidence), places=6)

    def test_openai_clip_predictor_uses_paper_fixed_logit_scale(self) -> None:
        class FakeImageProcessor:
            @staticmethod
            def __call__(*, images, do_rescale, return_tensors):
                del do_rescale, return_tensors
                return {
                    "pixel_values": torch.tensor(
                        [
                            [float(image[..., 0].mean()), float(image[..., 1].mean())]
                            for image in images
                        ],
                        dtype=torch.float32,
                    )
                }

        class FakeClipModel(torch.nn.Module):
            @staticmethod
            def get_image_features(*, pixel_values):
                return SimpleNamespace(pooler_output=pixel_values)

        predictor = object.__new__(_OpenAIClipImageNetClassifier)
        torch.nn.Module.__init__(predictor)
        predictor.device = torch.device("cpu")
        predictor.processor = SimpleNamespace(image_processor=FakeImageProcessor())
        predictor.model = FakeClipModel()
        predictor.register_buffer(
            "text_features",
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            persistent=False,
        )

        image = np.zeros((1, 3, 2, 2), dtype=np.float32)
        image[0, 0] = 1.0
        scores = predictor.predict(image)

        np.testing.assert_allclose(
            scores,
            np.array([[100.0, 0.0]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_victim_adapter_routes_siglip2_around_art_registry(self) -> None:
        fake_predictor = Mock(spec=torch.nn.Module)
        fake_predictor.parameters.return_value = []
        with patch(
            "attack_runner_common._Siglip2ImageNetClassifier",
            return_value=fake_predictor,
        ) as classifier_cls:
            victim = VictimModelAdapter(
                model_name=SIGLIP2_IMAGENET_MODEL_ID,
                device="cpu",
                objective_mode="logit_margin_max",
            )

        classifier_cls.assert_called_once_with(
            model_id=SIGLIP2_IMAGENET_MODEL_ID,
            device=torch.device("cpu"),
        )
        self.assertIs(victim.model, fake_predictor)
        self.assertIs(victim.f_model, fake_predictor)
        fake_predictor.eval.assert_called_once_with()

    def test_victim_adapter_routes_openai_clip_around_art_registry(self) -> None:
        fake_predictor = Mock(spec=torch.nn.Module)
        fake_predictor.parameters.return_value = []
        with patch(
            "attack_runner_common._OpenAIClipImageNetClassifier",
            return_value=fake_predictor,
        ) as classifier_cls:
            victim = VictimModelAdapter(
                model_name=OPENAI_CLIP_IMAGENET_MODEL_ID,
                device="cpu",
                objective_mode="logit_margin_max",
            )

        classifier_cls.assert_called_once_with(
            model_id=OPENAI_CLIP_IMAGENET_MODEL_ID,
            device=torch.device("cpu"),
        )
        self.assertIs(victim.model, fake_predictor)
        self.assertIs(victim.f_model, fake_predictor)
        fake_predictor.eval.assert_called_once_with()

    def test_naturalness_parser_and_gemma_e4b_call_contract(self) -> None:
        self.assertEqual(
            parse_naturalness_eval_answer(
                'analysis\n{"natural": false, "feedback": "subject is distorted"}'
            ),
            (False, "subject is distorted"),
        )
        self.assertEqual(
            parse_naturalness_eval_answer('{"is_natural": true, "reason": ""}'),
            (True, ""),
        )
        self.assertEqual(
            parse_naturalness_eval_answer("unclear verdict"),
            (None, ""),
        )

        args = SimpleNamespace(
            class_name="goldfish",
            classifier_name="resnet50",
            gcg_scene_llm_backend="gemma4",
            gcg_scene_llm_model_id="google/gemma-4-E4B-it",
            gcg_scene_llm_device="auto",
            gcg_scene_llm_max_new_tokens=4096,
            gcg_scene_llm_thinking=True,
            gcg_eval_naturalness_llm_thinking=False,
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=('{"natural": true, "feedback": ""}', None),
        ) as query:
            natural, feedback, raw_answer, error = (
                evaluate_attack_success_naturalness(
                    image_path=Path("comparison.png"),
                    candidate_prompt="ignored by the restored legacy prompt",
                    args=args,
                    is_source_vs_edited_comparison=True,
                )
            )

        self.assertIs(natural, True)
        self.assertEqual(feedback, "")
        self.assertEqual(raw_answer, '{"natural": true, "feedback": ""}')
        self.assertIsNone(error)
        query_kwargs = query.call_args.kwargs
        self.assertEqual(query_kwargs["vlm_backend"], "gemma4")
        self.assertEqual(
            query_kwargs["vlm_model_id"],
            "google/gemma-4-E4B-it",
        )
        self.assertEqual(query_kwargs["max_new_tokens"], 512)
        self.assertFalse(query_kwargs["enable_thinking"])
        self.assertFalse(query_kwargs["do_sample"])
        self.assertIn("Left: source image", query_kwargs["question"])
        self.assertIn("Right: edited image", query_kwargs["question"])
        self.assertNotIn(
            "SEMANTIC-PRIVACY REQUIREMENT",
            query_kwargs["question"],
        )

        class_ablation_args = SimpleNamespace(
            **{
                **vars(args),
                "class_ablation": True,
                "class_name": None,
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=('{"natural": true, "feedback": ""}', None),
        ) as class_ablation_query:
            evaluate_attack_success_naturalness(
                image_path=Path("comparison.png"),
                candidate_prompt="ignored",
                args=class_ablation_args,
                is_source_vs_edited_comparison=True,
            )

        class_ablation_question = (
            class_ablation_query.call_args.kwargs["question"]
        )
        self.assertIn(
            "SEMANTIC-PRIVACY REQUIREMENT",
            class_ablation_question,
        )
        self.assertIn(
            "Do not identify, name, classify, describe, or infer any object",
            class_ablation_question,
        )
        self.assertIn(
            'Refer to image content only as "the main subject", '
            '"the source image",',
            class_ablation_question,
        )
        self.assertIn(
            "Do not provide any explanation or free-form feedback.",
            class_ablation_question,
        )

        with patch(
            "vlm_attack.query_vlm_text",
            return_value=("unclear verdict", None),
        ):
            natural, feedback, raw_answer, error = (
                evaluate_attack_success_naturalness(
                    image_path=Path("comparison.png"),
                    candidate_prompt="ignored",
                    args=args,
                    is_source_vs_edited_comparison=True,
                )
            )
        self.assertIsNone(natural)
        self.assertEqual(feedback, "")
        self.assertEqual(raw_answer, "unclear verdict")
        self.assertEqual(error, "naturalness_verdict_unparseable")

    def test_naturalness_comparison_is_source_left_and_full_edit_right(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.png"
            output_path = root / "comparison.png"
            Image.new("RGB", (11, 7), "red").save(source_path)
            candidate = {
                "candidate_selected_image": Image.new("RGB", (13, 9), "blue"),
                "candidate_classifier_image": Image.new("RGB", (224, 224), "green"),
            }

            save_naturalness_comparison(
                source_image_path=source_path,
                candidate=candidate,
                output_path=output_path,
            )

            with Image.open(output_path) as comparison:
                comparison = comparison.convert("RGB")
                self.assertEqual(comparison.size, (26, 9))
                self.assertEqual(comparison.getpixel((6, 4)), (255, 0, 0))
                self.assertEqual(comparison.getpixel((19, 4)), (0, 0, 255))

    def test_naturalness_feedback_survives_scene_feedback_merge(self) -> None:
        merged = merge_scene_vocab_feedback_history(
            existing_feedback=[],
            generated_words=["distort subject"],
            scored_candidates=[
                {
                    "candidate_word": "distort subject",
                    "candidate_objective": 2.0,
                    "naturalness_checked": True,
                    "naturalness_is_natural": False,
                    "naturalness_feedback": "the main object is warped",
                }
            ],
        )

        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0]["naturalness_is_natural"], False)
        self.assertEqual(
            merged[0]["naturalness_feedback"],
            "the main object is warped",
        )

    def test_verifier_rejects_first_raw_success_and_saves_second_natural_success(self) -> None:
        class CorrectBaselineClassifier:
            @staticmethod
            def objective_and_stats(_image_01, target_label=None):
                del target_label
                return 0.0, {
                    "pred_idx": 0,
                    "pred_conf": 1.0,
                    "pred_logit": 5.0,
                    "target_conf": 1.0,
                    "target_logit": 5.0,
                    "target_label_conf": None,
                    "target_label_logit": None,
                    "ce": 0.0,
                }

        class NaturalnessRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.naturalness_prompts = []
                self.comparison_pixels = []

            def setup(self, **_kwargs):
                return None

            def close(self):
                return None

            def reset_cwor_state(self):
                return None

            def generate_scene_vocab_words(self, **_kwargs):
                return (
                    ["unnatural edit", "natural edit"],
                    '{"candidates": ["unnatural edit", "natural edit"]}',
                    "generation prompt",
                    None,
                )

            def query_vlm_word(self, **_kwargs):
                raise AssertionError("scene-vocab candidates should be used")

            def evaluate_candidates(self, **_kwargs):
                self.last_prompt_query_count = 2
                red_float32 = np.zeros((3, 224, 224), dtype=np.float32)
                red_float32[0] = 1.0
                green_float32 = np.zeros((3, 224, 224), dtype=np.float32)
                green_float32[1] = 1.0
                return (
                    [
                        {
                            "candidate_word": "unnatural edit",
                            "candidate_prompt": "an object in the unnatural edit",
                            "candidate_objective": 10.0,
                            "pred_idx": 1,
                            "candidate_variant": "prompt",
                            "candidate_strip_index": 0,
                            "candidate_selected_image": Image.new(
                                "RGB", (32, 32), "red"
                            ),
                            "candidate_classifier_image": Image.new(
                                "RGB", (224, 224), "red"
                            ),
                            "candidate_classifier_input_float32": red_float32,
                        },
                        {
                            "candidate_word": "natural edit",
                            "candidate_prompt": "an object in the natural edit",
                            "candidate_objective": 1.0,
                            "pred_idx": 1,
                            "candidate_variant": "prompt",
                            "candidate_strip_index": 1,
                            "candidate_selected_image": Image.new(
                                "RGB", (32, 32), "green"
                            ),
                            "candidate_classifier_image": Image.new(
                                "RGB", (224, 224), "green"
                            ),
                            "candidate_classifier_input_float32": green_float32,
                        },
                    ],
                    None,
                    None,
                )

            def evaluate_naturalness(
                self,
                *,
                image_path,
                candidate_prompt,
                **_kwargs,
            ):
                self.naturalness_prompts.append(candidate_prompt)
                with Image.open(image_path) as comparison:
                    self.comparison_pixels.append(
                        comparison.convert("RGB").getpixel((48, 16))
                    )
                if "unnatural" in candidate_prompt:
                    return (
                        False,
                        "the main object is distorted",
                        '{"natural": false, "feedback": "the main object is distorted"}',
                        None,
                    )
                return True, "", '{"natural": true, "feedback": ""}', None

            def save_prompt_artifacts(self, **_kwargs):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "source.png"
            output_path = root / "attack_success.png"
            report_path = root / "report.json"
            Image.new("RGB", (20, 20), "white").save(input_path)
            np.save(
                output_path.with_suffix(".float32.npy"),
                np.zeros((3, 224, 224), dtype=np.float32),
                allow_pickle=False,
            )
            args, unknown = parse_core_args(
                [
                    "--model_path",
                    "bernini",
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
                    "2",
                    "--max_victim_queries",
                    "2",
                    "--classifier_label",
                    "0",
                    "--classifier_objective",
                    "ce_max",
                    "--gcg_candidate_source",
                    "gemma_scene_vocab",
                    "--gcg_scene_vocab_feedback",
                    "true",
                    "--gcg_early_stop_on_attack_success",
                    "true",
                    "--gcg_eval_naturalness_on_attack_success",
                    "true",
                    "--gcg_eval_naturalness_llm_thinking",
                    "false",
                    "--gcg_scene_llm_model_id",
                    "google/gemma-4-E4B-it",
                    "--device",
                    "cpu",
                ]
            )
            runtime = NaturalnessRuntime()
            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=CorrectBaselineClassifier(),
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertTrue(result["final_attack_success"])
            self.assertEqual(result["attack_success_query_count"], 2)
            self.assertEqual(result["final_prompt"], "an object in the natural edit")
            self.assertEqual(
                runtime.naturalness_prompts,
                [
                    "an object in the unnatural edit",
                    "an object in the natural edit",
                ],
            )
            self.assertEqual(runtime.comparison_pixels, [(255, 0, 0), (0, 128, 0)])
            with Image.open(output_path) as saved:
                self.assertEqual(
                    saved.convert("RGB").getpixel((112, 112)),
                    (0, 128, 0),
                )
            saved_float32 = np.load(
                output_path.with_suffix(".float32.npy"),
                allow_pickle=False,
            )
            self.assertTrue(np.all(saved_float32[0] == 0.0))
            self.assertTrue(np.all(saved_float32[1] == 1.0))

            report = json.loads(report_path.read_text(encoding="utf-8"))
            candidates = report["history"][0]["scored_candidates"]
            self.assertIs(candidates[0]["naturalness_is_natural"], False)
            self.assertEqual(
                candidates[0]["naturalness_feedback"],
                "the main object is distorted",
            )
            self.assertIs(candidates[1]["naturalness_is_natural"], True)
            self.assertIs(
                report["history"][0]["attack_success_image_natural"],
                True,
            )
            self.assertEqual(
                report["naturalness_verifier"]["model_id"],
                "google/gemma-4-E4B-it",
            )
            self.assertFalse((root / "naturalness").exists())
            self.assertNotIn("naturalness_comparison_path", candidates[0])
            self.assertNotIn("naturalness_comparison_path", candidates[1])

    def test_all_unnatural_raw_successes_continue_and_produce_no_success_artifact(self) -> None:
        class CorrectBaselineClassifier:
            @staticmethod
            def objective_and_stats(_image_01, target_label=None):
                del target_label
                return 0.0, {"pred_idx": 0}

        class AlwaysUnnaturalRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.generation_feedback = []
                self.eval_count = 0

            def setup(self, **_kwargs):
                return None

            def close(self):
                return None

            def reset_cwor_state(self):
                return None

            def generate_scene_vocab_words(self, *, step_idx, previous_feedback, **_kwargs):
                self.generation_feedback.append([dict(item) for item in previous_feedback])
                word = f"unnatural edit {int(step_idx)}"
                return [word], f'{{"candidates": ["{word}"]}}', "prompt", None

            def query_vlm_word(self, **_kwargs):
                raise AssertionError("scene-vocab candidates should be used")

            def evaluate_candidates(self, *, candidate_words, candidate_prompts, **_kwargs):
                self.last_prompt_query_count = 1
                color = "red" if self.eval_count == 0 else "blue"
                self.eval_count += 1
                return (
                    [
                        {
                            "candidate_word": candidate_words[0],
                            "candidate_prompt": candidate_prompts[0],
                            "candidate_objective": float(self.eval_count),
                            "pred_idx": 1,
                            "candidate_variant": "prompt",
                            "candidate_strip_index": 0,
                            "candidate_selected_image": Image.new(
                                "RGB", (32, 32), color
                            ),
                            "candidate_classifier_image": Image.new(
                                "RGB", (224, 224), color
                            ),
                        }
                    ],
                    None,
                    None,
                )

            def evaluate_naturalness(self, **_kwargs):
                return (
                    False,
                    "the main object is warped",
                    '{"natural": false, "feedback": "the main object is warped"}',
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
            np.save(
                output_path.with_suffix(".float32.npy"),
                np.ones((3, 224, 224), dtype=np.float32),
                allow_pickle=False,
            )
            args, unknown = parse_core_args(
                [
                    "--model_path",
                    "bernini",
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
                    "2",
                    "--gcg_batch_size",
                    "1",
                    "--max_victim_queries",
                    "2",
                    "--classifier_label",
                    "0",
                    "--classifier_objective",
                    "ce_max",
                    "--gcg_candidate_source",
                    "gemma_scene_vocab",
                    "--gcg_scene_vocab_feedback",
                    "true",
                    "--gcg_early_stop_on_attack_success",
                    "true",
                    "--gcg_eval_naturalness_on_attack_success",
                    "true",
                    "--device",
                    "cpu",
                ]
            )
            runtime = AlwaysUnnaturalRuntime()
            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=CorrectBaselineClassifier(),
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertFalse(result["final_attack_success"])
            self.assertEqual(result["history_len"], 2)
            self.assertEqual(runtime.eval_count, 2)
            self.assertFalse(output_path.exists())
            self.assertFalse(output_path.with_suffix(".float32.npy").exists())
            self.assertEqual(runtime.generation_feedback[0], [])
            second_step_feedback = runtime.generation_feedback[1]
            self.assertEqual(len(second_step_feedback), 1)
            self.assertIs(
                second_step_feedback[0]["naturalness_is_natural"],
                False,
            )
            self.assertEqual(
                second_step_feedback[0]["naturalness_feedback"],
                "the main object is warped",
            )

    def test_and_mode_continues_after_unnatural_prompt_success(self) -> None:
        class CorrectBaselineClassifier:
            @staticmethod
            def objective_and_stats(_image_01, target_label=None):
                del target_label
                return 0.0, {
                    "pred_idx": 0,
                    "pred_conf": 1.0,
                    "pred_logit": 5.0,
                    "target_conf": 1.0,
                    "target_logit": 5.0,
                    "target_label_conf": None,
                    "target_label_logit": None,
                    "ce": 0.0,
                }

        class AndNaturalnessRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.last_and_query_count = 0
                self.and_called = False

            def setup(self, **_kwargs):
                return None

            def close(self):
                return None

            def reset_cwor_state(self):
                return None

            def generate_scene_vocab_words(self, *, args, **_kwargs):
                args._scene_vocab_strategy_entries = [
                    {
                        "word": "prompt edit",
                        "strategy_name": "background_shift",
                        "strategy_title": "Background Shift",
                    }
                ]
                return ["prompt edit"], '{"candidates": ["prompt edit"]}', "prompt", None

            def query_vlm_word(self, **_kwargs):
                raise AssertionError("scene-vocab candidates should be used")

            def evaluate_candidates(self, **_kwargs):
                self.last_prompt_query_count = 1
                return (
                    [
                        {
                            "candidate_word": "prompt edit",
                            "candidate_prompt": "prompt edit",
                            "candidate_objective": 10.0,
                            "pred_idx": 1,
                            "candidate_variant": "prompt",
                            "candidate_strip_index": 0,
                            "candidate_selected_image": Image.new(
                                "RGB", (32, 32), "red"
                            ),
                            "candidate_classifier_image": Image.new(
                                "RGB", (224, 224), "red"
                            ),
                        }
                    ],
                    None,
                    None,
                )

            def evaluate_and_candidate(self, **_kwargs):
                self.and_called = True
                self.last_and_query_count = 1
                blue_float32 = np.zeros((3, 224, 224), dtype=np.float32)
                blue_float32[2] = 1.0
                return (
                    [
                        {
                            "candidate_word": "<CWOR>",
                            "candidate_prompt": "verified and edit",
                            "candidate_objective": 2.0,
                            "pred_idx": 1,
                            "candidate_variant": "cwor",
                            "cwor_strategy_query_offset": 1,
                            "candidate_selected_image": Image.new(
                                "RGB", (32, 32), "blue"
                            ),
                            "candidate_classifier_image": Image.new(
                                "RGB", (224, 224), "blue"
                            ),
                            "candidate_classifier_input_float32": blue_float32,
                        }
                    ],
                    None,
                )

            def evaluate_naturalness(self, *, candidate_prompt, **_kwargs):
                if candidate_prompt == "prompt edit":
                    return (
                        False,
                        "the subject disappeared",
                        '{"natural": false, "feedback": "the subject disappeared"}',
                        None,
                    )
                return True, "", '{"natural": true, "feedback": ""}', None

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
                    "--model_path",
                    "black-forest-labs/FLUX.2-klein-9b-kv",
                    "--attack_mode",
                    "and",
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
                    "--gcg_scene_vocab_prompts_per_strategy",
                    "1",
                    "--max_victim_queries",
                    "2",
                    "--classifier_label",
                    "0",
                    "--classifier_objective",
                    "ce_max",
                    "--gcg_early_stop_on_attack_success",
                    "true",
                    "--gcg_eval_naturalness_on_attack_success",
                    "true",
                    "--device",
                    "cpu",
                ]
            )
            runtime = AndNaturalnessRuntime()
            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=CorrectBaselineClassifier(),
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertTrue(runtime.and_called)
            self.assertTrue(result["final_attack_success"])
            self.assertEqual(result["attack_success_query_count"], 2)
            self.assertEqual(result["final_prompt"], "verified and edit")
            with Image.open(output_path) as saved:
                self.assertEqual(
                    saved.convert("RGB").getpixel((112, 112)),
                    (0, 0, 255),
                )

    def test_and_mode_charges_duplicate_but_not_empty_strategy_slot(self) -> None:
        class CorrectBaselineClassifier:
            @staticmethod
            def objective_and_stats(_image_01, target_label=None):
                del target_label
                return 0.0, {
                    "pred_idx": 0,
                    "pred_conf": 1.0,
                    "pred_logit": 5.0,
                    "target_conf": 1.0,
                    "target_logit": 5.0,
                    "target_label_conf": None,
                    "target_label_logit": None,
                    "ce": 0.0,
                }

        class DuplicateAndEmptyRuntime:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.last_and_query_count = 0
                self.evaluate_calls = 0

            def setup(self, **_kwargs):
                return None

            def close(self):
                return None

            def reset_cwor_state(self):
                return None

            def generate_scene_vocab_words(self, *, args, **_kwargs):
                args._scene_vocab_strategy_entries = [
                    {
                        "word": "repeat me",
                        "strategy_name": "background_shift",
                        "strategy_title": "Background Shift",
                    },
                    {
                        "word": "coated in brushed metal",
                        "strategy_name": "texture_material",
                        "strategy_title": "Texture & Material",
                    },
                ]
                args._scene_vocab_strategy_slots = [
                    {
                        "word": "repeat me",
                        "strategy_name": "background_shift",
                        "strategy_title": "Background Shift",
                        "empty": False,
                    },
                    {
                        "word": "",
                        "strategy_name": "weather_atmosphere",
                        "strategy_title": "Weather & Atmosphere",
                        "empty": True,
                    },
                    {
                        "word": "coated in brushed metal",
                        "strategy_name": "texture_material",
                        "strategy_title": "Texture & Material",
                        "empty": False,
                    },
                ]
                return (
                    ["repeat me", "coated in brushed metal"],
                    '{"strategies": {}}',
                    "prompt",
                    None,
                )

            def query_vlm_word(self, **_kwargs):
                raise AssertionError("empty strategy slots must not trigger fallback")

            def evaluate_candidates(self, *, candidate_words, candidate_prompts, **_kwargs):
                self.evaluate_calls += 1
                self.last_prompt_query_count = 1
                assert candidate_words == ["coated in brushed metal"]
                assert candidate_prompts == ["coated in brushed metal"]
                return (
                    [
                        {
                            "candidate_word": "coated in brushed metal",
                            "candidate_prompt": "coated in brushed metal",
                            "candidate_objective": 1.0,
                            "pred_idx": 0,
                            "candidate_variant": "prompt",
                        }
                    ],
                    None,
                    None,
                )

            def evaluate_and_candidate(self, **_kwargs):
                raise AssertionError("duplicate consumed the remaining AND query slot")

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
                    "--model_path",
                    "black-forest-labs/FLUX.2-klein-9b-kv",
                    "--attack_mode",
                    "and",
                    "--input_img_path",
                    str(input_path),
                    "--output_path",
                    str(output_path),
                    "--report_path",
                    str(report_path),
                    "--prompt",
                    "repeat me",
                    "--gcg_word",
                    "repeat",
                    "--gcg_steps",
                    "1",
                    "--gcg_batch_size",
                    "3",
                    "--gcg_scene_vocab_prompts_per_strategy",
                    "1",
                    "--max_victim_queries",
                    "2",
                    "--classifier_label",
                    "0",
                    "--gcg_eval_naturalness_on_attack_success",
                    "false",
                    "--device",
                    "cpu",
                ]
            )
            runtime = DuplicateAndEmptyRuntime()
            result = run_blackbox_attack_core(
                args=args,
                unknown_args=unknown,
                classifier=CorrectBaselineClassifier(),
                runtime=runtime,
                manage_runtime=True,
            )

            self.assertEqual(runtime.evaluate_calls, 1)
            self.assertEqual(result["victim_query_count"], 2)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            entry = report["history"][0]
            self.assertEqual(entry["strategy_slot_count"], 3)
            self.assertEqual(entry["strategy_empty_skipped_count"], 1)
            self.assertEqual(entry["strategy_duplicate_skipped_count"], 1)
            self.assertEqual(entry["strategy_duplicate_query_count"], 1)
            self.assertEqual(entry["candidate_text_count"], 1)
            self.assertEqual(entry["victim_queries_this_step"], 2)

    def test_class_ablation_defaults_off_and_parses_in_runner_and_core(self) -> None:
        runner_default = build_runner_parser().parse_args([])
        runner_enabled = build_runner_parser().parse_args(
            ["--class_ablation", "true"]
        )
        core_default, _ = parse_core_args(
            ["--model_path", "black-forest-labs/FLUX.2-klein-9b-kv"]
        )
        core_enabled, _ = parse_core_args(
            [
                "--model_path",
                "black-forest-labs/FLUX.2-klein-9b-kv",
                "--class_ablation",
                "true",
            ]
        )

        self.assertFalse(runner_default.class_ablation)
        self.assertTrue(runner_enabled.class_ablation)
        self.assertFalse(core_default.class_ablation)
        self.assertTrue(core_enabled.class_ablation)

        propagated_cli = build_core_cli(
            cfg=runner_enabled,
            hf_token="",
            sample_image_path=Path("input.png"),
            output_path=Path("output.png"),
            report_path=Path("report.json"),
            classifier_label=1,
            sample_target_label=None,
            sample_run_name="class-ablation-test",
        )
        option_index = propagated_cli.index("--class_ablation")
        self.assertEqual(propagated_cli[option_index + 1], "1")
        self.assertEqual(propagated_cli.count("--class_ablation"), 1)

    def test_naturalness_verifier_defaults_on_and_can_be_disabled(self) -> None:
        core_default, _ = parse_core_args(
            ["--model_path", "black-forest-labs/FLUX.2-klein-9b-kv"]
        )
        core_disabled, _ = parse_core_args(
            [
                "--model_path",
                "black-forest-labs/FLUX.2-klein-9b-kv",
                "--gcg_eval_naturalness_on_attack_success",
                "false",
            ]
        )

        self.assertTrue(core_default.gcg_eval_naturalness_on_attack_success)
        self.assertFalse(core_disabled.gcg_eval_naturalness_on_attack_success)

        config_root = Path(__file__).resolve().parents[1] / "configs"
        for config_name in (
            "flux2_and_attack_nips.yaml",
            "flux2_and_attack_nips_swin.yaml",
            "flux2_and_attack_nips_vim.yaml",
        ):
            with self.subTest(config=config_name):
                config = yaml.safe_load(
                    (config_root / config_name).read_text(encoding="utf-8")
                )
                self.assertIs(
                    config["run_args"][
                        "gcg_eval_naturalness_on_attack_success"
                    ],
                    True,
                )

    def test_strategy_mllm_modes_parse_and_propagate(self) -> None:
        runner_args = build_runner_parser().parse_args(
            ["--strategy_mllm_mode", "qwen3-vl-4b-instruct"]
        )
        core_args, _ = parse_core_args(
            [
                "--model_path",
                "black-forest-labs/FLUX.2-klein-9b-kv",
                "--strategy_mllm_mode",
                "qwen3_vl_4b_instruct",
            ]
        )

        self.assertEqual(
            normalize_strategy_mllm_mode("Qwen/Qwen3-VL-4B-Instruct"),
            "qwen3_vl_4b_instruct",
        )
        self.assertEqual(runner_args.strategy_mllm_mode, "qwen3_vl_4b_instruct")
        self.assertEqual(core_args.strategy_mllm_mode, "qwen3_vl_4b_instruct")

        propagated_cli = build_core_cli(
            cfg=runner_args,
            hf_token="",
            sample_image_path=Path("input.png"),
            output_path=Path("output.png"),
            report_path=Path("report.json"),
            classifier_label=1,
            sample_target_label=None,
            sample_run_name="qwen3-vl-strategy-test",
        )
        option_index = propagated_cli.index("--strategy_mllm_mode")
        self.assertEqual(
            propagated_cli[option_index + 1],
            "qwen3_vl_4b_instruct",
        )
        self.assertEqual(propagated_cli.count("--strategy_mllm_mode"), 1)

        internvl_runner_args = build_runner_parser().parse_args(
            ["--strategy_mllm_mode", "InternVL3.5-4B"]
        )
        internvl_core_args, _ = parse_core_args(
            [
                "--model_path",
                "black-forest-labs/FLUX.2-klein-9b-kv",
                "--strategy_mllm_mode",
                "internvl3_5_4b",
            ]
        )
        self.assertEqual(
            internvl_runner_args.strategy_mllm_mode,
            "internvl3_5_4b",
        )
        self.assertEqual(
            internvl_core_args.strategy_mllm_mode,
            "internvl3_5_4b",
        )
        internvl_instruct_args = build_runner_parser().parse_args(
            ["--strategy_mllm_mode", "InternVL3.5-4B-Instruct"]
        )
        self.assertEqual(
            internvl_instruct_args.strategy_mllm_mode,
            "internvl3_5_4b_instruct",
        )

    def test_qwen3_vl_strategy_mode_matches_naturalness_verifier(self) -> None:
        resolved = resolve_strategy_mllm_runtime(
            mode="qwen3_vl_4b_instruct",
            configured_backend="gemma4",
            configured_model_id="google/gemma-4-E4B-it",
            configured_thinking=True,
            configured_do_sample=True,
        )
        self.assertEqual(
            resolved,
            (
                "qwen3_vl_4b_instruct",
                "qwen",
                QWEN3_VL_4B_INSTRUCT_MODEL_ID,
                False,
                True,
            ),
        )

        args = SimpleNamespace(
            strategy_mllm_mode="qwen3_vl_4b_instruct",
            class_ablation=False,
            class_name="goldfish",
            classifier_name="resnet50",
            model_path="black-forest-labs/FLUX.2-klein-9b-kv",
            bernini_edit_prompt_mode=False,
            qwen_edit_prompt_mode=False,
            qwen_strategy_and_enable=False,
            gcg_occurrence=0,
            gcg_scene_vocab_topic=None,
            gcg_scene_vocab_size=3,
            gcg_scene_vocab_prompts_per_strategy=1,
            gcg_scene_vocab_enabled_strategies="all",
            gcg_slot_candidate_max_words=5,
            gcg_scene_vocab_feedback=False,
            gcg_scene_feedback_limit=1000,
            cwor_enable=True,
            gcg_scene_llm_backend="gemma4",
            gcg_scene_llm_model_id="google/gemma-4-E4B-it",
            gcg_scene_llm_device="cpu",
            gcg_scene_llm_max_new_tokens=128,
            gcg_scene_llm_thinking=True,
            gcg_scene_llm_do_sample=True,
            gcg_eval_naturalness_llm_thinking=False,
        )
        strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": ["move into a forest"],
                    "weather_atmosphere": ["add soft rain"],
                    "texture_material": ["coated in brushed metal"],
                }
            }
        )

        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(strategy_answer, None),
        ) as query:
            words, _, _, strategy_error = generate_scene_vocab_words(
                args=args,
                step_idx=0,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertIsNone(strategy_error)
        self.assertEqual(
            words,
            [
                "move into a forest",
                "add soft rain",
                "coated in brushed metal",
            ],
        )
        self.assertEqual(
            [item["strategy_name"] for item in args._scene_vocab_strategy_entries],
            [
                "background_shift",
                "weather_atmosphere",
                "texture_material",
            ],
        )
        strategy_call = query.call_args.kwargs
        self.assertEqual(strategy_call["vlm_backend"], "qwen")
        self.assertEqual(
            strategy_call["vlm_model_id"],
            QWEN3_VL_4B_INSTRUCT_MODEL_ID,
        )
        self.assertFalse(strategy_call["enable_thinking"])
        self.assertTrue(strategy_call["do_sample"])

        object_wrapped_strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": {
                        "candidates": ["move into a forest"]
                    },
                    "weather_atmosphere": {
                        "candidates": ["add soft rain"]
                    },
                    "texture_material": {
                        "candidates": ["coated in brushed metal"]
                    },
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(object_wrapped_strategy_answer, None),
        ):
            wrapped_words, _, _, wrapped_error = generate_scene_vocab_words(
                args=args,
                step_idx=1,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertIsNone(wrapped_error)
        self.assertEqual(
            wrapped_words,
            [
                "move into a forest",
                "add soft rain",
                "coated in brushed metal",
            ],
        )

        repeated_current_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": ["move into a warehouse"],
                    "weather_atmosphere": ["add soft rain"],
                    "texture_material": ["coated in brushed metal"],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            side_effect=[
                (repeated_current_answer, None),
                (repeated_current_answer, None),
                (strategy_answer, None),
            ],
        ) as repeated_current_query:
            repeated_current_words, _, repeated_current_prompt, repeated_current_error = (
                generate_scene_vocab_words(
                    args=args,
                    step_idx=2,
                    current_prompt="move into a warehouse",
                    current_word="background",
                    slot_kind="scene",
                    best_objective=0.0,
                    previous_feedback=[],
                    reference_image_path=Path("unused-reference.png"),
                    fallback_word="outdoor",
                )
            )

        self.assertIsNone(repeated_current_error)
        self.assertEqual(
            repeated_current_words,
            [
                "move into a warehouse",
                "add soft rain",
                "coated in brushed metal",
            ],
        )
        self.assertEqual(repeated_current_query.call_count, 1)
        self.assertNotIn("CONTRACT CORRECTION", repeated_current_prompt)
        self.assertEqual(
            [slot["empty"] for slot in args._scene_vocab_strategy_slots],
            [False, False, False],
        )

        empty_then_valid_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": [],
                    "weather_atmosphere": [],
                    "texture_material": [],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            side_effect=[
                (empty_then_valid_answer, None),
                (strategy_answer, None),
            ],
        ) as retry_query:
            retry_words, _, retry_prompt, retry_error = generate_scene_vocab_words(
                args=args,
                step_idx=2,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertIsNone(retry_error)
        self.assertEqual(retry_words, [])
        self.assertEqual(retry_query.call_count, 1)
        self.assertNotIn("CONTRACT CORRECTION", retry_prompt)
        self.assertEqual(
            [slot["empty"] for slot in args._scene_vocab_strategy_slots],
            [True, True, True],
        )

        incomplete_strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": ["move into a forest"],
                    "weather_atmosphere": ["add soft rain"],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(incomplete_strategy_answer, None),
        ):
            incomplete_words, _, _, incomplete_error = generate_scene_vocab_words(
                args=args,
                step_idx=1,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertEqual(
            incomplete_words,
            ["move into a forest", "add soft rain"],
        )
        self.assertIsNone(incomplete_error)
        self.assertEqual(
            [slot["empty"] for slot in args._scene_vocab_strategy_slots],
            [False, False, True],
        )

        empty_strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": [],
                    "weather_atmosphere": [],
                    "texture_material": [],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(empty_strategy_answer, None),
        ):
            empty_words, _, _, empty_error = generate_scene_vocab_words(
                args=args,
                step_idx=2,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertEqual(empty_words, [])
        self.assertIsNone(empty_error)
        self.assertEqual(
            [slot["empty"] for slot in args._scene_vocab_strategy_slots],
            [True, True, True],
        )

        too_many_strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": [
                        "move into a forest",
                        "move into a warehouse",
                    ],
                    "weather_atmosphere": ["add soft rain"],
                    "texture_material": ["coated in brushed metal"],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(too_many_strategy_answer, None),
        ):
            too_many_words, _, _, too_many_error = generate_scene_vocab_words(
                args=args,
                step_idx=3,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertEqual(
            too_many_words,
            [
                "move into a forest",
                "add soft rain",
                "coated in brushed metal",
            ],
        )
        self.assertIsNone(too_many_error)

        unknown_group_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": ["move into a forest"],
                    "weather_atmosphere": ["add soft rain"],
                    "texture_material": ["coated in brushed metal"],
                    "strategies": ["not a real strategy"],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(unknown_group_answer, None),
        ):
            unknown_words, _, _, unknown_error = generate_scene_vocab_words(
                args=args,
                step_idx=4,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertEqual(unknown_words, wrapped_words)
        self.assertIsNone(unknown_error)

        with patch(
            "vlm_attack.query_vlm_text",
            return_value=('{"natural": true, "feedback": ""}', None),
        ) as query:
            evaluate_attack_success_naturalness(
                image_path=Path("comparison.png"),
                candidate_prompt="ignored",
                args=args,
                is_source_vs_edited_comparison=True,
            )

        verifier_call = query.call_args.kwargs
        self.assertEqual(verifier_call["vlm_backend"], "qwen")
        self.assertEqual(
            verifier_call["vlm_model_id"],
            QWEN3_VL_4B_INSTRUCT_MODEL_ID,
        )
        self.assertFalse(verifier_call["enable_thinking"])
        self.assertFalse(verifier_call["do_sample"])

    def test_internvl3_5_mode_matches_naturalness_verifier(self) -> None:
        resolved = resolve_strategy_mllm_runtime(
            mode="OpenGVLab/InternVL3_5-4B",
            configured_backend="gemma4",
            configured_model_id="google/gemma-4-E4B-it",
            configured_thinking=True,
            configured_do_sample=True,
        )
        self.assertEqual(
            resolved,
            (
                "internvl3_5_4b",
                "internvl",
                INTERNVL3_5_4B_MODEL_ID,
                False,
                True,
            ),
        )
        instruct_resolved = resolve_strategy_mllm_runtime(
            mode="OpenGVLab/InternVL3_5-4B-Instruct",
            configured_backend="gemma4",
            configured_model_id="google/gemma-4-E4B-it",
            configured_thinking=True,
            configured_do_sample=True,
        )
        self.assertEqual(
            instruct_resolved,
            (
                "internvl3_5_4b_instruct",
                "internvl",
                INTERNVL3_5_4B_INSTRUCT_MODEL_ID,
                False,
                True,
            ),
        )

        args = SimpleNamespace(
            strategy_mllm_mode="internvl3_5_4b_instruct",
            class_ablation=False,
            class_name="goldfish",
            classifier_name="resnet50",
            model_path="black-forest-labs/FLUX.2-klein-9b-kv",
            bernini_edit_prompt_mode=False,
            qwen_edit_prompt_mode=False,
            qwen_strategy_and_enable=False,
            gcg_occurrence=0,
            gcg_scene_vocab_topic=None,
            gcg_scene_vocab_size=3,
            gcg_scene_vocab_prompts_per_strategy=1,
            gcg_scene_vocab_enabled_strategies="all",
            gcg_slot_candidate_max_words=5,
            gcg_scene_vocab_feedback=False,
            gcg_scene_feedback_limit=1000,
            cwor_enable=True,
            gcg_scene_llm_backend="gemma4",
            gcg_scene_llm_model_id="google/gemma-4-E4B-it",
            gcg_scene_llm_device="cpu",
            gcg_scene_llm_max_new_tokens=128,
            gcg_scene_llm_thinking=True,
            gcg_scene_llm_do_sample=True,
            gcg_eval_naturalness_llm_thinking=True,
        )
        strategy_answer = json.dumps(
            {
                "strategies": {
                    "background_shift": ["move into a forest"],
                    "weather_atmosphere": ["add soft rain"],
                    "texture_material": ["coated in brushed metal"],
                }
            }
        )
        with patch(
            "vlm_attack.query_vlm_text",
            return_value=(strategy_answer, None),
        ) as strategy_query:
            words, _, _, strategy_error = generate_scene_vocab_words(
                args=args,
                step_idx=0,
                current_prompt="a photo of goldfish in the background",
                current_word="background",
                slot_kind="scene",
                best_objective=0.0,
                previous_feedback=[],
                reference_image_path=Path("unused-reference.png"),
                fallback_word="outdoor",
            )

        self.assertIsNone(strategy_error)
        self.assertEqual(
            words,
            [
                "move into a forest",
                "add soft rain",
                "coated in brushed metal",
            ],
        )
        self.assertEqual(
            strategy_query.call_args.kwargs["vlm_backend"],
            "internvl",
        )
        self.assertEqual(
            strategy_query.call_args.kwargs["vlm_model_id"],
            INTERNVL3_5_4B_INSTRUCT_MODEL_ID,
        )

        with patch(
            "vlm_attack.query_vlm_text",
            return_value=('{"natural": true, "feedback": ""}', None),
        ) as query:
            evaluate_attack_success_naturalness(
                image_path=Path("comparison.png"),
                candidate_prompt="ignored",
                args=args,
                is_source_vs_edited_comparison=True,
            )

        verifier_call = query.call_args.kwargs
        self.assertEqual(verifier_call["vlm_backend"], "internvl")
        self.assertEqual(
            verifier_call["vlm_model_id"],
            INTERNVL3_5_4B_INSTRUCT_MODEL_ID,
        )
        self.assertFalse(verifier_call["enable_thinking"])
        self.assertFalse(verifier_call["do_sample"])

    def test_internvl_uses_official_custom_chat_runtime(self) -> None:
        class FakePreTrainedModel:
            pass

        raw_model = Mock()
        raw_model.eval.return_value = raw_model
        raw_model.to.return_value = raw_model
        tokenizer = object()
        fake_auto_model = Mock()
        fake_auto_model.from_pretrained.side_effect = (
            lambda *args, **kwargs: (
                self.assertEqual(
                    FakePreTrainedModel.all_tied_weights_keys,
                    {},
                )
                or raw_model
            )
        )
        fake_auto_tokenizer = Mock()
        fake_auto_tokenizer.from_pretrained.return_value = tokenizer
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = fake_auto_model
        fake_transformers.AutoTokenizer = fake_auto_tokenizer
        fake_modeling_utils = ModuleType("transformers.modeling_utils")
        fake_modeling_utils.PreTrainedModel = FakePreTrainedModel

        with patch.dict(
            sys.modules,
            {
                "transformers": fake_transformers,
                "transformers.modeling_utils": fake_modeling_utils,
            },
        ):
            model, processor, ask_fn, uses_pipeline = load_vlm_runtime(
                backend="internvl",
                model_id=INTERNVL3_5_4B_MODEL_ID,
                vlm_dtype=torch.float32,
                vlm_device=torch.device("cpu"),
                allow_blip=True,
            )

        self.assertIsInstance(model, _InternVLChatRuntime)
        self.assertIs(model.model, raw_model)
        self.assertIs(model.tokenizer, tokenizer)
        self.assertIsNone(processor)
        self.assertIs(ask_fn, _ask_with_internvl_chat)
        self.assertTrue(uses_pipeline)
        self.assertFalse(
            hasattr(FakePreTrainedModel, "all_tied_weights_keys")
        )
        model_call = fake_auto_model.from_pretrained.call_args
        self.assertEqual(
            model_call.args,
            (INTERNVL3_5_4B_MODEL_ID,),
        )
        self.assertTrue(model_call.kwargs["trust_remote_code"])
        self.assertTrue(model_call.kwargs["low_cpu_mem_usage"])
        self.assertIs(model_call.kwargs["dtype"], torch.float32)
        raw_model.eval.assert_called_once_with()
        raw_model.to.assert_called_once_with(torch.device("cpu"))
        fake_auto_tokenizer.from_pretrained.assert_called_once_with(
            INTERNVL3_5_4B_MODEL_ID,
            trust_remote_code=True,
            use_fast=False,
        )

    def test_internvl_custom_chat_uses_official_image_prompt_contract(self) -> None:
        raw_model = Mock()
        raw_model.chat.return_value = '{"natural": true, "feedback": ""}'
        tokenizer = object()
        runtime = _InternVLChatRuntime(
            model=raw_model,
            tokenizer=tokenizer,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pixels = torch.ones((1, 3, 448, 448), dtype=torch.float32)

        with patch(
            "vlm_runtime._prepare_internvl_pixel_values",
            return_value=pixels,
        ) as prepare:
            answer = _ask_with_internvl_chat(
                image=Image.new("RGB", (32, 24)),
                question="Return JSON.",
                model=runtime,
                processor=None,
                device=torch.device("cpu"),
                max_new_tokens=64,
                enable_thinking=False,
                do_sample=False,
            )

        self.assertEqual(answer, '{"natural": true, "feedback": ""}')
        prepare.assert_called_once()
        raw_model.chat.assert_called_once_with(
            tokenizer,
            pixels,
            "<image>\nReturn JSON.",
            {"max_new_tokens": 64, "do_sample": False},
        )

    def test_qwen3_vl_pipeline_uses_multimodal_chat_messages(self) -> None:
        fake_pipeline = Mock(
            return_value=[
                {
                    "generated_text": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "answer"}],
                        }
                    ]
                }
            ]
        )
        image = Image.new("RGB", (4, 4), "white")

        answer = _ask_with_qwen_pipeline(
            image=image,
            question="question",
            model=fake_pipeline,
            processor=None,
            device=torch.device("cpu"),
            max_new_tokens=32,
            enable_thinking=False,
            do_sample=True,
        )

        self.assertEqual(answer, "answer")
        call = fake_pipeline.call_args
        self.assertEqual(
            call.kwargs["generate_kwargs"],
            {"max_new_tokens": 32, "do_sample": True},
        )
        self.assertNotIn("enable_thinking", call.kwargs)
        messages = call.kwargs["text"]
        self.assertIs(messages[0]["content"][0]["image"], image)
        self.assertEqual(messages[0]["content"][1]["text"], "question")

    def test_clean_correct_filter_option_defaults_off_and_parses(self) -> None:
        default_args = build_runner_parser().parse_args([])
        enabled_args = build_runner_parser().parse_args(
            [
                "--attack_only_clean_correct",
                "true",
                "--clean_correct_sample_size",
                "100",
                "--clean_correct_sample_seed",
                "2",
            ]
        )

        self.assertFalse(default_args.attack_only_clean_correct)
        self.assertEqual(default_args.clean_correct_sample_size, 0)
        self.assertIsNone(default_args.clean_correct_sample_seed)
        self.assertTrue(enabled_args.attack_only_clean_correct)
        self.assertEqual(enabled_args.clean_correct_sample_size, 100)
        self.assertEqual(enabled_args.clean_correct_sample_seed, 2)

    def test_clean_correct_seeded_sampling_is_exact_and_deterministic(self) -> None:
        population = set(range(250))

        seed0_first = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=0,
        )
        seed0_second = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=0,
        )
        seed1 = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=1,
        )

        self.assertEqual(len(seed0_first), 100)
        self.assertEqual(seed0_first, seed0_second)
        self.assertNotEqual(seed0_first, seed1)
        self.assertTrue(seed0_first <= population)

    def test_clean_correct_seeded_sampling_rejects_an_undersized_pool(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requested=100 available=99",
        ):
            sample_clean_correct_indices(
                set(range(99)),
                sample_size=100,
                seed=0,
            )

    def test_qwen_clean_correct_window_options_default_and_parse(self) -> None:
        default_args = build_qwen_runner_parser().parse_args([])
        selected_args = build_qwen_runner_parser().parse_args(
            [
                "--attack_only_clean_correct",
                "true",
                "--clean_correct_skip",
                "100",
                "--clean_correct_count",
                "100",
            ]
        )

        self.assertEqual(default_args.clean_correct_skip, 0)
        self.assertEqual(default_args.clean_correct_count, 0)
        self.assertEqual(selected_args.clean_correct_skip, 100)
        self.assertEqual(selected_args.clean_correct_count, 100)

    def test_qwen_clean_correct_seed_options_default_and_parse(self) -> None:
        default_args = build_qwen_runner_parser().parse_args([])
        selected_args = build_qwen_runner_parser().parse_args(
            [
                "--attack_only_clean_correct",
                "true",
                "--clean_correct_sample_size",
                "100",
                "--clean_correct_sample_seed",
                "7",
            ]
        )

        self.assertEqual(default_args.clean_correct_sample_size, 0)
        self.assertIsNone(default_args.clean_correct_sample_seed)
        self.assertEqual(selected_args.clean_correct_sample_size, 100)
        self.assertEqual(selected_args.clean_correct_sample_seed, 7)

    def test_qwen_clean_correct_window_preserves_candidate_file_order(self) -> None:
        selected = qwen_runner.slice_clean_correct_indices(
            candidate_indices=[8, 3, 5, 1, 9, 2],
            clean_correct_indices={8, 5, 1, 9, 2},
            skip=2,
            count=2,
        )

        self.assertEqual(selected, [1, 9])

    def test_qwen_clean_correct_window_requires_the_exact_requested_count(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "available_after_skip=1",
        ):
            qwen_runner.slice_clean_correct_indices(
                candidate_indices=[8, 3, 5],
                clean_correct_indices={8, 5},
                skip=1,
                count=2,
            )

    def test_qwen_query_summary_reports_success_and_all_sample_means(self) -> None:
        metrics = qwen_runner.summarize_attack_query_metrics(
            [
                {
                    "final_attack_success": True,
                    "victim_query_count": 4,
                    "attack_success_query_count": 3,
                },
                {
                    "final_attack_success": False,
                    "victim_query_count": 100,
                },
                {
                    "attack_success_before_failure": True,
                    "partial_core_report": {
                        "victim_query_count": 7,
                        "attack_success_query_count": 5,
                    },
                },
            ]
        )

        self.assertEqual(metrics["attack_success_count"], 2)
        self.assertAlmostEqual(metrics["attack_success_rate_percent"], 200.0 / 3.0)
        self.assertEqual(metrics["successful_attack_query_count_recorded"], 2)
        self.assertEqual(metrics["successful_attack_query_mean"], 4.0)
        self.assertEqual(metrics["all_sample_query_count_recorded"], 3)
        self.assertEqual(metrics["all_sample_query_mean"], 37.0)

    def test_qwen_query_summary_marks_incomplete_average_unknown(self) -> None:
        metrics = qwen_runner.summarize_attack_query_metrics(
            [
                {
                    "final_attack_success": True,
                    "victim_query_count": 2,
                    "attack_success_query_count": 2,
                },
                {"status": "failed", "error": "runtime error"},
            ]
        )

        self.assertEqual(metrics["all_sample_query_count_recorded"], 1)
        self.assertIsNone(metrics["all_sample_query_mean"])

    def test_bernini_query_summary_reports_success_and_all_sample_means(self) -> None:
        metrics = bernini_runner.summarize_attack_query_metrics(
            [
                {
                    "final_attack_success": True,
                    "victim_query_count": 4,
                    "attack_success_query_count": 3,
                },
                {
                    "final_attack_success": False,
                    "victim_query_count": 100,
                },
                {
                    "attack_success_before_failure": True,
                    "partial_core_report": {
                        "victim_query_count": 7,
                        "attack_success_query_count": 5,
                    },
                },
            ]
        )

        self.assertEqual(metrics["attack_success_count"], 2)
        self.assertAlmostEqual(metrics["attack_success_rate_percent"], 200.0 / 3.0)
        self.assertEqual(metrics["successful_attack_query_count_recorded"], 2)
        self.assertEqual(metrics["successful_attack_query_mean"], 4.0)
        self.assertEqual(metrics["all_sample_query_count_recorded"], 3)
        self.assertEqual(metrics["all_sample_query_mean"], 37.0)

    def test_class_ablation_config_is_isolated_from_the_baseline(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "configs"
        baseline = yaml.safe_load(
            (config_root / "flux2_and_attack_nips.yaml").read_text(
                encoding="utf-8"
            )
        )
        ablation = yaml.safe_load(
            (
                config_root
                / "flux2_and_attack_nips_no_explicit_class_guidance.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertNotIn("class_ablation", baseline["run_args"])
        self.assertIs(ablation["run_args"]["class_ablation"], True)
        comparable_ablation = dict(ablation["run_args"])
        comparable_ablation.pop("class_ablation")
        self.assertEqual(comparable_ablation, baseline["run_args"])

    def test_adv_res_and_siglip2_clean100_configs_share_required_rules(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "configs"
        expected_victims = {
            "flux2_and_attack_nips_adv_res.yaml": "adv_res",
            "flux2_and_attack_nips_siglip2.yaml": "siglip2",
        }
        for config_name, expected_victim in expected_victims.items():
            with self.subTest(config=config_name):
                payload = yaml.safe_load(
                    (config_root / config_name).read_text(encoding="utf-8")
                )
                args = payload["run_args"]
                self.assertEqual(args["victim_model"], expected_victim)
                self.assertEqual(
                    args["gcg_scene_vocab_enabled_strategies"],
                    "all",
                )
                self.assertIs(args["gcg_scene_llm_do_sample"], True)
                self.assertIs(
                    args["gcg_eval_naturalness_on_attack_success"],
                    True,
                )
                self.assertIs(
                    args["gcg_eval_naturalness_llm_thinking"],
                    False,
                )
                self.assertIs(args["attack_only_clean_correct"], True)
                self.assertEqual(args["clean_correct_sample_size"], 100)
                self.assertEqual(args["clean_correct_sample_seed"], 0)
                self.assertEqual(args["manual_seed"], 0)

    def test_class_ablation_neutralizes_all_supported_placeholders(self) -> None:
        markers = (
            "<class>",
            "{class}",
            "{class_name}",
            "{target_class_name}",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                args = SimpleNamespace(
                    prompt=f"a photo of {marker} in the background",
                    prompts=None,
                    class_name="goldfish",
                    class_ablation=True,
                    classifier_name="resnet50",
                    classifier_label=1,
                )
                with patch(
                    "vlm_attack_blackbox_core.infer_imagenet_class_name"
                ) as infer_class:
                    resolved = _resolve_prompt(args)

                infer_class.assert_not_called()
                self.assertEqual(
                    resolved,
                    "a photo of the subject in the background",
                )
                self.assertEqual(args.prompt, resolved)
                self.assertIsNone(args.class_name)
                self.assertNotIn(marker, resolved)

    def test_default_prompt_resolution_and_class_instruction_are_preserved(self) -> None:
        args = SimpleNamespace(
            prompt="a photo of <class> in the background",
            prompts=None,
            class_name=None,
            class_ablation=False,
            classifier_name="resnet50",
            classifier_label=1,
        )
        with patch(
            "vlm_attack_blackbox_core.infer_imagenet_class_name",
            return_value="goldfish",
        ) as infer_class:
            resolved = _resolve_prompt(args)

        infer_class.assert_called_once()
        self.assertEqual(resolved, "a photo of goldfish in the background")
        self.assertEqual(args.class_name, "goldfish")

    def test_repository_categories_csv_supports_non_torchvision_victims(self) -> None:
        self.assertEqual(infer_imagenet_class_name("adv_res", 0), "tench")
        self.assertEqual(infer_imagenet_class_name("siglip2", 1), "goldfish")

    def test_class_ablation_removes_explicit_class_text_from_llm_prompt(self) -> None:
        raw_answer = json.dumps(
            {
                "strategies": [
                    {
                        "name": "background_shift",
                        "candidates": ["move into a forest"],
                    },
                    {
                        "name": "weather_atmosphere",
                        "candidates": ["add soft rain"],
                    },
                    {
                        "name": "texture_material",
                        "candidates": ["coated in brushed metal"],
                    },
                ]
            }
        )

        def make_args(class_ablation: bool) -> SimpleNamespace:
            return SimpleNamespace(
                class_ablation=class_ablation,
                class_name="goldfish",
                classifier_name="resnet50",
                model_path="black-forest-labs/FLUX.2-klein-9b-kv",
                bernini_edit_prompt_mode=False,
                qwen_edit_prompt_mode=False,
                qwen_strategy_and_enable=False,
                gcg_occurrence=0,
                gcg_scene_vocab_topic=None,
                gcg_scene_vocab_size=3,
                gcg_scene_vocab_prompts_per_strategy=1,
                gcg_scene_vocab_enabled_strategies="all",
                gcg_slot_candidate_max_words=5,
                gcg_scene_vocab_feedback=False,
                gcg_scene_feedback_limit=1000,
                cwor_enable=True,
                gcg_scene_llm_backend="gemma4",
                gcg_scene_llm_model_id="test-model",
                gcg_scene_llm_device="cpu",
                gcg_scene_llm_max_new_tokens=128,
                gcg_scene_llm_thinking=False,
                gcg_scene_llm_do_sample=False,
            )

        captured_prompts = []

        def fake_query_vlm_text(**kwargs):
            captured_prompts.append(kwargs["question"])
            return raw_answer, None

        with patch(
            "vlm_attack.query_vlm_text",
            side_effect=fake_query_vlm_text,
        ):
            for enabled in (False, True):
                generate_scene_vocab_words(
                    args=make_args(enabled),
                    step_idx=0,
                    current_prompt="a photo of goldfish in the background",
                    current_word="background",
                    slot_kind="scene",
                    best_objective=0.0,
                    previous_feedback=[],
                    reference_image_path=Path("unused-reference.png"),
                    fallback_word="outdoor",
                )

        baseline_prompt, ablation_prompt = captured_prompts
        class_instruction = (
            "Every candidate must be semantically related to the target class"
        )
        self.assertIn(class_instruction, baseline_prompt)
        self.assertIn("goldfish", baseline_prompt)
        self.assertIn("<class>", baseline_prompt)

        self.assertNotIn(class_instruction, ablation_prompt)
        self.assertNotIn("goldfish", ablation_prompt)
        for marker in (
            "<class>",
            "{class}",
            "{class_name}",
            "{target_class_name}",
        ):
            self.assertNotIn(marker, ablation_prompt)
        self.assertIn(
            'Prompt template with slot marker: "a photo of the subject in the <scene>"',
            ablation_prompt,
        )

    def test_flux2_runner_treats_hf_token_as_optional(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "flux2_attack_runner.py"
        ).read_text(encoding="utf-8")

        self.assertIn("resolve_optional_hf_token(cfg.hf_token)", source)
        self.assertNotIn("resolve_hf_token(cfg.hf_token)", source)

    def test_flux2_runner_resumes_only_matching_verified_report(self) -> None:
        cfg = SimpleNamespace(
            attack_mode="and",
            strategy_mllm_mode="qwen3_vl_4b_instruct",
            gcg_scene_vocab_enabled_strategies="all",
            gcg_scene_vocab_prompts_per_strategy=1,
            gcg_scene_llm_do_sample=True,
            gcg_eval_naturalness_on_attack_success=True,
            gcg_eval_naturalness_llm_thinking=False,
            scene_vlm_do_sample=False,
            manual_seed=0,
        )
        payload = {
            "final_attack_success": True,
            "args": {
                "attack_mode": "and",
                "strategy_mllm_mode": "qwen3_vl_4b_instruct",
                "gcg_scene_vocab_enabled_strategies": "all",
                "gcg_scene_vocab_prompts_per_strategy": 1,
                "gcg_scene_llm_do_sample": True,
                "gcg_eval_naturalness_on_attack_success": True,
                "gcg_eval_naturalness_llm_thinking": False,
                "seed": 0,
            },
            "strategy_generator": {
                "mode": "qwen3_vl_4b_instruct",
                "do_sample": True,
            },
            "naturalness_verifier": {
                "enabled": True,
                "mode": "qwen3_vl_4b_instruct",
                "do_sample": False,
            },
            "history": [
                {
                    "raw_vlm_answer": "{}",
                    "strategy_query_accounting_version": 3,
                    "strategy_slot_count": 3,
                    "strategy_duplicate_skipped_count": 1,
                    "strategy_duplicate_query_count": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            resumed = load_resumable_report(
                report_path,
                cfg=cfg,
                sample_index=7,
                image_id="image",
                true_label=1,
                target_label=2,
                sample_dir=Path(tmpdir),
            )
            self.assertIsNotNone(resumed)
            self.assertEqual(resumed["sample_index"], 7)
            self.assertTrue(resumed["resumed_from_existing_report"])

            payload["args"]["seed"] = 1
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                load_resumable_report(
                    report_path,
                    cfg=cfg,
                    sample_index=7,
                    image_id="image",
                    true_label=1,
                    target_label=2,
                    sample_dir=Path(tmpdir),
                )
            )

    def test_flux2_runner_resumes_none_strategy_report_with_zero_slots(self) -> None:
        cfg = SimpleNamespace(
            attack_mode="vlm",
            strategy_mllm_mode="internvl3_5_4b_instruct",
            gcg_scene_vocab_enabled_strategies="none",
            gcg_scene_vocab_prompts_per_strategy=0,
            gcg_scene_llm_do_sample=False,
            gcg_eval_naturalness_on_attack_success=True,
            gcg_eval_naturalness_llm_thinking=False,
            scene_vlm_do_sample=False,
            manual_seed=0,
        )
        payload = {
            "final_attack_success": False,
            "args": {
                "attack_mode": "vlm",
                "strategy_mllm_mode": "internvl3_5_4b_instruct",
                "gcg_scene_vocab_enabled_strategies": "none",
                "gcg_scene_vocab_prompts_per_strategy": 0,
                "gcg_scene_llm_do_sample": False,
                "gcg_eval_naturalness_on_attack_success": True,
                "gcg_eval_naturalness_llm_thinking": False,
                "seed": 0,
            },
            "strategy_generator": {
                "mode": "internvl3_5_4b_instruct",
                "do_sample": False,
            },
            "naturalness_verifier": {
                "enabled": True,
                "mode": "internvl3_5_4b_instruct",
                "do_sample": False,
            },
            "history": [
                {
                    "raw_vlm_answer": "{}",
                    "strategy_query_accounting_version": 3,
                    "strategy_slot_count": 0,
                    "strategy_duplicate_skipped_count": 0,
                    "strategy_duplicate_query_count": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            resumed = load_resumable_report(
                report_path,
                cfg=cfg,
                sample_index=0,
                image_id="image",
                true_label=1,
                target_label=2,
                sample_dir=Path(tmpdir),
            )
            self.assertIsNotNone(resumed)
            self.assertTrue(resumed["resumed_from_existing_report"])

    def test_flux2_resume_config_reads_core_only_sampling_args(self) -> None:
        cfg = SimpleNamespace(
            attack_mode="and",
            strategy_mllm_mode="qwen3_vl_4b_instruct",
            gcg_scene_vocab_enabled_strategies="all",
            gcg_scene_vocab_prompts_per_strategy=1,
            manual_seed=2,
            model_path="black-forest-labs/FLUX.2-klein-9b-kv",
        )
        resume_cfg = build_resume_validation_config(
            cfg,
            [
                "--gcg_scene_llm_do_sample",
                "true",
                "--scene_vlm_do_sample",
                "false",
                "--gcg_eval_naturalness_on_attack_success",
                "true",
                "--gcg_eval_naturalness_llm_thinking",
                "false",
            ],
        )
        self.assertTrue(resume_cfg.gcg_scene_llm_do_sample)
        self.assertFalse(resume_cfg.scene_vlm_do_sample)
        self.assertTrue(
            resume_cfg.gcg_eval_naturalness_on_attack_success
        )
        self.assertFalse(resume_cfg.gcg_eval_naturalness_llm_thinking)
        self.assertEqual(resume_cfg.manual_seed, 2)

    def test_flux2_pipeline_only_passes_nonempty_hf_token(self) -> None:
        captured_load_kwargs = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_path, **kwargs):
                del model_path
                captured_load_kwargs.append(dict(kwargs))
                return cls()

        fake_diffusers = SimpleNamespace(
            Flux2KleinPipeline=FakePipeline,
            Flux2KleinKVPipeline=FakePipeline,
        )

        def fake_import_module(name):
            if name == "diffusers":
                return fake_diffusers
            return Mock()

        with patch(
            "flux2_blackbox_runtime.importlib.import_module",
            side_effect=fake_import_module,
        ):
            for hf_token in ("", "hf-test-token"):
                with self.subTest(hf_token=bool(hf_token)):
                    Flux2KleinRenderSession(
                        args=SimpleNamespace(
                            model_path="black-forest-labs/FLUX.2-klein-9b-kv",
                            hf_token=hf_token,
                            cpu_offload=False,
                        ),
                        has_input_image=False,
                    )

        self.assertNotIn("token", captured_load_kwargs[0])
        self.assertEqual(captured_load_kwargs[1]["token"], "hf-test-token")

    def test_flux2_kv_pipeline_omits_unsupported_guidance_scale(self) -> None:
        class FakeKVPipeline:
            def __call__(
                self,
                *,
                prompt=None,
                image=None,
                num_inference_steps=4,
                max_sequence_length=512,
                output_type="pil",
            ):
                del prompt, image, num_inference_steps, max_sequence_length, output_type

        session = object.__new__(Flux2KleinRenderSession)
        session.args = SimpleNamespace(
            num_inference_steps=4,
            max_sequence_length=512,
            guidance_scale=3.0,
            height=1024,
            width=1024,
        )
        session.has_input_image = False
        session.pipe = FakeKVPipeline()

        kwargs = session._pipeline_kwargs()

        self.assertNotIn("guidance_scale", kwargs)

    def test_flux2_pipeline_keeps_supported_guidance_scale(self) -> None:
        class FakePipeline:
            def __call__(
                self,
                *,
                prompt=None,
                guidance_scale=4.0,
                num_inference_steps=4,
                max_sequence_length=512,
                output_type="pil",
            ):
                del prompt, guidance_scale, num_inference_steps, max_sequence_length, output_type

        session = object.__new__(Flux2KleinRenderSession)
        session.args = SimpleNamespace(
            num_inference_steps=4,
            max_sequence_length=512,
            guidance_scale=3.25,
            height=1024,
            width=1024,
        )
        session.has_input_image = False
        session.pipe = FakePipeline()

        kwargs = session._pipeline_kwargs()

        self.assertEqual(kwargs["guidance_scale"], 3.25)

    def test_qwen_batch_parser_and_core_args_propagate(self) -> None:
        cfg = build_qwen_runner_parser().parse_args(
            [
                "--qwen_batch_size",
                "3",
                "--qwen_batch_fallback",
                "false",
            ]
        )
        cfg.qwen_attack_mode = "vlm"
        core_args = argparse.Namespace()

        apply_qwen_runtime_args(core_args, cfg)

        self.assertEqual(core_args.qwen_batch_size, 3)
        self.assertFalse(core_args.qwen_batch_fallback)

    def test_qwen_launcher_forwards_batch_options(self) -> None:
        launcher = (ISOLATED_ROOT / "run_vlm_attack.sh").read_text(encoding="utf-8")

        self.assertIn('"qwen_batch_size": "QWEN_BATCH_SIZE"', launcher)
        self.assertIn('"qwen_batch_fallback": "QWEN_BATCH_FALLBACK"', launcher)
        self.assertIn(
            "append_optional QWEN_BATCH_SIZE --qwen_batch_size",
            launcher,
        )
        self.assertIn(
            "append_optional QWEN_BATCH_FALLBACK --qwen_batch_fallback",
            launcher,
        )

    def test_qwen_launcher_forwards_clean_correct_window_options(self) -> None:
        launcher = (ISOLATED_ROOT / "run_vlm_attack.sh").read_text(encoding="utf-8")

        self.assertIn('"clean_correct_skip": "CLEAN_CORRECT_SKIP"', launcher)
        self.assertIn('"clean_correct_count": "CLEAN_CORRECT_COUNT"', launcher)
        self.assertIn(
            "append_optional CLEAN_CORRECT_SKIP --clean_correct_skip",
            launcher,
        )
        self.assertIn(
            "append_optional CLEAN_CORRECT_COUNT --clean_correct_count",
            launcher,
        )

    def test_launcher_forwards_clean_correct_seed_options(self) -> None:
        launcher = (ISOLATED_ROOT / "run_vlm_attack.sh").read_text(encoding="utf-8")

        self.assertIn(
            '"clean_correct_sample_size": "CLEAN_CORRECT_SAMPLE_SIZE"',
            launcher,
        )
        self.assertIn(
            '"clean_correct_sample_seed": "CLEAN_CORRECT_SAMPLE_SEED"',
            launcher,
        )
        self.assertIn(
            '--clean_correct_sample_size "$CLEAN_CORRECT_SAMPLE_SIZE"',
            launcher,
        )
        self.assertIn(
            'CMD+=(--clean_correct_sample_seed "$CLEAN_CORRECT_SAMPLE_SEED")',
            launcher,
        )
        self.assertEqual(
            launcher.count(
                'CMD+=(--clean_correct_sample_size "$CLEAN_CORRECT_SAMPLE_SIZE")'
            ),
            2,
        )
        self.assertEqual(
            launcher.count(
                'CMD+=(--clean_correct_sample_seed "$CLEAN_CORRECT_SAMPLE_SEED")'
            ),
            2,
        )

    def test_qwen_seeded_clean100_launcher_samples_from_all_candidates(self) -> None:
        launcher = (
            ISOLATED_ROOT / "run_qwen_vlm_res_seeded_clean100.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('SAMPLE_SIZE="${CLEAN_CORRECT_SAMPLE_SIZE:-100}"', launcher)
        self.assertIn('SAMPLE_SEED="${CLEAN_CORRECT_SAMPLE_SEED:-${SEED:-0}}"', launcher)
        self.assertIn('export SAMPLE_INDICES_FILE=""', launcher)
        self.assertIn('export START_INDEX=0', launcher)
        self.assertIn('--clean_correct_sample_size "$SAMPLE_SIZE"', launcher)
        self.assertIn('--clean_correct_sample_seed "$SAMPLE_SEED"', launcher)
        self.assertIn('--manual_seed "$GENERATION_SEED"', launcher)
        self.assertIn(
            'exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh"',
            launcher,
        )

    def test_bernini_seeded_clean100_launchers_fix_seed_values(self) -> None:
        common = (
            ISOLATED_ROOT / "run_bernini_vlm_res_seeded_clean100.sh"
        ).read_text(encoding="utf-8")
        seed0 = (
            ISOLATED_ROOT / "run_bernini_vlm_res_seed0_clean100.sh"
        ).read_text(encoding="utf-8")
        seed1 = (
            ISOLATED_ROOT / "run_bernini_vlm_res_seed1_clean100.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('SAMPLE_SIZE="${CLEAN_CORRECT_SAMPLE_SIZE:-100}"', common)
        self.assertIn('SAMPLE_SEED="${CLEAN_CORRECT_SAMPLE_SEED:-${SEED:-0}}"', common)
        self.assertIn('export SAMPLE_INDICES_FILE=""', common)
        self.assertIn('--clean_correct_sample_size "$SAMPLE_SIZE"', common)
        self.assertIn('--clean_correct_sample_seed "$SAMPLE_SEED"', common)
        self.assertIn('--manual_seed "$GENERATION_SEED"', common)
        self.assertIn(
            'exec bash "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh"',
            common,
        )
        self.assertIn('export CLEAN_CORRECT_SAMPLE_SEED=0', seed0)
        self.assertIn('export MANUAL_SEED=0', seed0)
        self.assertIn('export CLEAN_CORRECT_SAMPLE_SEED=1', seed1)
        self.assertIn('export MANUAL_SEED=1', seed1)

    def test_bernini_remote_bootstrap_installs_art_and_pins_compatible_source(self) -> None:
        bootstrap = (
            ISOLATED_ROOT / "run_bernini_vlm_res_remote.sh"
        ).read_text(encoding="utf-8")
        requirements = (
            ISOLATED_ROOT.parent / "requirements-bernini-faas.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("-r requirements-faas.txt", requirements)
        self.assertIn("decord", requirements)
        self.assertIn("requirements-bernini-faas.txt", bootstrap)
        self.assertIn(
            "from art.estimators.classification import PyTorchClassifier",
            bootstrap,
        )
        self.assertIn("https://github.com/bytedance/Bernini.git", bootstrap)
        self.assertIn("9a366af09c93a94a014e7c6df9782155a67908ef", bootstrap)
        self.assertIn("ByteDance/Bernini-R-Diffusers", bootstrap)
        self.assertNotIn("python -m venv", bootstrap)

    def test_qwen_second_clean100_launcher_uses_target_200_window(self) -> None:
        launcher_path = (
            ISOLATED_ROOT / "run_qwen_vlm_res_second_clean100.sh"
        )
        launcher = launcher_path.read_text(encoding="utf-8")
        remote_bootstrap = (
            ISOLATED_ROOT / "run_qwen_vlm_res_remote.sh"
        ).read_text(encoding="utf-8")
        faas_requirements = (
            ISOLATED_ROOT.parent / "requirements-faas.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("target_200_indices.json", launcher)
        self.assertIn("--clean_correct_skip 100", launcher)
        self.assertIn("--clean_correct_count 100", launcher)
        self.assertIn(
            'exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh"',
            launcher,
        )
        self.assertNotIn(
            'exec bash "$SCRIPT_DIR/run_vlm_attack.sh"',
            launcher,
        )
        self.assertIn("--attack_only_clean_correct true", remote_bootstrap)
        self.assertIn("--qwen_batch_size 1", remote_bootstrap)
        self.assertIn(
            "from art.estimators.classification import PyTorchClassifier",
            remote_bootstrap,
        )
        self.assertIn(
            "adversarial-robustness-toolbox",
            faas_requirements,
        )

    def test_qwen_after_clean200_launcher_uses_all_dataset_candidates(self) -> None:
        launcher = (
            ISOLATED_ROOT / "run_qwen_vlm_res_after_clean200.sh"
        ).read_text(encoding="utf-8")
        remote_bootstrap = (
            ISOLATED_ROOT / "run_qwen_vlm_res_remote.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('export SAMPLE_INDICES_FILE=""', launcher)
        self.assertIn('CLEAN_CORRECT_COUNT="${CLEAN_CORRECT_COUNT:-0}"', launcher)
        self.assertIn("--clean_correct_skip 200", launcher)
        self.assertIn(
            '--clean_correct_count "$CLEAN_CORRECT_COUNT"',
            launcher,
        )
        self.assertIn(
            'exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh"',
            launcher,
        )
        self.assertIn(
            'SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE-',
            remote_bootstrap,
        )
        self.assertIn(
            'CMD+=(--sample_indices_file "$SAMPLE_INDICES_FILE")',
            remote_bootstrap,
        )

    def test_qwen_prompt_embedding_batches_are_padded_with_attention_mask(self) -> None:
        short = torch.ones((1, 2, 3), dtype=torch.float32)
        long = torch.full((1, 4, 3), 2.0, dtype=torch.float32)

        embeddings, mask = _pad_prompt_batches(
            [
                (short, None),
                (long, torch.ones((1, 4), dtype=torch.long)),
            ]
        )

        self.assertEqual(tuple(embeddings.shape), (2, 4, 3))
        self.assertIsNotNone(mask)
        self.assertEqual(mask.tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertTrue(torch.equal(embeddings[0, :2], short[0]))
        self.assertTrue(torch.equal(embeddings[0, 2:], torch.zeros((2, 3))))

    def test_qwen_custom_batch_executes_one_batched_denoising_loop(self) -> None:
        class FakeImageProcessor:
            @staticmethod
            def resize(image, height, width):
                del height, width
                return image.copy()

            @staticmethod
            def preprocess(image, height, width):
                del image, height, width
                return torch.zeros((1, 3, 4, 4), dtype=torch.float32)

            @staticmethod
            def postprocess(decoded, output_type):
                self.assertEqual(output_type, "pil")
                return [
                    Image.new("RGB", (2, 2), (index, index, index))
                    for index in range(int(decoded.shape[0]))
                ]

        class FakeScheduler:
            order = 1
            config = {}

            def __init__(self):
                self.timesteps = torch.tensor([], dtype=torch.float32)
                self.begin_index = None

            def set_timesteps(self, *, sigmas, device, mu):
                del mu
                self.timesteps = torch.arange(
                    len(sigmas),
                    0,
                    -1,
                    dtype=torch.float32,
                    device=device,
                )

            def set_begin_index(self, index):
                self.begin_index = int(index)

            @staticmethod
            def step(noise_pred, timestep, latents, return_dict):
                del noise_pred, timestep, return_dict
                return (latents,)

        class FakeTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(in_channels=4, guidance_embeds=False)
                self.batch_sizes = []

            def forward(
                self,
                *,
                hidden_states,
                timestep,
                guidance,
                encoder_hidden_states_mask,
                encoder_hidden_states,
                img_shapes,
                attention_kwargs,
                return_dict,
            ):
                del (
                    timestep,
                    guidance,
                    encoder_hidden_states_mask,
                    encoder_hidden_states,
                    img_shapes,
                    attention_kwargs,
                    return_dict,
                )
                self.batch_sizes.append(int(hidden_states.shape[0]))
                return (torch.zeros_like(hidden_states),)

        class FakeVae:
            dtype = torch.float32
            config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0], z_dim=1)

            @staticmethod
            def decode(latents, return_dict):
                del return_dict
                return (latents.repeat(1, 3, 1, 1, 1),)

        class FakePipe:
            def __init__(self):
                self._execution_device = torch.device("cpu")
                self.vae_scale_factor = 2
                self.latent_channels = 1
                self.image_processor = FakeImageProcessor()
                self.scheduler = FakeScheduler()
                self.transformer = FakeTransformer()
                self.vae = FakeVae()
                self.encoded_prompts = []
                self.prepare_batch_size = None
                self.freed = False

            def encode_prompt(
                self,
                *,
                image,
                prompt,
                device,
                num_images_per_prompt,
                max_sequence_length,
            ):
                del image, device, num_images_per_prompt, max_sequence_length
                self.encoded_prompts.append(prompt)
                length = 2 if prompt == "short" else 3
                return torch.ones((1, length, 4)), None

            def prepare_latents(
                self,
                images,
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generators,
                latents,
            ):
                del (
                    images,
                    num_channels_latents,
                    height,
                    width,
                    dtype,
                    device,
                    generators,
                    latents,
                )
                self.prepare_batch_size = int(batch_size)
                return (
                    torch.zeros((batch_size, 2, 4), dtype=torch.float32),
                    torch.zeros((batch_size, 1, 4), dtype=torch.float32),
                )

            @staticmethod
            def _unpack_latents(latents, height, width, vae_scale_factor):
                del height, width, vae_scale_factor
                return torch.zeros(
                    (int(latents.shape[0]), 1, 1, 2, 2),
                    dtype=latents.dtype,
                )

            def maybe_free_model_hooks(self):
                self.freed = True

        pipe = FakePipe()
        images = render_qwen_image_edit_batch(
            pipe=pipe,
            prompts=["short", "long"],
            image=Image.new("RGB", (32, 32), "white"),
            generators=[
                torch.Generator().manual_seed(1),
                torch.Generator().manual_seed(2),
            ],
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=2,
            max_sequence_length=512,
            guidance_scale=1.0,
        )

        self.assertEqual(len(images), 2)
        self.assertEqual(pipe.encoded_prompts, ["short", "long", " "])
        self.assertEqual(pipe.prepare_batch_size, 2)
        self.assertEqual(pipe.scheduler.begin_index, 0)
        self.assertEqual(pipe.transformer.batch_sizes, [2, 2, 2, 2])
        self.assertTrue(pipe.freed)

    def test_qwen_pipeline_only_passes_guidance_scale_to_distilled_models(self) -> None:
        class FakePipeline:
            def __init__(self, *, guidance_embeds: bool):
                self.transformer = SimpleNamespace(
                    config=SimpleNamespace(guidance_embeds=guidance_embeds)
                )
                self.call_kwargs = None

            def __call__(self, **kwargs):
                self.call_kwargs = dict(kwargs)
                return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

        for guidance_embeds in (False, True):
            with self.subTest(guidance_embeds=guidance_embeds):
                session = object.__new__(QwenImageEditRenderSession)
                session.args = SimpleNamespace(
                    device="cpu",
                    qwen_true_cfg_scale=4.0,
                    qwen_negative_prompt=" ",
                    num_inference_steps=4,
                    max_sequence_length=512,
                    guidance_scale=1.75,
                    qwen_num_images_per_prompt=1,
                )
                session.pipe = FakePipeline(guidance_embeds=guidance_embeds)

                image = session._pipe_call(
                    prompt="test prompt",
                    image=Image.new("RGB", (8, 8), "black"),
                    seed=1,
                )

                self.assertEqual(image.size, (8, 8))
                if guidance_embeds:
                    self.assertEqual(session.pipe.call_kwargs["guidance_scale"], 1.75)
                else:
                    self.assertNotIn("guidance_scale", session.pipe.call_kwargs)

    def test_qwen_render_batches_distinct_prompts_and_preserves_seed_order(self) -> None:
        session = object.__new__(QwenImageEditRenderSession)
        session.args = SimpleNamespace(
            device="cpu",
            seed=40,
            cpu_offload=False,
            qwen_batch_size=2,
            qwen_batch_fallback=False,
            qwen_true_cfg_scale=4.0,
            qwen_negative_prompt=" ",
            num_inference_steps=4,
            max_sequence_length=512,
            guidance_scale=1.0,
        )
        session.pipe = object()
        condition = Image.new("RGB", (16, 16), "white")
        session._load_condition_image = Mock(return_value=condition)
        sequential_image = Image.new("RGB", (16, 16), "blue")
        session._pipe_call = Mock(return_value=sequential_image)
        batch_images = [
            Image.new("RGB", (16, 16), "red"),
            Image.new("RGB", (16, 16), "green"),
        ]

        with patch(
            "qwen2_blackbox_runtime.render_qwen_image_edit_batch",
            return_value=batch_images,
        ) as batch_render:
            images = session.render_images(prompts=["first", "second", "third"])

        self.assertEqual([image.getpixel((0, 0)) for image in images], [
            (255, 0, 0),
            (0, 128, 0),
            (0, 0, 255),
        ])
        batch_kwargs = batch_render.call_args.kwargs
        self.assertEqual(batch_kwargs["prompts"], ["first", "second"])
        self.assertEqual(
            [generator.initial_seed() for generator in batch_kwargs["generators"]],
            [40, 41],
        )
        session._pipe_call.assert_called_once_with(
            prompt="third",
            image=condition,
            seed=42,
        )

    def test_qwen_batch_failure_retries_complete_set_sequentially(self) -> None:
        session = object.__new__(QwenImageEditRenderSession)
        session.args = SimpleNamespace(
            device="cpu",
            seed=7,
            cpu_offload=False,
            qwen_batch_size=2,
            qwen_batch_fallback=True,
        )
        session.pipe = object()
        condition = Image.new("RGB", (12, 12), "white")
        session._load_condition_image = Mock(return_value=condition)
        session._pipe_call = Mock(
            side_effect=[
                Image.new("RGB", (12, 12), "red"),
                Image.new("RGB", (12, 12), "blue"),
            ]
        )

        with patch(
            "qwen2_blackbox_runtime.render_qwen_image_edit_batch",
            side_effect=RuntimeError("synthetic batch failure"),
        ):
            images = session.render_images(prompts=["first", "second"])

        self.assertEqual(len(images), 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in session._pipe_call.call_args_list],
            [7, 8],
        )

    def test_qwen_cpu_offload_keeps_sequential_rendering(self) -> None:
        session = object.__new__(QwenImageEditRenderSession)
        session.args = SimpleNamespace(
            device="cpu",
            seed=9,
            cpu_offload=True,
            qwen_batch_size=3,
            qwen_batch_fallback=True,
        )
        session.pipe = object()
        session._load_condition_image = Mock(
            return_value=Image.new("RGB", (8, 8), "white")
        )
        session._pipe_call = Mock(
            side_effect=[
                Image.new("RGB", (8, 8), "red"),
                Image.new("RGB", (8, 8), "blue"),
            ]
        )

        with patch(
            "qwen2_blackbox_runtime.render_qwen_image_edit_batch"
        ) as batch_render:
            images = session.render_images(prompts=["first", "second"])

        self.assertEqual(len(images), 2)
        batch_render.assert_not_called()
        self.assertEqual(session._pipe_call.call_count, 2)

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
        self.assertEqual(len(paths), 34)

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

    def test_csv_metadata_replaces_eager_image_dataset_without_changing_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images.csv").write_text(
                "ImageId,TrueLabel,TargetClass\n"
                "image10,7,9\n"
                "image2,3,5\n",
                encoding="utf-8",
            )
            image_ids, true_labels, target_labels = load_nips_ground_truth(root)

        self.assertEqual(image_ids, ["image2", "image10"])
        self.assertEqual(true_labels, [2, 6])
        self.assertEqual(target_labels, [4, 8])
        self.assertEqual(
            list(
                iter_nips_metadata_batches(
                    image_ids,
                    true_labels,
                    target_labels,
                    batch_size=1,
                )
            ),
            [(["image2"], [2], [4]), (["image10"], [6], [8])],
        )

    def test_clean_correct_filter_selects_only_victim_correct_images(self) -> None:
        class BrightnessPredictor:
            def __init__(self):
                self.batch_sizes = []

            def predict(self, image_batch, batch_size):
                self.batch_sizes.append(int(batch_size))
                logits = np.zeros((image_batch.shape[0], 1000), dtype=np.float32)
                predicted = (image_batch.mean(axis=(1, 2, 3)) > 0.5).astype(np.int64)
                logits[np.arange(image_batch.shape[0]), predicted] = 1.0
                return logits

        victim = object.__new__(VictimModelAdapter)
        victim.model_name = "resnet50"
        victim.input_res = 224
        victim.f_model = BrightnessPredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir)
            Image.new("RGB", (16, 16), "black").save(images_dir / "dark.png")
            Image.new("RGB", (16, 16), "white").save(images_dir / "bright.png")

            selected, records = select_clean_correct_indices(
                victim=victim,
                images_dir=images_dir,
                image_ids=["dark", "bright", "missing"],
                true_labels=[0, 0, 0],
                candidate_indices=[0, 1, 2],
                batch_size=2,
            )

        self.assertEqual(selected, {0})
        self.assertEqual(
            [item["status"] for item in records],
            ["selected", "skipped", "error"],
        )
        self.assertEqual(records[0]["clean_pred_idx"], 0)
        self.assertEqual(records[1]["clean_pred_idx"], 1)
        self.assertIn("missing_source_image", records[2]["error"])
        self.assertEqual(victim.f_model.batch_sizes, [2])

    def test_clean_correct_seeded_sampling_is_exact_and_deterministic(self) -> None:
        population = set(range(250))

        seed0_first = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=0,
        )
        seed0_second = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=0,
        )
        seed1 = sample_clean_correct_indices(
            population,
            sample_size=100,
            seed=1,
        )

        self.assertEqual(len(seed0_first), 100)
        self.assertEqual(seed0_first, seed0_second)
        self.assertNotEqual(seed0_first, seed1)
        self.assertTrue(seed0_first <= population)

    def test_clean_correct_seeded_sampling_rejects_an_undersized_pool(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requested=100 available=99",
        ):
            sample_clean_correct_indices(
                set(range(99)),
                sample_size=100,
                seed=0,
            )

    def test_runners_do_not_eagerly_load_dataset_images(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "flux2_attack_runner.py",
            "qwen2_attack_runner.py",
            "bernini_attack_runner.py",
        ):
            source = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("ImageNet_Compatible", source, msg=name)
            self.assertNotIn("DataLoader", source, msg=name)

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

    def test_prepared_queue_enforces_query_accounting_v3(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        queue_dir = (
            project_root
            / ".aris"
            / "queues"
            / "pro6000_flux2_mllm_dosample_seed012_clean100_20260731_151453"
        )
        queue_text = (queue_dir / "queue.sh").read_text(encoding="utf-8")
        verifier_text = (queue_dir / "verify_job.py").read_text(
            encoding="utf-8"
        )
        manifest = json.loads(
            (queue_dir / "manifest.json").read_text(encoding="utf-8")
        )

        self.assertNotIn("legacy_current_prompt_filtered", queue_text)
        self.assertNotIn("legacy_current_prompt_filtered", verifier_text)
        self.assertIn(
            "strategy_query_accounting_version",
            verifier_text,
        )
        self.assertIn(
            "duplicate_query_count != duplicate_count",
            verifier_text,
        )
        self.assertIn(
            "strategy_duplicate_query_count",
            verifier_text,
        )
        completion_contract = manifest["completion_contract"]
        self.assertEqual(
            completion_contract["strategy_query_accounting_version"],
            3,
        )
        self.assertIn(
            "consume exactly one",
            completion_contract["duplicate_candidate_policy"],
        )
        self.assertIn(
            "consume zero",
            completion_contract["empty_candidate_policy"],
        )
        self.assertNotIn("wave2", queue_text.lower())
        self.assertIn(
            '[[ ! -e "$run_root" && ! -e "$attempt_record" ]] && break',
            queue_text,
        )
        self.assertIn(
            'attempt_cohort_file="$run_root/cohort_100_indices.json"',
            queue_text,
        )
        self.assertIn(
            'cp -- "$COHORT_FILE" "$attempt_cohort_file"',
            queue_text,
        )
        self.assertIn(
            '--sample_indices_file "$attempt_cohort_file"',
            queue_text,
        )
        self.assertIn("--resume_existing_reports true", queue_text)
        self.assertIn(
            "resume partial attempt without rerunning verified reports",
            queue_text,
        )

    def test_revised_seed01_queues_cover_exact_eight_cells(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        completed_queue_dir = (
            project_root
            / ".aris"
            / "queues"
            / "pro6000_flux2_mllm_dosample_seed012_clean100_20260731_151453"
        )
        followup_queue_dir = (
            project_root
            / ".aris"
            / "queues"
            / "pro6000_flux2_mllm_modes_seed01_followup_20260731_191906"
        )
        completed_manifest = json.loads(
            (completed_queue_dir / "manifest.json").read_text(encoding="utf-8")
        )
        followup_manifest = json.loads(
            (followup_queue_dir / "manifest.json").read_text(encoding="utf-8")
        )
        completed_queue_text = (completed_queue_dir / "queue.sh").read_text(
            encoding="utf-8"
        )
        followup_queue_text = (followup_queue_dir / "queue.sh").read_text(
            encoding="utf-8"
        )

        completed_jobs = completed_manifest["jobs"]
        self.assertEqual(
            [(job["mode"], job["seed"]) for job in completed_jobs],
            [
                ("qwen3_vl_4b_instruct", 0),
                ("qwen3_vl_4b_instruct", 1),
            ],
        )
        self.assertIn("seed 2", completed_manifest["experiment"]["scope_revision"])
        self.assertNotIn("seed2", completed_queue_text)

        followup_jobs = followup_manifest["jobs"]
        self.assertEqual(len(followup_jobs), 6)
        self.assertEqual(
            [
                (
                    job["mode"],
                    job["attack_mode"],
                    job["strategies"],
                    job["strategy_do_sample"],
                    job["seed"],
                )
                for job in followup_jobs
            ],
            [
                ("qwen3_vl_4b_instruct", "vlm", "none", False, 0),
                ("qwen3_vl_4b_instruct", "vlm", "none", False, 1),
                ("internvl3_5_4b_instruct", "vlm", "none", False, 0),
                ("internvl3_5_4b_instruct", "vlm", "none", False, 1),
                ("internvl3_5_4b_instruct", "and", "all", True, 0),
                ("internvl3_5_4b_instruct", "and", "all", True, 1),
            ],
        )
        self.assertEqual(
            followup_manifest["common"]["query_accounting_version"],
            3,
        )
        self.assertNotIn("wave2", followup_queue_text.lower())
        self.assertNotIn("seed2", followup_queue_text)
        self.assertIn('JOB_ATTACK_MODES=(vlm vlm vlm vlm and and)', followup_queue_text)
        self.assertIn('JOB_STRATEGIES=(none none none none all all)', followup_queue_text)
        self.assertIn('--scene_vlm_do_sample false', followup_queue_text)
        self.assertIn('--manual_seed "$seed"', followup_queue_text)
        self.assertIn('--sample_indices_file "$attempt_cohort_file"', followup_queue_text)


class AttackImageSavingTests(unittest.TestCase):
    def test_in_memory_candidates_match_the_previous_lossless_strip_pixels(self) -> None:
        first_array = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        second_array = np.flip(first_array, axis=1).copy()
        source_images = [Image.fromarray(first_array), Image.fromarray(second_array)]

        with tempfile.TemporaryDirectory() as tmpdir:
            strip_path = Path(tmpdir) / "legacy_strip.png"
            Image.fromarray(np.hstack([first_array, second_array])).save(strip_path)
            with Image.open(strip_path) as opened:
                legacy_rgb = opened.convert("RGB")
                legacy_tiles = [legacy_rgb.crop((0, 0, 8, 6)), legacy_rgb.crop((8, 0, 16, 6))]

        class InMemoryRenderSession:
            def __init__(self) -> None:
                self.last_prompt_query_count = 0
                self.calls = []

            def render_images(self, *, prompts, mixed_initial_edit_cache=None):
                self.calls.append((list(prompts), mixed_initial_edit_cache))
                return [image.copy() for image in source_images]

        class RecordingClassifier:
            input_res = 224

            def __init__(self) -> None:
                self.inputs = []

            def objective_and_stats(self, image_01, target_label=None):
                del target_label
                self.inputs.append(image_01.detach().cpu().clone())
                index = len(self.inputs) - 1
                return float(index), {
                    "pred_idx": index,
                    "pred_conf": 0.5,
                    "pred_logit": float(index),
                    "target_conf": 0.5,
                    "target_logit": 0.0,
                    "ce": 0.0,
                }

        session = InMemoryRenderSession()
        classifier = RecordingClassifier()
        results, error, metadata = evaluate_attack_candidates(
            args=argparse.Namespace(device="cpu"),
            classifier=classifier,
            candidate_words=["first", "second"],
            candidate_prompts=["prompt one", "prompt two"],
            has_input_image=True,
            render_session=session,
            capture_classifier_tile_image=True,
        )

        self.assertIsNone(error)
        self.assertIsNone(metadata)
        self.assertEqual(session.calls, [(["prompt one", "prompt two"], None)])
        self.assertEqual(session.last_prompt_query_count, 2)
        self.assertEqual(len(results), 2)
        for index, result in enumerate(results):
            self.assertTrue(
                np.array_equal(
                    np.asarray(result["candidate_selected_image"]),
                    np.asarray(legacy_tiles[index]),
                )
            )
            self.assertEqual(result["candidate_strip_index"], index)
            self.assertEqual(result["candidate_selected_image"].size, (8, 6))
            self.assertEqual(result["candidate_classifier_image"].size, (224, 224))

    def test_production_prompt_candidate_retains_exact_victim_artifacts(self) -> None:
        class FakePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        class InMemoryRenderSession:
            last_prompt_query_count = 0

            @staticmethod
            def render_images(*, prompts, mixed_initial_edit_cache=None):
                del mixed_initial_edit_cache
                return [
                    Image.new("RGB", (317, 191), (13, 127, 241))
                    for _ in prompts
                ]

        victim = object.__new__(VictimModelAdapter)
        victim.model_name = "resnet50"
        victim.device = torch.device("cpu")
        victim.objective_mode = "ce_max"
        victim.label = 0
        victim.input_res = 224
        victim.f_model = FakePredictor()
        victim._evaluation_callback = None
        victim._evaluation_attempt_count = 0
        victim._last_evaluation_payload = None

        session = InMemoryRenderSession()
        results, error, _ = evaluate_attack_candidates(
            args=argparse.Namespace(device="cpu"),
            classifier=victim,
            candidate_words=["forest"],
            candidate_prompts=["forest"],
            has_input_image=True,
            render_session=session,
            capture_classifier_tile_image=True,
        )

        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        candidate = results[0]
        expected_float32 = victim._preprocess(
            image_to_tensor_01(candidate["candidate_selected_image"])
        )[0]
        self.assertTrue(
            np.array_equal(
                candidate["candidate_classifier_input_float32"],
                expected_float32,
            )
        )
        expected_uint8 = (
            np.clip(expected_float32, 0.0, 1.0).transpose(1, 2, 0) * 255.0
        ).round().astype(np.uint8)
        self.assertTrue(
            np.array_equal(
                np.asarray(candidate["candidate_classifier_image"]),
                expected_uint8,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_paths = save_blackbox_prompt_artifacts(
                run_dir=Path(tmpdir),
                step_idx=0,
                candidate_source="gemma_scene_vocab",
                prompt_text="prompt",
                raw_answer='{"candidates": ["forest"]}',
                feedback_used=[],
                generated_words=["forest"],
                filtered_words=["forest"],
                scored_candidates=results,
                vlm_error=None,
                score_error=None,
            )
            response_path = Path(tmpdir) / artifact_paths["response_json_path"]
            response = json.loads(response_path.read_text(encoding="utf-8"))
            saved_candidate = response["scored_candidates"][0]
            self.assertNotIn("candidate_classifier_input_float32", saved_candidate)
            self.assertNotIn("candidate_classifier_image", saved_candidate)

    def test_flux_and_candidate_retains_exact_victim_artifacts(self) -> None:
        class FakePredictor:
            @staticmethod
            def predict(image_batch, batch_size):
                del image_batch, batch_size
                return np.asarray([[0.0, 5.0]], dtype=np.float32)

        victim = object.__new__(VictimModelAdapter)
        victim.model_name = "resnet50"
        victim.device = torch.device("cpu")
        victim.objective_mode = "ce_max"
        victim.label = 0
        victim.input_res = 224
        victim.f_model = FakePredictor()
        victim._evaluation_callback = None
        victim._evaluation_attempt_count = 0
        victim._last_evaluation_payload = None

        rendered = Image.new("RGB", (331, 205), (219, 31, 97))
        session = object.__new__(Flux2KleinRenderSession)
        session.args = SimpleNamespace(
            flux2_strategy_cwor_merge_mode="and",
            _flux2_strategy_cwor_query_budget=1,
            cwor_target_label=None,
            device="cpu",
        )
        session.last_and_query_count = 0
        session._render_prompt_image = lambda _prompt: rendered.copy()
        prompt_candidates = [
            {
                "candidate_prompt": "forest",
                "candidate_objective": 2.0,
                "candidate_variant": "prompt",
                "candidate_strategy_name": "background",
            },
            {
                "candidate_prompt": "soft lighting",
                "candidate_objective": 1.0,
                "candidate_variant": "prompt",
                "candidate_strategy_name": "lighting",
            },
        ]

        results, error = session.evaluate_and_candidate(
            classifier=victim,
            cwor_result_prompt="",
            cwor_base_confidence=None,
            original_objective=0.0,
            prompt_candidates=prompt_candidates,
            cwor_step_index=1,
        )

        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        expected_float32 = victim._preprocess(image_to_tensor_01(rendered))[0]
        self.assertTrue(
            np.array_equal(
                results[0]["candidate_classifier_input_float32"],
                expected_float32,
            )
        )
        expected_uint8 = (
            np.clip(expected_float32, 0.0, 1.0).transpose(1, 2, 0) * 255.0
        ).round().astype(np.uint8)
        self.assertTrue(
            np.array_equal(
                np.asarray(results[0]["candidate_classifier_image"]),
                expected_uint8,
            )
        )

    def test_production_candidate_path_contains_no_strip_serialization(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "vlm_attack.py",
            "flux2_blackbox_runtime.py",
            "qwen2_blackbox_runtime.py",
            "bernini_blackbox_runtime.py",
        ):
            source = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("np.hstack", source, msg=name)
            self.assertNotIn("split_prompt_strip", source, msg=name)
            self.assertNotIn("candidate_strip.png", source, msg=name)

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
        source = torch.rand(1, 3, 333, 400)
        expected_classifier_input = victim._preprocess(source)[0]
        _, stats = victim.objective_and_stats(source)

        self.assertEqual(len(callback_payloads), 1)
        payload = callback_payloads[0]
        self.assertEqual(payload["pred_idx"], stats["pred_idx"])
        self.assertEqual(payload["candidate_classifier_image_size"], 224)
        self.assertIsInstance(payload["candidate_classifier_image"], Image.Image)
        self.assertEqual(payload["candidate_classifier_image"].size, (224, 224))
        classifier_input = payload["candidate_classifier_input_float32"]
        self.assertEqual(classifier_input.dtype, np.float32)
        self.assertEqual(classifier_input.shape, (3, 224, 224))
        self.assertTrue(np.array_equal(classifier_input, expected_classifier_input))

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
                float32_path = output_path.with_suffix(".float32.npy")
                test_case.assertTrue(float32_path.is_file())
                saved_float32 = np.load(float32_path, allow_pickle=False)
                expected_float32 = classifier._preprocess(
                    torch.full((1, 3, 320, 240), 0.25)
                )[0]
                test_case.assertTrue(np.array_equal(saved_float32, expected_float32))
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
                    "--gcg_eval_naturalness_on_attack_success",
                    "false",
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
            self.assertEqual(
                result["attack_success_float32_path"],
                "attack_success.float32.npy",
            )

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
                    "--gcg_eval_naturalness_on_attack_success", "false",
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
            self.assertTrue(output_path.with_suffix(".float32.npy").is_file())
            preserved = load_preserved_attack_report(report_path)
            self.assertIsNotNone(preserved)
            self.assertTrue(preserved["final_attack_success"])
            self.assertEqual(preserved["victim_query_count"], 2)
            self.assertEqual(preserved["attack_success_query_count"], 2)
            self.assertEqual(
                preserved["attack_success_float32_path"],
                "attack_success.float32.npy",
            )

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

    def test_float32_classifier_input_round_trips_bit_exactly(self) -> None:
        classifier_input = np.linspace(
            0.0,
            1.0,
            num=3 * 224 * 224,
            dtype=np.float32,
        ).reshape(3, 224, 224)
        candidate = {
            "candidate_classifier_input_float32": classifier_input,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "images" / "attack_success.png"
            returned_path = save_evaluated_attack_float32(candidate, output_path)

            self.assertEqual(
                returned_path,
                output_path.with_suffix(".float32.npy"),
            )
            restored = np.load(returned_path, allow_pickle=False)
            self.assertEqual(restored.dtype, np.float32)
            self.assertEqual(restored.shape, (3, 224, 224))
            self.assertTrue(np.array_equal(restored, classifier_input))

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
                    "--gcg_eval_naturalness_on_attack_success", "false",
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
                    "--gcg_eval_naturalness_on_attack_success", "false",
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
                attack_success_float32_path=None,
                attack_success_float32_error=None,
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

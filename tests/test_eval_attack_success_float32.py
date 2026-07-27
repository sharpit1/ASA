from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from eval.eval_attack_success_float32 import (
    compute_untarget_asr,
    evaluate_run,
    load_classifier_input,
)


class Float32AttackEvaluationTests(unittest.TestCase):
    def test_untarget_asr_uses_only_clean_correct_samples_as_denominator(self) -> None:
        samples = [
            {"true_label": 1, "clean_pred_idx": 1, "adv_pred_idx": 2},
            {"true_label": 2, "clean_pred_idx": 2, "adv_pred_idx": 2},
            {"true_label": 3, "clean_pred_idx": 3, "adv_pred_idx": None},
            {"true_label": 4, "clean_pred_idx": 0, "adv_pred_idx": 5},
        ]

        result = compute_untarget_asr(samples)

        self.assertEqual(result["untarget_asr_denominator"], 3)
        self.assertEqual(result["untarget_asr_numerator"], 1)
        self.assertAlmostEqual(result["untarget_asr"], 100.0 / 3.0)

    def test_untarget_asr_is_none_when_no_clean_sample_is_correct(self) -> None:
        result = compute_untarget_asr(
            [{"true_label": 1, "clean_pred_idx": 0, "adv_pred_idx": 2}]
        )

        self.assertEqual(result["untarget_asr_denominator"], 0)
        self.assertEqual(result["untarget_asr_numerator"], 0)
        self.assertIsNone(result["untarget_asr"])

    def test_untarget_asr_honors_explicit_prompt_adjusted_success(self) -> None:
        result = compute_untarget_asr(
            [
                {
                    "true_label": 1,
                    "clean_pred_idx": 1,
                    "adv_pred_idx": 2,
                    "attack_success": False,
                }
            ]
        )

        self.assertEqual(result["untarget_asr_denominator"], 1)
        self.assertEqual(result["untarget_asr_numerator"], 0)
        self.assertEqual(result["untarget_asr"], 0.0)

    def test_float32_sidecar_is_loaded_without_clipping(self) -> None:
        classifier_input = np.zeros((3, 224, 224), dtype=np.float32)
        classifier_input[0, 0, 0] = np.float32(1.0000004)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attack_success.float32.npy"
            np.save(path, classifier_input, allow_pickle=False)

            loaded = load_classifier_input(path)

        self.assertTrue(np.array_equal(loaded, classifier_input))
        self.assertGreater(float(loaded.max()), 1.0)

    def test_evaluate_run_applies_clean_correct_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            dataset_root = root / "dataset"
            images_dir = dataset_root / "images"
            images_dir.mkdir(parents=True)
            results = []
            for index, true_label in enumerate((1, 2, 3, 4)):
                image_id = f"image_{index}"
                Image.new("RGB", (8, 8), color=(index, index, index)).save(
                    images_dir / f"{image_id}.png"
                )
                sample_dir = run_root / f"sample_{index:04d}"
                sample_dir.mkdir(parents=True)
                if index != 2:
                    sidecar_path = (
                        sample_dir / "images" / "attack_success.float32.npy"
                    )
                    sidecar_path.parent.mkdir()
                    np.save(
                        sidecar_path,
                        np.zeros((3, 224, 224), dtype=np.float32),
                        allow_pickle=False,
                    )
                results.append(
                    {
                        "sample_index": index,
                        "image_id": image_id,
                        "true_label": true_label,
                    }
                )
            run_root.mkdir(exist_ok=True)
            (run_root / "run_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "victim_model": "resnet50",
                        "results": results,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "eval.eval_attack_success_float32.VictimModelAdapter",
                    return_value=Mock(),
                ),
                patch(
                    "eval.eval_attack_success_float32.predict_label",
                    side_effect=(1, 2, 2, 2, 3, 0, 5),
                ),
            ):
                payload = evaluate_run(
                    run_root=run_root,
                    device="cpu",
                    prompt_label_correct=False,
                )

        self.assertEqual(payload["sample_count"], 4)
        self.assertEqual(payload["float32_sidecar_count"], 3)
        self.assertEqual(payload["untarget_asr_denominator"], 3)
        self.assertEqual(payload["untarget_asr_numerator"], 1)
        self.assertAlmostEqual(payload["untarget_asr"], 100.0 / 3.0)

    def test_evaluate_run_excludes_prediction_named_in_source_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            dataset_root = root / "dataset"
            images_dir = dataset_root / "images"
            images_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(images_dir / "image_0.png")
            (dataset_root / "categories.csv").write_text(
                "CategoryId,CategoryName\n3,library\n",
                encoding="utf-8",
            )
            sample_dir = run_root / "sample_0000"
            sidecar_path = sample_dir / "images" / "attack_success.float32.npy"
            sidecar_path.parent.mkdir(parents=True)
            np.save(
                sidecar_path,
                np.zeros((3, 224, 224), dtype=np.float32),
                allow_pickle=False,
            )
            (sample_dir / "report.json").write_text(
                json.dumps({"optimized_prompt": "place it in a library"}),
                encoding="utf-8",
            )
            (run_root / "run_summary.json").write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "victim_model": "resnet50",
                        "results": [
                            {
                                "sample_index": 0,
                                "image_id": "image_0",
                                "true_label": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "eval.eval_attack_success_float32.VictimModelAdapter",
                    return_value=Mock(),
                ),
                patch(
                    "eval.eval_attack_success_float32.predict_label",
                    side_effect=(1, 2),
                ),
            ):
                payload = evaluate_run(run_root=run_root, device="cpu")

        self.assertEqual(
            payload["attack_success_without_prompt_correction_count"], 1
        )
        self.assertEqual(
            payload["attack_success_excluded_by_prompt_correction_count"], 1
        )
        self.assertEqual(payload["untarget_asr_numerator"], 0)
        self.assertEqual(payload["samples"][0]["prompt_correct_labels"], [2])
        self.assertTrue(payload["samples"][0]["prompt_label_match"])


if __name__ == "__main__":
    unittest.main()

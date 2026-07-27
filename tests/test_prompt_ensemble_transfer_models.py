import unittest
from unittest import mock

import torch

from eval.eval_prompt_transfer_in_image import (
    build_prompt_ensemble_classifier,
    normalize_model_name,
)


class PromptEnsembleTransferModelTest(unittest.TestCase):
    def test_clip_aliases(self):
        for alias in (
            "clip",
            "openai-clip",
            "clip-vit-base-patch16",
            "clip-vit-b-16",
            "openai/clip-vit-base-patch16",
        ):
            self.assertEqual(normalize_model_name(alias), "clip")

    def test_siglip2_aliases(self):
        for alias in (
            "siglip2",
            "siglip2-base-patch16-224",
            "google/siglip2-base-patch16-224",
        ):
            self.assertEqual(normalize_model_name(alias), "siglip2")

    def test_prompt_ensemble_builder_routes_canonical_models(self):
        clip_sentinel = object()
        siglip_sentinel = object()
        with (
            mock.patch("eval.eval_prompt_transfer_in_image.logger.log"),
            mock.patch(
                "isolated_vlm_attack.attack_runner_common._OpenAIClipImageNetClassifier",
                return_value=clip_sentinel,
            ) as clip_builder,
            mock.patch(
                "isolated_vlm_attack.attack_runner_common._Siglip2ImageNetClassifier",
                return_value=siglip_sentinel,
            ) as siglip_builder,
        ):
            self.assertIs(
                build_prompt_ensemble_classifier("clip", "cpu"),
                clip_sentinel,
            )
            self.assertIs(
                build_prompt_ensemble_classifier("siglip2", "cpu"),
                siglip_sentinel,
            )

        clip_builder.assert_called_once_with(
            model_id="openai/clip-vit-base-patch16",
            device=torch.device("cpu"),
        )
        siglip_builder.assert_called_once_with(
            model_id="google/siglip2-base-patch16-224",
            device=torch.device("cpu"),
        )

    def test_prompt_ensemble_builder_rejects_other_models(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            build_prompt_ensemble_classifier("resnet50", "cpu")


if __name__ == "__main__":
    unittest.main()

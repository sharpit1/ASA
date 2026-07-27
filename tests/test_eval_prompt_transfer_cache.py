import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eval.eval_prompt_transfer_in_image import (
    get_render_cache_path,
    is_valid_label_payload_cache,
    load_clean_batch_from_image_paths,
    load_label_payload_cache,
    save_label_payload_cache,
)


class PromptTransferCacheTest(unittest.TestCase):
    def _payload(self, image_path: Path):
        return {
            "ground_label": 3,
            "ground_class_name": "test-class",
            "artifact_type": "prompt",
            "artifact_prompt_for_log": "introduce light mist",
            "clean_batch": np.zeros((1, 3, 8, 8), dtype=np.float32),
            "adv_batch": np.ones((1, 3, 8, 8), dtype=np.float32),
            "label_batch": np.asarray([3], dtype=np.int64),
            "image_paths": [str(image_path)],
            "generation_failures": 0,
        }

    def test_clean_batch_can_be_omitted_and_reloaded(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            image_path = temp_dir / "clean.png"
            Image.new("RGB", (12, 10), color=(10, 20, 30)).save(image_path)

            save_label_payload_cache(
                cache_dir=temp_dir,
                payload=self._payload(image_path),
                save_clean_batch=False,
            )

            cache_path = get_render_cache_path(temp_dir, 3)
            with np.load(cache_path, allow_pickle=False) as raw:
                self.assertNotIn("clean_batch", raw.files)
                self.assertFalse(bool(raw["clean_batch_saved"][0]))

            loaded = load_label_payload_cache(temp_dir, 3)
            self.assertIsNone(loaded["clean_batch"])
            reconstructed = load_clean_batch_from_image_paths(
                loaded["image_paths"],
                eval_size=8,
            )
            self.assertEqual(reconstructed.shape, (1, 3, 8, 8))
            self.assertEqual(reconstructed.dtype, np.float32)
            self.assertTrue(is_valid_label_payload_cache(cache_path, expected_count=1))
            self.assertFalse(is_valid_label_payload_cache(cache_path, expected_count=2))

    def test_clean_batch_on_remains_backward_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            image_path = temp_dir / "clean.png"
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image_path)

            save_label_payload_cache(
                cache_dir=temp_dir,
                payload=self._payload(image_path),
                save_clean_batch=True,
            )

            loaded = load_label_payload_cache(temp_dir, 3)
            self.assertIsNotNone(loaded["clean_batch"])
            self.assertEqual(loaded["clean_batch"].shape, (1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()

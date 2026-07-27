"""Count clean-correct NIPS2017 samples using the supplied common module verbatim."""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from data.nips2017 import get_rgb_01_transform, load_ground_truth, load_image
from vlm_runner.attack_runner_common import VictimModelAdapter


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    image_ids, true_labels, _ = load_ground_truth(os.path.join(dataset_dir, "images.csv"))
    # Preserve the original 299px RGB image in [0, 1].  The supplied adapter's
    # _preprocess() is the only component that performs the 299→224 resize.
    transform = get_rgb_01_transform()
    batch_size = int(os.environ.get("VLM_CLEAN_BATCH_SIZE", "32"))

    requested_victim = os.environ.get("VLM_CLEAN_VICTIM", "").strip()
    victims = (requested_victim,) if requested_victim else ("siglip2", "clip-vit-b-16")
    for victim in victims:
        adapter = VictimModelAdapter(model_name=victim, device="cuda", objective_mode="ce_max")
        correct = 0
        total = len(image_ids)
        for start in tqdm(range(0, total, batch_size), desc=f"clean {victim}", unit="batch"):
            end = min(start + batch_size, total)
            images = torch.cat(
                [load_image(os.path.join(dataset_dir, "images"), image_id, transform) for image_id in image_ids[start:end]],
                dim=0,
            )
            # This is the exact supplied adapter's resize/clamp path followed
            # by its supplied prompt-ensemble classifier.
            classifier_input = adapter._preprocess(images)
            scores = adapter.f_model.predict(classifier_input, batch_size=batch_size)
            pred = np.argmax(scores, axis=1)
            labels = np.asarray(true_labels[start:end], dtype=np.int64)
            correct += int(np.sum(pred == labels))

        print(f"{victim}: clean_correct={correct}/{total}, clean_accuracy={correct / total * 100:.2f}%")
        del adapter
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

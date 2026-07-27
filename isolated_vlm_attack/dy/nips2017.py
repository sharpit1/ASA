"""
NIPS 2017 Adversarial Competition dataset loader.
Loads images in natsort order to match AdvAD evaluation scripts.

Source: AdvAD/ImageNet_Compatible_Dataset.py
"""

import os
import csv

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from natsort import index_natsorted


def load_ground_truth(csv_path):
    """
    Load image metadata from NIPS 2017 CSV.
    Returns lists in natsort order (matching AdvAD eval scripts).

    Labels are converted from 1-indexed (CSV) to 0-indexed (PyTorch).
    """
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    sorted_indices = index_natsorted(image_id_list)
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]

    return image_id_list, label_ori_list, label_tar_list


def get_transform(image_size=224):
    """
    Image transform matching AdvAD/DifAttack convention:
    Resize to (image_size, image_size) with bilinear interpolation, then ToTensor → [0, 1].
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def get_rgb_01_transform():
    """Convert an RGB PNG to a raw CHW float tensor in [0, 1], without resize."""

    def _transform(image: Image.Image) -> torch.Tensor:
        return transforms.functional.pil_to_tensor(image.convert("RGB")).float().div_(255.0)

    return _transform


def load_image(img_dir, image_id, transform):
    """Load a single image, apply transform, return [1, 3, H, W] tensor."""
    path = os.path.join(img_dir, image_id + '.png')
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1, 3, H, W] in [0, 1]


# Default dataset paths (relative to project root)
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'images.csv')
DEFAULT_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'images')

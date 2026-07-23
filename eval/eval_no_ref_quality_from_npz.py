import torch
import argparse
import numpy as np 
from PIL import Image
import random
import difflib
import re
import sys
from pathlib import Path
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import trange, tqdm
import numpy
import numpy as np
import math
import pyiqa

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import logger
from eval.npz_loader import (
    is_8bit_quantized_01,
    load_adv_images_from_npz,
    quantize_to_8bit_01,
)


import datetime
import time

parser = argparse.ArgumentParser(description='Test Image Quality!')
parser.add_argument('--img_path', type=str, default='./', help='cnn')
parser.add_argument(
    '--metric',
    type=str,
    default='musiq,nima-vgg16-ava,hyperiqa,musiq-ava,tres',
    help='pyiqa metric names. Comma-separated values are supported.',
)
parser.add_argument('--seed', type=int, default=42, help='cnn')


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


_LEGACY_METRIC_ALIASES = {
    # Legacy name used in older scripts.
    "musiq-koniq": ["musiq", "musiq-paq2piq", "musiq-spaq", "musiq-ava", "nima-koniq"],
    # Friendly aliases frequently used in papers or internal notes.
    "nima-ava": ["nima-vgg16-ava", "nima"],
}


def _available_pyiqa_metrics():
    if hasattr(pyiqa, "list_models"):
        return set(pyiqa.list_models())
    if hasattr(pyiqa, "list_metrics"):
        return set(pyiqa.list_metrics())
    return set()


def _normalize_metric_name(metric_name):
    return re.sub(r"\s+", "", str(metric_name)).lower().replace("_", "-")


def _resolve_metric_name(metric_name, available_metrics):
    if metric_name in available_metrics:
        return metric_name, None

    normalized = _normalize_metric_name(metric_name)
    for name in available_metrics:
        if _normalize_metric_name(name) == normalized:
            return name, None

    if normalized in _LEGACY_METRIC_ALIASES:
        for candidate in _LEGACY_METRIC_ALIASES[normalized]:
            if candidate in available_metrics:
                note = (
                    f"Metric '{metric_name}' is unavailable in this pyiqa version. "
                    f"Using compatible fallback '{candidate}'."
                )
                return candidate, note

    close_matches = difflib.get_close_matches(normalized, sorted(available_metrics), n=5)
    raise ValueError(
        f"Metric '{metric_name}' is not supported by installed pyiqa. "
        f"Try one of: {close_matches if close_matches else sorted(list(available_metrics))[:10]}"
    )


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(path).convert('RGB')
            return img
    except IOError:
        print('Cannot load image ' + path)


def resolve_eval_dir(file_name):
    if not file_name:
        raise ValueError("file_name must be provided.")

    path = Path(file_name).expanduser()
    if path.exists():
        return path
    if path.is_absolute():
        return path

    relative = Path(str(file_name).lstrip("/\\"))
    legacy_path = Path("outputs/nips2017") / relative
    if legacy_path.exists():
        return legacy_path
    return legacy_path

def eval_no_ref_quality_from_npz(file_name=None, re_logger=True, quant=False):
    file_dir = resolve_eval_dir(file_name)
    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = file_dir / ("eval_" + str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=str(save_dir))

    npz_file = file_dir / "adversarial_examples.npz"
    all_adv_images_01 = load_adv_images_from_npz(npz_file, output_layout="nhwc")
    if quant:
        if is_8bit_quantized_01(all_adv_images_01):
            logger.log("Adversarial images already 8-bit quantized. Skip requantization.")
        else:
            logger.log("Adversarial images are not 8-bit quantized. Apply requantization.")
            all_adv_images_01 = quantize_to_8bit_01(all_adv_images_01)

    all_adv_images = np.clip(all_adv_images_01 * 255.0, 0, 255)

    all_adv_images = torch.from_numpy(all_adv_images).to(device)

    image_num = all_adv_images.shape[0]

    requested_metrics = [m.strip() for m in str(args.metric).split(",") if m.strip()]
    if not requested_metrics:
        requested_metrics = ["musiq", "nima-vgg16-ava", "hyperiqa", "musiq-ava", "tres"]

    available_metrics = _available_pyiqa_metrics()
    metrics = []
    seen = set()
    for metric_name in requested_metrics:
        resolved_name, note = _resolve_metric_name(metric_name, available_metrics)
        if note is not None:
            logger.log(note)
        if resolved_name not in seen:
            metrics.append(resolved_name)
            seen.add(resolved_name)
    all_res = []
    lower_better = []

    logger.log("******* Evaluating {} No-ref quality ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))

    for metric in metrics:
        iqa_metric = pyiqa.create_metric(metric, device=device)
        lower_better.append(iqa_metric.lower_better)
        res_now = 0
        for i in tqdm(range(image_num)):
            # img2 = torch.Tensor(np.array(img_loader(os.path.join(img2_root, f_list[i])).resize((224, 224)))).unsqueeze(0).to(device).permute(0, 3, 1, 2)/255.0
            img2 = all_adv_images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            score_now = iqa_metric(img2)
            res_now += score_now
        res = res_now.item()/image_num
        all_res.append(res)

    for i in range(len(metrics)):
        logger.log("{} ({}): {:.4f}".format(metrics[i], lower_better[i], all_res[i]))
        # print('name', res_now/len(f_list))


if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_no_ref_quality_from_npz(file_name=dir, re_logger=True, quant=False)
        eval_no_ref_quality_from_npz(file_name=dir, re_logger=True, quant=True)

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import datetime
import time
from tqdm import tqdm

from pytorch_fid_score_new.fid_score_new import return_fid_from_data
from utils import logger
from PIL import Image

from natsort import index_natsorted
from eval.npz_loader import (
    is_8bit_quantized_01,
    load_adv_images_from_npz,
    quantize_to_8bit_01,
)

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
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

def count_images_in_directory(directory):
    files = os.listdir(directory)
    file_count = len([f for f in files if os.path.isfile(os.path.join(directory, f))])
    return file_count


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

def eval_fid_from_npz(file_name=None, re_logger=True, quant=False):

    file_dir = resolve_eval_dir(file_name)
    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = file_dir / ("eval_" + str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=str(save_dir))

    logger.log("******* Evaluating {} FID ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))

    npz_file = file_dir / "adversarial_examples.npz"
    all_adv_images = load_adv_images_from_npz(npz_file, output_layout="nchw")
    image_num = all_adv_images.shape[0]

    if quant:
        if is_8bit_quantized_01(all_adv_images):
            logger.log("Adversarial images already 8-bit quantized. Skip requantization.")
        else:
            logger.log("Adversarial images are not 8-bit quantized. Apply requantization.")
            all_adv_images = quantize_to_8bit_01(all_adv_images)

    images_root = "./data/nips2017/images/"  # The clean images' root directory.
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./data/nips2017/images.csv')

    image_size = all_adv_images.shape[2]

    all_clean_images = []
    # print("loading original images")
    for i in tqdm(range(image_num)):
        original_image = Image.open(images_root + image_id_list[i] + '.png').convert('RGB')
        original_image = original_image.resize((image_size, image_size), resample=Image.BILINEAR)
        original_image = np.array(original_image)
        all_clean_images.append(original_image)
    all_clean_images = np.stack(all_clean_images, axis=0)
    all_clean_images = all_clean_images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    fid = return_fid_from_data(all_adv_images, all_clean_images)
    logger.log("FID: {}".format(fid))
    logger.log("******* Done *******")
    return float(fid)

if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_fid_from_npz(file_name=dir, re_logger=True, quant=False)
        eval_fid_from_npz(file_name=dir, re_logger=True, quant=True)

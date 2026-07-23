import csv
import os
import sys
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch as th
import lpips

import datetime
import time
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

def eval_quality_from_npz(file_name=None, re_logger=True, quant=False):

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

    '''test for saving image actually, equal performance'''
    # for i in range(all_adv_images.shape[0]):
    #     np_now = np.array(all_adv_images[i], dtype=np.uint8)
    #     print(np_now.shape)
    #     image = Image.fromarray(np_now)
    #     image.save("temp_test.png", compress_level=0)
    #     image_new = Image.open("temp_test.png")
    #     all_adv_images[i] = np.array(image_new, dtype=np.float32)

    images_root = "./data/nips2017/images/"  # The clean images' root directory.
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./data/nips2017/images.csv')

    image_num = all_adv_images.shape[0]
    image_size = all_adv_images.shape[2]

    all_l_inf = 0
    all_l_inf_0to1 = 0
    all_l2 = 0
    all_l2_0to1 = 0
    l_inf_max = 0
    l_inf_0to1_max = 0

    all_psnr_float = 0
    all_ssim_float = 0
    psnr_float_inf_num = 0

    loss_fn = lpips.LPIPS(net='alex').cuda()
    all_lpips = 0
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    failed_num = 0

    logger.log("******* Evaluating {} quality ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))
    for i in tqdm(range(image_num)):
        image_id_now = image_id_list[i]
        label_ori_now = label_ori_list[i]
        label_tar_now = label_tar_list[i]

        clean_image = Image.open(images_root + image_id_list[i] + '.png').convert('RGB')
        clean_image = clean_image.resize((image_size, image_size), resample=Image.BILINEAR)
        clean_image = np.array(clean_image).astype(np.float32)

        adv_image = all_adv_images[i]

        if (adv_image == clean_image).all():
            failed_num += 1
            continue

        # psnr
        psnr_float = compare_psnr(clean_image, adv_image, data_range=255.)
        all_psnr_float += psnr_float
        # logger.log("PSNR (float): {}".format(psnr_float))
        # ssim
        ssim_float = compare_ssim(clean_image, adv_image, data_range=255., channel_axis=-1)
        all_ssim_float += ssim_float
        # logger.log("SSIM (float): {}".format(ssim_float))

        # LPIPS
        adv_image = image_transform(adv_image).cuda().unsqueeze(0)
        clean_image = image_transform(clean_image).cuda().unsqueeze(0)
        lpips_value = loss_fn(adv_image / 255., clean_image / 255.)
        all_lpips += lpips_value

        perturbation = adv_image - clean_image
        # L_inf
        l_inf = th.norm(perturbation, float('inf'))
        all_l_inf += l_inf
        if l_inf_max < l_inf.item():
            l_inf_max = l_inf.item()
        # logger.log("L_inf: {} (float 0-255)".format(l_inf))
        # L_inf (0-1)
        l_inf_0to1 = th.norm(perturbation / 255., float('inf'))
        all_l_inf_0to1 += l_inf_0to1
        if l_inf_0to1_max < l_inf_0to1.item():
            l_inf_0to1_max = l_inf_0to1.item()
        # logger.log("L_inf: {} (float 0-1)".format(l_inf_0to1))
        # L_2
        l2 = th.norm(perturbation, p=2)
        all_l2 += l2
        # logger.log("L_2: {} (float 0-255)".format(l2))
        # L2 (0-1)
        l2_0to1 = th.norm(perturbation / 255., p=2)
        all_l2_0to1 += l2_0to1
        # logger.log("L_2: {} (float 0-1)".format(l2_0to1))

    image_num = image_num - failed_num
    logger.log("Failed Img Num: {}".format(failed_num))

    logger.log("Avg Linf (0-255): {:.3f}".format(all_l_inf / image_num))
    logger.log("Max Linf (0-255): {:.3f}".format(l_inf_max))
    logger.log("Avg Linf (0-1): {:.5f}".format(all_l_inf_0to1 / image_num))
    logger.log("Max Linf (0-1): {:.5f}".format(l_inf_0to1_max))
    logger.log("Avg L2   (0-255): {:.2f}".format(all_l2 / image_num))
    logger.log("Avg L2   (0-1): {:.2f}".format(all_l2_0to1 / image_num))
    logger.log("Avg PSNR: {:.2f}".format(all_psnr_float / image_num))
    logger.log("Avg SSIM: {:.5f}".format(all_ssim_float / image_num))
    logger.log("Avg LPIPS: {:.5f}".format(all_lpips.item() / image_num))

    logger.log("******* Done *******")

if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx",
    ]
    for dir in dir_list:
        eval_quality_from_npz(file_name=dir, quant=True)
        eval_quality_from_npz(file_name=dir, quant=False)

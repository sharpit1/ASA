"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch as th

import datetime
import time
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import logger
from PIL import Image

from eval.attacked_models import model_selection
from art.estimators.classification import PyTorchClassifier
from natsort import index_natsorted
from eval.eval_defense import (
    DefendedModel,
    HGDNIPSDefense,
    NIPSR3Defense,
    NIPSR3TF1Defense,
    is_hgd_defense_name,
    is_nips_r3_defense_name,
    is_nips_r3_tf1_defense_name,
)
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


def get_source_model_name(file_name):
    if not file_name:
        return None
    path = Path(file_name).expanduser()
    if path.is_absolute() or path.exists():
        return path.name or None

    normalized = os.path.normpath(str(file_name)).strip(os.sep)
    if not normalized or normalized == ".":
        return None
    return normalized.split(os.sep)[0]


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


def load_victim_query_counts(file_dir, image_num):
    run_summary_path = os.path.join(file_dir, "run_summary.json")
    if not os.path.exists(run_summary_path):
        return None

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)

    query_counts = np.full(image_num, np.nan, dtype=np.float32)
    for result in run_summary.get("results", []):
        sample_index = result.get("sample_index")
        victim_query_count = result.get("victim_query_count")
        if sample_index is None or victim_query_count is None:
            continue
        sample_index = int(sample_index)
        if 0 <= sample_index < image_num:
            query_counts[sample_index] = float(victim_query_count)

    return query_counts


def summary_value_to_bool(value):
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_asa_run_stats(file_dir, image_num):
    run_summary_path = os.path.join(file_dir, "run_summary.json")
    if not os.path.exists(run_summary_path):
        return None, None

    with open(run_summary_path, "r") as f:
        run_summary = json.load(f)

    query_exhausted = np.zeros(image_num, dtype=bool)
    gemma_call_counts = np.full(image_num, np.nan, dtype=np.float32)
    for fallback_index, result in enumerate(run_summary.get("results", [])):
        sample_index = result.get("sample_index", fallback_index)
        try:
            sample_index = int(sample_index)
        except Exception:
            sample_index = fallback_index
        if not (0 <= sample_index < image_num):
            sample_index = fallback_index
        if not (0 <= sample_index < image_num):
            continue

        exhausted = result.get("victim_query_exhausted", result.get("query_exhausted"))
        if exhausted is None:
            exhausted = result.get("early_stop_reason") == "victim_query_budget_exhausted"
        query_exhausted[sample_index] = summary_value_to_bool(exhausted)

        gemma_call_count = None
        for key in (
            "gemma_call_count",
            "gemma_calls",
            "gemma_query_count",
            "gemma_query_calls",
            "source_gemma_call_count",
            "history_len",
        ):
            if result.get(key) is not None:
                gemma_call_count = result.get(key)
                break
        if gemma_call_count is not None:
            gemma_call_counts[sample_index] = float(gemma_call_count)

    return query_exhausted, gemma_call_counts


def load_npz_array_from_keys(npz_file, image_num, keys):
    npz_data = np.load(npz_file, allow_pickle=False)
    try:
        if not isinstance(npz_data, np.lib.npyio.NpzFile):
            return None
        for key in keys:
            if key in npz_data.files:
                values = np.asarray(npz_data[key])
                break
        else:
            return None
    finally:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            npz_data.close()

    if values.size != image_num:
        raise ValueError(
            "Expected one of {} in {} to have {} values, but got shape {}.".format(
                list(keys),
                npz_file,
                image_num,
                values.shape,
            )
        )
    return values.reshape(-1)


def load_source_query_counts_from_npz(npz_file, image_num, key="source_query"):
    npz_data = np.load(npz_file, allow_pickle=False)
    try:
        if not isinstance(npz_data, np.lib.npyio.NpzFile) or key not in npz_data.files:
            return None
        raw_query_counts = np.asarray(npz_data[key])
    finally:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            npz_data.close()

    if not np.issubdtype(raw_query_counts.dtype, np.number):
        return None

    raw_query_counts = raw_query_counts.astype(np.float32, copy=False)

    if raw_query_counts.size != image_num:
        raise ValueError(
            "Expected '{}' in {} to have {} values, but got shape {}.".format(
                key,
                npz_file,
                image_num,
                raw_query_counts.shape,
            )
        )

    return raw_query_counts.reshape(-1)


def load_source_prompts_from_npz(npz_file, image_num, key="source_prompt"):
    npz_data = np.load(npz_file, allow_pickle=False)
    try:
        if not isinstance(npz_data, np.lib.npyio.NpzFile) or key not in npz_data.files:
            return None
        source_prompts = np.asarray(npz_data[key]).astype(str)
    finally:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            npz_data.close()

    if source_prompts.size != image_num:
        raise ValueError(
            "Expected '{}' in {} to have {} values, but got shape {}.".format(
                key,
                npz_file,
                image_num,
                source_prompts.shape,
            )
        )
    return source_prompts.reshape(-1)


def load_sample_names_from_npz(npz_file, image_num, key="sample_names"):
    npz_data = np.load(npz_file, allow_pickle=False)
    try:
        if not isinstance(npz_data, np.lib.npyio.NpzFile) or key not in npz_data.files:
            return None
        sample_names = np.asarray(npz_data[key]).astype(str)
    finally:
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            npz_data.close()

    if sample_names.size != image_num:
        raise ValueError(
            "Expected '{}' in {} to have {} values, but got shape {}.".format(
                key,
                npz_file,
                image_num,
                sample_names.shape,
            )
        )
    return sample_names.reshape(-1).tolist()


def load_labels_txt(labels_txt, image_num, labels_one_based=True):
    labels = []
    with open(labels_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                label = int(stripped)
                labels.append(label - 1 if labels_one_based else label)

    if len(labels) < image_num:
        raise ValueError(
            "Expected at least {} labels in {}, but found {}.".format(
                image_num,
                labels_txt,
                len(labels),
            )
        )
    return labels[:image_num]


def resolve_clean_image_path(images_root, image_id):
    images_root = Path(images_root)
    image_id = str(image_id)
    candidate = images_root / image_id
    if candidate.suffix and candidate.exists():
        return candidate

    for suffix in (".png", ".jpg", ".jpeg", ".JPEG"):
        candidate = images_root / "{}{}".format(image_id, suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Clean image not found for {} under {}".format(image_id, images_root))


def load_category_phrases(csv_filename):
    excluded_phrases = {"light"}
    category_phrases = {}
    with open(csv_filename, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            try:
                label_idx = int(row["CategoryId"]) - 1
            except Exception:
                continue
            raw_name = str(row.get("CategoryName", "")).strip()
            phrases = [
                phrase.strip().lower()
                for phrase in raw_name.split(",")
                if phrase.strip() and phrase.strip().lower() not in excluded_phrases
            ]
            if label_idx >= 0 and phrases:
                category_phrases[label_idx] = phrases
    return category_phrases


def phrase_in_prompt(phrase, prompt):
    pattern = r"(?<![a-z0-9]){}(?![a-z0-9])".format(re.escape(phrase.lower()))
    return re.search(pattern, prompt.lower()) is not None


def build_prompt_correct_label_sets(source_prompts, category_phrases):
    prompt_correct_label_sets = []
    for prompt in source_prompts:
        prompt_text = str(prompt)
        labels = {
            label_idx
            for label_idx, phrases in category_phrases.items()
            if any(phrase_in_prompt(phrase, prompt_text) for phrase in phrases)
        }
        prompt_correct_label_sets.append(labels)
    return prompt_correct_label_sets


def predictions_in_correct_labels(pred_labels, labels, prompt_correct_label_sets=None):
    if prompt_correct_label_sets is None:
        return pred_labels == labels
    return np.asarray(
        [
            int(pred_label) == int(label) or int(pred_label) in prompt_correct_label_sets[i]
            for i, (pred_label, label) in enumerate(zip(pred_labels, labels))
        ],
        dtype=bool,
    )


def eval_asr_from_npz(
    file_name=None,
    re_logger=True,
    quant=False,
    model_name=None,
    use_npz_source_query=False,
    use_prompt_label_as_correct=True,
    npz_path=None,
    defense_name=None,
    jpeg_quality=75,
    bit_depth=4,
    hgd_source_dir=None,
    hgd_checkpoint_dir=None,
    hgd_models=None,
    nips_r3_torch_weights_dir=None,
    nips_r3_source_dir=None,
    nips_r3_checkpoint_dir=None,
    nips_r3_python=None,
    nips_r3_batch_size=16,
    nips_r3_predict_batch_size=1000,
    clean_images_root="./data/nips2017/images/",
    ground_truth_csv="./data/nips2017/images.csv",
    labels_txt=None,
    labels_one_based=True,
    fixed_clean_correct_masks=None,
    asr_denominator="clean",
):

    assert model_name is not None
    if isinstance(model_name, str):
        models_transfer_name = [model_name]
    elif isinstance(model_name, (list, tuple)):
        # Accept a list/tuple of model names, and flatten one accidental nesting level.
        models_transfer_name = []
        for item in model_name:
            if isinstance(item, (list, tuple)):
                models_transfer_name.extend(list(item))
            else:
                models_transfer_name.append(item)
    else:
        raise TypeError("model_name must be str or list/tuple of str, got {}".format(type(model_name)))

    hgd_requested = is_hgd_defense_name(defense_name)
    nips_r3_requested = is_nips_r3_defense_name(defense_name)
    nips_r3_tf1_requested = is_nips_r3_tf1_defense_name(defense_name)
    if hgd_requested:
        models_transfer_name = ["hgd"]
    elif nips_r3_requested:
        models_transfer_name = ["nips-r3"]
    elif nips_r3_tf1_requested:
        models_transfer_name = ["nips-r3-tf1"]

    if file_name is None and npz_path is None:
        raise ValueError("file_name or npz_path must be provided.")

    npz_path = Path(npz_path).expanduser() if npz_path is not None else None
    file_dir = resolve_eval_dir(file_name) if file_name is not None else npz_path.parent

    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        eval_prefix = "eval"
        if defense_name:
            safe_defense_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(defense_name)).strip("_")
            eval_prefix = "eval_defense_{}".format(safe_defense_name or "unknown")
        save_dir = file_dir / (eval_prefix + "_" + str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=str(save_dir))


    logger.log("******* Evaluating {} ASR ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))
    logger.log("Defense: {}".format(defense_name if defense_name else "none"))
    if hgd_requested:
        logger.log("HGD is a model-level defense; --model is ignored and the HGD ensemble is evaluated.")
    if nips_r3_requested:
        logger.log("NIPS-R3 is a model-level defense; --model is ignored and the torch-converted mmd sub-ensemble is evaluated.")
    if nips_r3_tf1_requested:
        logger.log("NIPS-R3 TF1 is a model-level defense; --model is ignored and the upstream mmd defense is evaluated.")

    if npz_path is not None:
        npz_file = npz_path
    else:
        npz_file = file_dir / "adversarial_examples_correction.npz"
        if not npz_file.exists():
            npz_file = file_dir / "adversarial_examples_new.npz"
        if not npz_file.exists():
            npz_file = file_dir / "adversarial_examples.npz"
    if not npz_file.exists():
        raise FileNotFoundError("Adversarial examples not found: {}".format(npz_file))
    logger.log("Using adversarial examples: {}".format(npz_file.name))

    # Keep external adversarial examples as-is (user-provided data is already in [0, 1]).
    all_adv_images = load_adv_images_from_npz(npz_file, output_layout="nchw", normalize=True)
    logger.log(
        "Loaded adversarial images range : min={:.6f}, max={:.6f}".format(
            float(all_adv_images.min()), float(all_adv_images.max())
        )
    )

    # if quant:
    #     if is_8bit_quantized_01(all_adv_images):
    #         logger.log("Adversarial images already 8-bit quantized. Skip requantization.")
    #     else:
    #         logger.log("Adversarial images are not 8-bit quantized. Apply requantization.")
    #         all_adv_images = quantize_to_8bit_01(all_adv_images)

    image_num = all_adv_images.shape[0]
    source_model_name = get_source_model_name(file_name)
    if source_model_name not in models_transfer_name and file_dir.parent.name in models_transfer_name:
        source_model_name = file_dir.parent.name
    victim_query_counts = None
    query_count_source = None
    if use_npz_source_query:
        victim_query_counts = load_source_query_counts_from_npz(npz_file, image_num)
        if victim_query_counts is not None:
            query_count_source = "adversarial_examples.npz[source_query]"

    if victim_query_counts is None:
        victim_query_counts = load_victim_query_counts(str(file_dir), image_num)
        if victim_query_counts is not None:
            query_count_source = "run_summary.json[victim_query_count]"

    if query_count_source is not None:
        logger.log("Loaded query counts from {}.".format(query_count_source))
    elif use_npz_source_query:
        logger.log("No numeric query counts found in adversarial_examples.npz[source_query] or run_summary.json.")

    logger.log("source model for query stats: {}".format(source_model_name))

    is_asa_eval = "ASA" or "nips2017" in str(file_name)
    asa_query_exhausted = None
    asa_gemma_call_counts = None
    if is_asa_eval:
        asa_query_exhausted, asa_gemma_call_counts = load_asa_run_stats(str(file_dir), image_num)
        npz_query_exhausted = load_npz_array_from_keys(
            npz_file,
            image_num,
            ("victim_query_exhausted", "query_exhausted"),
        )
        if npz_query_exhausted is not None:
            npz_query_exhausted = np.asarray(
                [summary_value_to_bool(value) for value in npz_query_exhausted],
                dtype=bool,
            )
            asa_query_exhausted = (
                npz_query_exhausted
                if asa_query_exhausted is None
                else np.logical_or(asa_query_exhausted, npz_query_exhausted)
            )
        npz_gemma_call_counts = load_npz_array_from_keys(
            npz_file,
            image_num,
            (
                "source_gemma_call_count",
                "gemma_call_count",
                "gemma_calls",
                "gemma_query_count",
                "gemma_query_calls",
            ),
        )
        if npz_gemma_call_counts is not None and np.issubdtype(npz_gemma_call_counts.dtype, np.number):
            asa_gemma_call_counts = npz_gemma_call_counts.astype(np.float32, copy=False)
        if asa_query_exhausted is not None:
            logger.log("ASA query exhausted samples: {}".format(int(np.sum(asa_query_exhausted))))
        if asa_gemma_call_counts is not None and np.any(~np.isnan(asa_gemma_call_counts)):
            logger.log("Loaded ASA gemma call counts.")

    prompt_correct_label_sets = None
    if use_prompt_label_as_correct:
        source_prompts = load_source_prompts_from_npz(npz_file, image_num)
        if source_prompts is None:
            logger.log("ASA prompt-label correctness enabled, but adversarial_examples.npz[source_prompt] was not found.")
        else:
            category_phrases = load_category_phrases('./data/nips2017/categories.csv')
            prompt_correct_label_sets = build_prompt_correct_label_sets(source_prompts, category_phrases)
            matched_count = sum(1 for labels in prompt_correct_label_sets if labels)
            logger.log(
                "ASA prompt-label correctness enabled: matched category phrases in {}/{} prompts.".format(
                    matched_count,
                    image_num,
                )
            )

    if labels_txt is not None:
        images_root = clean_images_root
        image_id_list = load_sample_names_from_npz(npz_file, image_num)
        if image_id_list is None:
            image_id_list = [str(index + 1) for index in range(image_num)]
        label_ori_list = load_labels_txt(labels_txt, image_num, labels_one_based=labels_one_based)
        label_tar_list = [0 for _ in range(image_num)]
        logger.log("Using clean images root: {}".format(images_root))
        logger.log(
            "Using labels txt: {} ({})".format(
                labels_txt,
                "1-based" if labels_one_based else "0-based",
            )
        )
    else:
        images_root = clean_images_root
        image_id_list, label_ori_list, label_tar_list = load_ground_truth(ground_truth_csv)
        logger.log("Using clean images root: {}".format(images_root))
        logger.log("Using ground truth csv: {}".format(ground_truth_csv))

    all_ori_labels = []
    all_tar_labels = []
    all_clean_images = []

    image_size = all_adv_images.shape[2]
    print("loading original images")
    for i in tqdm(range(image_num)):
        original_image = Image.open(resolve_clean_image_path(images_root, image_id_list[i])).convert('RGB')
        original_image = original_image.resize((image_size, image_size), resample=Image.BILINEAR)
        original_image = np.array(original_image)
        all_clean_images.append(original_image)

        label_ori_now = label_ori_list[i]
        all_ori_labels.append(np.array(label_ori_now)[None])
        label_tar_now = label_tar_list[i]
        all_tar_labels.append(np.array(label_tar_now)[None])

    all_clean_images = np.stack(all_clean_images, axis=0)
    all_clean_images = all_clean_images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    all_ori_labels = np.concatenate(all_ori_labels, axis=0)
    all_tar_labels = np.concatenate(all_tar_labels, axis=0)

    assert all_adv_images.shape[2] == all_adv_images.shape[3] == all_clean_images.shape[2] == all_clean_images.shape[3]
    img_res = all_adv_images.shape[2]
    logger.log("image res: {}x{}".format(img_res, img_res))
    logger.log("eval models: {}".format(models_transfer_name))

    result = model_transfer(
        all_clean_images,
        all_adv_images,
        all_ori_labels,
        all_tar_labels,
        res=img_res,
        models_transfer_name=models_transfer_name,
        nb_classes=1000,
        source_model_name=source_model_name,
        victim_query_counts=victim_query_counts,
        asa_query_exhausted=asa_query_exhausted,
        asa_gemma_call_counts=asa_gemma_call_counts,
        prompt_correct_label_sets=prompt_correct_label_sets,
        defense_name=defense_name,
        jpeg_quality=jpeg_quality,
        bit_depth=bit_depth,
        hgd_source_dir=hgd_source_dir,
        hgd_checkpoint_dir=hgd_checkpoint_dir,
        hgd_models=hgd_models,
        nips_r3_torch_weights_dir=nips_r3_torch_weights_dir,
        nips_r3_source_dir=nips_r3_source_dir,
        nips_r3_checkpoint_dir=nips_r3_checkpoint_dir,
        nips_r3_python=nips_r3_python,
        nips_r3_batch_size=nips_r3_batch_size,
        nips_r3_predict_batch_size=nips_r3_predict_batch_size,
        fixed_clean_correct_masks=fixed_clean_correct_masks,
        asr_denominator=asr_denominator,
    )
    return result

def model_transfer(
    clean_img,
    adv_img,
    label,
    label_tar,
    res,
    models_transfer_name,
    nb_classes=1000,
    save_path=None,
    source_model_name=None,
    victim_query_counts=None,
    asa_query_exhausted=None,
    asa_gemma_call_counts=None,
    prompt_correct_label_sets=None,
    defense_name=None,
    jpeg_quality=75,
    bit_depth=4,
    hgd_source_dir=None,
    hgd_checkpoint_dir=None,
    hgd_models=None,
    nips_r3_torch_weights_dir=None,
    nips_r3_source_dir=None,
    nips_r3_checkpoint_dir=None,
    nips_r3_python=None,
    nips_r3_batch_size=16,
    nips_r3_predict_batch_size=1000,
    fixed_clean_correct_masks=None,
    asr_denominator="clean",
):
    if asr_denominator not in {"clean", "total"}:
        raise ValueError("asr_denominator must be 'clean' or 'total', got {}".format(asr_denominator))

    all_clean_accuracy = []
    all_adv_accuracy = []
    all_untarget_asr = []
    all_clean_correct_counts = []
    all_asr_denominators = []
    all_fixed_clean_baselines = []
    all_success_query_sums = []
    all_success_query_means = []
    all_gemma_call_means = []
    clean_correct_masks_by_model = {}
    device_type = 'gpu' if th.cuda.is_available() else 'cpu'
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    hgd_requested = is_hgd_defense_name(defense_name)
    nips_r3_requested = is_nips_r3_defense_name(defense_name)
    nips_r3_tf1_requested = is_nips_r3_tf1_defense_name(defense_name)

    for name in tqdm(models_transfer_name):
        # logger.log("*********transfer to {}********".format(name))
        preprocess_np = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        if hgd_requested:
            model = HGDNIPSDefense(
                source_dir=hgd_source_dir,
                checkpoint_dir=hgd_checkpoint_dir,
                model_names=hgd_models,
            ).to(device)
            pred_offset = 0
            preprocess_np = (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        elif nips_r3_requested:
            model = NIPSR3Defense(
                weights_dir=nips_r3_torch_weights_dir,
            ).to(device)
            pred_offset = 0
            preprocess_np = (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        elif nips_r3_tf1_requested:
            model = NIPSR3TF1Defense(
                source_dir=nips_r3_source_dir,
                checkpoint_dir=nips_r3_checkpoint_dir,
                python_executable=nips_r3_python,
                batch_size=nips_r3_batch_size,
            ).to(device)
            pred_offset = 0
            preprocess_np = (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        else:
            model = model_selection(name)
            pred_offset = 1 if name in {"adv_inc", "adv_res"} else 0

        if defense_name and not hgd_requested and not nips_r3_requested and not nips_r3_tf1_requested:
            model = DefendedModel(
                model=model,
                defense=defense_name,
                image_size=res,
                normalize=True,
                jpeg_quality=jpeg_quality,
                bit_depth=bit_depth,
            )
            preprocess_np = (
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        model.eval()

        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=th.nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=preprocess_np,
            device_type=device_type,
        )

        if hgd_requested:
            predict_batch_size = 4
        elif nips_r3_requested:
            predict_batch_size = min(nips_r3_predict_batch_size, 4)
        elif nips_r3_tf1_requested:
            predict_batch_size = nips_r3_predict_batch_size
        else:
            predict_batch_size = 50
        clean_pred = f_model.predict(clean_img, batch_size=predict_batch_size)
        clean_pred_label = np.argmax(clean_pred, axis=1) - pred_offset
        clean_correct_mask = clean_pred_label == label
        clean_correct_masks_by_model[name] = clean_correct_mask.copy()
        accuracy = np.sum(clean_correct_mask) / len(label)
        all_clean_accuracy.append(accuracy * 100)
        all_clean_correct_counts.append(int(np.sum(clean_correct_mask)))

        adv_pred = f_model.predict(adv_img, batch_size=predict_batch_size)
        print(f"adv_img_max:{adv_img.max()}")
        adv_pred_label = np.argmax(adv_pred, axis=1) - pred_offset
        adv_correct_mask = predictions_in_correct_labels(adv_pred_label, label, prompt_correct_label_sets)
        fixed_clean_correct_mask = None
        if asr_denominator == "total":
            success_mask = ~adv_correct_mask
            clean_correct_count = len(label)
        else:
            if fixed_clean_correct_masks is not None:
                if isinstance(fixed_clean_correct_masks, dict):
                    fixed_clean_correct_mask = fixed_clean_correct_masks.get(name)
                else:
                    fixed_clean_correct_mask = fixed_clean_correct_masks[i]
                fixed_clean_correct_mask = np.asarray(fixed_clean_correct_mask, dtype=bool)
                if fixed_clean_correct_mask.shape[0] != label.shape[0]:
                    raise ValueError(
                        "fixed clean mask for {} has length {}, expected {}".format(
                            name,
                            fixed_clean_correct_mask.shape[0],
                            label.shape[0],
                        )
                    )

            asr_clean_correct_mask = fixed_clean_correct_mask if fixed_clean_correct_mask is not None else clean_correct_mask
            success_mask = np.logical_and(asr_clean_correct_mask, ~adv_correct_mask)
            clean_correct_count = np.sum(asr_clean_correct_mask)
        if asa_query_exhausted is not None:
            success_mask = np.logical_and(success_mask, ~asa_query_exhausted)
        accuracy = np.sum(adv_correct_mask) / len(label)
        all_adv_accuracy.append(accuracy * 100)
        all_asr_denominators.append(int(clean_correct_count))
        all_fixed_clean_baselines.append(fixed_clean_correct_mask is not None)
        untarget_asr = float(np.sum(success_mask) / clean_correct_count * 100) if clean_correct_count > 0 else float("nan")
        all_untarget_asr.append(untarget_asr)

        success_query_sum = None
        success_query_mean = None
        if name == source_model_name and victim_query_counts is not None:
            success_queries = victim_query_counts[success_mask]
            success_queries = success_queries[~np.isnan(success_queries)]
            if success_queries.size > 0:
                success_query_sum = float(np.sum(success_queries))
                success_query_mean = float(np.mean(success_queries))
        all_success_query_sums.append(success_query_sum)
        all_success_query_means.append(success_query_mean)

        gemma_call_mean = None
        if name == source_model_name and asa_gemma_call_counts is not None:
            gemma_calls = asa_gemma_call_counts[success_mask]
            gemma_calls = gemma_calls[~np.isnan(gemma_calls)]
            if gemma_calls.size > 0:
                gemma_call_mean = float(np.mean(gemma_calls))
        all_gemma_call_means.append(gemma_call_mean)

    logger.log("***********************************************************")
    for i in range(len(models_transfer_name)):
        log_message = "{}\nclean acc: {:.2f}, adv acc: {:.2f}, untarget_asr: {:.2f}".format(
            models_transfer_name[i],
            all_clean_accuracy[i],
            all_adv_accuracy[i],
            all_untarget_asr[i],
        )
        if asr_denominator == "total":
            log_message += ", total_count: {}/{}".format(all_asr_denominators[i], len(label))
        elif all_fixed_clean_baselines[i]:
            log_message += ", fixed_clean_count: {}/{}".format(all_asr_denominators[i], len(label))
        else:
            log_message += ", clean_count: {}/{}".format(all_clean_correct_counts[i], len(label))
        if all_success_query_sums[i] is not None:
            log_message += ", success_query_sum: {:.2f}, success_query_avg: {:.2f}".format(
                all_success_query_sums[i],
                all_success_query_means[i],
            )
        if all_gemma_call_means[i] is not None:
            log_message += ", gemma_call_avg: {:.2f}".format(all_gemma_call_means[i])
        logger.log(log_message)
    logger.log("***********************************************************")
    metrics = {}
    for i, name in enumerate(models_transfer_name):
        metrics[name] = {
            "clean_acc": all_clean_accuracy[i],
            "adv_acc": all_adv_accuracy[i],
            "untarget_asr": all_untarget_asr[i],
            "clean_correct_count": all_clean_correct_counts[i],
            "asr_denominator": all_asr_denominators[i],
            "asr_denominator_mode": asr_denominator,
            "fixed_clean_baseline": all_fixed_clean_baselines[i],
            "success_query_sum": all_success_query_sums[i],
            "success_query_mean": all_success_query_means[i],
            "gemma_call_mean": all_gemma_call_means[i],
        }
    return {
        "metrics": metrics,
        "clean_correct_masks": clean_correct_masks_by_model,
    }


if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_asr_from_npz(file_name=dir, quant=False)
        eval_asr_from_npz(file_name=dir, quant=True)

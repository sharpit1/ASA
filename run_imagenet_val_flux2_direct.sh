#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-$ROOT_DIR/data/ILSVRC2012_img_val}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/imagenet_val/flux2_kv_mist_dust_10perclass_20260725}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
RENDER_BATCH_SIZE="${RENDER_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-80000}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
TRANSFER_MODELS="${TRANSFER_MODELS:-resnet50 wrn50 inception_v3 convnext vgg19 vit swin deit}"

export ASA_PROJECT_ROOT="${ASA_PROJECT_ROOT:-$ROOT_DIR}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT_DIR/.cache/huggingface}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if ! [[ "$SAMPLES_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SAMPLES_PER_CLASS must be a positive integer." >&2
  exit 2
fi

data_ready() {
  "$PYTHON_BIN" - "$IMAGENET_VAL_DIR" "$SAMPLES_PER_CLASS" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = int(sys.argv[2])
if not root.is_dir():
    raise SystemExit(1)
class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
if len(class_dirs) != 1000:
    raise SystemExit(1)
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
for class_dir in class_dirs:
    count = sum(
        1
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if count < required:
        raise SystemExit(1)
PY
}

gpu_ready() {
  "$PYTHON_BIN" - "$MIN_FREE_VRAM_MIB" <<'PY'
import sys
import torch

minimum_mib = int(sys.argv[1])
free_bytes, total_bytes = torch.cuda.mem_get_info()
free_mib = int(free_bytes // (1024 * 1024))
total_mib = int(total_bytes // (1024 * 1024))
print(f"[imagenet-edit] GPU memory free={free_mib} MiB total={total_mib} MiB")
raise SystemExit(0 if free_mib >= minimum_mib else 1)
PY
}

while ! data_ready; do
  echo "[imagenet-edit] waiting for ImageNet val: $IMAGENET_VAL_DIR (1000 classes, >=${SAMPLES_PER_CLASS} images/class)"
  sleep "$WAIT_INTERVAL_SECONDS"
done
echo "[imagenet-edit] ImageNet val is ready: $IMAGENET_VAL_DIR"

while ! gpu_ready; do
  echo "[imagenet-edit] waiting for at least ${MIN_FREE_VRAM_MIB} MiB free VRAM"
  sleep "$WAIT_INTERVAL_SECONDS"
done

read -r -a MODEL_ARGS <<< "$TRANSFER_MODELS"
mkdir -p "$OUTPUT_DIR"
echo "[imagenet-edit] output_dir=$OUTPUT_DIR"
echo "[imagenet-edit] prompts=2 source_images=$((1000 * SAMPLES_PER_CLASS)) edited_images=$((2 * 1000 * SAMPLES_PER_CLASS))"
echo "[imagenet-edit] transfer_models=${MODEL_ARGS[*]}"

exec "$PYTHON_BIN" eval/edit_imagenet_val_flux2.py \
  --imagenet_val_dir "$IMAGENET_VAL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --prompt "introduce light mist" \
  --prompt "coated in fine dust" \
  --models "${MODEL_ARGS[@]}" \
  --samples_per_class "$SAMPLES_PER_CLASS" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --render_batch_size "$RENDER_BATCH_SIZE" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --save_clean_batch off \
  --resume

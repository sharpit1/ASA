#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: python interpreter not found." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/flux2_and_attack_nips.yaml}"
DATASET_ROOT="${DATASET_ROOT:-data/nips2017}"
DATASET_NAME="${DATASET_NAME:-nips2017}"
SOURCE_MODEL="${SOURCE_MODEL:-resnet50}"
TRANSFER_MODELS="${TRANSFER_MODELS:-resnet50,wrn50,inception_v3,convnext,vgg19,vit,swin,deit}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
RESULT_DIR="${RESULT_DIR:-results}"
RUN_NAME="${RUN_NAME:-faas_${SOURCE_MODEL}_$(date +%Y%m%d_%H%M%S)}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
FAAS_DRY_RUN="${FAAS_DRY_RUN:-0}"
NPZ_BATCH_SIZE="${NPZ_BATCH_SIZE:-64}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT_DIR/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-$ROOT_DIR/ckpt/torch_hub}"
BASE_CONSTRAINTS_PATH=""

make_absolute() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$ROOT_DIR" "$value"
  fi
}

CONFIG="$(make_absolute "$CONFIG")"
DATASET_ROOT="$(make_absolute "$DATASET_ROOT")"
OUTPUT_ROOT="$(make_absolute "$OUTPUT_ROOT")"
RESULT_DIR="$(make_absolute "$RESULT_DIR")"
HF_CACHE_ROOT="$(make_absolute "$HF_CACHE_ROOT")"
TORCH_HOME="$(make_absolute "$TORCH_HOME")"
RUN_ROOT="$OUTPUT_ROOT/$DATASET_NAME/$SOURCE_MODEL/$RUN_NAME"
RESULT_TXT="$RESULT_DIR/$RUN_NAME.txt"

HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME

mkdir -p "$RESULT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
exec > >(tee "$RESULT_TXT") 2>&1

on_exit() {
  local rc="$?"
  if [[ -n "$BASE_CONSTRAINTS_PATH" && -f "$BASE_CONSTRAINTS_PATH" ]]; then
    rm -f "$BASE_CONSTRAINTS_PATH"
  fi
  printf '[faas] exit_code=%s\n' "$rc"
  printf '[faas] result_txt=%s\n' "$RESULT_TXT"
}
trap on_exit EXIT

capture_provider_stack() {
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
import json

packages = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "vllm",
    "transformers",
    "accelerate",
    "datasets",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pillow",
    "opencv-python-headless",
)
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
print(json.dumps(versions, sort_keys=True))
PY
}

capture_pip_check() {
  "$PYTHON_BIN" - <<'PY'
import json
import subprocess
import sys

completed = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
issues = []
if completed.returncode != 0:
    issues = sorted(
        {
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        }
    )
print(json.dumps(issues, sort_keys=True))
PY
}

case "$INSTALL_DEPS" in
  1|true|yes|on)
    echo "[faas] stage=install_dependencies"
    echo "[faas] installing requirements-faas.txt without replacing the base torch stack"
    BASE_STACK_JSON="$(capture_provider_stack)"
    BASE_PIP_CHECK_JSON="$(capture_pip_check)"
    echo "[faas] base_stack_before=$BASE_STACK_JSON"
    echo "[faas] pip_check_before=$BASE_PIP_CHECK_JSON"
    BASE_CONSTRAINTS_PATH="$(mktemp "${TMPDIR:-/tmp}/asa-faas-base-constraints.XXXXXX")"
    printf '%s' "$BASE_STACK_JSON" | "$PYTHON_BIN" -c \
      'import json,sys; data=json.load(sys.stdin); print("\n".join(f"{key}=={value}" for key,value in sorted(data.items())))' \
      > "$BASE_CONSTRAINTS_PATH"
    PIP_CONSTRAINT="$BASE_CONSTRAINTS_PATH" "$PYTHON_BIN" -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --prefer-binary \
      -r "$ROOT_DIR/requirements-faas.txt"
    POST_INSTALL_STACK_JSON="$(capture_provider_stack)"
    POST_INSTALL_PIP_CHECK_JSON="$(capture_pip_check)"
    echo "[faas] base_stack_after=$POST_INSTALL_STACK_JSON"
    echo "[faas] pip_check_after=$POST_INSTALL_PIP_CHECK_JSON"
    if [[ "$BASE_STACK_JSON" != "$POST_INSTALL_STACK_JSON" ]]; then
      echo "ERROR: dependency installation changed the provider base stack." >&2
      exit 2
    fi
    if [[ "$BASE_PIP_CHECK_JSON" != "$POST_INSTALL_PIP_CHECK_JSON" ]]; then
      echo "ERROR: dependency installation changed the pip check result." >&2
      exit 2
    fi
    if [[ "$POST_INSTALL_PIP_CHECK_JSON" != "[]" ]]; then
      echo "[faas] WARNING: preserving pre-existing provider pip issues: $POST_INSTALL_PIP_CHECK_JSON"
    fi
    ;;
  0|false|no|off|'') ;;
  *)
    echo "ERROR: INSTALL_DEPS must be boolean-like (got '$INSTALL_DEPS')." >&2
    exit 2
    ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi
for required in images images.csv categories.csv gt.csv; do
  if [[ ! -e "$DATASET_ROOT/$required" ]]; then
    echo "ERROR: NIPS2017 dataset entry missing: $DATASET_ROOT/$required" >&2
    exit 2
  fi
done
if ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_SAMPLES must be a positive integer (got '$MAX_SAMPLES')." >&2
  exit 2
fi
if ! [[ "$NPZ_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NPZ_BATCH_SIZE must be a positive integer (got '$NPZ_BATCH_SIZE')." >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: refusing to mix with an existing run: $RUN_ROOT" >&2
  exit 2
fi

case ",$TRANSFER_MODELS," in
  *",$SOURCE_MODEL,"*) ;;
  *)
    echo "ERROR: TRANSFER_MODELS must include SOURCE_MODEL=$SOURCE_MODEL." >&2
    exit 2
    ;;
esac

echo "[faas] stage=configuration"
echo "[faas] python=$PYTHON_BIN"
echo "[faas] config=$CONFIG"
echo "[faas] dataset=$DATASET_ROOT"
echo "[faas] source_model=$SOURCE_MODEL"
echo "[faas] transfer_models=$TRANSFER_MODELS"
echo "[faas] max_samples=$MAX_SAMPLES"
echo "[faas] run_root=$RUN_ROOT"
echo "[faas] hf_cache_root=$HF_CACHE_ROOT"
echo "[faas] torch_home=$TORCH_HOME"

case "$FAAS_DRY_RUN" in
  1|true|yes|on)
    echo "[faas] stage=attack_dry_run"
    env \
      CONFIG="$CONFIG" \
      DATASET_ROOT="$DATASET_ROOT" \
      DATASET_NAME="$DATASET_NAME" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      RUN_NAME="$RUN_NAME" \
      MAX_SAMPLES="$MAX_SAMPLES" \
      VICTIM_MODEL="$SOURCE_MODEL" \
      HF_CACHE_ROOT="$HF_CACHE_ROOT" \
      RUN_VLM_ATTACK_DRY_RUN=1 \
      "$ROOT_DIR/run_vlm_attack.sh"
    echo "[faas] dry_run_complete"
    exit 0
    ;;
  0|false|no|off|'') ;;
  *)
    echo "ERROR: FAAS_DRY_RUN must be boolean-like (got '$FAAS_DRY_RUN')." >&2
    exit 2
    ;;
esac

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN_FILE:-}" ]]; then
  echo "ERROR: configure HF_TOKEN (preferably as a FaaS secret) or HF_TOKEN_FILE." >&2
  exit 2
fi

echo "[faas] stage=environment_preflight"
"$PYTHON_BIN" - <<'PY'
import json

import torch
import torchvision
import transformers
import diffusers
import timm
import yaml
from art.estimators.classification import PyTorchClassifier
from diffusers import Flux2KleinKVPipeline
from natsort import natsorted
from pytorch_fid import fid_score
from transformers import AutoModelForImageTextToText, AutoProcessor

del (
    AutoModelForImageTextToText,
    AutoProcessor,
    Flux2KleinKVPipeline,
    PyTorchClassifier,
    fid_score,
    natsorted,
    timm,
    yaml,
)

versions = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
}
print("[faas] environment_versions=" + json.dumps(versions, sort_keys=True))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
matrix = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
witness = matrix @ matrix.T
torch.cuda.synchronize()
if witness.shape != (8, 8) or not torch.isfinite(witness).all():
    raise SystemExit("CUDA kernel witness produced an invalid result")
print(
    "[faas] CUDA_WITNESS "
    f"shape={tuple(witness.shape)} "
    f"sum={float(witness.sum().item()):.1f} "
    f"device={torch.cuda.get_device_name(torch.cuda.current_device())}"
)
PY

echo "[faas] stage=source_attack"
env \
  CONFIG="$CONFIG" \
  DATASET_ROOT="$DATASET_ROOT" \
  DATASET_NAME="$DATASET_NAME" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RUN_NAME="$RUN_NAME" \
  MAX_SAMPLES="$MAX_SAMPLES" \
  VICTIM_MODEL="$SOURCE_MODEL" \
  HF_CACHE_ROOT="$HF_CACHE_ROOT" \
  "$ROOT_DIR/run_vlm_attack.sh"

SUMMARY_PATH="$RUN_ROOT/run_summary.json"
"$PYTHON_BIN" - "$SUMMARY_PATH" "$MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not summary_path.is_file():
    raise SystemExit(f"missing run summary: {summary_path}")
payload = json.loads(summary_path.read_text(encoding="utf-8"))
processed = int(payload.get("total_processed", -1))
failed = int(payload.get("fail_count", -1))
if processed != expected or failed != 0:
    raise SystemExit(
        f"source attack incomplete: processed={processed}/{expected}, fail_count={failed}"
    )
print(
    "[faas] source_attack_complete "
    f"processed={processed} attack_success={payload.get('attack_success_count')}"
)
PY

echo "[faas] stage=build_npz"
"$PYTHON_BIN" "$ROOT_DIR/create_adversarial_examples_npz.py" \
  --root "$RUN_ROOT" \
  --output "$RUN_ROOT/adversarial_examples.npz" \
  --image-mode auto \
  --size 224 \
  --batch-size "$NPZ_BATCH_SIZE" \
  --dtype float32 \
  --overwrite

echo "[faas] stage=source_transfer_eval"
"$PYTHON_BIN" "$ROOT_DIR/eval/run_source_transfer.py" \
  --run-root "$RUN_ROOT" \
  --models "$TRANSFER_MODELS"

echo "[faas] pipeline_complete"

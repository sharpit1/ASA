#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap and run the Qwen Image Edit VLM attack from any ASA checkout path.
# The kau/pytorch-master image is expected to provide the CUDA/PyTorch stack;
# this script installs only the project-specific additions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen_edit_vlm_res.yaml}"
SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE:-outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000/target_200_indices.json}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export HF_CACHE_ROOT
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/transformers}"
export TORCH_HOME
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
cd "$PROJECT_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
  token_file="$HF_TOKEN_FILE"
  [[ "$token_file" != /* ]] && token_file="$PROJECT_ROOT/$token_file"
  [[ -f "$token_file" ]] || die "HF token file not found: $token_file"
  HF_TOKEN="$(tr -d '\r\n' < "$token_file")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    return 1
  fi
}

if [[ -n "$PYTHON_BIN" ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  PYTHON_BIN="$(find_system_python)" \
    || die "python3/python is required"
  echo "[setup] using provider system Python: $PYTHON_BIN"
fi

[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
[[ -f "$PROJECT_ROOT/requirements-faas.txt" ]] \
  || die "missing requirements file: requirements-faas.txt"
[[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]] \
  || die "missing isolated config: isolated_vlm_attack/$CONFIG_PATH"
[[ "$SAMPLE_INDICES_FILE" != /* ]] \
  || die "SAMPLE_INDICES_FILE must be relative to the ASA project root"
[[ -f "$PROJECT_ROOT/$SAMPLE_INDICES_FILE" ]] \
  || die "missing sample indices file: $SAMPLE_INDICES_FILE"
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] \
  || die "MIN_FREE_GIB must be a non-negative integer"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi is unavailable inside the remote container"
  if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader; then
    die "nvidia-smi failed; fix the NVIDIA driver/NVML container mapping first"
  fi

  echo "[check] validating the kau/pytorch-master base package versions"
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import os
import sys

expected = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "triton": "3.6.0",
    "vllm": "0.22.0",
    "transformers": "5.10.1",
    "accelerate": "1.13.0",
    "datasets": "4.8.5",
    "tokenizers": "0.22.2",
    "safetensors": "0.7.0",
    "huggingface-hub": "1.17.0",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "matplotlib": "3.10.9",
    "pillow": "12.2.0",
    "opencv-python-headless": "4.13.0.92",
}

mismatches = []
for package, wanted in expected.items():
    try:
        found = metadata.version(package)
    except metadata.PackageNotFoundError:
        found = "NOT INSTALLED"
    if found != wanted:
        mismatches.append(f"{package}: expected {wanted}, found {found}")

if mismatches:
    print("Base image package mismatch:", file=sys.stderr)
    for item in mismatches:
        print(f"  - {item}", file=sys.stderr)
    if os.environ.get("ALLOW_BASE_VERSION_MISMATCH") != "1":
        raise SystemExit(
            "Set ALLOW_BASE_VERSION_MISMATCH=1 only after verifying compatibility."
        )

import torch

cuda_build = str(torch.version.cuda or "")
if not cuda_build.startswith("13.0"):
    raise SystemExit(f"expected a CUDA 13.0 PyTorch build, found {cuda_build!r}")
print(f"[check] torch={torch.__version__} torch.version.cuda={cuda_build}")
PY

  echo "[check] running a seeded CUDA kernel witness"
  "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
torch.manual_seed(0)
x = torch.randn((8, 8), device="cuda")
y = x @ x
torch.cuda.synchronize()
print(
    "WITNESS",
    tuple(y.shape),
    torch.cuda.get_device_name(0),
    float(y.abs().sum().item()),
)
PY

  echo "[setup] installing project additions without replacing the provider stack"
  "$PYTHON_BIN" -m pip install \
    --upgrade-strategy only-if-needed \
    -r "$PROJECT_ROOT/requirements-faas.txt"
  "$PYTHON_BIN" -m pip check

  echo "[check] validating Qwen runtime imports"
  "$PYTHON_BIN" - <<'PY'
import diffusers
import timm
import yaml
from diffusers import QwenImageEditPlusPipeline

if diffusers.__version__ != "0.37.1":
    raise SystemExit(
        f"requirements-faas.txt expects diffusers 0.37.1, found {diffusers.__version__}"
    )
print("[check] QwenImageEditPlusPipeline import OK")
PY

  qwen_cache="$HF_HUB_CACHE/models--Qwen--Qwen-Image-Edit-2511"
  gemma_cache="$HF_HUB_CACHE/models--google--gemma-4-E4B-it"
  if [[ ! -d "$qwen_cache/snapshots" || ! -d "$gemma_cache/snapshots" ]]; then
    free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
    [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
    free_gib=$((free_kib / 1024 / 1024))
    if (( free_gib < MIN_FREE_GIB )); then
      die "only ${free_gib} GiB is free; at least ${MIN_FREE_GIB} GiB is required for first model download"
    fi
    echo "[check] free disk space: ${free_gib} GiB"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is unset. Gemma download may require accepted model terms and a token." >&2
  fi

  echo "[download] caching Qwen generator and Gemma naturalness verifier"
  "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ["HF_HUB_CACHE"]
for repo_id in ("Qwen/Qwen-Image-Edit-2511", "google/gemma-4-E4B-it"):
    print(f"[download] {repo_id}", flush=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(f"[download] ready: {snapshot_path}", flush=True)
PY

  echo "[download] caching torchvision ResNet-50 weights"
  "$PYTHON_BIN" - <<'PY'
from torchvision.models import ResNet50_Weights, resnet50

model = resnet50(weights=ResNet50_Weights.DEFAULT)
del model
print("[download] ResNet-50 weights ready")
PY
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "[done] environment and model caches are prepared"
  exit 0
fi

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/qwen_vlm_res_$(date +%Y%m%d_%H%M%S).log}"

CMD=(
  env
  "PYTHON_BIN=$PYTHON_BIN"
  "ASA_PROJECT_ROOT=$PROJECT_ROOT"
  "HF_CACHE_ROOT=$HF_CACHE_ROOT"
  "HF_HOME=$HF_HOME"
  "HF_HUB_CACHE=$HF_HUB_CACHE"
  "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
  "TORCH_HOME=$TORCH_HOME"
  bash "$SCRIPT_DIR/run_vlm_attack.sh"
  --config "$CONFIG_PATH"
  --attack_mode vlm
  --victim_model resnet50
  --attack_only_clean_correct true
  --gcg_scene_vocab_enabled_strategies none
  --class_ablation false
  --gcg_eval_naturalness_on_attack_success true
  --gcg_eval_naturalness_llm_thinking false
  --max_samples 1000
  --qwen_batch_size 1
  --sample_indices_file "$SAMPLE_INDICES_FILE"
  "$@"
)

echo "[run] project root: $PROJECT_ROOT"
echo "[run] log: $LOG_FILE"
printf "[run] command:"
printf " %q" "${CMD[@]}"
printf "\n"

set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
run_rc=${PIPESTATUS[0]}
set -e
exit "$run_rc"

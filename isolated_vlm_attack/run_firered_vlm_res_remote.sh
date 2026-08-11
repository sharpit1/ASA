#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare a fresh FaaS checkout for FireRed-Image-Edit-1.1 and run the
# isolated launcher with the provider's system Python/CUDA stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PROJECT_ROOT/.cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
CONFIG_PATH="${CONFIG_PATH:-configs/firered_vlm_attack_nips.yaml}"
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

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    return 1
  fi
}

normalize_pip_check() {
  sed \
    -e '/^No broken requirements found\.$/d' \
    -e '/^[[:space:]]*$/d' \
    | LC_ALL=C sort -u
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

if [[ -n "$PYTHON_BIN" ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  PYTHON_BIN="$(find_system_python)" || die "python3/python is required"
  echo "[setup] using provider system Python: $PYTHON_BIN"
fi

[[ -x "$PYTHON_BIN" ]] || die "python is not executable: $PYTHON_BIN"
[[ -f "$PROJECT_ROOT/requirements-faas.txt" ]] \
  || die "missing requirements file: requirements-faas.txt"
[[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]] \
  || die "missing FireRed config: isolated_vlm_attack/$CONFIG_PATH"
[[ -f "$SCRIPT_DIR/firered_attack_runner.py" ]] \
  || die "missing FireRed runner: isolated_vlm_attack/firered_attack_runner.py"
[[ -f "$SCRIPT_DIR/firered_blackbox_runtime.py" ]] \
  || die "missing FireRed runtime: isolated_vlm_attack/firered_blackbox_runtime.py"
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] \
  || die "MIN_FREE_GIB must be a non-negative integer"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi is unavailable inside the FaaS container"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader \
    || die "nvidia-smi failed; fix the NVIDIA container mapping first"

  provider_pip_check="$("$PYTHON_BIN" -m pip check 2>&1 || true)"
  echo "[setup] installing ASA additions into provider Python"
  "$PYTHON_BIN" -m pip install \
    --upgrade-strategy only-if-needed \
    -r "$PROJECT_ROOT/requirements-faas.txt"
  installed_pip_check="$("$PYTHON_BIN" -m pip check 2>&1 || true)"
  new_pip_conflicts="$(
    comm -13 \
      <(printf '%s\n' "$provider_pip_check" | normalize_pip_check) \
      <(printf '%s\n' "$installed_pip_check" | normalize_pip_check)
  )"
  if [[ -n "$new_pip_conflicts" ]]; then
    echo "ERROR: project installation introduced dependency conflicts:" >&2
    printf '  - %s\n' "$new_pip_conflicts" >&2
    exit 1
  fi

  echo "[check] validating CUDA, FireRed pipeline, and ART imports"
  "$PYTHON_BIN" - <<'PY'
import diffusers
import torch
import yaml
from art.estimators.classification import PyTorchClassifier
from diffusers import QwenImageEditPlusPipeline

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if diffusers.__version__ != "0.37.1":
    raise SystemExit(
        f"requirements-faas.txt expects diffusers 0.37.1, found {diffusers.__version__}"
    )
print(
    "[check] FireRed pipeline and ART imports OK",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"gpu={torch.cuda.get_device_name(0)}",
)
PY

  firered_cache="$HF_HUB_CACHE/models--FireRedTeam--FireRed-Image-Edit-1.1"
  gemma_cache="$HF_HUB_CACHE/models--google--gemma-4-E4B-it"
  if [[ ! -d "$firered_cache/snapshots" || ! -d "$gemma_cache/snapshots" ]]; then
    free_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
    [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
    free_gib=$((free_kib / 1024 / 1024))
    if (( free_gib < MIN_FREE_GIB )); then
      die "only ${free_gib} GiB is free; at least ${MIN_FREE_GIB} GiB is required for first model download"
    fi
    echo "[check] free disk space: ${free_gib} GiB"
  fi

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[setup] HF_TOKEN is unset; public model access will be used" >&2
  fi

  echo "[download] caching FireRed generator and Gemma verifier"
  "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ["HF_HUB_CACHE"]
for repo_id in (
    "FireRedTeam/FireRed-Image-Edit-1.1",
    "google/gemma-4-E4B-it",
):
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
  echo "[done] FireRed FaaS environment and model caches are prepared"
  exit 0
fi

LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/firered_vlm_res_$(date +%Y%m%d_%H%M%S).log}"

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
)
CMD+=("$@")

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

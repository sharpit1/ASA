#!/usr/bin/env bash
set -Eeuo pipefail

# Run one NIPS2017 sample through the Bernini 1024x1024 FaaS bootstrap.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="configs/bernini_vlm_attack_nips_1024.yaml"
OUTPUT_ROOT="${FAAS_OUTPUT_ROOT:-/app/output/sharpit1}"
SMOKE_INDEX="${SMOKE_INDEX:-0}"
RUN_NAME="${RUN_NAME:-bernini_vlm_res_1024_smoke_$(date +%Y%m%d_%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$SMOKE_INDEX" =~ ^[0-9]+$ && "$SMOKE_INDEX" -lt 1000 ]] \
  || die "SMOKE_INDEX must be an integer from 0 to 999"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "RUN_NAME may contain only letters, numbers, dots, underscores and hyphens"
[[ "$OUTPUT_ROOT" == /* ]] \
  || die "FAAS_OUTPUT_ROOT must be an absolute path"
[[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]] \
  || die "missing 1024 config: isolated_vlm_attack/$CONFIG_PATH"
[[ -f "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh" ]] \
  || die "missing FaaS bootstrap: run_bernini_vlm_res_remote.sh"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export CONFIG_PATH
export OUTPUT_ROOT
export HEIGHT=1024
export WIDTH=1024
export BERNINI_MAX_IMAGE_SIZE=1024
export START_INDEX="$SMOKE_INDEX"
export MAX_SAMPLES=1
export ATTACK_ONLY_CLEAN_CORRECT=0
export CLEAN_CORRECT_SAMPLE_SIZE=0
unset END_INDEX SAMPLE_INDICES SAMPLE_INDICES_FILE CLEAN_CORRECT_SAMPLE_SEED
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

echo "[bernini_1024_smoke] project_root=$PROJECT_ROOT"
echo "[bernini_1024_smoke] config=$CONFIG_PATH"
echo "[bernini_1024_smoke] output_dir=$OUTPUT_ROOT/nips2017/resnet50/$RUN_NAME"
echo "[bernini_1024_smoke] sample_index=$SMOKE_INDEX"
echo "[bernini_1024_smoke] generation_size=${WIDTH}x${HEIGHT}"
echo "[bernini_1024_smoke] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --output_root "$OUTPUT_ROOT" \
  --start_index "$SMOKE_INDEX" \
  --max_samples 1 \
  --attack_only_clean_correct false \
  --clean_correct_sample_size 0 \
  --height 1024 \
  --width 1024 \
  --bernini_max_image_size 1024 \
  --run_name "$RUN_NAME" \
  "$@"

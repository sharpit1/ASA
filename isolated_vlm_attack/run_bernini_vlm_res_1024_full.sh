#!/usr/bin/env bash
set -Eeuo pipefail

# Run the full NIPS2017 range through the Bernini 1024x1024 FaaS bootstrap.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="configs/bernini_vlm_attack_nips_1024.yaml"
FULL_SAMPLE_COUNT="${FULL_SAMPLE_COUNT:-1000}"
RUN_NAME="${RUN_NAME:-bernini_vlm_res_1024_full_$(date +%Y%m%d_%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$FULL_SAMPLE_COUNT" =~ ^[0-9]+$ \
    && "$FULL_SAMPLE_COUNT" -ge 1 \
    && "$FULL_SAMPLE_COUNT" -le 1000 ]] \
  || die "FULL_SAMPLE_COUNT must be an integer from 1 to 1000"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "RUN_NAME may contain only letters, numbers, dots, underscores and hyphens"
[[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]] \
  || die "missing 1024 config: isolated_vlm_attack/$CONFIG_PATH"
[[ -f "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh" ]] \
  || die "missing FaaS bootstrap: run_bernini_vlm_res_remote.sh"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export CONFIG_PATH
export HEIGHT=1024
export WIDTH=1024
export BERNINI_MAX_IMAGE_SIZE=1024
export START_INDEX=0
export MAX_SAMPLES="$FULL_SAMPLE_COUNT"
export ATTACK_ONLY_CLEAN_CORRECT=0
export CLEAN_CORRECT_SAMPLE_SIZE=0
unset END_INDEX SAMPLE_INDICES SAMPLE_INDICES_FILE CLEAN_CORRECT_SAMPLE_SEED
export LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/$RUN_NAME.log}"

echo "[bernini_1024_full] project_root=$PROJECT_ROOT"
echo "[bernini_1024_full] config=$CONFIG_PATH"
echo "[bernini_1024_full] sample_range=0-$((FULL_SAMPLE_COUNT - 1))"
echo "[bernini_1024_full] generation_size=${WIDTH}x${HEIGHT}"
echo "[bernini_1024_full] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --start_index 0 \
  --max_samples "$FULL_SAMPLE_COUNT" \
  --attack_only_clean_correct false \
  --clean_correct_sample_size 0 \
  --height 1024 \
  --width 1024 \
  --bernini_max_image_size 1024 \
  --run_name "$RUN_NAME" \
  "$@"

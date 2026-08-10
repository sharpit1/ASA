#!/usr/bin/env bash
set -Eeuo pipefail

# Attack every clean-correct sample found in the NIPS2017 candidate range.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="configs/bernini_vlm_attack_nips_1024.yaml"
OUTPUT_ROOT="${FAAS_OUTPUT_ROOT:-/app/output/sharpit1}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-1000}"
RUN_NAME="${RUN_NAME:-bernini_vlm_res_1024_clean_correct_full_$(date +%Y%m%d_%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$CANDIDATE_COUNT" =~ ^[0-9]+$ \
    && "$CANDIDATE_COUNT" -ge 1 \
    && "$CANDIDATE_COUNT" -le 1000 ]] \
  || die "CANDIDATE_COUNT must be an integer from 1 to 1000"
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
export START_INDEX=0
export MAX_SAMPLES="$CANDIDATE_COUNT"
export ATTACK_ONLY_CLEAN_CORRECT=1
export CLEAN_CORRECT_SAMPLE_SIZE=0
unset END_INDEX SAMPLE_INDICES SAMPLE_INDICES_FILE CLEAN_CORRECT_SAMPLE_SEED
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

echo "[bernini_1024_full] project_root=$PROJECT_ROOT"
echo "[bernini_1024_full] config=$CONFIG_PATH"
echo "[bernini_1024_full] output_dir=$OUTPUT_ROOT/nips2017/resnet50/$RUN_NAME"
echo "[bernini_1024_full] candidate_range=0-$((CANDIDATE_COUNT - 1))"
echo "[bernini_1024_full] selection=all_clean_correct"
echo "[bernini_1024_full] generation_size=${WIDTH}x${HEIGHT}"
echo "[bernini_1024_full] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_bernini_vlm_res_remote.sh" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --output_root "$OUTPUT_ROOT" \
  --start_index 0 \
  --max_samples "$CANDIDATE_COUNT" \
  --attack_only_clean_correct true \
  --clean_correct_sample_size 0 \
  --height 1024 \
  --width 1024 \
  --bernini_max_image_size 1024 \
  --run_name "$RUN_NAME" \
  "$@"

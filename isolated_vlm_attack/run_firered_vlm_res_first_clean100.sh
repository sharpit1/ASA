#!/usr/bin/env bash
set -Eeuo pipefail

# Attack the first 100 clean-correct NIPS2017 samples in dataset order with
# FireRed VLM mode and no strategy slots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="configs/firered_vlm_attack_nips.yaml"
OUTPUT_ROOT="${FAAS_OUTPUT_ROOT:-/app/output/sharpit1}"
CLEAN_COUNT="${CLEAN_CORRECT_COUNT:-100}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-1000}"
GENERATION_SEED="${MANUAL_SEED:-49}"
RUN_NAME="${RUN_NAME:-firered_vlm_res_first_clean${CLEAN_COUNT}_$(date +%Y%m%d_%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$CLEAN_COUNT" =~ ^[0-9]+$ && "$CLEAN_COUNT" -ge 1 ]] \
  || die "CLEAN_CORRECT_COUNT must be a positive integer"
[[ "$CANDIDATE_COUNT" =~ ^[0-9]+$ \
    && "$CANDIDATE_COUNT" -ge "$CLEAN_COUNT" \
    && "$CANDIDATE_COUNT" -le 1000 ]] \
  || die "CANDIDATE_COUNT must be between $CLEAN_COUNT and 1000"
[[ "$GENERATION_SEED" =~ ^-?[0-9]+$ ]] \
  || die "MANUAL_SEED must be an integer"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "RUN_NAME may contain only letters, numbers, dots, underscores and hyphens"
[[ "$OUTPUT_ROOT" == /* ]] \
  || die "FAAS_OUTPUT_ROOT must be an absolute path"
[[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]] \
  || die "missing FireRed config: isolated_vlm_attack/$CONFIG_PATH"
[[ -f "$SCRIPT_DIR/run_firered_vlm_res_remote.sh" ]] \
  || die "missing FaaS bootstrap: run_firered_vlm_res_remote.sh"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export CONFIG_PATH
unset SAMPLE_INDICES SAMPLE_INDICES_FILE END_INDEX CLEAN_CORRECT_SAMPLE_SEED
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

echo "[firered_first_clean100] project_root=$PROJECT_ROOT"
echo "[firered_first_clean100] config=$CONFIG_PATH"
echo "[firered_first_clean100] output_dir=$OUTPUT_ROOT/nips2017/resnet50/$RUN_NAME"
echo "[firered_first_clean100] candidate_range=0-$((CANDIDATE_COUNT - 1))"
echo "[firered_first_clean100] selection=first_clean_correct_in_dataset_order"
echo "[firered_first_clean100] clean_correct_skip=0"
echo "[firered_first_clean100] clean_correct_count=$CLEAN_COUNT"
echo "[firered_first_clean100] manual_seed=$GENERATION_SEED"
echo "[firered_first_clean100] attack_mode=vlm strategies=none"
echo "[firered_first_clean100] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_firered_vlm_res_remote.sh" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --output_root "$OUTPUT_ROOT" \
  --start_index 0 \
  --max_samples "$CANDIDATE_COUNT" \
  --attack_only_clean_correct true \
  --clean_correct_sample_size 0 \
  --clean_correct_skip 0 \
  --clean_correct_count "$CLEAN_COUNT" \
  --manual_seed "$GENERATION_SEED" \
  --gcg_scene_vocab_enabled_strategies none \
  --class_ablation false \
  --gcg_eval_naturalness_on_attack_success true \
  --gcg_eval_naturalness_llm_thinking false \
  --firered_batch_size 1 \
  --run_name "$RUN_NAME" \
  "$@"

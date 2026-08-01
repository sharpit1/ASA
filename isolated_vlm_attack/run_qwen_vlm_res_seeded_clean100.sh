#!/usr/bin/env bash
set -euo pipefail

# Run one reproducible Qwen Image Edit attack cohort. Examples:
#   SEED=0 bash isolated_vlm_attack/run_qwen_vlm_res_seeded_clean100.sh
#   SEED=1 SKIP_PREPARE=1 bash isolated_vlm_attack/run_qwen_vlm_res_seeded_clean100.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SAMPLE_SEED="${CLEAN_CORRECT_SAMPLE_SEED:-${SEED:-0}}"
SAMPLE_SIZE="${CLEAN_CORRECT_SAMPLE_SIZE:-100}"
GENERATION_SEED="${MANUAL_SEED:-$SAMPLE_SEED}"
CANDIDATE_COUNT="${MAX_SAMPLES:-1000}"
RUN_NAME="${RUN_NAME:-qwen_vlm_res_seeded_clean${SAMPLE_SIZE}_seed${SAMPLE_SEED}_$(date +%Y%m%d_%H%M%S)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$SAMPLE_SEED" =~ ^-?[0-9]+$ ]] \
  || die "SEED/CLEAN_CORRECT_SAMPLE_SEED must be an integer"
[[ "$GENERATION_SEED" =~ ^-?[0-9]+$ ]] \
  || die "MANUAL_SEED must be an integer"
[[ "$SAMPLE_SIZE" =~ ^[0-9]+$ && "$SAMPLE_SIZE" -ge 1 ]] \
  || die "CLEAN_CORRECT_SAMPLE_SIZE must be a positive integer"
[[ "$CANDIDATE_COUNT" =~ ^[0-9]+$ && "$CANDIDATE_COUNT" -ge "$SAMPLE_SIZE" ]] \
  || die "MAX_SAMPLES must be an integer greater than or equal to $SAMPLE_SIZE"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"

# Seed sampling must use the full candidate range, not the fixed 100/200 files
# used by the older ordered-window launchers.
export SAMPLE_INDICES=""
export SAMPLE_INDICES_FILE=""
export START_INDEX=0
export END_INDEX=""
export CLEAN_CORRECT_SKIP=0
export CLEAN_CORRECT_COUNT=0

echo "[qwen_seeded_clean100] project_root=$PROJECT_ROOT"
echo "[qwen_seeded_clean100] candidate_range=0-$((CANDIDATE_COUNT - 1))"
echo "[qwen_seeded_clean100] clean_correct_sample_size=$SAMPLE_SIZE"
echo "[qwen_seeded_clean100] clean_correct_sample_seed=$SAMPLE_SEED"
echo "[qwen_seeded_clean100] manual_seed=$GENERATION_SEED"
echo "[qwen_seeded_clean100] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh" \
  --clean_correct_sample_size "$SAMPLE_SIZE" \
  --clean_correct_sample_seed "$SAMPLE_SEED" \
  --manual_seed "$GENERATION_SEED" \
  --max_samples "$CANDIDATE_COUNT" \
  --run_name "$RUN_NAME" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

# Run one reproducible Bernini attack cohort. Examples:
#   SEED=0 bash isolated_vlm_attack/run_bernini_vlm_res_seeded_clean100.sh
#   SEED=1 bash isolated_vlm_attack/run_bernini_vlm_res_seeded_clean100.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SAMPLE_SEED="${CLEAN_CORRECT_SAMPLE_SEED:-${SEED:-0}}"
SAMPLE_SIZE="${CLEAN_CORRECT_SAMPLE_SIZE:-100}"
GENERATION_SEED="${MANUAL_SEED:-$SAMPLE_SEED}"
CANDIDATE_COUNT="${MAX_SAMPLES:-1000}"
RUN_NAME="${RUN_NAME:-bernini_vlm_res_seeded_clean${SAMPLE_SIZE}_seed${SAMPLE_SEED}_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-${CONFIG:-configs/bernini_vlm_attack_nips.yaml}}"

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

if [[ -z "${PYTHON_BIN:-}" && -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  export PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi
export ASA_PROJECT_ROOT="$PROJECT_ROOT"

# Seed sampling must use the full candidate range, not a fixed indices file.
export SAMPLE_INDICES=""
export SAMPLE_INDICES_FILE=""
export START_INDEX=0
export END_INDEX=""

echo "[bernini_seeded_clean100] project_root=$PROJECT_ROOT"
echo "[bernini_seeded_clean100] candidate_range=0-$((CANDIDATE_COUNT - 1))"
echo "[bernini_seeded_clean100] clean_correct_sample_size=$SAMPLE_SIZE"
echo "[bernini_seeded_clean100] clean_correct_sample_seed=$SAMPLE_SEED"
echo "[bernini_seeded_clean100] manual_seed=$GENERATION_SEED"
echo "[bernini_seeded_clean100] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_vlm_attack.sh" \
  --config "$CONFIG_PATH" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --attack_only_clean_correct true \
  --gcg_scene_vocab_enabled_strategies none \
  --gcg_scene_vocab_prompts_per_strategy 0 \
  --gcg_eval_naturalness_on_attack_success true \
  --gcg_eval_naturalness_llm_thinking false \
  --clean_correct_sample_size "$SAMPLE_SIZE" \
  --clean_correct_sample_seed "$SAMPLE_SEED" \
  --manual_seed "$GENERATION_SEED" \
  --max_samples "$CANDIDATE_COUNT" \
  --run_name "$RUN_NAME" \
  "$@"

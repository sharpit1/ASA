#!/usr/bin/env bash
set -Eeuo pipefail

# Run one FireRed VLM/none sample through the FaaS bootstrap.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="configs/firered_vlm_attack_nips.yaml"
OUTPUT_ROOT="${FAAS_OUTPUT_ROOT:-/app/output/sharpit1}"
SMOKE_INDEX="${SMOKE_INDEX:-0}"
RUN_NAME="${RUN_NAME:-firered_vlm_res_smoke_$(date +%Y%m%d_%H%M%S)}"

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
  || die "missing FireRed config: isolated_vlm_attack/$CONFIG_PATH"
[[ -f "$SCRIPT_DIR/run_firered_vlm_res_remote.sh" ]] \
  || die "missing FaaS bootstrap: run_firered_vlm_res_remote.sh"

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export CONFIG_PATH
unset SAMPLE_INDICES SAMPLE_INDICES_FILE END_INDEX CLEAN_CORRECT_SAMPLE_SEED
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
export LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

echo "[firered_smoke] project_root=$PROJECT_ROOT"
echo "[firered_smoke] config=$CONFIG_PATH"
echo "[firered_smoke] output_dir=$OUTPUT_ROOT/nips2017/resnet50/$RUN_NAME"
echo "[firered_smoke] sample_index=$SMOKE_INDEX"
echo "[firered_smoke] attack_mode=vlm strategies=none"
echo "[firered_smoke] run_name=$RUN_NAME"

exec bash "$SCRIPT_DIR/run_firered_vlm_res_remote.sh" \
  --attack_mode vlm \
  --victim_model resnet50 \
  --output_root "$OUTPUT_ROOT" \
  --start_index "$SMOKE_INDEX" \
  --max_samples 1 \
  --attack_only_clean_correct false \
  --clean_correct_sample_size 0 \
  --gcg_scene_vocab_enabled_strategies none \
  --class_ablation false \
  --gcg_eval_naturalness_on_attack_success true \
  --gcg_eval_naturalness_llm_thinking false \
  --firered_batch_size 1 \
  --run_name "$RUN_NAME" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_RUN_DIR="${QWEN_SOURCE_RUN_DIR:-$PROJECT_ROOT/outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000}"
INDICES_FILE="${SAMPLE_INDICES_FILE:-$SOURCE_RUN_DIR/target_200_indices.json}"

if [[ ! -f "$INDICES_FILE" ]]; then
  echo "ERROR: source indices file not found: $INDICES_FILE" >&2
  exit 1
fi

export ASA_PROJECT_ROOT="$PROJECT_ROOT"

echo "[qwen_second_clean100] project_root=$PROJECT_ROOT"
echo "[qwen_second_clean100] indices_file=$INDICES_FILE"
echo "[qwen_second_clean100] clean_correct_window=101-200"

exec bash "$SCRIPT_DIR/run_vlm_attack.sh" \
  --config configs/qwen_edit_vlm_res.yaml \
  --attack_mode vlm \
  --victim_model resnet50 \
  --attack_only_clean_correct true \
  --clean_correct_skip 100 \
  --clean_correct_count 100 \
  --gcg_scene_vocab_enabled_strategies none \
  --class_ablation false \
  --gcg_eval_naturalness_on_attack_success true \
  --gcg_eval_naturalness_llm_thinking false \
  --max_samples 1000 \
  --qwen_batch_size 1 \
  --sample_indices_file "$INDICES_FILE" \
  "$@"

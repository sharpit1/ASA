#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/qlab/ds/ASA
ISOLATED_ROOT="$PROJECT_ROOT/isolated_vlm_attack"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
HF_CACHE_ROOT="$PROJECT_ROOT/.cache/huggingface"
QUEUE_ID=pro6000_resnet_qwen_takeover_clean200_20260727_221500
QUEUE_DIR="$PROJECT_ROOT/.aris/queues/$QUEUE_ID"
QWEN_ROOT="$PROJECT_ROOT/outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000"
QWEN_INDICES="$QWEN_ROOT/target_200_indices.json"
QWEN_CONFIG=configs/qwen_edit_vlm_res.yaml

mkdir -p "$QUEUE_DIR"

state() {
  printf '{"queue_id":"%s","state":"%s","updated_at":"%s"}\n' \
    "$QUEUE_ID" "$1" "$(date -Is)" > "$QUEUE_DIR/state.json"
}

echo "$$" > "$QUEUE_DIR/queue.pid"
date -Is > "$QUEUE_DIR/queue.started_at"
state RUNNING_QWEN

printf '%s\n' \
  "PYTHON_BIN=$PYTHON_BIN ASA_PROJECT_ROOT=$PROJECT_ROOT HF_CACHE_ROOT=$HF_CACHE_ROOT bash run_vlm_attack.sh --config $QWEN_CONFIG --attack_mode vlm --victim_model resnet50 --attack_only_clean_correct true --gcg_scene_vocab_enabled_strategies none --class_ablation false --gcg_eval_naturalness_on_attack_success true --gcg_eval_naturalness_llm_thinking false --max_samples 1000 --sample_indices_file $QWEN_INDICES" \
  > "$QUEUE_DIR/qwen.command.txt"

(
  echo "$BASHPID" > "$QUEUE_DIR/qwen.pid"
  date -Is > "$QUEUE_DIR/qwen.started_at"
  cd "$ISOLATED_ROOT"
  PYTHON_BIN="$PYTHON_BIN" \
  ASA_PROJECT_ROOT="$PROJECT_ROOT" \
  HF_CACHE_ROOT="$HF_CACHE_ROOT" \
  HF_HOME="$HF_CACHE_ROOT" \
  HF_HUB_CACHE="$HF_CACHE_ROOT/hub" \
  TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers" \
  bash run_vlm_attack.sh \
    --config "$QWEN_CONFIG" \
    --attack_mode vlm \
    --victim_model resnet50 \
    --attack_only_clean_correct true \
    --gcg_scene_vocab_enabled_strategies none \
    --class_ablation false \
    --gcg_eval_naturalness_on_attack_success true \
    --gcg_eval_naturalness_llm_thinking false \
    --max_samples 1000 \
    --sample_indices_file "$QWEN_INDICES"
) 2>&1 | tee -a "$QUEUE_DIR/qwen.log"

rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$QUEUE_DIR/qwen.exit_code"
date -Is > "$QUEUE_DIR/qwen.finished_at"

if [[ "$rc" -eq 0 ]]; then
  state COMPLETED
else
  state FAILED_QWEN
fi

date -Is > "$QUEUE_DIR/queue.finished_at"
exit "$rc"

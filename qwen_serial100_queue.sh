#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/qlab/ds/ASA
ISOLATED_ROOT="$PROJECT_ROOT/isolated_vlm_attack"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
HF_CACHE_ROOT="$PROJECT_ROOT/.cache/huggingface"
QUEUE_ID=pro6000_resnet_qwen_serial_clean100_20260727_233554
QUEUE_DIR="$PROJECT_ROOT/.aris/queues/$QUEUE_ID"
QWEN_ROOT="$PROJECT_ROOT/outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000"
SOURCE_INDICES="$QWEN_ROOT/target_200_indices.json"
TARGET_INDICES="$QWEN_ROOT/target_100_indices_serial_20260727_233554.json"
REMAINING_INDICES="$QWEN_ROOT/remaining_target100_serial_20260727_233554.json"
MANIFEST="$QWEN_ROOT/serial100_manifest_20260727_233554.json"
QWEN_CONFIG=configs/qwen_edit_vlm_res.yaml

mkdir -p "$QUEUE_DIR"

state() {
  printf '{"queue_id":"%s","state":"%s","updated_at":"%s"}\n' \
    "$QUEUE_ID" "$1" "$(date -Is)" > "$QUEUE_DIR/state.json"
}

prepare_indices() {
  "$PYTHON_BIN" - "$SOURCE_INDICES" "$TARGET_INDICES" "$REMAINING_INDICES" "$MANIFEST" "$QWEN_ROOT" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source_path, target_path, remaining_path, manifest_path, run_root = map(Path, sys.argv[1:])
payload = json.loads(source_path.read_text(encoding="utf-8"))
if isinstance(payload, dict):
    payload = payload.get("indices", payload.get("sample_indices"))
if not isinstance(payload, list):
    raise TypeError("target_200_indices.json must contain a list of indices")

source = [int(item) for item in payload]
if len(source) < 100:
    raise ValueError(f"expected at least 100 source indices, found {len(source)}")
if len(source) != len(set(source)):
    raise ValueError("source indices contain duplicates")

target = source[:100]
completed = [
    idx for idx in target
    if (run_root / f"sample_{idx:04d}" / "report.json").is_file()
]
completed_set = set(completed)
remaining = [idx for idx in target if idx not in completed_set]

def write_indices(path: Path, values: list[int]) -> str:
    body = json.dumps(values, separators=(",", ":")) + "\n"
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

target_sha256 = write_indices(target_path, target)
remaining_sha256 = write_indices(remaining_path, remaining)
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_indices_file": str(source_path),
    "run_root": str(run_root),
    "selection_rule": "first 100 target_200 indices, excluding indices with an existing report.json",
    "target_count": len(target),
    "completed_count": len(completed),
    "remaining_count": len(remaining),
    "target_indices": target,
    "completed_indices": completed,
    "remaining_indices": remaining,
    "target_indices_file": str(target_path),
    "remaining_indices_file": str(remaining_path),
    "target_sha256": target_sha256,
    "remaining_sha256": remaining_sha256,
}
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "target_count": len(target),
    "completed_count": len(completed),
    "remaining_count": len(remaining),
    "target_sha256": target_sha256,
    "remaining_sha256": remaining_sha256,
}, separators=(",", ":")))
PY
}

if [[ "${1:-}" == "--prepare" ]]; then
  prepare_indices
  exit 0
fi

prepare_indices > "$QUEUE_DIR/indices.prepare.json"

remaining_count="$(
  "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$REMAINING_INDICES"
)"
if [[ "$remaining_count" -eq 0 ]]; then
  state COMPLETED_NOOP
  date -Is > "$QUEUE_DIR/queue.finished_at"
  exit 0
fi

echo "$$" > "$QUEUE_DIR/queue.pid"
date -Is > "$QUEUE_DIR/queue.started_at"
state RUNNING_QWEN_SERIAL100

printf '%s\n' \
  "PYTHON_BIN=$PYTHON_BIN ASA_PROJECT_ROOT=$PROJECT_ROOT HF_CACHE_ROOT=$HF_CACHE_ROOT bash run_vlm_attack.sh --config $QWEN_CONFIG --attack_mode vlm --victim_model resnet50 --attack_only_clean_correct true --gcg_scene_vocab_enabled_strategies none --class_ablation false --gcg_eval_naturalness_on_attack_success true --gcg_eval_naturalness_llm_thinking false --max_samples 1000 --qwen_batch_size 1 --sample_indices_file $REMAINING_INDICES" \
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
    --qwen_batch_size 1 \
    --sample_indices_file "$REMAINING_INDICES"
) 2>&1 | tee -a "$QUEUE_DIR/qwen.log"

rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$QUEUE_DIR/qwen.exit_code"
date -Is > "$QUEUE_DIR/qwen.finished_at"

if [[ "$rc" -eq 0 ]]; then
  state COMPLETED
else
  state FAILED_QWEN_SERIAL100
fi

date -Is > "$QUEUE_DIR/queue.finished_at"
exit "$rc"

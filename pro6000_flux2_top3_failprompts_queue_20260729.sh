#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/qlab/ds/ASA
PYTHON_BIN="$ROOT/.venv/bin/python"
QUEUE_ID=pro6000_flux2_top3_failprompts_imagenet1k_10perclass_20260729
QUEUE_ROOT="$ROOT/.aris/queues/$QUEUE_ID"
OUTPUT_ROOT="$ROOT/outputs/imagenet_val/flux2_top3_failprompts_imagenet1k_10perclass_20260729"
MIN_FREE_VRAM_MIB=80000
MIN_FREE_DISK_GIB=45
WAIT_SECONDS=60
OOM_RETRY_DELAY_SECONDS=120
OOM_MAX_ATTEMPTS=3

PROMPTS=(
  "coated in fine grit"
  "introduce light humidity"
  "add light humidity"
)
SLUGS=(
  "coated_in_fine_grit"
  "introduce_light_humidity"
  "add_light_humidity"
)

EXPECTED_EDIT_SHA256=245fef6cb6b515126f773a9e41e4cc8d8f8136cf8aa578d91430b4a5c7797aac
EXPECTED_ATTACKED_MODELS_SHA256=511356782abe0972d4d1dead7237b86d93a6b0b7e1032fe53f12e794bbdd1172
EXPECTED_HELPER_SHA256=aa850f556276dd9e129d565dd34a113b6a1ffe1f484877fd82dd6ddb81eda38b

mkdir -p "$QUEUE_ROOT/status" "$QUEUE_ROOT/logs"
exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  printf '%s queue already active: %s\n' "$(date -Iseconds)" "$QUEUE_ID"
  exit 75
fi
exec >>"$QUEUE_ROOT/queue.log" 2>&1

timestamp() {
  date -Iseconds
}

record() {
  printf '%s %s\n' "$(timestamp)" "$*"
}

set_state() {
  local state_name="$1"
  local detail="${2:-}"
  "$PYTHON_BIN" - "$QUEUE_ROOT/state.json" "$QUEUE_ID" "$state_name" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "queue_id": sys.argv[2],
    "state": sys.argv[3],
    "detail": sys.argv[4],
    "updated_at": datetime.now().astimezone().isoformat(),
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, path)
PY
}

VALIDATION_MONITOR_PID=""
finish() {
  local code=$?
  if [[ -n "$VALIDATION_MONITOR_PID" ]] && kill -0 "$VALIDATION_MONITOR_PID" 2>/dev/null; then
    kill "$VALIDATION_MONITOR_PID" 2>/dev/null || true
    wait "$VALIDATION_MONITOR_PID" 2>/dev/null || true
  fi
  printf '%s\n' "$code" >"$QUEUE_ROOT/status/watcher.exit"
  timestamp >"$QUEUE_ROOT/status/watcher.finished_at"
  if (( code != 0 )); then
    set_state FAILED "watcher exit=$code"
  fi
  record "watcher finished exit=$code"
}
trap finish EXIT

verify_source_hashes() {
  local edit_hash
  local attacked_hash
  local helper_hash
  edit_hash="$(sha256sum "$ROOT/eval/edit_imagenet_val_flux2.py" | awk '{print $1}')"
  attacked_hash="$(sha256sum "$ROOT/eval/attacked_models.py" | awk '{print $1}')"
  helper_hash="$(sha256sum "$ROOT/eval/eval_prompt_transfer_in_image.py" | awk '{print $1}')"
  record "source hashes edit=$edit_hash attacked_models=$attacked_hash helper=$helper_hash"
  [[ "$edit_hash" == "$EXPECTED_EDIT_SHA256" ]]
  [[ "$attacked_hash" == "$EXPECTED_ATTACKED_MODELS_SHA256" ]]
  [[ "$helper_hash" == "$EXPECTED_HELPER_SHA256" ]]
  "$PYTHON_BIN" - "$QUEUE_ROOT/source_hashes.json" "$edit_hash" "$attacked_hash" "$helper_hash" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "observed_at": datetime.now().astimezone().isoformat(),
    "files": {
        "eval/edit_imagenet_val_flux2.py": sys.argv[2],
        "eval/attacked_models.py": sys.argv[3],
        "eval/eval_prompt_transfer_in_image.py": sys.argv[4],
    },
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(tmp, path)
PY
}

check_disk() {
  local available_kib
  local required_kib
  available_kib="$(df --output=avail "$ROOT" | awk 'NR==2 {print $1}')"
  required_kib=$((MIN_FREE_DISK_GIB * 1024 * 1024))
  record "disk available_kib=$available_kib required_kib=$required_kib"
  (( available_kib >= required_kib ))
}

wait_for_gpu() {
  local gpu_line
  local free_mib
  local total_mib
  set_state WAITING_FOR_GPU "require free_vram_mib >= $MIN_FREE_VRAM_MIB"
  while true; do
    if gpu_line="$("$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
x = torch.ones((8, 8), device="cuda")
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"{free_bytes // 1048576} {total_bytes // 1048576} {float(x.sum())}")
PY
    )"; then
      read -r free_mib total_mib witness_sum <<<"$gpu_line"
      record "GPU witness free=${free_mib}MiB total=${total_mib}MiB sum=${witness_sum}"
      if (( free_mib >= MIN_FREE_VRAM_MIB )); then
        return
      fi
    else
      record "CUDA readiness probe failed; retrying"
    fi
    sleep "$WAIT_SECONDS"
  done
}

validate_prompt() {
  local slug="$1"
  local prompt="$2"
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$QUEUE_ROOT" "$slug" "$prompt" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
queue_root = Path(sys.argv[2])
slug = sys.argv[3]
prompt = sys.argv[4]
expected_prompts = [
    "coated in fine grit",
    "introduce light humidity",
    "add light humidity",
]
expected_slugs = [
    "coated_in_fine_grit",
    "introduce_light_humidity",
    "add_light_humidity",
]

manifest_path = root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["prompts"] == expected_prompts, manifest["prompts"]
assert manifest["prompt_slugs"] == expected_slugs, manifest["prompt_slugs"]
assert manifest["samples_per_class"] == 10
assert manifest["max_classes"] == 0
assert manifest["source_image_count"] == 10000
assert manifest["edited_image_count_expected"] == 30000
assert manifest["save_clean_batch"] == "off"
assert manifest["transfer_models"] == ["resnet50"]
assert manifest["seed"] == 123
assert manifest["render_size"] == 1024
assert manifest["eval_size"] == 224
assert manifest["num_inference_steps"] == 4

prompt_root = root / slug
summary = json.loads((prompt_root / "prompt_summary.json").read_text(encoding="utf-8"))
assert summary["prompt"] == prompt
assert summary["transfer"] == {}
cache_files = sorted((prompt_root / "_render_cache").glob("label_*.npz"))
assert len(cache_files) == 1000, len(cache_files)
generation_failures = 0
edited_image_count = 0
for expected_label, cache_path in enumerate(cache_files):
    assert cache_path.name == f"label_{expected_label:04d}.npz", cache_path
    with np.load(cache_path, allow_pickle=True) as payload:
        assert "clean_batch" not in payload.files, cache_path
        assert "clean_batch_saved" in payload.files, cache_path
        assert not bool(np.asarray(payload["clean_batch_saved"]).reshape(-1)[0]), cache_path
        assert "generation_failures" in payload.files, cache_path
        failures = int(np.asarray(payload["generation_failures"]).reshape(-1)[0])
        assert failures == 0, (cache_path, failures)
        generation_failures += failures
        assert "adv_batch" in payload.files, cache_path
        adv_batch = np.asarray(payload["adv_batch"])
        assert adv_batch.shape[0] == 10, (cache_path, adv_batch.shape)
        edited_image_count += int(adv_batch.shape[0])
        assert "image_paths" in payload.files, cache_path
        assert len(np.asarray(payload["image_paths"]).reshape(-1)) == 10, cache_path
        assert "ground_label" in payload.files, cache_path
        assert int(np.asarray(payload["ground_label"]).reshape(-1)[0]) == expected_label

clean_paths = list(prompt_root.rglob("*clean_batch*"))
assert not clean_paths, clean_paths[:10]
marker = {
    "prompt": prompt,
    "slug": slug,
    "validated_at": datetime.now().astimezone().isoformat(),
    "cache_file_count": len(cache_files),
    "edited_image_count": edited_image_count,
    "generation_failures": generation_failures,
    "clean_batch_file_count": 0,
    "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
}
path = queue_root / "status" / f"{slug}.validated.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, path)
print("PROMPT_VALID", json.dumps(marker, ensure_ascii=False))
PY
}

monitor_prompt_completion() {
  local i
  for i in "${!SLUGS[@]}"; do
    local slug="${SLUGS[$i]}"
    local prompt="${PROMPTS[$i]}"
    local summary_path="$OUTPUT_ROOT/$slug/prompt_summary.json"
    local marker_path="$QUEUE_ROOT/status/$slug.validated.json"
    if [[ -f "$marker_path" ]]; then
      record "validation marker already exists slug=$slug"
      continue
    fi
    while [[ ! -f "$summary_path" ]]; do
      sleep 30
    done
    record "validating completed prompt slug=$slug"
    validate_prompt "$slug" "$prompt"
  done
}

record "queue watcher started"
timestamp >"$QUEUE_ROOT/status/watcher.started_at"
verify_source_hashes
"$PYTHON_BIN" "$ROOT/eval/edit_imagenet_val_flux2.py" --help >/dev/null
check_disk
wait_for_gpu
check_disk

monitor_prompt_completion &
VALIDATION_MONITOR_PID=$!

attempt=1
while (( attempt <= OOM_MAX_ATTEMPTS )); do
  set_state RUNNING "render attempt=$attempt/$OOM_MAX_ATTEMPTS"
  timestamp >"$QUEUE_ROOT/status/run.attempt_${attempt}.started_at"
  attempt_log="$QUEUE_ROOT/logs/render.attempt_${attempt}.log"
  record "starting render attempt=$attempt output=$OUTPUT_ROOT"
  set +e
  ASA_PROJECT_ROOT="$ROOT" \
  HF_CACHE_ROOT="$ROOT/.cache/huggingface" \
  HF_HOME="$ROOT/.cache/huggingface" \
  HF_HUB_CACHE="$ROOT/.cache/huggingface/hub" \
  TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    "$PYTHON_BIN" "$ROOT/eval/edit_imagenet_val_flux2.py" \
      --imagenet_val_dir "$ROOT/data/ILSVRC2012_img_val" \
      --output_dir "$OUTPUT_ROOT" \
      --prompt "${PROMPTS[0]}" \
      --prompt "${PROMPTS[1]}" \
      --prompt "${PROMPTS[2]}" \
      --models resnet50 \
      --samples_per_class 10 \
      --max_classes 0 \
      --start_iteration 0 \
      --render_size 1024 \
      --eval_size 224 \
      --num_inference_steps 4 \
      --render_batch_size 4 \
      --batch_size 64 \
      --seed 123 \
      --save_clean_batch off \
      --skip_transfer_eval \
      --resume \
      2>&1 | tee -a "$attempt_log"
  run_exit="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$run_exit" >"$QUEUE_ROOT/status/run.attempt_${attempt}.exit"
  timestamp >"$QUEUE_ROOT/status/run.attempt_${attempt}.finished_at"
  if (( run_exit == 0 )); then
    break
  fi
  if ! grep -Eq 'torch\.OutOfMemoryError|CUDA out of memory' "$attempt_log"; then
    record "non-OOM render failure exit=$run_exit"
    exit "$run_exit"
  fi
  if (( attempt >= OOM_MAX_ATTEMPTS )); then
    record "OOM retry budget exhausted"
    exit "$run_exit"
  fi
  set_state RETRY_WAIT "OOM attempt=$attempt; retrying resumably"
  record "OOM detected; preserving caches and retrying after ${OOM_RETRY_DELAY_SECONDS}s"
  sleep "$OOM_RETRY_DELAY_SECONDS"
  wait_for_gpu
  check_disk
  attempt=$((attempt + 1))
done

set_state VALIDATING "waiting for per-prompt validation"
wait "$VALIDATION_MONITOR_PID"
VALIDATION_MONITOR_PID=""
for slug in "${SLUGS[@]}"; do
  test -f "$QUEUE_ROOT/status/$slug.validated.json"
done
test -f "$OUTPUT_ROOT/all_prompt_summary.json"
timestamp >"$QUEUE_ROOT/status/queue.completed_at"
set_state COMPLETED "3 prompts, 3000 caches, 30000 edited images"
record "all prompt caches completed and validated"

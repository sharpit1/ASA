#!/usr/bin/env bash
set -uo pipefail

ISOLATED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$ISOLATED_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
WAIT_PID="${WAIT_PID:-}"
WAIT_START_TICKS="${WAIT_START_TICKS:-}"
GPU_INDEX="${GPU_INDEX:-0}"
QUEUE_ID="${QUEUE_ID:-clean100_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-$PROJECT_ROOT/outputs/experiment_queues/$QUEUE_ID}"
STATE_FILE="$QUEUE_ROOT/state.tsv"
QUEUE_LOG="$QUEUE_ROOT/queue.log"

mkdir -p "$QUEUE_ROOT/logs"
exec 9>"$PROJECT_ROOT/outputs/experiment_queues/clean_correct_seed_sweep.lock"
if ! flock -n 9; then
  echo "ERROR: another clean-correct seed sweep is already active." >&2
  exit 1
fi

log() {
  local message="$1"
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$message" | tee -a "$QUEUE_LOG"
}

record_state() {
  local config_name="$1"
  local seed="$2"
  local status="$3"
  local run_name="$4"
  local detail="${5:-}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" \
    "$config_name" \
    "$seed" \
    "$status" \
    "$run_name" \
    "$detail" >> "$STATE_FILE"
}

same_wait_process_is_alive() {
  [[ -n "$WAIT_PID" && -r "/proc/$WAIT_PID/stat" ]] || return 1
  if [[ -z "$WAIT_START_TICKS" ]]; then
    return 0
  fi
  local current_start_ticks
  current_start_ticks="$(awk '{print $22}' "/proc/$WAIT_PID/stat" 2>/dev/null || true)"
  [[ "$current_start_ticks" == "$WAIT_START_TICKS" ]]
}

validate_summary() {
  local summary_path="$1"
  local expected_seed="$2"
  "$PYTHON_BIN" - "$summary_path" "$expected_seed" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"missing summary: {path}")
summary = json.loads(path.read_text(encoding="utf-8"))
checks = {
    "attack_only_clean_correct": summary.get("attack_only_clean_correct") is True,
    "clean_correct_sample_size": summary.get("clean_correct_sample_size") == 100,
    "clean_correct_sample_seed": summary.get("clean_correct_sample_seed") == seed,
    "clean_correct_sampled_count": summary.get("clean_correct_sampled_count") == 100,
    "selected_indices": len(summary.get("clean_correct_selected_indices", [])) == 100,
    "total_processed": summary.get("total_processed") == 100,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"summary validation failed ({','.join(failed)}): {path}")
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 1
fi

printf 'timestamp\tconfig\tseed\tstatus\trun_name\tdetail\n' > "$STATE_FILE"
log "queue_id=$QUEUE_ID gpu=$GPU_INDEX wait_pid=${WAIT_PID:-none}"

if same_wait_process_is_alive; then
  log "waiting for PID $WAIT_PID to finish"
  while same_wait_process_is_alive; do
    sleep 60
  done
  log "PID $WAIT_PID finished; waiting 30 seconds for GPU resources to settle"
  sleep 30
else
  log "wait target is not active; starting sweep immediately"
fi

configs=(
  "configs/flux2_and_attack_nips.yaml"
  "configs/flux2_and_attack_nips_swin.yaml"
  "configs/flux2_and_attack_nips_vim.yaml"
)
config_names=(
  "flux2_and_attack_nips"
  "flux2_and_attack_nips_swin"
  "flux2_and_attack_nips_vim"
)
victim_names=(
  "resnet50"
  "swin"
  "vim-small"
)

failures=0
for config_idx in "${!configs[@]}"; do
  config="${configs[$config_idx]}"
  config_name="${config_names[$config_idx]}"
  victim_name="${victim_names[$config_idx]}"
  config_path="$ISOLATED_DIR/$config"
  if [[ ! -f "$config_path" ]]; then
    log "missing config: $config_path"
    record_state "$config_name" "-" "blocked" "-" "missing_config"
    failures=$((failures + 1))
    continue
  fi

  for seed in 0 1 2; do
    run_name="${config_name}_clean100_seed${seed}_${QUEUE_ID}"
    job_log="$QUEUE_ROOT/logs/${config_name}_seed${seed}.log"
    summary_path="$PROJECT_ROOT/outputs/nips2017/$victim_name/$run_name/run_summary.json"
    record_state "$config_name" "$seed" "running" "$run_name"
    log "starting config=$config_name seed=$seed run_name=$run_name"

    if (
      cd "$PROJECT_ROOT"
      env \
        PYTHON_BIN="$PYTHON_BIN" \
        CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        ATTACK_ONLY_CLEAN_CORRECT=1 \
        CLEAN_CORRECT_SAMPLE_SIZE=100 \
        CLEAN_CORRECT_SAMPLE_SEED="$seed" \
        MANUAL_SEED="$seed" \
        MAX_SAMPLES=1000 \
        RUN_NAME="$run_name" \
        CONFIG="$config_path" \
        "$ISOLATED_DIR/run_vlm_attack.sh"
    ) >> "$job_log" 2>&1; then
      if validate_summary "$summary_path" "$seed" >> "$job_log" 2>&1; then
        record_state "$config_name" "$seed" "completed" "$run_name" "$summary_path"
        log "completed config=$config_name seed=$seed"
      else
        record_state "$config_name" "$seed" "failed_validation" "$run_name" "$summary_path"
        log "summary validation failed config=$config_name seed=$seed log=$job_log"
        failures=$((failures + 1))
      fi
    else
      exit_code=$?
      record_state "$config_name" "$seed" "failed" "$run_name" "exit_code=$exit_code"
      log "failed config=$config_name seed=$seed exit_code=$exit_code log=$job_log"
      failures=$((failures + 1))
    fi
  done
done

if [[ "$failures" -gt 0 ]]; then
  log "sweep finished with failures=$failures"
  exit 1
fi

log "sweep completed successfully jobs=9"

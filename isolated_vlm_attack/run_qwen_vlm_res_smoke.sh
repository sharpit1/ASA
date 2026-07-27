#!/usr/bin/env bash
set -Eeuo pipefail

# Run exactly one index through the remote Qwen VLM attack bootstrap.
# Results are isolated under a timestamped smoke directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_INDICES_FILE="${SOURCE_INDICES_FILE:-outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000/target_200_indices.json}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-qwen_vlm_res_smoke_$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUTPUT_DIR_REL="outputs/nips2017/resnet50/$SMOKE_RUN_ID"
SMOKE_INDICES_FILE="$SMOKE_OUTPUT_DIR_REL/smoke_index.json"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$SOURCE_INDICES_FILE" != /* ]] \
  || die "SOURCE_INDICES_FILE must be relative to the ASA project root"
[[ -f "$PROJECT_ROOT/$SOURCE_INDICES_FILE" ]] \
  || die "missing source indices file: $SOURCE_INDICES_FILE"
[[ "$SMOKE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "SMOKE_RUN_ID may contain only letters, numbers, dots, underscores and hyphens"
[[ -f "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh" ]] \
  || die "missing bootstrap runner: run_qwen_vlm_res_remote.sh"

if [[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]]; then
  JSON_PYTHON="$PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  JSON_PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  JSON_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  JSON_PYTHON="$(command -v python)"
else
  die "python3/python is required to prepare the one-index smoke selection"
fi

mkdir -p "$PROJECT_ROOT/$SMOKE_OUTPUT_DIR_REL"
SMOKE_INDEX="$(
  "$JSON_PYTHON" - \
    "$PROJECT_ROOT/$SOURCE_INDICES_FILE" \
    "$PROJECT_ROOT/$SMOKE_INDICES_FILE" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
payload = json.loads(source_path.read_text(encoding="utf-8"))
if isinstance(payload, dict):
    payload = payload.get("indices", payload.get("sample_indices"))
if not isinstance(payload, list) or not payload:
    raise SystemExit("source indices file must contain a non-empty list")

index = int(payload[0])
destination_path.write_text(
    json.dumps([index], separators=(",", ":")) + "\n",
    encoding="utf-8",
)
print(index)
PY
)"

[[ "$SMOKE_INDEX" =~ ^[0-9]+$ ]] \
  || die "selected smoke index is not a non-negative integer: $SMOKE_INDEX"

echo "[smoke] selected index: $SMOKE_INDEX"
echo "[smoke] output directory: $SMOKE_OUTPUT_DIR_REL"

SAMPLE_INDICES_FILE="$SMOKE_INDICES_FILE" \
LOG_FILE="$PROJECT_ROOT/$SMOKE_OUTPUT_DIR_REL/smoke.log" \
bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh" \
  --run_name "$SMOKE_RUN_ID" \
  --start_index "$SMOKE_INDEX" \
  --max_samples 1 \
  "$@"

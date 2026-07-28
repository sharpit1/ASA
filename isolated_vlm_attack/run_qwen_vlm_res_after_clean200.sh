#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_CORRECT_COUNT="${CLEAN_CORRECT_COUNT:-0}"

if ! [[ "$CLEAN_CORRECT_COUNT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLEAN_CORRECT_COUNT must be a non-negative integer" >&2
  exit 1
fi

# An empty value tells the remote bootstrap to use all dataset candidates.
export SAMPLE_INDICES_FILE=""

echo "[qwen_after_clean200] clean_correct_skip=200"
echo "[qwen_after_clean200] clean_correct_count=$CLEAN_CORRECT_COUNT"

exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh" \
  --clean_correct_skip 200 \
  --clean_correct_count "$CLEAN_CORRECT_COUNT" \
  "$@"

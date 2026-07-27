#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ASA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_RUN_DIR="${QWEN_SOURCE_RUN_DIR:-outputs/nips2017/resnet50/qwen_vlm_none_naturalness_clean500_20260727_015000}"
INDICES_FILE="${SAMPLE_INDICES_FILE:-$SOURCE_RUN_DIR/target_200_indices.json}"

if [[ "$INDICES_FILE" == /* ]]; then
  echo "ERROR: SAMPLE_INDICES_FILE must be relative to the ASA project root" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/$INDICES_FILE" ]]; then
  echo "ERROR: source indices file not found: $INDICES_FILE" >&2
  exit 1
fi

export ASA_PROJECT_ROOT="$PROJECT_ROOT"
export SAMPLE_INDICES_FILE="$INDICES_FILE"

echo "[qwen_second_clean100] project_root=$PROJECT_ROOT"
echo "[qwen_second_clean100] indices_file=$INDICES_FILE"
echo "[qwen_second_clean100] clean_correct_window=101-200"

exec bash "$SCRIPT_DIR/run_qwen_vlm_res_remote.sh" \
  --clean_correct_skip 100 \
  --clean_correct_count 100 \
  "$@"

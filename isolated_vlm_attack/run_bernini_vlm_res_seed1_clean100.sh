#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED=1
export CLEAN_CORRECT_SAMPLE_SEED=1
export MANUAL_SEED=1

exec bash "$SCRIPT_DIR/run_bernini_vlm_res_seeded_clean100.sh" "$@"

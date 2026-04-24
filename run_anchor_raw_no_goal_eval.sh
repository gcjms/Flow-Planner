#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_anchor_eval_common.sh"

anchor_eval_init
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/anchor_runs/raw_no_goal_eval}"
export OUTPUT_ROOT

run_anchor_eval_case "raw_no_goal_baseline" "$RAW_CKPT" "none"

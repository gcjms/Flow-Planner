#!/bin/bash
set -euo pipefail

# Train a scheduled-anchor-sampling planner finetune run, then evaluate it.
#
# Usage:
#   bash run_anchor_scheduled_sampling.sh 0.3
#   bash run_anchor_scheduled_sampling.sh 0.5
#
# The argument is p_max: the final probability of replacing oracle anchors with
# AnchorPredictor top-1 anchors during finetuning.

P_MAX="${1:-}"
if [ -z "$P_MAX" ]; then
  echo "Usage: bash run_anchor_scheduled_sampling.sh <p_max>" >&2
  echo "Example: bash run_anchor_scheduled_sampling.sh 0.3" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_anchor_eval_common.sh"

anchor_eval_init

P_TAG="$(python - "$P_MAX" <<'PY'
import sys
p = float(sys.argv[1])
print(str(p).replace(".", "p"))
PY
)"

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-/root/autodl-tmp/nuplan_npz}"
TRAIN_DATA_LIST="${TRAIN_DATA_LIST:-/root/autodl-tmp/nuplan_npz/train_list.json}"
VAL_DATA_DIR="${VAL_DATA_DIR:-/root/autodl-tmp/nuplan_npz}"
VAL_DATA_LIST="${VAL_DATA_LIST:-/root/autodl-tmp/nuplan_npz/val_list.json}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/anchor_runs/planner_ft_sched_p${P_TAG}}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"
DECODER_LR_MULT="${DECODER_LR_MULT:-0.1}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-80000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
RAMP_EPOCHS="${RAMP_EPOCHS:-$EPOCHS}"
NUM_WORKERS="${NUM_WORKERS:-4}"

anchor_require_file "$RAW_CKPT"
anchor_require_file "$ANCHOR_VOCAB_PATH"
anchor_require_file "$ANCHOR_PREDICTOR_CKPT"
anchor_require_dir "$TRAIN_DATA_DIR"
anchor_require_dir "$VAL_DATA_DIR"
anchor_require_file "$TRAIN_DATA_LIST"
anchor_require_file "$VAL_DATA_LIST"

mkdir -p "$SAVE_DIR"

train_cmd=(
  python finetune_anchor_planner.py
  --planner-config "$CONFIG_PATH"
  --planner-ckpt "$RAW_CKPT"
  --anchor-vocab-path "$ANCHOR_VOCAB_PATH"
  --anchor-predictor-ckpt "$ANCHOR_PREDICTOR_CKPT"
  --train-data-dir "$TRAIN_DATA_DIR"
  --train-data-list "$TRAIN_DATA_LIST"
  --val-data-dir "$VAL_DATA_DIR"
  --val-data-list "$VAL_DATA_LIST"
  --save-dir "$SAVE_DIR"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --lr "$LR"
  --decoder-lr-mult "$DECODER_LR_MULT"
  --encoder-lr-mult "$ENCODER_LR_MULT"
  --scheduled-sampling-p-max "$P_MAX"
  --scheduled-sampling-ramp-epochs "$RAMP_EPOCHS"
)

if [ -n "$MAX_TRAIN_SAMPLES" ]; then
  train_cmd+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [ -n "$MAX_VAL_SAMPLES" ]; then
  train_cmd+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi

{
  echo "=== Scheduled anchor sampling train ==="
  echo "date: $(date)"
  echo "p_max: $P_MAX"
  echo "save_dir: $SAVE_DIR"
  echo "raw_ckpt: $RAW_CKPT"
  echo "anchor_predictor_ckpt: $ANCHOR_PREDICTOR_CKPT"
  echo
  printf 'command:'
  printf ' %q' "${train_cmd[@]}"
  echo
  echo
} | tee "$SAVE_DIR/train.log"

"${train_cmd[@]}" 2>&1 | tee -a "$SAVE_DIR/train.log"

SCHED_CKPT="$SAVE_DIR/planner_anchor_best.pth"
anchor_require_file "$SCHED_CKPT"

SCHED_OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/anchor_runs/deploy_eval_sched_p${P_TAG}}"
(
  export PLANNER_FT_CKPT="$SCHED_CKPT"
  export OUTPUT_ROOT="$SCHED_OUTPUT_ROOT"
  bash "$SCRIPT_DIR/run_anchor_eval_suite.sh" \
    planner_ft_none \
    predicted_anchor \
    predicted_anchor_rerank_a \
    oracle_anchor \
    oracle_anchor_rerank
)

echo "=== Scheduled run complete ==="
echo "p_max: $P_MAX"
echo "best_ckpt: $SCHED_CKPT"
echo "eval_output_root: $SCHED_OUTPUT_ROOT"

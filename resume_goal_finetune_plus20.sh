#!/bin/bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/root/autodl-tmp/Flow-Planner/outputs/goal_finetune/2026-04-11_21-32-21}"
TARGET_EPOCH="${TARGET_EPOCH:-70}"
DEVICE="${DEVICE:-cuda}"
SCENE_DIR="${SCENE_DIR:-/root/autodl-tmp/hard_scenarios_v2}"
DATA_LIST="${DATA_LIST:-/root/autodl-tmp/hard_scenarios_v2/train_list.json}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50 60 70}"
REPORT_JSON="${REPORT_JSON:-$RUN_DIR/resume_eval_table.json}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/resume_plus20.log}"
GOAL_MODE="${GOAL_MODE:-gt_nearest}"
TRAINING_DATA="${TRAINING_DATA:-/root/autodl-tmp/hard_scenarios_v2}"
TRAINING_JSON="${TRAINING_JSON:-/root/autodl-tmp/hard_scenarios_v2/train_list.json}"
TENSORBOARD_LOG_PATH="${TENSORBOARD_LOG_PATH:-/root/autodl-tmp/Flow-Planner/tb_goal_resume}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

cd /root/autodl-tmp/Flow-Planner

export TRAINING_DATA
export TRAINING_JSON
export TENSORBOARD_LOG_PATH
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29529

{
  echo "=== Resume goal finetune started at $(date) ==="
  echo "RUN_DIR=$RUN_DIR"
  echo "TARGET_EPOCH=$TARGET_EPOCH"
  echo "GOAL_MODE=$GOAL_MODE"

  python -u -m flow_planner.trainer \
    --config-name goal_finetune \
    save_dir="$RUN_DIR" \
    resume_path="$RUN_DIR" \
    pretrained_checkpoint=null \
    train.epoch="$TARGET_EPOCH"

  echo
  echo "=== Checkpoint comparison at $(date) ==="
  python -u eval_goal_checkpoint_table.py \
    --run-dir "$RUN_DIR" \
    --scene-dir "$SCENE_DIR" \
    --data-list "$DATA_LIST" \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --goal-mode "$GOAL_MODE" \
    --epochs $EVAL_EPOCHS \
    --output-json "$REPORT_JSON"
} 2>&1 | tee -a "$LOG_FILE"

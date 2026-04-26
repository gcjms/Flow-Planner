#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/Flow-Planner

# Activate conda env
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

PLANNER_CONFIG="checkpoints/config_goal.yaml"
PLANNER_CKPT="checkpoints/model_goal.pth"
GOAL_VOCAB="goal_vocab.npy"
TRAIN_DATA_DIR="/root/autodl-tmp/hard_scenarios_v2"
TRAIN_DATA_LIST="splits/goal_predictor_train.json"
VAL_DATA_DIR="/root/autodl-tmp/hard_scenarios_v2"
VAL_DATA_LIST="splits/goal_predictor_val.json"
SAVE_DIR="outputs/goal_predictor_run1"

echo ">>> Starting goal_predictor run1 training <<<"
echo "Train data: $TRAIN_DATA_LIST"
echo "Val data: $VAL_DATA_LIST"
echo "Save dir: $SAVE_DIR"
echo ""

python train_goal_predictor.py \
  --planner-config "$PLANNER_CONFIG" \
  --planner-ckpt "$PLANNER_CKPT" \
  --goal-vocab-path "$GOAL_VOCAB" \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --train-data-list "$TRAIN_DATA_LIST" \
  --val-data-dir "$VAL_DATA_DIR" \
  --val-data-list "$VAL_DATA_LIST" \
  --save-dir "$SAVE_DIR" \
  --device cuda \
  --batch-size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --freeze-backbone \
  --hidden-dim 256 \
  --dropout 0.1

echo ""
echo ">>> Training complete, running evaluation <<<"
echo ""

python eval_goal_predictor.py \
  --planner-config "$PLANNER_CONFIG" \
  --planner-ckpt "$PLANNER_CKPT" \
  --goal-vocab-path "$GOAL_VOCAB" \
  --predictor-ckpt "$SAVE_DIR/goal_predictor_best.pth" \
  --data-dir "$VAL_DATA_DIR" \
  --data-list "$VAL_DATA_LIST" \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --output-json "$SAVE_DIR/eval_run1.json"

echo ""
echo ">>> Run1 complete <<<"
echo "Results saved to: $SAVE_DIR"

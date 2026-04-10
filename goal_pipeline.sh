#!/bin/bash
# ============================================================
# Goal Conditioning Pipeline (Step 2 + Step 3)
# ============================================================
# Step 2: Finetune FlowPlanner with goal conditioning (50 epochs)
# Step 3: Generate DPO candidates with diverse goals
# ============================================================
set -e

# ---- Environment ----
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

export TRAINING_DATA=/root/autodl-tmp/hard_scenarios_v2
export TRAINING_JSON=/root/autodl-tmp/hard_scenarios_v2/train_list.json
export TENSORBOARD_LOG_PATH=/root/autodl-tmp/Flow-Planner/tb_goal
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29529

cd /root/autodl-tmp/Flow-Planner

LOG=/root/goal_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "  Goal Conditioning Pipeline"
echo "  Start: $(date)"
echo "=========================================="

# ============================================================
# Step 2: Finetune with goal conditioning
# ============================================================
echo ""
echo ">>> Step 2: Finetune model with goal_dim=2"
echo "    Data: $TRAINING_DATA (50K scenes)"
echo "    Pretrained: checkpoints/model.pth"
echo "    Goal vocab: goal_vocab.npy"
echo "    Epochs: 50, Batch: 32, LR: 5e-5"
echo ""

python -u flow_planner/trainer.py \
    --config-name goal_finetune \
    optimizer.lr=5e-5

echo ""
echo ">>> Step 2 COMPLETE at $(date)"

# ---- 找到最新 checkpoint ----
CKPT_DIR=$(find /root/autodl-tmp/Flow-Planner/outputs/goal_finetune -name "*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found after training!"
    exit 1
fi
echo ">>> Best checkpoint: $CKPT_DIR"

# 复制到规范位置
cp "$CKPT_DIR" /root/autodl-tmp/Flow-Planner/checkpoints/model_goal.pth
echo ">>> Copied to checkpoints/model_goal.pth"

# ============================================================
# Step 3: Generate DPO candidates with diverse goals
# ============================================================
echo ""
echo ">>> Step 3: Generate goal-diverse DPO candidates"
echo "    Using 5000 hardest scenarios from training data"
echo ""

# 用 5000 个场景生成候选
python -u -m flow_planner.dpo.generate_candidates_goal \
    --data_dir /root/autodl-tmp/hard_scenarios_v2 \
    --config_path /root/autodl-tmp/Flow-Planner/checkpoints/model_config.yaml \
    --ckpt_path /root/autodl-tmp/Flow-Planner/checkpoints/model_goal.pth \
    --vocab_path /root/autodl-tmp/Flow-Planner/goal_vocab.npy \
    --output_dir /root/autodl-tmp/dpo_candidates_goal \
    --num_candidates 5 \
    --max_scenarios 5000

echo ""
echo ">>> Step 3 COMPLETE at $(date)"
echo ""
echo "=========================================="
echo "  Pipeline finished at $(date)"
echo "  Next: run score_hybrid.py → train_dpo.py"
echo "=========================================="

# 不自动关机，等用户确认
echo ""
echo ">>> Pipeline done. Instance kept running for inspection."
echo ">>> To shut down: shutdown now"

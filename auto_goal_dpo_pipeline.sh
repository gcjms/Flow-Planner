#!/bin/bash
# =============================================================================
# Goal Conditioning + DPO Pipeline
# =============================================================================
# 完整流程:
#   Step 1: 聚类 GT 终点 → goal_vocab.npy
#   Step 2: (手动) 改 yaml + 重训带 goal 的 FM 模型
#   Step 3: Goal-diverse 候选生成
#   Step 4: 打分 → 构建偏好对
#   Step 5: DPO 训练
#   Step 6: 候选多样性评估 + DPO 后模型对比
#
# 前提:
#   - Step 2 已完成 (带 goal 的模型已训好)
#   - checkpoints/config_goal.yaml 和 model_goal.pth 存在
#   - goal_vocab.npy 存在
#   - dpo_mining 场景数据存在
#
# 使用方法:
#   chmod +x auto_goal_dpo_pipeline.sh
#   nohup bash auto_goal_dpo_pipeline.sh > /root/goal_pipeline_stdout.log 2>&1 &
#
# =============================================================================

set -euo pipefail
LOG=/root/goal_dpo_pipeline.log
REPORT=/root/goal_dpo_eval_report.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

echo "=== Goal + DPO Pipeline started at $(date) ===" | tee $LOG

export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache

# ===================== Paths =====================
GOAL_VOCAB=/root/autodl-tmp/Flow-Planner/goal_vocab.npy
CONFIG_GOAL=checkpoints/config_goal.yaml
CKPT_GOAL=checkpoints/model_goal.pth
SCENE_DIR=/root/autodl-tmp/dpo_mining
CANDIDATES_DIR=/root/autodl-tmp/dpo_candidates_goal
PREFERENCES_DIR=/root/autodl-tmp/dpo_preferences_goal
SCORED_DIR=$PREFERENCES_DIR/scored_dir
PREF_META=$PREFERENCES_DIR/preferences_multi_meta.jsonl
DPO_DIR=checkpoints/dpo_goal
DIVERSITY_EVAL_SCENES=200

# ===================== Hyperparameters =====================
N_CLUSTERS=64
GOAL_FRAME=39          # 0-indexed: 39 = 4s@10Hz, -1 = last frame (8s)
NUM_CANDIDATES=5
CFG_WEIGHT=1.8
DPO_BETA=5.0
DPO_EPOCHS=3
DPO_LR=5e-5
LORA_RANK=4
LORA_ALPHA=16
SFT_WEIGHT=0.1
NUM_T_SAMPLES=16
SPREAD_THRESHOLD=3.0
MAX_EVAL_SCENES=500
# ===========================================================

# =====================================================
# STEP 1: 聚类 (如果 goal_vocab.npy 不存在才跑)
# =====================================================
if [ ! -f "$GOAL_VOCAB" ]; then
    echo "[Step 1] Clustering GT endpoints..." | tee -a $LOG
    python -u -m flow_planner.goal.cluster_goals \
        --data_dir /root/autodl-tmp/nuplan_npz \
        --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
        --output_path $GOAL_VOCAB \
        --n_clusters $N_CLUSTERS \
        --goal_frame $GOAL_FRAME \
        2>&1 | tee -a $LOG
    echo "[Step 1] Done at $(date)" | tee -a $LOG
else
    echo "[Step 1] SKIP: $GOAL_VOCAB already exists" | tee -a $LOG
fi

# =====================================================
# 前置检查
# =====================================================
if [ ! -f "$CONFIG_GOAL" ]; then
    echo "[ERROR] $CONFIG_GOAL not found!" | tee -a $LOG
    echo "  请先完成 Step 2: 改 yaml + 重训带 goal 的模型" | tee -a $LOG
    echo "  参见 docs/goal_conditioning_guide.md Step 2" | tee -a $LOG
    exit 1
fi

if [ ! -f "$CKPT_GOAL" ]; then
    echo "[ERROR] $CKPT_GOAL not found!" | tee -a $LOG
    echo "  请先完成 Step 2: 重训带 goal 的模型" | tee -a $LOG
    exit 1
fi

if [ ! -d "$SCENE_DIR" ]; then
    echo "[ERROR] $SCENE_DIR not found! 需要 dpo_mining 场景数据" | tee -a $LOG
    exit 1
fi

echo "[Check] All prerequisites OK" | tee -a $LOG

# =====================================================
# STEP 3: Goal-diverse 候选生成
# =====================================================
echo "[Step 3] Generating goal-diverse candidates ($NUM_CANDIDATES per scene)..." | tee -a $LOG

python -u -m flow_planner.dpo.generate_candidates_goal \
    --data_dir $SCENE_DIR \
    --config_path $CONFIG_GOAL \
    --ckpt_path $CKPT_GOAL \
    --vocab_path $GOAL_VOCAB \
    --output_dir $CANDIDATES_DIR \
    --num_candidates $NUM_CANDIDATES \
    --cfg_weight $CFG_WEIGHT \
    2>&1 | tee -a $LOG

N_CANDS=$(ls $CANDIDATES_DIR/*_candidates.npz 2>/dev/null | wc -l)
echo "[Step 3] Generated $N_CANDS candidate files" | tee -a $LOG
echo "[Step 3] Done at $(date)" | tee -a $LOG

if [ "$N_CANDS" -lt 10 ]; then
    echo "[Step 3] FAILED: Too few candidates ($N_CANDS < 10)" | tee -a $LOG
    exit 1
fi

# =====================================================
# STEP 4: 打分 → 构建偏好对
# =====================================================
echo "[Step 4] Structured scoring candidates and building multi-pair preferences..." | tee -a $LOG

python -u -m flow_planner.dpo.score_hybrid \
    --candidates_dir $CANDIDATES_DIR \
    --output_dir $PREFERENCES_DIR \
    --scored_dir $SCORED_DIR \
    --use_structured_scores \
    --emit_traj_info \
    --skip_vlm \
    --spread_threshold $SPREAD_THRESHOLD \
    2>&1 | tee -a $LOG

python -u -m flow_planner.dpo.build_multi_pairs \
    --scored_dir $SCORED_DIR \
    --candidates_dir $CANDIDATES_DIR \
    --output_path $PREFERENCES_DIR/preferences_multi.npz \
    --meta_path $PREF_META \
    --top_good_per_cluster 1 \
    --subtle_bad_per_good 2 \
    2>&1 | tee -a $LOG

PREF_FILE=$PREFERENCES_DIR/preferences_multi.npz
if [ ! -f "$PREF_FILE" ]; then
    echo "[Step 4] FAILED: No structured multi-pair preferences generated!" | tee -a $LOG
    exit 1
fi

N_PAIRS=$(python -c "import numpy as np; d=np.load('$PREF_FILE', allow_pickle=True); print(d['chosen'].shape[0])")
DIM_COUNTS=$(python -c "import numpy as np; d=np.load('$PREF_FILE', allow_pickle=True); u,c=np.unique(d['dim_labels'], return_counts=True); print(dict(zip([str(x) for x in u], [int(x) for x in c])))")
echo "[Step 4] Built $N_PAIRS structured preference pairs" | tee -a $LOG
echo "[Step 4] Dimension mix: $DIM_COUNTS" | tee -a $LOG
echo "[Step 4] Done at $(date)" | tee -a $LOG

if [ "$N_PAIRS" -lt 20 ]; then
    echo "[Step 4] Too few pairs ($N_PAIRS < 20), aborting" | tee -a $LOG
    exit 1
fi

# =====================================================
# STEP 5: DPO 训练
# =====================================================
echo "[Step 5] DPO training (beta=$DPO_BETA, rank=$LORA_RANK, epochs=$DPO_EPOCHS)..." | tee -a $LOG

mkdir -p $DPO_DIR

python -u -m flow_planner.dpo.train_dpo \
    --config_path $CONFIG_GOAL \
    --ckpt_path $CKPT_GOAL \
    --preference_path $PREF_FILE \
    --scene_dir $SCENE_DIR \
    --output_dir $DPO_DIR \
    --beta $DPO_BETA \
    --epochs $DPO_EPOCHS \
    --lr $DPO_LR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --batch_size 8 \
    --sft_weight $SFT_WEIGHT \
    --num_t_samples $NUM_T_SAMPLES \
    --save_merged \
    2>&1 | tee /root/goal_dpo_train.log

echo "[Step 5] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 6: 评估
# =====================================================
echo "[Step 6] Diversity-focused evaluation..." | tee -a $LOG

echo "========================================" > $REPORT
echo "  Goal + DPO Evaluation Report" >> $REPORT
echo "  $(date)" >> $REPORT
echo "  Clusters: $N_CLUSTERS (goal_frame=$GOAL_FRAME)" >> $REPORT
echo "  Candidates per scene: $NUM_CANDIDATES" >> $REPORT
echo "  Preference pairs: $N_PAIRS" >> $REPORT
echo "  DPO: beta=$DPO_BETA, rank=$LORA_RANK, epochs=$DPO_EPOCHS" >> $REPORT
echo "========================================" >> $REPORT
echo "" >> $REPORT

MERGED_CKPT=$DPO_DIR/model_dpo_merged.pth

# 候选多样性评估: 原 goal 模型 vs DPO 后模型
echo "[Goal Diversity Evaluation: $DIVERSITY_EVAL_SCENES scenes]" >> $REPORT

for MODEL_INFO in \
    "Original:$CKPT_GOAL" \
    "Goal_DPO:$MERGED_CKPT"; do
    NAME="${MODEL_INFO%%:*}"
    CKPT="${MODEL_INFO##*:}"
    if [ -f "$CKPT" ]; then
        echo "  Evaluating $NAME..." | tee -a $LOG
        RESULT=$(python -u -m flow_planner.dpo.eval_goal_diversity \
            --ckpt_path "$CKPT" \
            --config_path $CONFIG_GOAL \
            --data_dir /root/autodl-tmp/hard_scenarios_v2 \
            --vocab_path $GOAL_VOCAB \
            --max_scenarios $DIVERSITY_EVAL_SCENES \
            --num_candidates $NUM_CANDIDATES \
            --cfg_weight $CFG_WEIGHT 2>&1 | grep -E "Summary|Summary|scenes_|endpoint_spread|pairwise_|score_|unique_" || true)
        echo "  $NAME:" >> $REPORT
        echo "$RESULT" >> $REPORT
        echo "" >> $REPORT
    else
        echo "  [SKIP] $NAME: $CKPT not found" >> $REPORT
    fi
done

# 训练日志摘要
echo "[Training Summary]" >> $REPORT
grep "Epoch .*/.*|" /root/goal_dpo_train.log >> $REPORT 2>/dev/null || true
echo "" >> $REPORT

echo "[Checkpoints]" >> $REPORT
ls -lh $DPO_DIR/*.pt $DPO_DIR/*.pth 2>/dev/null >> $REPORT || true
echo "" >> $REPORT

echo "=== Pipeline completed at $(date) ===" | tee -a $LOG
echo "=== Completed at $(date) ===" >> $REPORT

cat $REPORT

echo ""
echo "Done! Report saved to $REPORT"
echo "Merged model: $MERGED_CKPT"

#!/bin/bash
# On-Policy DPO Pipeline
# ========================
# Step 1: 从 50K 训练场景中采样 5000 个，每个推理 20 次，挖掘 borderline 对
# Step 2: 用 on-policy 偏好对训练 DPO
# Step 3: 合并权重
# Step 4: 开环碰撞测试（对比 baseline）
# Step 5: 闭环仿真
# Step 6: 生成报告 + 关机

set -o pipefail
LOG=/root/onpolicy_pipeline.log
REPORT=/root/onpolicy_eval_report.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

echo "=== On-Policy DPO Pipeline started at $(date) ===" | tee $LOG

export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache

# =====================================================
# STEP 1: 挖掘 on-policy 偏好对
# =====================================================
echo "[Step 1] Mining on-policy preference pairs..." | tee -a $LOG
echo "  5000 scenes × 20 samples each = 100K inferences" | tee -a $LOG

ONPOLICY_DATA=/root/autodl-tmp/onpolicy_pairs.npz

python -u -m flow_planner.dpo.generate_onpolicy_pairs \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --output_path $ONPOLICY_DATA \
    --max_scenes 5000 \
    --num_samples 20 \
    --device cuda \
    2>&1 | tee -a $LOG

# 检查是否成功生成了偏好对
if [ ! -f "$ONPOLICY_DATA" ]; then
    echo "[Step 1] FAILED: No on-policy pairs generated!" | tee -a $LOG
    echo "Pipeline aborted - no data to train on" | tee -a $LOG
    
    # 写报告
    echo "========================================" > $REPORT
    echo "  On-Policy DPO Report (FAILED)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "" >> $REPORT
    echo "Mining produced 0 on-policy pairs." >> $REPORT
    echo "Trajectory diversity too low for borderline detection." >> $REPORT
    echo "" >> $REPORT
    tail -30 $LOG >> $REPORT
    
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

N_PAIRS=$(python -c "import numpy as np; d=np.load('$ONPOLICY_DATA', allow_pickle=True); print(d['chosen'].shape[0])")
echo "[Step 1] Found $N_PAIRS on-policy pairs" | tee -a $LOG

if [ "$N_PAIRS" -lt 10 ]; then
    echo "[Step 1] Too few pairs ($N_PAIRS < 10), aborting" | tee -a $LOG
    echo "========================================" > $REPORT
    echo "  On-Policy DPO Report (INSUFFICIENT DATA)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "Only found $N_PAIRS on-policy pairs (need >= 10)" >> $REPORT
    tail -30 $LOG >> $REPORT
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

echo "[Step 1] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 2: DPO 训练（on-policy 数据）
# =====================================================
echo "[Step 2] Training DPO with on-policy pairs..." | tee -a $LOG

ONPOLICY_DIR=checkpoints/dpo_onpolicy
mkdir -p $ONPOLICY_DIR

python -u -m flow_planner.dpo.train_dpo \
    --config_path checkpoints/model_config.yaml \
    --ckpt_path checkpoints/model.pth \
    --dpo_data_path $ONPOLICY_DATA \
    --output_dir $ONPOLICY_DIR \
    --beta 0.5 \
    --epochs 3 \
    --lr 5e-5 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --batch_size 8 \
    --K 16 \
    2>&1 | tee /root/dpo_onpolicy_train.log

echo "[Step 2] DPO training done at $(date)" | tee -a $LOG

# =====================================================
# STEP 3: 合并 LoRA 权重
# =====================================================
echo "[Step 3] Merging on-policy LoRA weights..." | tee -a $LOG

python -u -c "
import torch, os
from flow_planner.dpo.lora import inject_lora, load_lora, merge_lora
from omegaconf import OmegaConf
from hydra.utils import instantiate

cfg = OmegaConf.load('checkpoints/model_config.yaml')
OmegaConf.update(cfg, 'data.dataset.train.future_downsampling_method', 'uniform', force_add=True)
OmegaConf.update(cfg, 'data.dataset.train.predicted_neighbor_num', 0, force_add=True)
model = instantiate(cfg.model)
ckpt = torch.load('checkpoints/model.pth', map_location='cpu', weights_only=False)
sd = ckpt.get('ema_state_dict', ckpt.get('state_dict', ckpt))
sd = {k.replace('module.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
inject_lora(model.model_decoder, rank=4, alpha=8)
lora_path = '$ONPOLICY_DIR/lora_best.pt'
if not os.path.exists(lora_path):
    lora_path = '$ONPOLICY_DIR/lora_epoch_3.pt'
if not os.path.exists(lora_path):
    lora_path = '$ONPOLICY_DIR/lora_epoch_1.pt'
print(f'Loading {lora_path}')
load_lora(model.model_decoder, lora_path)
merge_lora(model.model_decoder)
raw_sd = model.state_dict()
fixed_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
torch.save({'state_dict': fixed_sd}, '$ONPOLICY_DIR/model_onpolicy_merged.pth')
print('Merged model saved!')
" 2>&1 | tee -a $LOG

echo "[Step 3] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 4: 开环碰撞测试
# =====================================================
echo "[Step 4] Open-loop collision test..." | tee -a $LOG

echo "========================================" > $REPORT
echo "  On-Policy DPO Evaluation Report" >> $REPORT
echo "  $(date)" >> $REPORT
echo "========================================" >> $REPORT
echo "" >> $REPORT
echo "[Mining Results]" >> $REPORT
echo "  On-policy pairs found: $N_PAIRS" >> $REPORT
grep "Avg trajectory diversity" $LOG >> $REPORT 2>/dev/null
grep "Borderline scenes" $LOG >> $REPORT 2>/dev/null
echo "" >> $REPORT
echo "[Open-Loop: 500 Hard Scenarios]" >> $REPORT

for MODEL_INFO in \
    "Original:checkpoints/model.pth" \
    "DPO_offpolicy_run1:checkpoints/dpo_oracle_run1/model_dpo_merged.pth" \
    "DPO_onpolicy:$ONPOLICY_DIR/model_onpolicy_merged.pth"; do
    NAME="${MODEL_INFO%%:*}"
    CKPT="${MODEL_INFO##*:}"
    if [ -f "$CKPT" ]; then
        echo "Testing $NAME..." | tee -a $LOG
        RESULT=$(python -u -m flow_planner.dpo.generate_oracle_pairs \
            --ckpt_path $CKPT \
            --config_path checkpoints/model_config.yaml \
            --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
            --output_path /dev/null \
            --max_scenes 500 2>&1 | grep "Model collisions")
        echo "  $NAME: $RESULT" | tee -a $LOG
        echo "  $NAME: $RESULT" >> $REPORT
    fi
done

echo "" >> $REPORT

# =====================================================
# STEP 5: 闭环仿真 (on-policy model, 200 scenarios)
# =====================================================
echo "[Step 5] Closed-loop simulation..." | tee -a $LOG

MERGED_CKPT="$ONPOLICY_DIR/model_onpolicy_merged.pth"
SIM_SCRIPT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages/nuplan/planning/script/run_simulation.py
HYDRA_SEARCH="[pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"

if [ -f "$MERGED_CKPT" ]; then
    echo "[Closed-Loop: val14, 200 scenarios]" >> $REPORT
    
    python $SIM_SCRIPT \
        +simulation=closed_loop_nonreactive_agents \
        planner=flow_planner \
        planner.flow_planner.config_path=/root/autodl-tmp/Flow-Planner/checkpoints/model_config_resolved.yaml \
        planner.flow_planner.ckpt_path=/root/autodl-tmp/Flow-Planner/$MERGED_CKPT \
        planner.flow_planner.use_cfg=true \
        planner.flow_planner.cfg_weight=1.8 \
        planner.flow_planner.device=cuda \
        scenario_builder=nuplan \
        scenario_builder.data_root=/root/autodl-tmp/val_data/data/cache/val \
        scenario_filter=val14 \
        scenario_filter.limit_total_scenarios=200 \
        experiment_name=dpo_onpolicy_cl \
        output_dir=/root/autodl-tmp/dpo_onpolicy_eval \
        worker=ray_distributed worker.threads_per_node=4 \
        distributed_mode=SINGLE_NODE \
        number_of_gpus_allocated_per_simulation=0.25 \
        exit_on_failure=false \
        hydra.searchpath="$HYDRA_SEARCH" \
        2>&1 | tee /root/dpo_onpolicy_sim.log
    
    echo "  DPO on-policy closed-loop results:" >> $REPORT
    grep -E "final_score|Score|collision" /root/dpo_onpolicy_sim.log | tail -10 >> $REPORT 2>/dev/null
    
    # 如果闭环失败（config 问题），记录错误但继续
    if grep -q "InterpolationKeyError" /root/dpo_onpolicy_sim.log 2>/dev/null; then
        echo "  [WARN] Closed-loop failed due to config interpolation error" >> $REPORT
        echo "  Check /root/dpo_onpolicy_sim.log for details" >> $REPORT
    fi
else
    echo "  [SKIP] No merged model found, skipping closed-loop" >> $REPORT
fi

echo "" >> $REPORT

# =====================================================
# STEP 6: 训练摘要 + 关机
# =====================================================
echo "[Training Summary]" >> $REPORT
echo "On-Policy DPO (β=0.5, rank=4, 3 epochs):" >> $REPORT
grep "Epoch .*/3 |" /root/dpo_onpolicy_train.log >> $REPORT 2>/dev/null
echo "" >> $REPORT

echo "[All Checkpoints]" >> $REPORT
ls -lh $ONPOLICY_DIR/*.pt $ONPOLICY_DIR/*.pth 2>/dev/null >> $REPORT

echo "" >> $REPORT
echo "=== Completed at $(date) ===" >> $REPORT
echo "=== Completed at $(date) ===" >> $LOG

cat $REPORT

echo "Shutting down in 60 seconds..." | tee -a $LOG
sleep 60
shutdown now 2>/dev/null || poweroff 2>/dev/null

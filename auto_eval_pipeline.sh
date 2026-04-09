#!/bin/bash
# 全自动评估 Pipeline v3：
# run1 + run2 都跑闭环，用较少场景加速
# 绝不删除任何文件！

set -o pipefail
LOG=/root/eval_pipeline.log
REPORT=/root/dpo_eval_report.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

echo "=== Eval Pipeline v3 started at $(date) ===" | tee $LOG

export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache
HYDRA_SEARCH="[pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
SIM_SCRIPT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages/nuplan/planning/script/run_simulation.py

# =====================================================
# STEP 1: 等待 run2 训练完成
# =====================================================
echo "[Step 1] Waiting for run2 training..." | tee -a $LOG
MAX_WAIT=21600
waited=0
while pgrep -f "train_dpo.*oracle_pairs_4d" > /dev/null 2>&1; do
    if [ $waited -ge $MAX_WAIT ]; then
        echo "TIMEOUT" | tee -a $LOG
        break
    fi
    sleep 300
    waited=$((waited + 300))
    tail -1 /root/dpo_oracle_run2.log >> $LOG 2>/dev/null
done
echo "[Step 1] Training done at $(date)" | tee -a $LOG

# =====================================================
# STEP 2: 合并 run2 LoRA 权重
# =====================================================
echo "[Step 2] Merging run2 LoRA weights..." | tee -a $LOG
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
run2_dir = 'checkpoints/dpo_oracle_run2'
best = os.path.join(run2_dir, 'lora_best.pt')
if not os.path.exists(best):
    best = os.path.join(run2_dir, 'lora_epoch_1.pt')
print(f'Loading {best}')
load_lora(model.model_decoder, best)
merge_lora(model.model_decoder)
raw_sd = model.state_dict()
fixed_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
torch.save({'state_dict': fixed_sd}, os.path.join(run2_dir, 'model_run2_merged.pth'))
print('Done!')
" 2>&1 | tee -a $LOG

# =====================================================
# STEP 3: 开环碰撞测试 (3 models × 500 scenes)
# =====================================================
echo "[Step 3] Open-loop collision test..." | tee -a $LOG
echo "========================================" > $REPORT
echo "  DPO Evaluation Report" >> $REPORT
echo "  $(date)" >> $REPORT
echo "========================================" >> $REPORT

echo "" >> $REPORT
echo "[Open-Loop: 500 Hard Scenarios]" >> $REPORT

for MODEL_INFO in \
    "Original:checkpoints/model.pth" \
    "DPO_run1_b1_e3:checkpoints/dpo_oracle_run1/model_dpo_merged.pth" \
    "DPO_run2_b10_e1:checkpoints/dpo_oracle_run2/model_run2_merged.pth"; do
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
# STEP 4: 闭环仿真 (run1 + run2, val14 全量各跑一次)
# =====================================================
echo "[Step 4] Closed-loop simulations..." | tee -a $LOG
echo "[Closed-Loop: val14, 500 scenarios each]" >> $REPORT

run_closed_loop() {
    local NAME=$1
    local CKPT=$2
    local OUTDIR=$3
    
    echo "[4] Running $NAME closed-loop..." | tee -a $LOG
    python $SIM_SCRIPT \
        +simulation=closed_loop_nonreactive_agents \
        planner=flow_planner \
        planner.flow_planner.config_path=/root/autodl-tmp/Flow-Planner/checkpoints/model_config.yaml \
        planner.flow_planner.ckpt_path=$CKPT \
        planner.flow_planner.use_cfg=true \
        planner.flow_planner.cfg_weight=1.8 \
        planner.flow_planner.device=cuda \
        scenario_builder=nuplan \
        scenario_builder.data_root=/root/autodl-tmp/val_data/data/cache/val \
        scenario_filter=val14 \
        scenario_filter.limit_total_scenarios=500 \
        experiment_name=${NAME} \
        output_dir=$OUTDIR \
        worker=ray_distributed worker.threads_per_node=4 \
        distributed_mode=SINGLE_NODE \
        number_of_gpus_allocated_per_simulation=0.25 \
        exit_on_failure=false \
        hydra.searchpath="$HYDRA_SEARCH" \
        2>&1 | tee /root/${NAME}_sim.log
    
    echo "  $NAME results:" >> $REPORT
    grep -E "final_score|Score|collision" /root/${NAME}_sim.log | tail -10 >> $REPORT 2>/dev/null
    echo "" >> $REPORT
}

# Run1 闭环
run_closed_loop "dpo_run1" \
    "/root/autodl-tmp/Flow-Planner/checkpoints/dpo_oracle_run1/model_dpo_merged.pth" \
    "/root/autodl-tmp/dpo_eval_output"

# Run2 闭环
RUN2_MERGED="checkpoints/dpo_oracle_run2/model_run2_merged.pth"
if [ -f "$RUN2_MERGED" ]; then
    run_closed_loop "dpo_run2" \
        "/root/autodl-tmp/Flow-Planner/$RUN2_MERGED" \
        "/root/autodl-tmp/dpo_eval_output_run2"
fi

# =====================================================
# STEP 5: 训练摘要
# =====================================================
echo "" >> $REPORT
echo "[Training Summary]" >> $REPORT
echo "Run1 (β=1.0, rank=8, 3 epochs):" >> $REPORT
grep "Epoch .*/3 |" /root/dpo_oracle_train.log >> $REPORT 2>/dev/null
echo "Run2 (β=10.0, rank=4, 1 epoch):" >> $REPORT
grep "Epoch .*/1 |" /root/dpo_oracle_run2.log >> $REPORT 2>/dev/null

# =====================================================
# STEP 6: 文件清单 + 关机
# =====================================================
echo "" >> $REPORT
echo "[All Checkpoints]" >> $REPORT
ls -lh checkpoints/dpo_oracle_run1/*.pt checkpoints/dpo_oracle_run1/*.pth 2>/dev/null >> $REPORT
ls -lh checkpoints/dpo_oracle_run2/*.pt checkpoints/dpo_oracle_run2/*.pth 2>/dev/null >> $REPORT

echo "=== Completed at $(date) ===" >> $REPORT
echo "=== Completed at $(date) ===" >> $LOG
cat $REPORT

shutdown now 2>/dev/null || poweroff 2>/dev/null

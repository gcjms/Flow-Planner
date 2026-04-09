#!/bin/bash
# =============================================================================
# SDE + Multi-Objective DPO Pipeline
# =============================================================================
# 完整流程:
#   Step 0: SDE 多样性验证 (100 场景快速测试)
#   Step 1: SDE 模式挖掘 multi-objective preference pairs
#   Step 2: DPO 训练 (dimension-weighted loss)
#   Step 3: Merge LoRA
#   Step 4: 开环多维度评估 (对比原模型 vs DPO 模型)
#   Step 5: 闭环 NR-CLS 仿真
#   Step 6: 汇总报告 + 关机
#
# 使用方法:
#   1. 把整个项目上传/git pull 到 AutoDL
#   2. 确保 checkpoints/model.pth 和 model_config.yaml 存在
#   3. 确保 hard_scenarios_v2 数据在 /root/autodl-tmp/hard_scenarios_v2
#   4. 运行:
#        chmod +x auto_sde_dpo_pipeline.sh
#        nohup bash auto_sde_dpo_pipeline.sh > /root/sde_pipeline_stdout.log 2>&1 &
#
# =============================================================================

set -o pipefail
LOG=/root/sde_dpo_pipeline.log
REPORT=/root/sde_dpo_eval_report.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

echo "=== SDE + DPO Pipeline started at $(date) ===" | tee $LOG

export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache

# ===================== Hyperparameters =====================
SIGMA_BASE=0.3          # SDE noise strength
SDE_STEPS=20            # SDE integration steps (more steps = more diversity budget)
NUM_SEEDS=5             # random seeds per CFG weight
CFG_WEIGHTS="0.5,1.0,1.8,3.0"   # CFG weights to try
MAX_SCENES=5000         # scenes for pair mining
DPO_BETA=5.0
DPO_EPOCHS=2
LORA_RANK=4
LORA_ALPHA=8
DIM_WEIGHTS="collision:5,ttc:5,comfort:2"
# ===========================================================

# =====================================================
# STEP 0: SDE Diversity Verification (quick sanity check)
# =====================================================
echo "[Step 0] SDE diversity verification (100 scenes)..." | tee -a $LOG

python -u -m flow_planner.dpo.measure_sde_diversity \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --num_scenes 100 \
    --num_samples 20 \
    --sigma_base "0.1,0.3,0.5" \
    --sde_steps $SDE_STEPS \
    --cfg_weight 1.8 \
    --device cuda \
    2>&1 | tee /root/sde_diversity_report.log

echo "[Step 0] Diversity report saved to /root/sde_diversity_report.log" | tee -a $LOG
echo "[Step 0] Done at $(date)" | tee -a $LOG

# Check if SDE provides any improvement (parse last recommendation line)
if grep -q "WARNING: No sigma value" /root/sde_diversity_report.log; then
    echo "[Step 0] WARNING: SDE did not improve diversity!" | tee -a $LOG
    echo "  Pipeline continues but results may be similar to ODE-only DPO." | tee -a $LOG
fi

# =====================================================
# STEP 1: Mine preference pairs with SDE sampling
# =====================================================
echo "[Step 1] Mining multi-objective pairs with SDE (σ=$SIGMA_BASE, steps=$SDE_STEPS)..." | tee -a $LOG
echo "  $MAX_SCENES scenes x $NUM_SEEDS seeds x $(echo $CFG_WEIGHTS | tr ',' '\n' | wc -l) CFGs" | tee -a $LOG

MO_DATA=/root/autodl-tmp/sde_multiobjective_pairs.npz

python -u -m flow_planner.dpo.generate_multiobjective_pairs \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --output_path $MO_DATA \
    --max_scenes $MAX_SCENES \
    --num_seeds $NUM_SEEDS \
    --cfg_weights $CFG_WEIGHTS \
    --target_dims collision,ttc,comfort \
    --score_gap_threshold 0.12 \
    --sde \
    --sigma_base $SIGMA_BASE \
    --sde_steps $SDE_STEPS \
    --device cuda \
    2>&1 | tee -a $LOG

if [ ! -f "$MO_DATA" ]; then
    echo "[Step 1] FAILED: No preference pairs generated!" | tee -a $LOG
    echo "========================================" > $REPORT
    echo "  SDE + DPO Report (FAILED at Step 1)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "Mining produced 0 pairs. Check /root/sde_diversity_report.log" >> $REPORT
    tail -30 $LOG >> $REPORT
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

N_PAIRS=$(python -c "import numpy as np; d=np.load('$MO_DATA', allow_pickle=True); print(d['chosen'].shape[0])")
echo "[Step 1] Found $N_PAIRS pairs (SDE mode)" | tee -a $LOG

if [ "$N_PAIRS" -lt 20 ]; then
    echo "[Step 1] Too few pairs ($N_PAIRS < 20), aborting" | tee -a $LOG
    echo "========================================" > $REPORT
    echo "  SDE + DPO Report (INSUFFICIENT DATA)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "Only $N_PAIRS pairs with SDE. Try: increase sigma_base or lower score_gap_threshold." >> $REPORT
    tail -30 $LOG >> $REPORT
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

echo "[Step 1] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 2: DPO Training
# =====================================================
echo "[Step 2] DPO training (beta=$DPO_BETA, rank=$LORA_RANK, epochs=$DPO_EPOCHS)..." | tee -a $LOG

SDE_DPO_DIR=checkpoints/dpo_sde
mkdir -p $SDE_DPO_DIR

python -u -m flow_planner.dpo.train_dpo \
    --config_path checkpoints/model_config.yaml \
    --ckpt_path checkpoints/model.pth \
    --preference_path $MO_DATA \
    --output_dir $SDE_DPO_DIR \
    --beta $DPO_BETA \
    --epochs $DPO_EPOCHS \
    --lr 5e-5 \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --batch_size 8 \
    --sft_weight 0.1 \
    --num_t_samples 16 \
    --dim_weights "$DIM_WEIGHTS" \
    --save_merged \
    2>&1 | tee /root/sde_dpo_train.log

echo "[Step 2] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 3: Merge LoRA
# =====================================================
echo "[Step 3] Merging LoRA weights..." | tee -a $LOG

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
inject_lora(model.model_decoder, rank=$LORA_RANK, alpha=$LORA_ALPHA)
lora_path = '$SDE_DPO_DIR/lora_best.pt'
if not os.path.exists(lora_path):
    lora_path = '$SDE_DPO_DIR/lora_epoch_${DPO_EPOCHS}.pt'
if not os.path.exists(lora_path):
    lora_path = '$SDE_DPO_DIR/lora_epoch_1.pt'
print(f'Loading {lora_path}')
load_lora(model.model_decoder, lora_path)
merge_lora(model.model_decoder)
raw_sd = model.state_dict()
fixed_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
torch.save({'state_dict': fixed_sd}, '$SDE_DPO_DIR/model_sde_dpo_merged.pth')
print('Merged model saved!')
" 2>&1 | tee -a $LOG

echo "[Step 3] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 4: Open-loop multi-dim evaluation
# =====================================================
echo "[Step 4] Open-loop evaluation..." | tee -a $LOG

echo "========================================" > $REPORT
echo "  SDE + DPO Evaluation Report" >> $REPORT
echo "  $(date)" >> $REPORT
echo "  SDE: sigma=$SIGMA_BASE, steps=$SDE_STEPS" >> $REPORT
echo "  DPO: beta=$DPO_BETA, rank=$LORA_RANK, epochs=$DPO_EPOCHS" >> $REPORT
echo "  Pairs mined: $N_PAIRS" >> $REPORT
echo "========================================" >> $REPORT
echo "" >> $REPORT

echo "[SDE Diversity Summary]" >> $REPORT
tail -20 /root/sde_diversity_report.log >> $REPORT
echo "" >> $REPORT

echo "[Open-Loop Evaluation: 500 Hard Scenarios]" >> $REPORT

for MODEL_INFO in \
    "Original:checkpoints/model.pth" \
    "SDE_DPO:$SDE_DPO_DIR/model_sde_dpo_merged.pth"; do
    NAME="${MODEL_INFO%%:*}"
    CKPT="${MODEL_INFO##*:}"
    if [ -f "$CKPT" ]; then
        echo "Evaluating $NAME..." | tee -a $LOG
        RESULT=$(python -u -m flow_planner.dpo.eval_multidim \
            --ckpt_path $CKPT \
            --config_path checkpoints/model_config.yaml \
            --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
            --max_scenes 500 2>&1 | grep -E "SUMMARY|collision_rate|avg_ttc|avg_comfort|avg_progress|avg_route")
        echo "  $NAME:" | tee -a $LOG
        echo "$RESULT" | tee -a $LOG
        echo "  $NAME:" >> $REPORT
        echo "$RESULT" >> $REPORT
        echo "" >> $REPORT
    fi
done

# =====================================================
# STEP 5: Closed-loop NR-CLS simulation
# =====================================================
echo "[Step 5] Closed-loop NR-CLS simulation..." | tee -a $LOG

MERGED_CKPT="$SDE_DPO_DIR/model_sde_dpo_merged.pth"
SIM_SCRIPT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages/nuplan/planning/script/run_simulation.py
HYDRA_SEARCH="[pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"

if [ -f "$MERGED_CKPT" ]; then
    echo "[Closed-Loop NR-CLS: val14, 500 scenarios]" >> $REPORT

    python $SIM_SCRIPT \
        +simulation=closed_loop_nonreactive_agents \
        planner=flow_planner \
        planner.flow_planner.config_path=/root/autodl-tmp/Flow-Planner/checkpoints/model_config.yaml \
        planner.flow_planner.ckpt_path=/root/autodl-tmp/Flow-Planner/$MERGED_CKPT \
        planner.flow_planner.use_cfg=true \
        planner.flow_planner.cfg_weight=1.8 \
        planner.flow_planner.device=cuda \
        scenario_builder=nuplan \
        scenario_builder.data_root=/root/autodl-tmp/val_data/data/cache/val \
        scenario_filter=val14 \
        scenario_filter.limit_total_scenarios=500 \
        experiment_name=sde_dpo_cl \
        output_dir=/root/autodl-tmp/sde_dpo_eval \
        worker=ray_distributed worker.threads_per_node=4 \
        distributed_mode=SINGLE_NODE \
        number_of_gpus_allocated_per_simulation=0.25 \
        exit_on_failure=false \
        hydra.searchpath="$HYDRA_SEARCH" \
        2>&1 | tee /root/sde_dpo_sim.log

    echo "  SDE+DPO closed-loop results:" >> $REPORT
    grep -E "final_score|Score|collision" /root/sde_dpo_sim.log | tail -10 >> $REPORT 2>/dev/null

    if grep -q "InterpolationKeyError" /root/sde_dpo_sim.log 2>/dev/null; then
        echo "  [WARN] Closed-loop failed due to config interpolation error" >> $REPORT
    fi
else
    echo "  [SKIP] No merged model found, skipping closed-loop" >> $REPORT
fi

echo "" >> $REPORT

# =====================================================
# STEP 6: Training summary + shutdown
# =====================================================
echo "[Training Logs]" >> $REPORT
grep "Epoch .*/2 |" /root/sde_dpo_train.log >> $REPORT 2>/dev/null
grep -A3 "\[.*collision\]" /root/sde_dpo_train.log | tail -9 >> $REPORT 2>/dev/null
echo "" >> $REPORT

echo "[Mining Stats]" >> $REPORT
grep "AvgDiv:" $LOG | tail -3 >> $REPORT 2>/dev/null
grep "Multi-Objective Mining Complete" -A10 $LOG >> $REPORT 2>/dev/null
echo "" >> $REPORT

echo "[All Checkpoints]" >> $REPORT
ls -lh $SDE_DPO_DIR/*.pt $SDE_DPO_DIR/*.pth 2>/dev/null >> $REPORT
echo "" >> $REPORT

echo "=== Completed at $(date) ===" >> $REPORT
echo "=== Completed at $(date) ===" >> $LOG

cat $REPORT

echo "Shutting down in 60 seconds..." | tee -a $LOG
sleep 60
shutdown now 2>/dev/null || poweroff 2>/dev/null

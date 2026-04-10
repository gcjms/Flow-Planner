#!/bin/bash
# Multi-Objective DPO Pipeline
# =============================
# Step 1: Mine multi-objective preference pairs (5 seeds x 4 CFG weights = 20 candidates/scene)
# Step 2: Train DPO with dimension-weighted loss
# Step 3: Merge LoRA weights
# Step 4: Multi-dimensional open-loop evaluation
# Step 5: Closed-loop NR-CLS simulation
# Step 6: Generate report + shutdown

set -o pipefail
LOG=/root/multiobjective_pipeline.log
REPORT=/root/multiobjective_eval_report.txt

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

echo "=== Multi-Objective DPO Pipeline started at $(date) ===" | tee $LOG

export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache

# =====================================================
# STEP 1: Mine multi-objective preference pairs
# =====================================================
echo "[Step 1] Mining multi-objective preference pairs..." | tee -a $LOG
echo "  5000 scenes x 20 candidates (5 seeds x 4 CFG) each" | tee -a $LOG

MO_DATA=/root/autodl-tmp/multiobjective_pairs.npz

python -u -m flow_planner.dpo.generate_multiobjective_pairs \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --output_path $MO_DATA \
    --max_scenes 5000 \
    --num_seeds 5 \
    --cfg_weights 0.5,1.0,1.8,3.0 \
    --target_dims collision,ttc,comfort \
    --score_gap_threshold 0.15 \
    --device cuda \
    2>&1 | tee -a $LOG

if [ ! -f "$MO_DATA" ]; then
    echo "[Step 1] FAILED: No multi-objective pairs generated!" | tee -a $LOG
    echo "========================================" > $REPORT
    echo "  Multi-Objective DPO Report (FAILED)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "Mining produced 0 multi-objective pairs." >> $REPORT
    tail -30 $LOG >> $REPORT
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

N_PAIRS=$(python -c "import numpy as np; d=np.load('$MO_DATA', allow_pickle=True); print(d['chosen'].shape[0])")
echo "[Step 1] Found $N_PAIRS multi-objective pairs" | tee -a $LOG

if [ "$N_PAIRS" -lt 20 ]; then
    echo "[Step 1] Too few pairs ($N_PAIRS < 20), aborting" | tee -a $LOG
    echo "========================================" > $REPORT
    echo "  Multi-Objective DPO Report (INSUFFICIENT DATA)" >> $REPORT
    echo "  $(date)" >> $REPORT
    echo "========================================" >> $REPORT
    echo "Only found $N_PAIRS pairs (need >= 20)" >> $REPORT
    tail -30 $LOG >> $REPORT
    cat $REPORT
    shutdown now 2>/dev/null || poweroff 2>/dev/null
    exit 1
fi

echo "[Step 1] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 2: DPO Training (multi-objective data)
# =====================================================
echo "[Step 2] Training DPO with multi-objective pairs..." | tee -a $LOG

MO_DIR=checkpoints/dpo_multiobjective
mkdir -p $MO_DIR

python -u -m flow_planner.dpo.train_dpo \
    --config_path checkpoints/model_config.yaml \
    --ckpt_path checkpoints/model.pth \
    --preference_path $MO_DATA \
    --output_dir $MO_DIR \
    --beta 5.0 \
    --epochs 2 \
    --lr 5e-5 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --batch_size 8 \
    --sft_weight 0.1 \
    --num_t_samples 16 \
    --dim_weights "collision:5,ttc:5,comfort:2" \
    --save_merged \
    2>&1 | tee /root/dpo_multiobjective_train.log

echo "[Step 2] DPO training done at $(date)" | tee -a $LOG

# =====================================================
# STEP 3: Merge LoRA weights
# =====================================================
echo "[Step 3] Merging multi-objective LoRA weights..." | tee -a $LOG

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
lora_path = '$MO_DIR/lora_best.pt'
if not os.path.exists(lora_path):
    lora_path = '$MO_DIR/lora_epoch_2.pt'
if not os.path.exists(lora_path):
    lora_path = '$MO_DIR/lora_epoch_1.pt'
print(f'Loading {lora_path}')
load_lora(model.model_decoder, lora_path)
merge_lora(model.model_decoder)
raw_sd = model.state_dict()
fixed_sd = {k.replace('module.', ''): v for k, v in raw_sd.items()}
torch.save({'state_dict': fixed_sd}, '$MO_DIR/model_mo_merged.pth')
print('Merged model saved!')
" 2>&1 | tee -a $LOG

echo "[Step 3] Done at $(date)" | tee -a $LOG

# =====================================================
# STEP 4: Multi-dimensional open-loop evaluation
# =====================================================
echo "[Step 4] Multi-dimensional open-loop evaluation..." | tee -a $LOG

echo "========================================" > $REPORT
echo "  Multi-Objective DPO Evaluation Report" >> $REPORT
echo "  $(date)" >> $REPORT
echo "========================================" >> $REPORT
echo "" >> $REPORT

echo "[Mining Results]" >> $REPORT
echo "  Multi-objective pairs found: $N_PAIRS" >> $REPORT
grep "Per-dimension" $LOG >> $REPORT 2>/dev/null
grep "Avg trajectory diversity" $LOG >> $REPORT 2>/dev/null
echo "" >> $REPORT

echo "[Open-Loop Evaluation: 500 Hard Scenarios]" >> $REPORT

for MODEL_INFO in \
    "Original:checkpoints/model.pth" \
    "DPO_MultiObj:$MO_DIR/model_mo_merged.pth"; do
    NAME="${MODEL_INFO%%:*}"
    CKPT="${MODEL_INFO##*:}"
    if [ -f "$CKPT" ]; then
        echo "Evaluating $NAME..." | tee -a $LOG
        RESULT=$(python -u -m flow_planner.dpo.eval_multidim \
            --ckpt_path $CKPT \
            --config_path checkpoints/model_config.yaml \
            --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
            --max_scenes 500 2>&1 | grep -E "SUMMARY|collision_rate|avg_ttc|avg_comfort|avg_progress")
        echo "  $NAME:" | tee -a $LOG
        echo "$RESULT" | tee -a $LOG
        echo "  $NAME:" >> $REPORT
        echo "$RESULT" >> $REPORT
    fi
done

echo "" >> $REPORT

# =====================================================
# STEP 5: Closed-loop NR-CLS simulation
# =====================================================
echo "[Step 5] Closed-loop simulation..." | tee -a $LOG

MERGED_CKPT="$MO_DIR/model_mo_merged.pth"
SIM_SCRIPT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages/nuplan/planning/script/run_simulation.py
HYDRA_SEARCH="[pkg://flow_planner.nuplan_simulation.scenario_filter,pkg://flow_planner.nuplan_simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"

if [ -f "$MERGED_CKPT" ]; then
    echo "[Closed-Loop: val14, 500 scenarios]" >> $REPORT

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
        experiment_name=dpo_multiobjective_cl \
        output_dir=/root/autodl-tmp/dpo_multiobjective_eval \
        worker=ray_distributed worker.threads_per_node=4 \
        distributed_mode=SINGLE_NODE \
        number_of_gpus_allocated_per_simulation=0.25 \
        exit_on_failure=false \
        hydra.searchpath="$HYDRA_SEARCH" \
        2>&1 | tee /root/dpo_multiobjective_sim.log

    echo "  DPO multi-objective closed-loop results:" >> $REPORT
    grep -E "final_score|Score|collision" /root/dpo_multiobjective_sim.log | tail -10 >> $REPORT 2>/dev/null

    if grep -q "InterpolationKeyError" /root/dpo_multiobjective_sim.log 2>/dev/null; then
        echo "  [WARN] Closed-loop failed due to config interpolation error" >> $REPORT
    fi
else
    echo "  [SKIP] No merged model found, skipping closed-loop" >> $REPORT
fi

echo "" >> $REPORT

# =====================================================
# STEP 6: Training summary + shutdown
# =====================================================
echo "[Training Summary]" >> $REPORT
echo "Multi-Objective DPO (beta=5.0, rank=4, 2 epochs, dim_weights=collision:5,ttc:5,comfort:2):" >> $REPORT
grep "Epoch .*/2 |" /root/dpo_multiobjective_train.log >> $REPORT 2>/dev/null
grep -A3 "\[.*collision\]" /root/dpo_multiobjective_train.log | tail -9 >> $REPORT 2>/dev/null
echo "" >> $REPORT

echo "[All Checkpoints]" >> $REPORT
ls -lh $MO_DIR/*.pt $MO_DIR/*.pth 2>/dev/null >> $REPORT

echo "" >> $REPORT
echo "=== Completed at $(date) ===" >> $REPORT
echo "=== Completed at $(date) ===" >> $LOG

cat $REPORT

echo "Shutting down in 60 seconds..." | tee -a $LOG
sleep 60
shutdown now 2>/dev/null || poweroff 2>/dev/null

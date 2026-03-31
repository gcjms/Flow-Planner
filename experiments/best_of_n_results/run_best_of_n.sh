#!/bin/bash
set -e
export NUPLAN_DEVKIT_ROOT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages
export NUPLAN_DATA_ROOT=/root/autodl-tmp/val_data/data/cache/val
export NUPLAN_MAPS_ROOT=/root/autodl-tmp/maps_raw/maps
export NUPLAN_EXP_ROOT=/root/autodl-tmp/best_of_n_output
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/root/autodl-tmp/ray_tmp
CONFIG_FILE=/root/Flow-Planner/checkpoints/model_config.yaml
CKPT_FILE=/root/Flow-Planner/checkpoints/model.pth
PYTHON=/root/miniconda3/envs/flow_planner/bin/python
NUM_CANDIDATES=5
CFG_WEIGHT=1.8

echo "==========================================="
echo "Best-of-N (N=5) - ray_distributed threads=4 gpu=0.25"
echo "Start: $(date)"
echo "==========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

rm -rf "$NUPLAN_EXP_ROOT"
mkdir -p "$NUPLAN_EXP_ROOT"

$PYTHON $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=flow_planner \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    planner.flow_planner.cfg_weight=$CFG_WEIGHT \
    planner.flow_planner.num_candidates=$NUM_CANDIDATES \
    planner.flow_planner.device=cuda \
    planner.flow_planner.use_cfg=true \
    scenario_builder=nuplan \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT \
    scenario_filter=val14 \
    output_dir=$NUPLAN_EXP_ROOT \
    experiment_name=closed_loop_nonreactive_agents_best_of_5 \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    distributed_mode=SINGLE_NODE \
    number_of_gpus_allocated_per_simulation=0.25 \
    enable_simulation_progress_bar=true \
    exit_on_failure=false \
    "hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    2>&1 | tee "$NUPLAN_EXP_ROOT/simulation.log"

echo ""
echo "[$(date)] Simulation complete!"
$PYTHON /root/scripts/compute_nr_cls_bon.py
echo "[$(date)] All done!"

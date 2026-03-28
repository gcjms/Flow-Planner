#!/bin/bash
# =============================================================================
# Per-Scenario-Type CFG Weight Grid Search
# =============================================================================
# 对 w ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} 分别跑闭环仿真（val14, 1118 场景）
# 跑完后用 analyze_w_by_scenario.py 分析每种场景类型的 NR Score
#
# 用法:
#   bash scripts/run_w_grid_search.sh          # 完整实验 (val14, ~18-24h)
#   bash scripts/run_w_grid_search.sh --debug  # 快速验证 (6场景, ~10min)
# =============================================================================

set -e

# 解析命令行参数
DEBUG_MODE=false
for arg in "$@"; do
    case $arg in
        --debug) DEBUG_MODE=true ;;
    esac
done

# ============================================
# 用户配置区（根据你的环境修改）
# ============================================
# AutoDL 环境配置
export NUPLAN_DEVKIT_ROOT=/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages
export NUPLAN_DATA_ROOT=/root/autodl-tmp/nuplan/dataset
export NUPLAN_MAPS_ROOT=/root/autodl-tmp/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/root/autodl-tmp/w_grid_search_output
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# 模型路径 - 根据实际情况修改
CONFIG_FILE=/root/autodl-tmp/training_output_combined/outputs/FlowPlannerTraining/flow_planner_standard/2026-03-21_14-00-42/.hydra/config.yaml
CKPT_FILE=/root/autodl-tmp/training_output_combined/outputs/FlowPlannerTraining/flow_planner_standard/2026-03-21_14-00-42/epoch_120.pth

# Python 路径
PYTHON=/root/miniconda3/envs/flow_planner/bin/python

# w 候选值
W_VALUES=(0.5 1.0 1.5 2.0 2.5 3.0)

# 仿真配置
if [ "$DEBUG_MODE" = true ]; then
    SPLIT=debug_w_grid
    echo "[DEBUG MODE] Using debug scenario filter (6 scenarios)"
else
    SPLIT=val14
fi
CHALLENGE=closed_loop_nonreactive_agents
PLANNER=flow_planner
SCENARIO_BUILDER=nuplan
# 128 CPU + 独占 4090 → 最大化并行度
# threads_per_node = 32 可以让 ray 并行跑 32 个场景
# number_of_gpus_allocated_per_simulation = 0.125 → 允许 8 个 worker 共享 GPU
THREADS=32
GPU_PER_SIM=0.125
# ============================================

echo "=========================================="
echo "Per-Scenario-Type CFG Weight Grid Search"
echo "=========================================="
echo "W values: ${W_VALUES[*]}"
echo "Split: $SPLIT"
echo "Total scenarios: ~1118"
echo "Estimated time: ~3-4 hours per w value"
echo "=========================================="

# 激活 conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

for W in "${W_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date)] Starting simulation with w=$W"
    echo "=========================================="

    OUTPUT_DIR=${NUPLAN_EXP_ROOT}/w_${W}

    # 跳过已完成的 w
    METRICS_DIR="${OUTPUT_DIR}/metrics"
    if [ -d "$METRICS_DIR" ]; then
        METRIC_COUNT=$(find "$METRICS_DIR" -name "*.pickle*" 2>/dev/null | wc -l)
        if [ "$METRIC_COUNT" -ge 1000 ]; then
            echo "[SKIP] w=$W already completed ($METRIC_COUNT metrics found)"
            continue
        fi
    fi

    # 清理之前未完成的结果
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    $PYTHON $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
        +simulation=$CHALLENGE \
        planner=$PLANNER \
        planner.flow_planner.config_path=$CONFIG_FILE \
        planner.flow_planner.ckpt_path=$CKPT_FILE \
        planner.flow_planner.cfg_weight=$W \
        scenario_builder=$SCENARIO_BUILDER \
        scenario_builder.data_root=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \
        scenario_filter=$SPLIT \
        output_dir=$OUTPUT_DIR \
        experiment_uid="w_grid_search/w_${W}" \
        verbose=true \
        worker=ray_distributed \
        worker.threads_per_node=$THREADS \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=$GPU_PER_SIM \
        enable_simulation_progress_bar=true \
        hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
        2>&1 | tee "${OUTPUT_DIR}/simulation.log"

    echo "[$(date)] Completed w=$W"
done

echo ""
echo "=========================================="
echo "[$(date)] All simulations complete!"
echo "=========================================="
echo "Running analysis..."

# 运行分析脚本
$PYTHON /root/Flow-Planner/scripts/analyze_w_by_scenario.py \
    --results_dir $NUPLAN_EXP_ROOT \
    --output_file $NUPLAN_EXP_ROOT/w_grid_search_report.md

echo "[$(date)] Analysis complete! Report: $NUPLAN_EXP_ROOT/w_grid_search_report.md"

# 自动关机
echo "[$(date)] Shutting down in 30 seconds..."
sleep 30
shutdown -h now

#!/bin/bash
# ============================================================
# FlowPlanner 闭环仿真验证 — 一键脚本
# 自动运行 w=1.0,1.5,1.8,2.0,2.5,3.0 六种配置
# ============================================================
#
# 使用前请修改以下路径：
# ============================================================

# === 必须修改的路径 ===
NUPLAN_DEVKIT="/root/nuplan-devkit"          # nuplan-devkit 安装路径
NUPLAN_DATA_ROOT="/root/nuplan/dataset"      # nuPlan 数据集根目录
NUPLAN_MAPS_ROOT="/root/nuplan/maps"         # nuPlan 地图目录

# === 通常不需要修改 ===
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT_PATH="$PROJECT_ROOT/checkpoints/model.pth"
CONFIG_PATH="$PROJECT_ROOT/checkpoints/model_config.yaml"
OUTPUT_DIR="$PROJECT_ROOT/simulation_outputs"
RESULTS_DIR="$PROJECT_ROOT/cfg_validation_results"

# === CFG 权重列表 ===
W_VALUES="1.0 1.5 1.8 2.0 2.5 3.0"

# === 仿真参数 ===
SCENARIO_FILTER="val14_split"    # 完整验证集
# SCENARIO_FILTER="val14_split"  # 用这个跑 mini 快速验证
DEVICE="cuda"

# ============================================================

set -e

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/validation_log.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# === 环境检查 ===
log "============================================================"
log "FlowPlanner Closed-Loop CFG Validation"
log "============================================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi failed. No GPU detected."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
log "GPU: $GPU_INFO"

# Check model checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    log "ERROR: Model checkpoint not found at $CKPT_PATH"
    log "Please download from HuggingFace: ttwhy/flow-planner"
    exit 1
fi
log "Checkpoint: $CKPT_PATH"

# Check nuplan-devkit
if [ ! -d "$NUPLAN_DEVKIT" ]; then
    log "ERROR: nuplan-devkit not found at $NUPLAN_DEVKIT"
    log "Please clone: git clone https://github.com/motional/nuplan-devkit.git"
    exit 1
fi
log "nuplan-devkit: $NUPLAN_DEVKIT"

# Check data
if [ ! -d "$NUPLAN_DATA_ROOT" ]; then
    log "ERROR: nuPlan data not found at $NUPLAN_DATA_ROOT"
    log "Please download Val split from https://www.nuscenes.org/nuplan"
    exit 1
fi
log "Data: $NUPLAN_DATA_ROOT"
log "Maps: $NUPLAN_MAPS_ROOT"
log "Scenario filter: $SCENARIO_FILTER"
log "W values: $W_VALUES"
log "============================================================"

# === 激活环境 ===
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flow_planner 2>/dev/null || {
    log "WARNING: flow_planner env not found, using current environment"
}

export NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT
export PROJECT_ROOT

# === 依次运行每个 w 值 ===
for W in $W_VALUES; do
    EXPERIMENT_NAME="cfg_w${W}"
    RESULT_FILE="$RESULTS_DIR/cfg_${W}_results.txt"

    log ""
    log "============================================================"
    log "Running w=$W  (experiment: $EXPERIMENT_NAME)"
    log "============================================================"

    START_TIME=$(date +%s)

    python "$NUPLAN_DEVKIT/nuplan/planning/script/run_simulation.py" \
        +simulation=closed_loop_nonreactive_agents \
        planner=flow_planner \
        planner.flow_planner.config_path="$CONFIG_PATH" \
        planner.flow_planner.ckpt_path="$CKPT_PATH" \
        planner.flow_planner.use_cfg=true \
        planner.flow_planner.cfg_weight="$W" \
        planner.flow_planner.device="$DEVICE" \
        scenario_builder=nuplan \
        scenario_filter="$SCENARIO_FILTER" \
        experiment_name="$EXPERIMENT_NAME" \
        output_dir="$OUTPUT_DIR" \
        2>&1 | tee "$RESULT_FILE"

    END_TIME=$(date +%s)
    DURATION=$(( (END_TIME - START_TIME) / 60 ))
    log "w=$W done in ${DURATION} minutes"

    # 提取最终分数
    FINAL_SCORE=$(grep "final_score" "$RESULT_FILE" | tail -1)
    if [ -n "$FINAL_SCORE" ]; then
        log "w=$W result: $FINAL_SCORE"
    fi
done

# === 汇总结果 ===
log ""
log "============================================================"
log "ALL VALIDATIONS COMPLETE - SUMMARY"
log "============================================================"

echo ""
echo "cfg_weight | Overall NR Score | Collision-Free | Drivable | Detail Line"
echo "---------- | ---------------- | -------------- | -------- | -----------"

for W in $W_VALUES; do
    RESULT_FILE="$RESULTS_DIR/cfg_${W}_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        FINAL=$(grep "final_score" "$RESULT_FILE" | tail -1)
        if [ -n "$FINAL" ]; then
            # 解析: final_score|1115.0|0.8119|0.9381|0.9345|0.8565
            IFS='|' read -ra PARTS <<< "$FINAL"
            SCORE=${PARTS[2]:-"N/A"}
            COLLISION=${PARTS[3]:-"N/A"}
            DRIVABLE=${PARTS[4]:-"N/A"}
            echo "w=$W      | $SCORE             | $COLLISION       | $DRIVABLE | $FINAL"
        else
            echo "w=$W      | PARSE ERROR | - | - | (check $RESULT_FILE)"
        fi
    else
        echo "w=$W      | NOT RUN | - | - | -"
    fi
done | tee -a "$LOG_FILE"

log ""
log "Results saved to: $RESULTS_DIR/"
log "Done!"

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

# === CFG 权重列表（8 个值，精细扫描） ===
W_VALUES="0.5 1.0 1.5 1.8 2.0 2.5 3.0 4.0"

# === 仿真参数 ===
SCENARIO_FILTER="val14_split"    # 完整验证集（~1115 场景）
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

# === 先跑 no-CFG baseline ===
log ""
log "============================================================"
log "Running BASELINE (no CFG, use_cfg=false)"
log "============================================================"

RESULT_FILE="$RESULTS_DIR/cfg_no_cfg_results.txt"
START_TIME=$(date +%s)

python "$NUPLAN_DEVKIT/nuplan/planning/script/run_simulation.py" \
    +simulation=closed_loop_nonreactive_agents \
    planner=flow_planner \
    planner.flow_planner.config_path="$CONFIG_PATH" \
    planner.flow_planner.ckpt_path="$CKPT_PATH" \
    planner.flow_planner.use_cfg=false \
    planner.flow_planner.cfg_weight=1.0 \
    planner.flow_planner.device="$DEVICE" \
    scenario_builder=nuplan \
    scenario_filter="$SCENARIO_FILTER" \
    experiment_name="cfg_no_cfg" \
    output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$RESULT_FILE"

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))
log "no_cfg done in ${DURATION} minutes"

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

# ============================================================
# 汇总结果
# ============================================================
log ""
log "============================================================"
log "ALL VALIDATIONS COMPLETE - SUMMARY"
log "============================================================"

SUMMARY_FILE="$RESULTS_DIR/summary.md"
echo "# CFG Weight Closed-Loop Validation Results" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "| cfg_weight | Overall NR | Collision-Free | Drivable Area | TTC/Comfort |" >> "$SUMMARY_FILE"
echo "|:---:|:---:|:---:|:---:|:---:|" >> "$SUMMARY_FILE"

ALL_CONFIGS="no_cfg $W_VALUES"
for W in $ALL_CONFIGS; do
    RESULT_FILE="$RESULTS_DIR/cfg_${W}_results.txt"
    if [ -f "$RESULT_FILE" ]; then
        FINAL=$(grep "final_score" "$RESULT_FILE" | tail -1)
        if [ -n "$FINAL" ]; then
            IFS='|' read -ra PARTS <<< "$FINAL"
            SCORE=${PARTS[2]:-"N/A"}
            COLLISION=${PARTS[3]:-"N/A"}
            DRIVABLE=${PARTS[4]:-"N/A"}
            TTC=${PARTS[5]:-"N/A"}
            echo "| w=$W | $SCORE | $COLLISION | $DRIVABLE | $TTC |" >> "$SUMMARY_FILE"
            echo "w=$W | NR=$SCORE | Collision=$COLLISION | Drivable=$DRIVABLE | TTC=$TTC"
        fi
    fi
done | tee -a "$LOG_FILE"

# === 场景类型细分分析 ===
echo "" >> "$SUMMARY_FILE"
echo "## Per-Scenario-Type Breakdown" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

SCENARIO_TYPES="changing_lane starting_left_turn starting_right_turn high_lateral_acceleration near_multiple_vehicles stopping_with_lead stationary_in_traffic high_magnitude_speed"

for TYPE in $SCENARIO_TYPES; do
    echo "" >> "$SUMMARY_FILE"
    echo "### $TYPE" >> "$SUMMARY_FILE"
    echo "| cfg_weight | NR Score | Count |" >> "$SUMMARY_FILE"
    echo "|:---:|:---:|:---:|" >> "$SUMMARY_FILE"

    for W in $ALL_CONFIGS; do
        RESULT_FILE="$RESULTS_DIR/cfg_${W}_results.txt"
        if [ -f "$RESULT_FILE" ]; then
            TYPE_LINE=$(grep "^${TYPE}|" "$RESULT_FILE" | tail -1)
            if [ -n "$TYPE_LINE" ]; then
                IFS='|' read -ra PARTS <<< "$TYPE_LINE"
                COUNT=${PARTS[1]:-"?"}
                SCORE=${PARTS[2]:-"?"}
                echo "| w=$W | $SCORE | $COUNT |" >> "$SUMMARY_FILE"
            fi
        fi
    done
done

echo "" >> "$SUMMARY_FILE"
echo "## Conclusion" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "If the NR scores across different w values differ by < 1%, then adaptive CFG is NOT worthwhile." >> "$SUMMARY_FILE"
echo "If specific scenario types (e.g., left turns, lane changes) show > 2% difference, then scenario-specific w could be valuable." >> "$SUMMARY_FILE"

log ""
log "Summary saved to: $SUMMARY_FILE"
log "Full results in: $RESULTS_DIR/"
log ""
log "============================================================"
log "DONE! Total configs tested: $(echo $ALL_CONFIGS | wc -w)"
log "============================================================"


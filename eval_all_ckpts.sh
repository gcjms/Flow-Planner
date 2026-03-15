#!/bin/bash
# ============================================================
# 批量评估所有 checkpoint，找出最优模型
# 用法: bash eval_all_ckpts.sh [训练输出目录]
# ============================================================
set -e

# 配置
PYTHON="/home/gcjms/miniconda3/envs/flow_planner/bin/python"
PROJECT_ROOT="/home/gcjms/Flow-Planner"
DATA_DIR="/home/gcjms/nuplan/dataset/processed_boston_npz"
VAL_JSON="$DATA_DIR/flow_planner_val.json"
CFG_WEIGHT=1.8

# 获取最新的训练输出目录
if [ -n "$1" ]; then
    CKPT_DIR="$1"
else
    CKPT_DIR=$(ls -td "$PROJECT_ROOT/training_output/outputs/FlowPlannerTraining/flow_planner_standard/"*/ | head -1)
fi

CONFIG_PATH="$CKPT_DIR/.hydra/config.yaml"
RESULT_FILE="$CKPT_DIR/eval_results_summary.txt"

echo "=================================================="
echo " 批量 Checkpoint 评估"
echo "=================================================="
echo "  训练目录:  $CKPT_DIR"
echo "  验证数据:  $VAL_JSON"
echo "  CFG Weight: $CFG_WEIGHT"
echo "=================================================="

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 找不到 config: $CONFIG_PATH"
    exit 1
fi

# 初始化结果文件
echo "Checkpoint Evaluation Results" > "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

BEST_ADE=99999
BEST_CKPT=""

# 遍历所有 checkpoint（按 epoch 排序）
for ckpt in $(ls "$CKPT_DIR"/model_epoch_*.pth 2>/dev/null | sort -t_ -k3 -n); do
    CKPT_NAME=$(basename "$ckpt")
    EPOCH=$(echo "$CKPT_NAME" | grep -oP 'epoch_\K[0-9]+')
    
    echo ""
    echo "--- 评估 $CKPT_NAME ---"
    
    # 运行评估
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0 \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TRAINING_DATA="$DATA_DIR" \
        $PYTHON "$PROJECT_ROOT/eval_open_loop.py" \
            --config_path "$CONFIG_PATH" \
            --ckpt_path "$ckpt" \
            --data_dir "$DATA_DIR" \
            --data_list "$VAL_JSON" \
            --cfg_weight $CFG_WEIGHT \
            2>&1)
    
    # 提取指标
    ADE=$(echo "$OUTPUT" | grep "ADE (m)" | head -1 | awk '{print $NF}')
    FDE=$(echo "$OUTPUT" | grep "FDE (m)" | head -1 | awk '{print $NF}')
    ADE_1s=$(echo "$OUTPUT" | grep "ADE@1s" | awk '{print $NF}')
    ADE_3s=$(echo "$OUTPUT" | grep "ADE@3s" | awk '{print $NF}')
    HEADING=$(echo "$OUTPUT" | grep "Heading Error (deg)" | awk '{print $NF}')
    
    echo "  ADE=$ADE  FDE=$FDE  ADE@1s=$ADE_1s  ADE@3s=$ADE_3s  Heading=${HEADING}°"
    
    # 写入结果
    printf "Epoch %3s | ADE=%-8s FDE=%-8s ADE@1s=%-8s ADE@3s=%-8s Heading=%-8s\n" \
        "$EPOCH" "$ADE" "$FDE" "$ADE_1s" "$ADE_3s" "$HEADING" >> "$RESULT_FILE"
    
    # 比较找最优
    IS_BETTER=$(echo "$ADE < $BEST_ADE" | bc -l 2>/dev/null || echo "0")
    if [ "$IS_BETTER" -eq 1 ]; then
        BEST_ADE="$ADE"
        BEST_CKPT="$CKPT_NAME"
        BEST_EPOCH="$EPOCH"
    fi
done

# 也评估 latest.pth
if [ -f "$CKPT_DIR/latest.pth" ]; then
    echo ""
    echo "--- 评估 latest.pth ---"
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0 \
        PROJECT_ROOT="$PROJECT_ROOT" \
        TRAINING_DATA="$DATA_DIR" \
        $PYTHON "$PROJECT_ROOT/eval_open_loop.py" \
            --config_path "$CONFIG_PATH" \
            --ckpt_path "$CKPT_DIR/latest.pth" \
            --data_dir "$DATA_DIR" \
            --data_list "$VAL_JSON" \
            --cfg_weight $CFG_WEIGHT \
            2>&1)
    
    ADE=$(echo "$OUTPUT" | grep "ADE (m)" | head -1 | awk '{print $NF}')
    FDE=$(echo "$OUTPUT" | grep "FDE (m)" | head -1 | awk '{print $NF}')
    echo "  ADE=$ADE  FDE=$FDE"
    printf "latest    | ADE=%-8s FDE=%-8s\n" "$ADE" "$FDE" >> "$RESULT_FILE"
fi

# 输出总结
echo "" >> "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"
echo "🏆 Best: Epoch $BEST_EPOCH ($BEST_CKPT) | ADE=$BEST_ADE" >> "$RESULT_FILE"

echo ""
echo "=================================================="
echo "🏆 最优 Checkpoint: Epoch $BEST_EPOCH"
echo "   文件: $BEST_CKPT"
echo "   ADE:  $BEST_ADE"
echo "=================================================="
echo ""
echo "完整结果已保存: $RESULT_FILE"
cat "$RESULT_FILE"

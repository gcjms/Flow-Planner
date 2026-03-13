#!/bin/bash
# ============================================================
# Flow-Planner 一键: 数据处理 → 训练 → 评估
# 适用于: AutoDL 单卡 4090
#
# 用法: bash run_all.sh [--epochs 200] [--batch_size 32]
#                       [--skip_data_process] [--skip_train]
# ============================================================
set -e

# ==================== 配置区 ====================
# 路径配置 (根据你的 AutoDL 环境修改)
PROJECT_ROOT="$HOME/Flow-Planner"
NUPLAN_DATA="$HOME/nuplan/dataset/nuplan-v1.1/splits/mini"
NUPLAN_MAPS="$HOME/nuplan/dataset/maps"
NPZ_OUTPUT="$HOME/nuplan/dataset/processed_npz"
TRAINING_OUTPUT="$PROJECT_ROOT/training_output"

# 训练配置
EPOCHS=200
BATCH_SIZE=32
SAVE_EVERY=20         # 每N个epoch保存一次checkpoint
VAL_RATIO=0.2         # 验证集比例
SEED=42
CFG_WEIGHT=1.8

# Python 路径
PYTHON="python"
# 如果用 conda 环境:
# PYTHON="$HOME/miniconda3/envs/flow_planner/bin/python"

# ==================== 解析命令行参数 ====================
SKIP_DATA_PROCESS=false
SKIP_TRAIN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --skip_data_process) SKIP_DATA_PROCESS=true; shift ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        --val_ratio) VAL_RATIO="$2"; shift 2 ;;
        --cfg_weight) CFG_WEIGHT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ==================== 环境变量 ====================
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT
export SAVE_DIR="$TRAINING_OUTPUT"
export TENSORBOARD_LOG_PATH="$TRAINING_OUTPUT/tb_logs"
export TRAINING_DATA="$NPZ_OUTPUT"

echo "=========================================="
echo "Flow-Planner 全流程脚本"
echo "=========================================="
echo "  项目路径:    $PROJECT_ROOT"
echo "  nuPlan数据:  $NUPLAN_DATA"
echo "  NPZ输出:     $NPZ_OUTPUT"
echo "  训练输出:    $TRAINING_OUTPUT"
echo "  Epochs:      $EPOCHS"
echo "  Batch Size:  $BATCH_SIZE"
echo "  Val Ratio:   $VAL_RATIO"
echo "=========================================="

# ==================== Step 1: 数据处理 ====================
if [ "$SKIP_DATA_PROCESS" = false ]; then
    echo ""
    echo "[Step 1/4] 数据处理: .db → .npz"
    echo "-------------------------------------------"
    
    mkdir -p "$NPZ_OUTPUT"
    
    $PYTHON "$PROJECT_ROOT/data_process.py" \
        --data_path "$NUPLAN_DATA" \
        --map_path "$NUPLAN_MAPS" \
        --save_dir "$NPZ_OUTPUT" \
        --map_version "nuplan-maps-v1.0"
    
    echo "✅ 数据处理完成: $(ls "$NPZ_OUTPUT"/*.npz 2>/dev/null | wc -l) 个 npz 文件"
else
    echo ""
    echo "[Step 1/4] 跳过数据处理 (--skip_data_process)"
fi

# ==================== Step 2: Train/Val 切分 ====================
echo ""
echo "[Step 2/4] 数据集切分: train/val"
echo "-------------------------------------------"

# 查找训练JSON (可能已有或需要生成)
FULL_JSON="$NPZ_OUTPUT/flow_planner_training.json"
TRAIN_JSON="$NPZ_OUTPUT/flow_planner_train.json"
VAL_JSON="$NPZ_OUTPUT/flow_planner_val.json"

$PYTHON -c "
import json, random, os, glob

random.seed($SEED)

# 尝试加载已有的 JSON
if os.path.exists('$FULL_JSON'):
    with open('$FULL_JSON') as f:
        all_files = json.load(f)
else:
    # 自动从 npz 目录生成
    all_files = sorted([os.path.basename(f) for f in glob.glob('$NPZ_OUTPUT/*.npz')])
    with open('$FULL_JSON', 'w') as f:
        json.dump(all_files, f, indent=4)

random.shuffle(all_files)
split_idx = int(len(all_files) * (1 - $VAL_RATIO))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

with open('$TRAIN_JSON', 'w') as f:
    json.dump(train_files, f, indent=4)
with open('$VAL_JSON', 'w') as f:
    json.dump(val_files, f, indent=4)

print(f'  Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}')
"

export TRAINING_JSON="$TRAIN_JSON"

# ==================== Step 3: 训练 ====================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[Step 3/4] 开始训练 ($EPOCHS epochs, batch_size=$BATCH_SIZE)"
    echo "-------------------------------------------"
    
    cd "$PROJECT_ROOT/flow_planner/run_script"
    
    $PYTHON -m torch.distributed.run \
        --nnodes 1 --nproc-per-node 1 --standalone \
        ../trainer.py --config-name flow_planner_standard \
        train.epoch=$EPOCHS \
        train.batch_size=$BATCH_SIZE \
        train.save_utd=$SAVE_EVERY \
        save_every_since=0
    
    cd "$PROJECT_ROOT"
    echo "✅ 训练完成"
else
    echo ""
    echo "[Step 3/4] 跳过训练 (--skip_train)"
fi

# ==================== Step 4: 评估 ====================
echo ""
echo "[Step 4/4] Open-Loop 评估 (ADE/FDE)"
echo "-------------------------------------------"

# 找到最新的训练输出目录
LATEST_CONFIG=$(find "$TRAINING_OUTPUT/outputs" -name "config.yaml" -path "*/.hydra/*" 2>/dev/null | sort | tail -1)
LATEST_CKPT_DIR=$(dirname "$(dirname "$LATEST_CONFIG")")
LATEST_CKPT="$LATEST_CKPT_DIR/latest.pth"

if [ ! -f "$LATEST_CKPT" ]; then
    echo "❌ 找不到 checkpoint: $LATEST_CKPT"
    exit 1
fi

echo "  使用 checkpoint: $LATEST_CKPT"
echo "  使用 config:     $LATEST_CONFIG"

# 在训练集上评估
echo ""
echo "--- Train Set ($(cat $TRAIN_JSON | $PYTHON -c 'import json,sys;print(len(json.load(sys.stdin)))') samples) ---"
$PYTHON "$PROJECT_ROOT/eval_open_loop.py" \
    --config_path "$LATEST_CONFIG" \
    --ckpt_path "$LATEST_CKPT" \
    --data_dir "$NPZ_OUTPUT" \
    --data_list "$TRAIN_JSON" \
    --cfg_weight $CFG_WEIGHT \
    2>&1 | tee "$TRAINING_OUTPUT/eval_train.log"

# 在验证集上评估
echo ""
echo "--- Val Set ($(cat $VAL_JSON | $PYTHON -c 'import json,sys;print(len(json.load(sys.stdin)))') samples) ---"
$PYTHON "$PROJECT_ROOT/eval_open_loop.py" \
    --config_path "$LATEST_CONFIG" \
    --ckpt_path "$LATEST_CKPT" \
    --data_dir "$NPZ_OUTPUT" \
    --data_list "$VAL_JSON" \
    --cfg_weight $CFG_WEIGHT \
    2>&1 | tee "$TRAINING_OUTPUT/eval_val.log"

# ==================== 汇总 ====================
echo ""
echo "=========================================="
echo "✅ 全流程完成！"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  Checkpoints:  $LATEST_CKPT_DIR/"
echo "  Train Eval:   $TRAINING_OUTPUT/eval_train.log"
echo "  Val Eval:     $TRAINING_OUTPUT/eval_val.log"
echo "  TensorBoard:  tensorboard --logdir $TRAINING_OUTPUT/tb_logs"
echo ""
echo "如需只跑评估 (跳过训练):"
echo "  bash run_all.sh --skip_data_process --skip_train"

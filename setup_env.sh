#!/bin/bash
# ============================================================
# Flow-Planner AutoDL 环境安装脚本
# 适用于: AutoDL 单卡 4090, Ubuntu, CUDA 12.x
#
# 用法: bash setup_env.sh
# ============================================================
set -e

echo "=========================================="
echo "Flow-Planner 环境安装"
echo "=========================================="

# ---------- 1. 创建 conda 环境 ----------
echo "[1/6] 创建 conda 环境 (Python 3.9)..."
conda create -n flow_planner python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate flow_planner

# ---------- 2. 安装 PyTorch ----------
echo "[2/6] 安装 PyTorch 2.3.0 (CUDA 12.1)..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# ---------- 3. 安装项目依赖 ----------
echo "[3/6] 安装项目依赖..."
pip install \
    casadi \
    einops==0.8.0 \
    flow-matching==1.0.10 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    scipy==1.13.1 \
    tensorboard==2.11.2 \
    timm==1.0.10 \
    wandb==0.17.4 \
    torchdiffeq==0.2.5 \
    shapely==2.0.7 \
    geopandas==1.0.1 \
    pandas \
    tqdm

# ---------- 4. 安装 nuplan-devkit ----------
echo "[4/6] 安装 nuplan-devkit..."
if [ ! -d "$HOME/nuplan-devkit" ]; then
    cd $HOME
    git clone https://github.com/motional/nuplan-devkit.git
    cd nuplan-devkit
    pip install -e .
else
    echo "nuplan-devkit 已存在，跳过 clone"
    cd $HOME/nuplan-devkit
    pip install -e .
fi

# ---------- 5. 安装 Flow-Planner ----------
echo "[5/6] 安装 Flow-Planner..."
cd $HOME/Flow-Planner
pip install -e .

# ---------- 6. 验证安装 ----------
echo "[6/6] 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import flow_matching
import hydra
import timm
import shapely
print('All imports OK ✅')
"

echo ""
echo "=========================================="
echo "✅ 环境安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 下载 nuplan 数据到 ~/nuplan/dataset/"
echo "  2. 运行: bash run_all.sh"

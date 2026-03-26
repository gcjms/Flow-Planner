# FlowPlanner 闭环仿真验证指南

## 目标

用 nuPlan 官方仿真器验证不同 CFG 权重 `w` 对闭环性能（NR Score）的影响。

---

## 硬件要求

- **GPU**: NVIDIA GPU，显存 ≥ 8GB（推荐 RTX 3090 / 4090）
- **内存**: ≥ 32GB
- **磁盘**: ≥ 100GB（nuPlan Val 数据集 + 仿真输出）
- **操作系统**: Linux（Ubuntu 20.04+）

---

## 环境搭建（约 30 分钟）

### Step 1: 克隆仓库

```bash
git clone https://github.com/gcjms/Flow-Planner.git
cd Flow-Planner
```

### Step 2: 创建 Conda 环境

```bash
conda create -n flow_planner python=3.9 -y
conda activate flow_planner

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### Step 3: 安装 nuPlan-devkit（用于闭环仿真）

```bash
cd ..
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
cd ../Flow-Planner
```

### Step 4: 下载数据和权重

#### 4a. nuPlan Val 数据集

从 [nuPlan 官网](https://www.nuscenes.org/nuplan) 下载 **Val split** 的 `.db` 文件和地图数据。

```bash
# 示例目录结构：
# /data/nuplan/
#   ├── dataset/
#   │   └── nuplan-v1.1/
#   │       └── splits/
#   │           └── val/  ← .db 文件放这里
#   └── maps/
#       └── nuplan-maps-v1.0/  ← 地图文件
```

#### 4b. 模型权重

权重可从 HuggingFace 下载（或用仓库中自带的 checkpoints 目录）：

```bash
# 如果仓库中没有 checkpoints/model.pth：
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='ttwhy/flow-planner', filename='model.pth', local_dir='checkpoints/')
hf_hub_download(repo_id='ttwhy/flow-planner', filename='model_config.yaml', local_dir='checkpoints/')
"
```

---

## 一键运行闭环仿真

### 方式 1: 使用一键脚本（推荐）

```bash
# 编辑脚本中的路径（见下面说明），然后：
bash run_closed_loop_validation.sh
```

脚本会自动依次跑 w=1.0, 1.5, 1.8, 2.0, 2.5, 3.0 共 6 种配置。

### 方式 2: 手动运行

```bash
conda activate flow_planner

# 设置环境变量
export NUPLAN_DATA_ROOT=/data/nuplan/dataset    # nuPlan 数据目录
export NUPLAN_MAPS_ROOT=/data/nuplan/maps        # 地图目录
export PROJECT_ROOT=$(pwd)

# 运行闭环仿真（以 w=2.5 为例）
python $NUPLAN_DEVKIT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=flow_planner \
    planner.flow_planner.config_path=$PROJECT_ROOT/checkpoints/model_config.yaml \
    planner.flow_planner.ckpt_path=$PROJECT_ROOT/checkpoints/model.pth \
    planner.flow_planner.use_cfg=true \
    planner.flow_planner.cfg_weight=2.5 \
    planner.flow_planner.device=cuda \
    scenario_builder=nuplan \
    scenario_filter=val14_split \
    experiment_name=cfg_w2.5 \
    output_dir=$PROJECT_ROOT/simulation_outputs
```

---

## 预期结果

仿真完成后，会输出类似以下日志：

```
final_score|1115.0|0.8119|0.9381|0.9345|0.8565
```

对应：场景数 | Overall Score | 无碰撞率 | 可行驶区域率 | TTC/舒适度

### 已有参考数据

| cfg_weight | NR Score | Collision-Free | Drivable |
|:---:|:---:|:---:|:---:|
| 1.8 | 81.19% | 93.8% | 93.5% |
| 2.5 | 80.71% | 93.5% | 93.4% |

---

## 期望验证的配置

| 配置 | 说明 |
|------|------|
| `w=1.0` | CFG 关闭（baseline） |
| `w=1.5` | 轻度引导 |
| `w=1.8` | 论文默认值 |
| `w=2.0` | |
| `w=2.5` | 我们之前最佳 |
| `w=3.0` | 强引导 |

---

## 常见问题

### Q: `bokeh` 报错，numpy 版本冲突
```bash
pip install bokeh==2.4.3 numpy==1.23.5
```

### Q: `.db` 文件报 `corrupt` 或 `OperationalError`
重新下载对应的 `.db` 文件，可能下载时损坏。

### Q: 显存不足
```bash
# planner 推理显存约 500MB，如果仿真器本身也用 GPU 可能需要 12GB+
# 可以尝试减少仿真线程：
+simulation.number_of_cpus_for_simulation=4
```

### Q: 仿真特别慢
正常速度约 **5-10 分钟/场景**，1115 场景约 **6-12 小时**。可以先用 mini split 测试：
```bash
scenario_filter=val14_split  # ← 改为 mini split 先快速测试
```

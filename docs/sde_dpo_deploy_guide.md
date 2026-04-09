# SDE + Multi-Objective DPO 部署指南

## 这是什么

在 Flow-Planner 的 ODE 推理过程中注入可控随机扰动（SDE 模式），增加候选轨迹的多样性，然后用多目标 DPO 做偏好优化。

核心改动：
- `flow_planner/model/flow_planner_model/flow_utils/flow_ode.py` — 新增 `generate_sde()` 方法
- `flow_planner/dpo/generate_multiobjective_pairs.py` — 支持 `--sde` 模式 + 修复 CFG 输入 bug
- `flow_planner/dpo/measure_sde_diversity.py` — SDE 多样性验证脚本（新文件）
- `auto_sde_dpo_pipeline.sh` — 端到端自动化脚本（新文件）

## 前置条件

确认以下文件/目录存在：

```
/root/autodl-tmp/Flow-Planner/                # 项目根目录
├── checkpoints/
│   ├── model.pth                              # 模型权重
│   └── model_config.yaml                      # 模型配置
├── flow_planner/                              # 代码目录
│   ├── dpo/
│   │   ├── generate_multiobjective_pairs.py   # 已更新
│   │   ├── measure_sde_diversity.py           # 新文件
│   │   ├── train_dpo.py                       # 已有
│   │   └── ...
│   └── model/flow_planner_model/flow_utils/
│       └── flow_ode.py                        # 已更新
├── auto_sde_dpo_pipeline.sh                   # 新文件
/root/autodl-tmp/hard_scenarios_v2/            # 场景数据（.npz 文件）
/root/autodl-tmp/maps_raw/maps/                # nuPlan 地图
/root/autodl-tmp/val_data/data/cache/          # 验证数据
```

## 运行方式

### 方式一：一键全流程（推荐）

```bash
cd /root/autodl-tmp/Flow-Planner
chmod +x auto_sde_dpo_pipeline.sh
nohup bash auto_sde_dpo_pipeline.sh > /root/sde_pipeline_stdout.log 2>&1 &
```

全流程包含 6 步：
1. **Step 0**: SDE 多样性验证（100 场景，~15 min）
2. **Step 1**: SDE 模式挖掘偏好对（5000 场景，~3-5 h）
3. **Step 2**: DPO 训练（~30 min）
4. **Step 3**: LoRA 合并（~2 min）
5. **Step 4**: 开环多维度评估（~15 min）
6. **Step 5**: 闭环 NR-CLS 仿真（~1-2 h）

完成后自动关机。总计约 5-8 小时。

查看进度：

```bash
tail -f /root/sde_dpo_pipeline.log
```

### 方式二：只跑多样性测试（确认 SDE 有效）

```bash
conda activate flow_planner
cd /root/autodl-tmp/Flow-Planner

python -u -m flow_planner.dpo.measure_sde_diversity \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --num_scenes 100 \
    --num_samples 20 \
    --sigma_base "0.1,0.3,0.5" \
    --sde_steps 20 \
    --cfg_weight 1.8 \
    --device cuda
```

约 15-20 分钟。输出两张表格：

**表 1：几何多样性**（越大越好）
- RMSE(m): 候选轨迹间平均 RMSE
- EndPt(m): 终点散布度
- MaxLat(m): 最大横向散布
- MidPt(m): 中点散布度

**表 2：Reward 多样性**（越大越好，说明 DPO 能学到东西）
- Rew_std: 综合 reward 标准差
- Col_std: 碰撞分数标准差
- #Collide: 碰撞轨迹数量

脚本最后会给出推荐的 sigma_base 值。

### 方式三：单独挖 pair（不训 DPO）

```bash
python -u -m flow_planner.dpo.generate_multiobjective_pairs \
    --ckpt_path checkpoints/model.pth \
    --config_path checkpoints/model_config.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --output_path /root/autodl-tmp/sde_pairs.npz \
    --max_scenes 5000 \
    --num_seeds 5 \
    --cfg_weights 0.5,1.0,1.8,3.0 \
    --target_dims collision,ttc,comfort \
    --score_gap_threshold 0.12 \
    --sde \
    --sigma_base 0.3 \
    --sde_steps 20 \
    --device cuda
```

不加 `--sde` 则用原始 ODE 模式（确定性采样）。

## 关键超参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `sigma_base` | measure / generate | 0.3 | SDE 噪声强度。0.1=保守，0.5=激进。先用 measure 脚本测最优值 |
| `sde_steps` | measure / generate | 20 | SDE 积分步数。越多噪声越可控（推荐 20-50）。部署推理仍用原始 4 步 ODE |
| `score_gap_threshold` | generate | 0.12 | 构造偏好对的最小维度分差。太大 → pair 太少，太小 → 噪声 pair |
| `DPO_BETA` | pipeline | 5.0 | DPO loss 温度。越大越保守（不敢偏离参考模型） |
| `DIM_WEIGHTS` | pipeline | collision:5,ttc:5,comfort:2 | 各维度 DPO loss 权重 |
| `cfg_weights` | generate | 0.5,1.0,1.8,3.0 | CFG 权重列表。不同权重产生不同偏好的轨迹 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `/root/sde_diversity_report.log` | SDE 多样性测试完整输出 |
| `/root/sde_dpo_pipeline.log` | Pipeline 运行日志 |
| `/root/sde_dpo_eval_report.txt` | 最终评估报告（开环 + 闭环对比） |
| `/root/sde_dpo_train.log` | DPO 训练详细日志（含 per-dimension metrics） |
| `checkpoints/dpo_sde/model_sde_dpo_merged.pth` | 最终合并模型 |
| `checkpoints/dpo_sde/lora_best.pt` | 最优 LoRA 权重 |

## 调试

**如果 Step 0 报告 SDE 没有改善多样性：**
- 增大 sigma_base（试 0.5, 0.8, 1.0）
- 增大 sde_steps（试 30, 50）
- 检查 /root/sde_diversity_report.log 中的具体数值

**如果 Step 1 产生的 pair 数量太少（< 100）：**
- 降低 score_gap_threshold（试 0.08, 0.05）
- 增加 num_seeds（试 8, 10）
- 增大 sigma_base

**如果 DPO 训练后开环指标没有改善或恶化：**
- 增大 beta（试 8.0, 10.0）使训练更保守
- 减少 epochs 到 1
- 检查 train.log 中的 per-dimension accuracy 是否都 > 50%

## 技术背景

SDE 模式的核心公式：

```
标准 ODE:  x_{t+dt} = x_t + dt · v(x_t, t)
SDE 模式:  x_{t+dt} = x_t + dt · v(x_t, t) + σ(t) · √dt · ε

σ(t) = σ_base · (1 - t)   (线性衰减: 早期大扰动探索、后期小扰动保质量)
```

参考论文：
- Flow-GRPO (arXiv:2505.05470): ODE→SDE 转换 + GRPO 在线 RL
- FlowDrive (arXiv:2509.21961): 流步间扰动注入增加轨迹多样性
- DiverseFlow (CVPR 2025): DPP 耦合采样增加模式覆盖

本次改动只影响 **离线候选生成**（挖 pair 时），不影响部署推理（仍用确定性 ODE）。

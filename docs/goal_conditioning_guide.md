# Goal Conditioning + DPO 完整部署指南

## 概览

基于 GoalFlow (CVPR 2025) 的思路，在 Flow Planner 的 decoder 中加入 goal point conditioning。
不同的 goal point 生成决策级不同的轨迹（左绕/右绕/刹停），解决 DPO 的轨迹多样性不足问题。

## 完整流程（共 6 步）

```
Step 1: 聚类 GT 终点 → goal_vocab.npy (跑一次)
Step 2: 改 yaml + 重训 FM 模型 (带 goal conditioning)
Step 3: 用不同 goal 生成 DPO 候选轨迹
Step 4: Scorer 打分 → 构建偏好对
Step 5: LoRA + DPO 训练
Step 6: 合并权重 + 评估
```

## 文件清单

### 新增文件
```
flow_planner/goal/__init__.py                    # 包标记
flow_planner/goal/cluster_goals.py               # Step 1: 离线聚类脚本
flow_planner/goal/goal_utils.py                  # 工具函数
flow_planner/dpo/generate_candidates_goal.py     # Step 3: Goal-diverse 候选生成
```

### 修改文件
```
flow_planner/model/flow_planner_model/decoder.py      # 加了 goal_proj MLP
flow_planner/model/flow_planner_model/flow_planner.py  # 训练/推理传递 goal_point
```

### 已有文件（直接使用）
```
flow_planner/dpo/score_hybrid.py     # Step 4: 打分
flow_planner/dpo/train_dpo.py        # Step 5: DPO 训练
flow_planner/dpo/lora.py             # LoRA 实现
flow_planner/dpo/dpo_loss.py         # DPO loss
flow_planner/trainer.py              # FM 模型训练入口
```

---

## Step 1: 聚类 — 建立 goal 词典（跑一次）

```bash
python -m flow_planner.goal.cluster_goals \
    --data_dir /root/autodl-tmp/nuplan_npz \
    --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
    --output_path /root/Flow-Planner/goal_vocab.npy \
    --n_clusters 64
```

输出: `goal_vocab.npy` — shape (64, 2)，64 个 ego-centric (x, y) 聚类中心。

**验证**：脚本会打印每个聚类的中心坐标和样本数，确认分布合理。

---

## Step 2: 重训 FM 模型（带 goal conditioning）

### 2a. 修改 yaml 配置

在 `flow_planner/script/model/flow_planner.yaml` 中加两处：

**顶层加 `goal_vocab_path`：**
```yaml
_target_: flow_planner.model.flow_planner_model.flow_planner.FlowPlanner

goal_vocab_path: /root/Flow-Planner/goal_vocab.npy   # ← 新增

neighbor_num: 32
# ... 其余不变
```

**`model_decoder` 段加 `goal_dim: 2`：**
```yaml
model_decoder:
  _target_: flow_planner.model.flow_planner_model.decoder.FlowPlannerDecoder
  hidden_dim: 256
  goal_dim: 2              # ← 新增：启用 goal conditioning
  # ... 其余不变
```

### 2b. 启动训练

```bash
cd /root/Flow-Planner

# 跟原来训练命令一样，只是 yaml 里多了 goal 配置
torchrun --nproc_per_node=8 \
    -m flow_planner.trainer \
    --config-name flow_planner_standard
```

模型会自动：
1. 加载 goal_vocab.npy
2. 每个训练样本从 GT 终点查最近聚类中心作为 goal 条件
3. CFG 随机 drop goal（跟 neighbors 共用 cfg_flags）

**验证**：训练日志应该能看到 goal_vocab 加载成功，loss 正常下降。

---

## Step 3: Goal-diverse 候选生成

```bash
python -m flow_planner.dpo.generate_candidates_goal \
    --data_dir /root/autodl-tmp/dpo_mining \
    --config_path /root/Flow-Planner/checkpoints/config_goal.yaml \
    --ckpt_path /root/Flow-Planner/checkpoints/model_goal.pth \
    --vocab_path /root/Flow-Planner/goal_vocab.npy \
    --output_dir /root/autodl-tmp/dpo_candidates_goal \
    --num_candidates 5
```

每个场景输出一个 `*_candidates.npz`，包含：
- `candidates`: (5, T, D) — 5 条不同决策方向的轨迹
- `goal_labels`: (5, 2) — 每条轨迹对应的 goal point

**验证**：看几个场景的 candidates，5 条轨迹的终点应该明显不同（不像之前几乎一样）。

---

## Step 4: 打分 → 构建偏好对

```bash
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir /root/autodl-tmp/dpo_candidates_goal \
    --output_dir /root/autodl-tmp/dpo_preferences_goal \
    --skip_vlm \
    --spread_threshold 3.0
```

输出：
- `preferences.npz`: chosen/rejected 轨迹对
- `preference_details.json`: 每个场景的打分细节

**验证**：查看 `preference_details.json`，确认 chosen 和 rejected 的分数差距明显（之前的问题就是分数差距太小）。

---

## Step 5: DPO 训练

```bash
python -m flow_planner.dpo.train_dpo \
    --config_path /root/Flow-Planner/checkpoints/config_goal.yaml \
    --ckpt_path /root/Flow-Planner/checkpoints/model_goal.pth \
    --preference_path /root/autodl-tmp/dpo_preferences_goal/preferences.npz \
    --scene_dir /root/autodl-tmp/dpo_mining \
    --output_dir /root/Flow-Planner/checkpoints/dpo_goal \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --beta 5.0 \
    --sft_weight 0.1 \
    --num_t_samples 16 \
    --lora_rank 4 \
    --lora_alpha 16 \
    --save_merged
```

**关键指标**（看 TensorBoard 或日志）：
- `dpo/accuracy` > 0.6 → 模型在学
- `dpo/delta_mean` 持续上升 → chosen 比 rejected 越来越好
- `dpo/loss` 下降 → 正常

**输出**：
- `checkpoints/dpo_goal/lora_best.pt` — 最佳 LoRA 权重
- `checkpoints/dpo_goal/model_dpo_merged.pth` — 合并后的完整模型（推理用）

---

## Step 6: 评估

合并后的模型可以直接当原模型用（不需要 goal point）：

```bash
# 开环评估
python -m flow_planner.dpo.eval_multidim \
    --ckpt_path /root/Flow-Planner/checkpoints/dpo_goal/model_dpo_merged.pth \
    --config_path /root/Flow-Planner/checkpoints/config_goal.yaml \
    --scene_dir /root/autodl-tmp/hard_scenarios_v2 \
    --max_scenes 500
```

---

## 向后兼容

- `goal_dim: 0` (默认) = 不启用 goal conditioning，与原模型完全一致
- `goal_vocab_path: null` (默认) = 不使用 goal，原有 checkpoint 可正常加载
- 老 checkpoint 加载到新代码时，goal_proj 参数会被 missing keys 跳过，不影响

## 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_clusters (K) | 64 | nuPlan 数据下 64 够用, 可试 32/128 |
| goal_dim | 2 | (x, y) 坐标, 不需要 heading |
| cfg_prob | 0.3 | 沿用原值, goal 跟 neighbors 一起被 mask |
| num_candidates | 5 | DPO 候选数, 每个对应不同 goal |
| dpo beta | 5.0 | DPO 温度, 太小学不动太大过拟合 |
| lora_rank | 4 | LoRA 秩, 4 够了 |
| sft_weight | 0.1 | 防遗忘正则 |

## 注意事项

1. **Step 2 是必须的** — 不重训模型，goal embedding 永远是零，换 goal 没有效果
2. **Step 2 的 config_goal.yaml** — 训练完成后把训练用的 yaml 拷贝一份到 checkpoints/ 备用
3. **Step 3 的 ckpt** — 必须用 Step 2 训出来的带 goal 的模型，不能用原模型
4. **Step 5 的 --scene_dir** — 需要指向包含原始 NPZ 场景文件的目录（用于 encoder 计算条件）

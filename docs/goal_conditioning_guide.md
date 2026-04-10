# Goal Conditioning 部署指南

## 概览

基于 GoalFlow 思路，在 Flow Planner 的 decoder 中加入 goal point conditioning。
不同的 goal point 条件会生成决策级不同的轨迹（左绕/右绕/刹停），解决 DPO 的轨迹多样性问题。

## 文件清单

### 新增文件
```
flow_planner/goal/__init__.py           # 空文件
flow_planner/goal/cluster_goals.py      # Step 1: 离线聚类脚本
flow_planner/goal/goal_utils.py         # 工具函数
flow_planner/dpo/generate_candidates_goal.py  # Step 3: DPO 候选生成
```

### 修改文件
```
flow_planner/model/flow_planner_model/decoder.py      # 加了 goal_proj MLP
flow_planner/model/flow_planner_model/flow_planner.py  # 训练/推理传递 goal_point
```

## 三步部署流程

### Step 1: 聚类 — 跑一次

```bash
python -m flow_planner.goal.cluster_goals \
    --data_dir /root/autodl-tmp/nuplan_npz \
    --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
    --output_path /root/Flow-Planner/goal_vocab.npy \
    --n_clusters 64
```

输出: `goal_vocab.npy` (64, 2) — 64 个 goal point 聚类中心

### Step 2: 重训模型 — 加 goal conditioning

在训练配置 yaml 中加两处:

**model_decoder 加 `goal_dim: 2`:**

```yaml
# flow_planner/script/model/flow_planner.yaml

model_decoder:
  _target_: flow_planner.model.flow_planner_model.decoder.FlowPlannerDecoder
  hidden_dim: 256
  goal_dim: 2              # ← 新增：启用 goal conditioning
  # ... 其余不变
```

**model 加 `goal_vocab_path`:**

```yaml
# flow_planner/script/model/flow_planner.yaml

_target_: flow_planner.model.flow_planner_model.flow_planner.FlowPlanner

goal_vocab_path: /root/Flow-Planner/goal_vocab.npy   # ← 新增

# ... 其余不变
```

然后正常启动训练:
```bash
# 和之前完全一样的训练命令
torchrun --nproc_per_node=8 train.py ...
```

模型会自动:
1. 加载 goal_vocab.npy
2. 每个训练样本从 GT 终点查最近聚类中心作为 goal 条件
3. CFG 随机 drop goal (跟 drop neighbors 共用同一个 cfg_flags)

### Step 3: DPO 候选生成

```bash
python -m flow_planner.dpo.generate_candidates_goal \
    --data_dir /root/autodl-tmp/dpo_mining \
    --config_path /root/Flow-Planner/checkpoints/config_goal.yaml \
    --ckpt_path /root/Flow-Planner/checkpoints/model_goal.pth \
    --vocab_path /root/Flow-Planner/goal_vocab.npy \
    --output_dir /root/autodl-tmp/dpo_candidates_goal \
    --num_candidates 5
```

每个场景生成 5 条轨迹，每条轨迹对应不同的 goal point。
输出 NPZ 包含:
- `candidates`: (5, T, D) — 5 条不同的轨迹
- `goal_labels`: (5, 2) — 每条轨迹的 goal point 坐标

之后接入原有的 `score_hybrid.py` 打分 → DPO 训练。

## 向后兼容

- `goal_dim: 0` (默认) = 不启用 goal conditioning，行为与原模型完全一致
- `goal_vocab_path: null` (默认) = 训练/推理不使用 goal，原有 checkpoint 可正常加载
- 老的 checkpoint 可以加载到新代码中 (goal_proj 参数会被 missing keys 跳过)

## 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| n_clusters (K) | 64 | nuPlan 数据量下 64 够用, 可试 32/128 |
| goal_dim | 2 | (x, y) 坐标, 不需要 heading |
| cfg_prob | 0.3 | 沿用原值, goal 跟 neighbors 一起被 mask |
| num_candidates | 5 | DPO 候选数, 每个对应不同 goal |

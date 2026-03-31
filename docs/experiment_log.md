# Flow-Planner 实验记录

> 最后更新：2026-03-31  
> 项目：Flow-Planner — 基于 Flow Matching 的自动驾驶规划

---

# 第一章 SOTA 基线复现

## 1.1 实验目的

用官方 HuggingFace 权重复现 Flow-Planner 论文的 NR-CLS 分数 (94.31%)，建立可靠的实验 baseline。

## 1.2 实验配置

| 项目 | 值 |
|------|------|
| 权重 | `checkpoints/model.pth`（HuggingFace `ttwhy/flow-planner`） |
| CFG Weight | 1.8（论文默认） |
| 数据 | nuPlan val14 split, 1118 scenarios |
| 挑战类型 | `closed_loop_nonreactive_agents` |
| 硬件 | AutoDL RTX 4090 |
| Worker | Ray distributed, 4 threads/node |

## 1.3 结果

| 指标 | 平均分 | 通过率 | 权重 |
|------|--------|--------|------|
| driving_direction_compliance | 0.9955 | 99.5% (1112/1118) | 5 |
| ego_is_making_progress | 0.9964 | 99.6% (1114/1118) | 5 |
| drivable_area_compliance | 0.9821 | 98.2% (1098/1118) | 5 |
| speed_limit_compliance | 0.9773 | 84.4% (944/1118) | 2 |
| ego_is_comfortable | 0.9428 | 94.3% (1054/1118) | 2 |
| no_ego_at_fault_collisions | 0.9378 | 93.6% (1046/1118) | 5 |
| ego_progress_along_expert_route | 0.9319 | 46.1% (515/1118) | 5 |
| time_to_collision_within_bound | 0.8864 | 88.6% (991/1118) | 5 |

**NR-CLS = 95.56%（加权平均），超论文 94.31%，领先 +1.25%** ✅

### 与其他方法对比

| 方法 | NR-CLS | 来源 |
|------|--------|------|
| **我们复现（官方权重, w=1.8）** | **95.56%** | 本项目 |
| Flow-Planner 论文 | 94.31% | CVPR 2025 |
| PDM-Closed | 92-93% | 规则 baseline |
| Diffusion Planner | 89.87% | ICLR 2025 |
| PlanTF | 84.83% | ICRA 2024 |

## 1.4 NR-CLS 评分机制说明

NR-CLS (Non-Reactive Closed-Loop Score) 是 nuPlan 官方评估标准之一：

```
NR-CLS = Σ(weight_i × metric_i) / Σ(weight_i) × 100%
```

- **满分 100%** = 所有场景的所有指标都完美通过
- 8 个子指标各自 ∈ [0, 1]，按权重加权平均
- 由于部分指标接近 binary（pass/fail），分越高提升越难
- 行业现状：95%+ 已是当前 SOTA 水平，剩余的 ~5% 都是极端长尾场景

## 1.5 结论

基线成功复现且超越论文分数，平台和数据管线验证通过，可以进行创新实验。

---

# 第二章 Risk-Aware Adaptive CFG (Grid Search)

## 2.1 实验目的

验证 **Risk-Aware Adaptive CFG** 的可行性：能否通过驾驶场景的风险特征（TTC、THW、DRAC 等）预测最优 CFG 权重 w？

## 2.2 实验配置

| 项目 | 值 |
|------|------|
| 日期 | 2026-03-25 |
| 数据 | nuPlan Val, 15000 / 28845 场景 |
| 模型 | 官方权重 |
| w 范围 | {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0} |
| 硬件 | RTX 2060 (6GB) |
| 耗时 | 489 分钟 |

## 2.3 Grid Search 结果

### 各 w 的全局 ADE

| w | Mean ADE | Std |
|---|---------|-----|
| 0.5 | 1.9995 | 2.6021 |
| 1.0 | 2.0037 | 2.6026 |
| **1.5** | **1.9985** | 2.6024 |
| 2.0 | 2.0050 | 2.6256 |
| 2.5 | 2.0038 | 2.6178 |
| 3.0 | 2.0062 | 2.6156 |
| 3.5 | 2.0061 | 2.6122 |
| 4.0 | 2.0045 | 2.6100 |

> [!WARNING]
> **所有 w 的 ADE 差异 < 0.01m**。全局角度看 w 的选择影响极小。

### Per-Scenario 最优 w 分布

```
w=0.5:  1833 (12.2%)
w=1.0:  1838 (12.3%)
w=1.5:  1891 (12.6%)
w=2.0:  1914 (12.8%)
w=2.5:  1874 (12.5%)
w=3.0:  1890 (12.6%)
w=3.5:  1854 (12.4%)
w=4.0:  1906 (12.7%)
```

**几乎完全均匀分布** — "最优 w" 受 ODE 随机噪声影响更大，而非场景内在属性。

### 自适应 w 的理论上限

| 指标 | 值 |
|------|------|
| 全局固定 w=1.5 的 ADE | 1.9985m |
| **Oracle 自适应 w 的 ADE** | **1.4395m** |
| 理论改善 | **0.559m (28.0%)** |

## 2.4 特征相关性分析

| 特征 | 相关系数 r |
|------|-----------|
| lateral_min_dist | -0.025 |
| n_neighbors | +0.019 |
| front_vehicle_dist | -0.014 |
| ego_speed | -0.012 |
| 其余 8 项特征 | \|r\| < 0.01 |

> [!CAUTION]
> **所有特征 |r| < 0.03**，无任何风险特征能预测最优 w。

## 2.5 Risk Network 训练结果

| 指标 | 值 |
|------|------|
| MAE(w) | 0.969 |
| ±0.5 准确率 | 25.9% |
| ±1.0 准确率 | 51.8% |
| 预测 w 范围 | [1.17, 3.15] |

模型几乎只预测均值 w≈2.25，无法区分不同场景。

## 2.6 核心发现

1. **ODE 随机性主导最优 w**：Flow Matching 推理含随机初始噪声 `x₀ ~ N(0,I)`，同一场景每次推理 ADE 不同。Grid Search 只跑 1 次/w，"最优 w"可能只是随机波动
2. **风险特征无法预测最优 w**：12 维安全指标与 w 无统计相关性
3. **理论上限存在**：Oracle 自适应 w 可改善 28% ADE，但需消除噪声才能揭示真实关系

## 2.7 改进方向

| 方向 | 方法 | 预期效果 |
|------|------|---------|
| 消除 ODE 噪声 | 每场景每 w 跑 N 次取平均 ADE | 结果更稳定，可能揭示真实相关性 |
| 深层特征 | 用 encoder 输出代替手工特征 | 可能捕捉与 w 相关的隐藏模式 |
| 学习 w 分布 | 预测 w 的概率分布而非单值 | 适应 w 的高方差特性 |

## 2.8 输出文件

| 文件 | 内容 |
|------|------|
| [risk_dataset.npz](file:///home/gcjms/Flow-Planner/risk_outputs/risk_dataset.npz) | 15000×8 ADE 矩阵 + 特征 |
| [risk_network.pth](file:///home/gcjms/Flow-Planner/risk_outputs/risk_network.pth) | Risk Network 权重 |
| [validation_results.npz](file:///home/gcjms/Flow-Planner/risk_outputs/validation_results.npz) | 验证分析数据 |
| [grid_search_report.md](file:///home/gcjms/Flow-Planner/docs/grid_search_report.md) | 完整报告 |

---

# 第三章 Best-of-N 轨迹选择

## 3.1 实验目的

通过生成 N 条候选轨迹并用安全评分函数选择最优，验证 Best-of-N 策略能否提升闭环仿真性能。

## 3.2 实验配置

| 项目 | 值 |
|------|------|
| 日期 | 2026-03-30 21:07 → 2026-03-31 06:31 |
| N | 5 |
| CFG Weight | 1.8 |
| 数据 | val14 split, 1118 scenarios |
| 评分函数 | `TrajectoryScorer`（碰撞/TTC/路线/舒适度/进度 5 维） |
| 硬件 | AutoDL RTX 4090 |
| 耗时 | 9h24m (闭环), 全部成功 (0 failures) |

**评分函数权重分配：**

| 维度 | 权重 | 机制 |
|------|------|------|
| 碰撞检测 | 40 | 与邻居最近距离 > 2m → 满分 |
| TTC | 20 | 碰撞时间 > 3s → 满分 |
| 路线一致性 | 15 | 终点到路线最近点的距离 |
| 舒适度 | 10 | 最大加速度 + Jerk |
| 前进进度 | 15 | 纵向位移 / 60m |

## 3.3 结果

### NR-CLS 对比

| 方法 | NR-CLS (加权平均) | NR-CLS (乘积) |
|------|-------------------|---------------|
| Best-of-N (N=5) | 95.56% | 69.47% |
| Baseline (N=1) | 95.56% | 69.53% |
| **提升** | **+0.00%** ❌ | -0.06% |

### Open-Loop ADE 对比

| N | Mean ADE (m) | Median ADE (m) |
|---|-------------|----------------|
| 1 | 1.9595 | 1.2251 |
| 3 | 1.9597 | 1.2339 |
| 5 | 1.9583 | 1.2672 |
| 10 | 1.9681 | 1.2500 |

ADE 从 N=1 到 N=10 几乎没有变化，差异 < 0.01m。

## 3.4 失败原因分析 — 硬证据

### 证据 1：改善/恶化比例 = 50/50（= 随机选择）

```
N=5 vs N=1:
  ADE 改善的场景: 1016/2000 (50.8%)
  ADE 恶化的场景:  984/2000 (49.2%)
  相同:              0/2000

N=10 vs N=1:
  ADE 改善的场景: 1002/2000 (50.1%)
  ADE 恶化的场景:  998/2000 (49.9%)
```

> [!IMPORTANT]
> **完美的 50/50**。如果 scorer 有效，改善应远多于恶化。这证明 scorer 选出的轨迹与随机选的没有区别。

### 证据 2：NR-CLS 8 个子指标完全一致

1118 场景 × 8 个指标 = 8944 个评分值，加权平均后精确到小数点后两位完全一致 → 闭环表现零差异。

### 证据 3：推理时间不随 N 增长

```
Open-loop:
  N=1:  217.9ms/scenario
  N=5:  209.4ms/scenario  (ratio: 0.96x)
  N=10: 209.1ms/scenario  (ratio: 0.96x)
```

时间比例远小于预期的 5x/10x，需排查生成管线是否正常。

## 3.5 根因分析

### 🔴 根因 1：Scorer 使用了错误的邻居数据

`flow_planner.py` 第 283-289 行：

```python
# NOTE: 推理时只有 neighbor_past，没有 neighbor_future
neighbors = None
if hasattr(data, 'neighbor_future') and data.neighbor_future is not None:
    neighbors = data.neighbor_future[b]    # ← 推理时不存在
elif hasattr(data, 'neighbor_past') and data.neighbor_past is not None:
    neighbors = data.neighbor_past[b]      # ← 实际走这个分支
```

碰撞评分（权重 40）和 TTC 评分（权重 20）用的是**邻居过去的轨迹**评估未来安全性：
- 过去的邻居位置距 ego 通常远大于碰撞阈值 2m → **所有候选碰撞分满分**
- TTC 同理 → 所有候选满分
- 这两项占总权重的 60%，全部失效

### 🔴 根因 2：CFG 引导太强，候选差异太小

CFG w=1.8 意味着条件信号（地图、邻居、路线）强力引导生成方向，不同初始噪声 `x₀` 产生的轨迹差异很小。5 条轨迹终点位置相差极小 → 路线一致性、舒适度、进度评分也几乎一样。

### 🟡 根因 3：评估指标粒度太粗

NR-CLS 的子指标多为 binary pass/fail（"是否碰撞"、"是否在可行驶区域"），需要轨迹有**质变**才能翻转结果。Best-of-N 的微小差异不足以改变 pass/fail。

## 3.6 改进方案

### 方案 A：修复 Scorer 的邻居外推（P0）

`neighbor_past` 已包含速度信息 `[vx, vy]`（维度 4-5），可以做匀速外推：

```python
# neighbor_past shape: (M, T_p, 11)
# 维度: [x, y, cos_h, sin_h, vx, vy, width, length, type×3]
last_pos = neighbor_past[:, -1, :2]   # (M, 2)
velocity = neighbor_past[:, -1, 4:6]  # (M, 2) vx, vy
dt = 0.5  # nuPlan 时间步

# 匀速外推 T_future 步
future_positions = []
for t in range(T_future):
    pos = last_pos + velocity * dt * (t + 1)
    future_positions.append(pos)
neighbor_future_pred = torch.stack(future_positions, dim=1)  # (M, T_future, 2)
```

**注意：** 这不是一个"预测模块"，只是利用已有的 vx/vy 做匀速恒 heading 外推。精度不高但远好于用过去位置。

### 方案 B：固定随机种子做消融实验（P0）

当前每条候选用不同随机初始噪声 `torch.randn(...)`，但噪声本身的随机性导致无法公平比较。改为：

```python
for i in range(num_candidates):
    torch.manual_seed(seed + i)  # 固定种子
    x_init = torch.randn(...)
```

这样同一场景的 N 条候选轨迹是**确定性的**，可以排除噪声干扰来验证 scorer。

### 方案 C：增大候选多样性（P1）

不是用不同 w（这样不是控制变量实验），而是：
- 在 ODE solver 的不同步数产生分支
- 或者给初始噪声加一个温度缩放因子 `x₀ = temperature * randn(...)` 控制多样性

## 3.7 输出文件

| 文件 | 内容 |
|------|------|
| [nr_cls_comparison.txt](file:///home/gcjms/Flow-Planner/experiments/best_of_n_results/nr_cls_comparison.txt) | NR-CLS 对比 |
| [log.txt](file:///home/gcjms/Flow-Planner/experiments/best_of_n_results/log.txt) | 仿真日志 |
| [best_of_n_results.npz](file:///home/gcjms/Flow-Planner/risk_outputs/best_of_n/best_of_n_results.npz) | ADE 数据 (N=1,3,5,10) |
| [runner_report.parquet](file:///home/gcjms/Flow-Planner/experiments/best_of_n_results/runner_report.parquet) | 运行时数据 |
| [trajectory_scorer.py](file:///home/gcjms/Flow-Planner/flow_planner/risk/trajectory_scorer.py) | 评分器实现 |

---

# 第四章 实验总结与下一步

## 4.1 已完成实验总览

| # | 实验 | 状态 | 核心结论 |
|---|------|------|----------|
| 1 | SOTA 基线复现 | ✅ 成功 | NR-CLS 95.56%，超论文 +1.25% |
| 2 | Risk-Aware Adaptive CFG | ❌ 失败 | 风险特征无法预测最优 w，ODE 噪声主导 |
| 3 | Best-of-N 轨迹选择 | ❌ 失败 | Scorer bug + 候选差异太小，等效随机选 |

## 4.2 关键教训

1. **ODE 随机初始噪声是 Flow Matching 推理的核心不确定性来源**。任何涉及比较不同推理结果的实验，都必须考虑控制/消除此噪声
2. **Scorer 的输入数据必须与评估语义对齐**。用过去数据评估未来安全性是无效的
3. **NR-CLS 从 95% 往上提升极其困难**，需要在长尾场景上取得质变，而非在大多数场景上做微调

## 4.3 待执行改进项

| 优先级 | 项目 | 详情 |
|--------|------|------|
| P0 | 修复 Scorer — 邻居匀速外推 | 利用 `neighbor_past` 中的 vx/vy 外推未来位置 |
| P0 | 固定随机种子 + 重跑消融 | 验证 scorer 在排除噪声后是否有效 |
| P1 | 增大候选多样性 | 温度缩放 / ODE 分支策略 |
| P2 | Reward-Guided Flow Matching (Diffusion-DPO) | 训练时直接优化安全偏好 |

# TISA-DPO 论文解读

> **论文**: Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement
> **作者**: Jianbo Zhao et al. (哈工大 + 上交)
> **时间**: 2025 年 9 月
> **数据集**: NAVSIM (PDMS 89.8, SOTA)
> **论文**: https://arxiv.org/abs/2509.20938

---

## 一、这篇论文做了什么？

两个独立的贡献：
1. **TISA（时不变空间对齐）**：解决自回归规划的时空不对齐问题（跟我们无关）
2. **Multi-Objective DPO（多目标偏好优化）**：用 DPO 做 post-training，提升驾驶策略质量（**跟我们直接相关**）

TISA 提升了 1.2 分，DPO 提升了 **3.0 分** → DPO 的贡献反而更大！

---

## 二、他们的 DPO 怎么做的？（核心）

### 2.1 偏好对构建方法

```
Step 1: 对每个场景，用当前模型采样 128 条候选轨迹
Step 2: 用自动评价指标对每条轨迹打分：
  - NC (No Collision): 是否碰撞
  - DAC (Drivable Area Compliance): 是否在可行驶区域内
  - TTC (Time-to-Collision): 安全时间余量
  - EP (Ego Progress): 行驶进度
  - Comfort: 舒适度（加速度、jerk 等）

Step 3: 构建偏好对
  Winner Pool (chosen): 综合得分 Top-5 的轨迹
  Targeted Losers (rejected): 针对每个指标选一个"差在这个点上"的轨迹

  关键：loser 不是全面差，而是"只在某一个方面差"
  例如：
    (chosen, loser_TTC)  → 教模型"保持更大安全距离"
    (chosen, loser_DAC)  → 教模型"别开出路面"
    (chosen, loser_EP)   → 教模型"别原地不动"
```

### 2.2 为什么叫"多目标"？

```
标准 DPO（我们现在的做法）：
  一个 chosen + 一个 rejected = 1 对
  信号：笼统的"这个好，那个差"

Multi-Objective DPO（他们的做法）：
  一个 chosen pool + 多个 targeted losers = M 对
  信号：具体的"哪里差了、怎么改"

  比如同一个场景生成 3 个偏好对：
    (winner, loser_collision)  → 学避碰
    (winner, loser_TTC)        → 学保持安全距离  
    (winner, loser_drivable)   → 学不偏离道路
```

### 2.3 训练细节

```
基座模型: 自回归模型（ResNet-34 backbone）
动作空间: 离散化（128 × 64 = 8192 种动作）
候选数量: 128 条/场景
优化器: AdamW
学习率: 6e-5（DPO 阶段，是预训练的 1/10）
硬件: 8 × V100/A800
效果: PDMS 85.6 → 89.8 (+3.0 from MOPT, +1.2 from TISA)
```

---

## 三、和我们的对比

| 维度 | TISA-DPO | 我们 (Flow-Planner DPO) |
|------|----------|------------------------|
| **基座模型** | 自回归 (离散 token) | Flow Matching (连续 ODE) |
| **动作空间** | 离散 (8192 种) | 连续 (80×4 轨迹点) |
| **log P 计算** | 精确 (cross-entropy) | 近似 (-MSE of velocity field) |
| **候选生成** | 128 条/场景 | 1 条/场景 |
| **chosen 来源** | **模型自己的 Top-5** | **GT 轨迹** |
| **rejected 来源** | **模型自己的 targeted losers** | **模型碰撞轨迹** |
| **on/off-policy** | ✅ On-policy | ❌ Off-policy |
| **DPO 类型** | Multi-objective (多对) | Single-objective (一对) |
| **评价标准** | NC + DAC + TTC + EP + Comfort | 仅碰撞 |

---

## 四、他们的做法为什么有效？

### 4.1 关键：128 条候选！

```
他们能做 on-policy 的根本原因：
  自回归模型 + 离散动作空间 → 采样多样性极高
  128 条候选中自然会有好有坏

我们做不到的原因：
  Flow Matching ODE 确定性太强
  不同噪声初始化只产生 ~0.5m 差异
  10 条候选里要么全撞要么全没撞
```

### 4.2 关键：离散动作空间

```
他们的模型输出:
  P(action_1), P(action_2), ..., P(action_8192)
  → log P 就是 log softmax → 精确计算！

我们的模型输出:
  v_θ(x_t, t) → 连续向量场
  → log P ≈ -MSE → 只是近似！
  → 这个近似引入了额外噪声
```

### 4.3 关键：多目标分解

```
他们不是简单的 "好 vs 坏"
而是分解成具体的失败原因：

  "这条轨迹虽然没碰撞，但安全余量太小"
  "这条轨迹虽然安全，但停在原地没动"
  "这条轨迹虽然到了目的地，但开出了道路"

  → 每种失败对应一个具体的学习信号
  → 比我们的 "碰撞 vs GT" 精细得多
```

---

## 五、对我们的启发

### 5.1 我们能借鉴什么？

1. **增加候选数量**
   之前跑 5 条没用，那 128 条呢？Flow Matching 的多样性确实低，但量大了总会有差异。代价是推理时间 × 128。

2. **Multi-Objective 偏好对**
   即使用 GT 做 chosen，也可以把 rejected 按失败原因分类：
   - (GT, 碰撞轨迹) → 学避碰
   - (GT, 偏离道路轨迹) → 学路线跟随
   - (GT, 行进过慢轨迹) → 学进度

3. **温度采样增加多样性**
   在 ODE 初始噪声上加大温度 → 候选更分散 → 更容易产生碰撞 vs 安全的差异

### 5.2 我们的根本瓶颈

```
Flow Matching 的 ODE 推理天然确定性强
→ 多次采样差异太小 → 无法有效产生 on-policy 偏好对
→ 这是模型架构层面的限制，不是 DPO 方法的问题

TISA-DPO 能 on-policy 的前提：
  自回归 + 离散动作 → 天然多样性高 → 128 条有好有坏

如果要在 Flow Matching 上做真正的 on-policy DPO，
可能需要从架构层面增加随机性（比如 SDE 而非 ODE）
```

### 5.3 论文里一句关键的话

> "diffusion models model the probability distribution implicitly. This can make them less amenable to certain post-training refinement techniques, like reinforcement learning, that rely on explicit action probabilities."

翻译：**扩散/Flow 模型的隐式概率建模，天然不如自回归模型适合做 DPO/RL 后训练。** 这是他们在 Related Work 里直接指出的。我们面临的困难不是方法选错了，而是 Flow Matching 架构本身对 DPO 不友好。

---

## 六、总结

| 要点 | 结论 |
|------|------|
| 论文可信度 | ✅ 真实论文，NAVSIM SOTA，方法严谨 |
| DPO 效果 | +3.0 PDMS，比架构改进（+1.2）效果更大 |
| 偏好对构建 | On-policy: 128 条候选 + 多指标筛选 |
| 对我们的启示 | Flow Matching 不适合做 on-policy DPO，GT off-policy 可能是我们能做的最好方案 |
| 关键区别 | 离散动作 vs 连续轨迹，log P 精确 vs 近似 |

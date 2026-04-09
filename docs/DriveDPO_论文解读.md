# DriveDPO 论文解读

> **论文**: DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving
> **作者**: Shuyao Shang et al. (中科院自动化所)
> **时间**: 2025 年 9 月 | **NeurIPS 2025**
> **数据集**: NAVSIM (PDMS **90.0**, SOTA) + Bench2Drive
> **论文**: https://arxiv.org/abs/2509.17940

---

## 一、核心问题

模仿学习（IL）有两个致命缺陷：
1. **方向不对称**：同样偏离 GT 0.5m，往左偏可能安全，往右偏可能碰撞。但 L2 Loss 惩罚一样大
2. **好坏不分**：看起来像人类轨迹但实际不安全的轨迹，IL 也会学

Score-based 方法（如 Hydra-MDP）虽然加了安全打分，但**打分和策略优化解耦**——每个 anchor 独立回归 score，没有直接优化策略分布。

## 二、DriveDPO 的两阶段方法

### Stage 1: Unified Policy Distillation（统一策略蒸馏）

```
把模仿+安全融合成一个统一的目标分布：

对每个 anchor 轨迹 a_i：
  1. Imitation Similarity = -||a_i - GT||  → softmax 归一化
  2. Safety Score = NAVSIM 仿真器评估 PDMS
  3. p_unified = softmax(w1 × log(Sim) + w2 × log(PDMS))
     其中 w1=0.1, w2=1.0（安全权重远大于模仿权重）

训练：最小化 KL(π_θ, p_unified)
→ 直接优化策略分布，而不是独立的 score heads
```

**核心洞察**：log 变换 + softmax 竞争机制，让低安全分数的 anchor 被大幅压制。

### Stage 2: Safety DPO（安全迭代 DPO）

```
对每个场景：
  1. 从当前策略采样 K=1024 条候选轨迹
  2. chosen = p_unified 得分最高的
  3. rejected = 巧妙选择（不是简单的最低分！）

Rejected 选择策略（两种，论文用了第一种）：

  方法 A（Imitation-Based）：
    在"跟人类轨迹很像，但 PDMS < 0.3"的轨迹中
    选离人类轨迹最近的那个
    → 教模型："长得像人类开的但其实不安全"
    
  方法 B（Distance-Based）：
    在"PDMS 低"的轨迹中选离 chosen 最近的
    → 教模型："跟好的轨迹很像但其实差一点就危险"
```

**这是最关键的设计！**

```
Vanilla DPO:  chosen = 最好, rejected = 最差
  → 区别太大，模型学到的是浅层差异

DriveDPO:     chosen = 最好, rejected = "看起来像好的但实际不安全"
  → 区别微妙，模型被迫学习安全边界
  → PDMS 提升了额外的 +1.2（vs vanilla 的 +0.5）
```

### 训练细节

| 参数 | 值 |
|------|------|
| Anchor 数量 N | 8192 (离散轨迹字典) |
| DPO 候选 K | 1024 条/场景 |
| β | 0.1 |
| 预训练 | 30 epochs |
| DPO 微调 | 10 epochs |
| 安全阈值 τ | 0.3 |
| 硬件 | 6 × NVIDIA L20 |
| LR | 1e-4 |
| w1 (imitation) | 0.1 |
| w2 (safety) | 1.0 |

### 结果

```
Without DPO (仅 Unified Distillation): PDMS = 88.8
With DPO:                                PDMS = 90.0 (+1.2)

各子指标提升：
  NC (No Collision):     +0.6
  DAC (Drivable Area):   +0.8
  TTC (Time-to-Collision): +1.2
```

---

## 三、和 TISA-DPO 的对比

| 维度 | DriveDPO | TISA-DPO |
|------|----------|----------|
| 会议 | NeurIPS 2025 | 预印本 |
| 基座 | VADv2 (Anchor Vocabulary) | 自回归 |
| 动作空间 | 8192 anchors | 128×64 kinematic |
| DPO 候选数 | **1024** | **128** |
| rejected 选择 | 巧妙（像人类但不安全的） | 多目标分解 |
| PDMS | **90.0** | **89.8** |
| 闭环验证 | ✅ Bench2Drive | ❌ |

---

## 四、对我们的关键启发

### 4.1 最重要的借鉴：Rejected 选择策略

```
我们现在的做法：
  chosen = GT
  rejected = 模型碰撞轨迹
  → 差距太大 → 学形状不学安全

DriveDPO 的做法：
  rejected = "看起来像 GT 但不安全的轨迹"
  → 差距微妙 → 学安全边界

我们可以改的方向：
  不用那些跟 GT 差别巨大的碰撞轨迹
  而是找模型自己的轨迹中"差一点就安全"的那些
  → 即使我们做不到 1024 条候选
  → 哪怕 10-20 条，按碰撞严重程度排序
  → rejected 选"差一点就不碰"的那个
```

### 4.2 Iterative DPO

```
DriveDPO 做了 10 个 epoch 的 iterative 训练
每个 epoch 用最新模型重新采样候选
→ 数据始终 on-policy

我们只做了一次性的 off-policy
→ 即使效果有限，也没有迭代更新
```

### 4.3 β 的设置

```
DriveDPO:     β = 0.1    (很小，允许策略偏移)
我们 run1:    β = 1.0    (中等)
我们 run2:    β = 10.0   (很大，强约束)

他们能用小 β 是因为：
  1. 离散空间，log P 精确
  2. 候选差异微妙，不需要大力约束
  3. Iterative DPO 持续纠正漂移

我们用大 β 是因为：
  1. 连续空间，log P 近似
  2. GT vs 碰撞差距大，小 β 容易跑飞
```

### 4.4 根本差异与我们的出路

```
DriveDPO 和 TISA-DPO 都成功的根因：离散动作空间

离散空间优势：
  - log P = log softmax → 精确
  - 采样多样性高（8192 anchors）
  - 天然适合 DPO

Flow Matching 劣势：
  - log P ≈ -MSE → 近似
  - ODE 确定性强，多样性低
  - 架构层面不适合 DPO

但也不是完全没有出路：
  - 我们的 Oracle off-policy 虽然不完美，但信号极强
  - 关键是要控制过拟合（β、epoch、数据量）
  - 如果闭环验证有效，off-policy GT 就够写论文了
    → "在 Flow Matching 上首次应用 DPO" 本身就有新颖性
```

---

## 五、论文里的一句值得引用的话

> "Imitation learning faces two critical safety issues:
> 1. Even slight deviations from human trajectories may lead to dangerous outcomes
> 2. Symmetric loss penalizes deviations equally in both directions, while the safety impact can differ substantially"

这段话完美描述了我们做 DPO 的动机！

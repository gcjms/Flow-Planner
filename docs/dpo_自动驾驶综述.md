# DPO + 自动驾驶 相关工作综述

## 一、DPO 是什么？

### 1.1 背景：从 RLHF 到 DPO

**RLHF (Reinforcement Learning from Human Feedback)** 的标准流程：
1. 训练一个 Reward Model（奖励模型），学习人类偏好
2. 用 PPO 等 RL 算法，最大化 Reward Model 的打分
3. 问题：训练不稳定、超参多、工程复杂

**DPO (Direct Preference Optimization)** 的革命：
- Rafailov et al., NeurIPS 2023
- 核心贡献：证明了 RLHF 的目标函数可以被重写为一个**闭式解**
- 结果：**完全跳过了 Reward Model 的训练**，直接从偏好对 (chosen, rejected) 优化策略
- 优点：训练稳定、代码简单、效果不输 PPO

### 1.2 DPO 核心公式

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \cdot \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 正在训练的策略（带 LoRA） |
| $\pi_{ref}$ | 冻结的参考策略（原始模型） |
| $y_w$ / $y_l$ | Chosen / Rejected 样本 |
| $\beta$ | 温度系数（0.1~0.5） |

### 1.3 DPO vs RLHF vs Best-of-N

| 方法 | 需要 Reward Model? | 训练稳定性 | 推理速度 | 工程复杂度 |
|------|-------------------|-----------|---------|-----------|
| RLHF (PPO) | ✅ 需要单独训练 | 低（容易崩） | 快（N=1）| 高 |
| Best-of-N | ❌ 用规则打分 | 不训练 | **慢**（N=5+）| 中 |
| **DPO** | ❌ 不需要 | **高** | **快**（N=1）| **低** |

---

## 二、DPO + 自动驾驶 关键论文

### 2.1 DriveDPO — NeurIPS 2025 ⭐⭐⭐
**最直接的参考对象**

- **论文**：DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving
- **核心思路**：
  1. 先用模仿学习 + 规则安全分数做联合蒸馏，得到一个"统一策略分布"
  2. 再用 DPO 做迭代式轨迹级偏好对齐
  3. 模型自己采样多条轨迹 → 用安全分数排序 → 构造 (chosen, rejected) → DPO 微调
- **结果**：NAVSIM 基准 SOTA
- **对我们的启发**：
  - 它的 DPO Loss 可以直接借鉴
  - 我们的 `TrajectoryScorer` 可以替代它的 "rule-based safety score"
  - 它验证了"自己采样自己打分自己训"（Self-Play DPO）的可行性

### 2.2 TrajHF — 2025
**轨迹生成 + RLHF 的完整流程参考**

- **论文**：Learning Personalized Driving Styles via Reinforcement Learning from Human Feedback
- **核心思路**：
  1. 用多条件去噪器（类似 Diffusion）生成多模态轨迹
  2. 用 RLHF 对齐人类的驾驶风格偏好
- **对我们的启发**：
  - 它在 Diffusion 轨迹生成器上做 RLHF，和我们在 Flow Matching 上做 DPO 非常相似
  - 证明了"扩散/流匹配 + 偏好对齐"这条技术路线是可行的

### 2.3 CuriousVLA — CVPR 2026 ⭐⭐
**今晚讨论过：探索多样性 + RL 后训练**

- **论文**：Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models
- **核心思路**：
  1. 发现自回归模型的"狭隘策略"问题（= 我们说的均值回归）
  2. 阶段一：Feasible Trajectory Expansion 强制多样性
  3. 阶段二：Adaptive Diversity-Aware Sampling + Spanning Driving Reward 做 RL
- **对我们的启发**：
  - 它解决的"Narrow Policy"问题和我们的 Best-of-N 失效（5 条全撞）是同一个痛点
  - 但它用的是 VLM + 自回归，我们用 Flow Matching，天然有更好的多样性
  - 我们可以引用它的问题定义，但用更优雅的 DPO 方案解决

### 2.4 READ — ICLR 2026
**RL 微调预训练扩散驾驶模型**

- **论文**：READ: Reinforcement-based Adaptive Driving Fine-tuning
- **核心思路**：
  1. 在预训练好的扩散驾驶模型（DiffusionDrive）上做 RL 后微调
  2. 用闭环仿真的奖励信号来更新扩散模型的参数
- **对我们的启发**：
  - 这是最接近我们技术路线的工作（Diffusion + RL fine-tuning）
  - 但它用的是完整的 RL（PPO），我们用更简洁的 DPO
  - **可以直接对标**：证明 DPO 比 PPO 在扩散模型上更简单高效

### 2.5 Gen-Drive — ICRA 2025
**"生成-评估"范式**

- **论文**：Gen-Drive: Generation then Evaluation for Driving
- **核心思路**：
  1. 用 Diffusion 生成多条候选轨迹（= 我们的 Best-of-N）
  2. 用一个学习到的 Reward Model 评价
  3. 用 RL 微调 Diffusion 模型
- **对我们的启发**：
  - 它的"生成-评估"框架和我们的 Best-of-N 思路完全一致
  - 但它额外训了一个 Reward Model，我们用 DPO 直接跳过

### 2.6 LoRD — ICRA 2025
**今晚讨论过：低秩适配层做域迁移**

- **论文**：LoRD: Adapting Differentiable Driving Policies to Distribution Shifts
- **核心思路**：在策略网络中插入 LoRA 旁路，低资源适配新场景
- **对我们的启发**：
  - 直接提供了 LoRA 插入自动驾驶模型的工程参考
  - 我们的 DPO 训练就是在 LoRA 层上做的

### 2.7 Plan-R1 — 2025 ⭐⭐⭐
**nuPlan 上的 GRPO 后训练，和我们最像的实验设置**

- **论文**：Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling (arXiv:2505.17659)
- **核心思路**：
  1. 阶段一：把轨迹坐标量化成 token，用自回归方式预训练
  2. 阶段二：用 **GRPO (Group Relative Policy Optimization)** 做 RL 后微调
  3. 提出 **VD-GRPO**（Variance-Decoupled GRPO）：修改标准化方式，保留安全奖励的绝对量级
- **结果**：nuPlan 闭环 SOTA
- **对我们的启发**：
  - **直接竞争对手！** 它也在 nuPlan 上做后训练对齐，但用的是 GRPO 而非 DPO
  - 它用规则奖励（碰撞/超速/车道偏离），和我们的 `TrajectoryScorer` 异曲同工
  - 我们可以对标它的实验设计：用相同的 nuPlan 场景集，证明 DPO + Flow Matching 更优

### 2.8 DIVER — 2025 ⭐⭐
**扩散模型 + RL 指导生成多样化轨迹**

- **论文**：DIVER: Reinforced Diffusion Breaks Imitation Bottlenecks in End-to-End Autonomous Driving
- **核心思路**：
  1. **Policy-Aware Diffusion Generator (PADG)**：条件化扩散模型，感知地图和邻居交互
  2. 用 GRPO 将扩散模型当作随机策略来优化
  3. 解决模仿学习的模式坍缩问题（mode collapse）
- **对我们的启发**：
  - 它直接在 Diffusion 上做 GRPO，我们在 Flow Matching 上做 DPO，技术路线极其相近
  - 它的多样性生成机制（PADG）可以参考
  - 我们可以论证 DPO 比 GRPO 更简洁（不需要 group 采样）

### 2.9 DiffusionDriveV2 — 2025
**扩散驾驶模型的 RL 后训练迭代**

- **论文**：DiffusionDriveV2 (2025.12)
- **核心思路**：
  1. 在 DiffusionDrive（CVPR 2025 Highlight）基础上
  2. 引入 intra-anchor 和 inter-anchor GRPO
  3. 解决"多样性-质量困境"（diversity-quality dilemma）
- **对我们的启发**：
  - 进一步佐证"Diffusion/Flow + RL 后训练"是 2025~2026 主流路线
  - 它的 anchor 机制和我们的 CFG guided 采样有异曲同工之处

### 2.10 SafeDPO — 2025
**带安全约束的 DPO 变体**

- **论文**：SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
- **核心思路**：
  1. 在标准 DPO Loss 中加入二元安全指示器（binary safety indicator）
  2. 通过重新排序偏好对实现安全约束
  3. 单阶段优化，不需要额外的 Reward 或 Cost Model
- **对我们的启发**：
  - 虽然原论文是给 LLM 设计的，但其"安全约束 DPO"的思想可以直接迁移
  - 我们可以把 collision=0 的场景标记为"绝对不安全"，在 Loss 中给予更高权重

---

## 三、全景对比表

| 论文 | 会议 | 生成模型 | 对齐方法 | 评测基准 | 核心创新 |
|------|------|---------|---------|---------|---------|
| DriveDPO | NeurIPS 2025 | Transformer | DPO | NAVSIM | 统一蒸馏 + 迭代 DPO |
| TrajHF | 2025 | Diffusion | RLHF | NAVSIM | 个性化驾驶风格对齐 |
| Plan-R1 | arXiv 2025 | Autoregressive | **VD-GRPO** | **nuPlan** | 方差解耦 GRPO |
| DIVER | 2025 | Diffusion | GRPO | NAVSIM | Policy-Aware 扩散 + RL |
| DiffusionDriveV2 | 2025 | Diffusion | GRPO | NAVSIM | Anchor-based GRPO |
| READ | ICLR 2026 | Diffusion | PPO | nuPlan | RL 微调预训练扩散 |
| CuriousVLA | CVPR 2026 | VLM 自回归 | RL | NAVSIM | Narrow Policy 诊断 |
| Gen-Drive | ICRA 2025 | Diffusion | RL | NAVSIM | 生成-评估范式 |
| SafeDPO | arXiv 2025 | LLM | DPO | - | 安全约束 DPO |
| LoRD | ICRA 2025 | 通用 | 域适应 | nuPlan | LoRA 插入驾驶策略 |
| **Ours (Flow-DPO)** | - | **Flow Matching** | **DPO** | **nuPlan** | **FM + DPO + LoRA** |

---

## 四、我们的定位：Flow-Planner + DPO

### 4.1 我们的独特性

在上面所有相关工作中，**没有任何一篇**同时做到：
1. 用 **Flow Matching**（而非 Diffusion 或自回归）做轨迹生成
2. 用 **DPO**（而非 PPO/GRPO）做偏好对齐
3. 用 **LoRA**（而非全量微调）做参数高效训练

这三个组合在一起就是我们的差异化卖点。

### 4.2 论文故事线

1. **问题**：引用 CuriousVLA 的 Narrow Policy + Best-of-N 在 collision=0 场景失效
2. **观察**：展示 TrajectoryScorer 分析结果，量化 Best-of-N 的瓶颈
3. **方法**：提出 Flow-DPO —— Flow Matching + DPO + LoRA 的轻量对齐方案
4. **优势**：
   - 比 Plan-R1/DIVER (GRPO) 不需要 group 采样，更简洁
   - 比 READ (PPO) 不需要 Reward Model 和在线仿真，更稳定
   - 比 DriveDPO (Transformer) 在连续空间生成更平滑
   - 比 Best-of-N 推理时快 5 倍（N=1 vs N=5）
5. **实验**：nuPlan val14 闭环 NR-CLS 量化对比

### 4.3 Goal Conditioning 诊断结论

我们最近对 `goal conditioning` 做了一次关键诊断，这个结果对后续论文 framing 很重要。

#### 诊断动机

此前我们观察到：

- 带 `goal_point` 重训后的 checkpoint 在开环规划指标上明显差于 base
- 但同一批 checkpoint 在 DPO 候选生成里又能明显提升轨迹多样性

这两个现象表面上矛盾。更合理的怀疑是：

> 模型训练时依赖 `goal` 条件，但评估时没有把 `goal_point` 喂进去，导致评估设置与训练假设不一致。

#### 三种设置

| 设置 | 含义 | 用途 |
|------|------|------|
| `train no-goal / eval no-goal` | 标准 base 模型 | 基线 |
| `train with-goal / eval no-goal` | 训练时有 goal，测试时不给 goal | 错配诊断 |
| `train with-goal / eval gt-nearest-goal` | 训练时有 goal，测试时用 GT endpoint 最近的 cluster 作为 goal | Oracle 条件诊断 |

其中第三种是 **oracle-style evaluation**：

- 它不是可部署评估，因为使用了 GT future 的终点信息
- 但它非常适合回答一个关键问题：
  - 如果 goal 给对了，模型本身到底会不会用这个条件？

#### 关键实验结果

在远程机上对同一批 `goal_finetune` checkpoint（`e50/e60/e70`）重新跑评估后，结论非常明确。

**错误设置：`goal_mode=none`**

| 模型 | ADE | FDE | CollRate | Progress |
|------|-----|-----|----------|----------|
| base | 2.5563 | 6.2287 | 7% | 0.6411 |
| e50 | 6.2955 | 16.5964 | 20% | 0.4454 |
| e60 | 5.4893 | 15.1340 | 20% | 0.4622 |
| e70 | 4.5311 | 12.3644 | 18% | 0.5021 |

表面结论像是：

> goal-conditioned model 明显退化。

但这组结果实际上测的是：

> 一个训练时依赖 goal 的模型，在测试时不给 goal，还能剩下多少性能。

**正确诊断设置：`goal_mode=gt_nearest`**

| 模型 | ADE | FDE | CollRate | Progress |
|------|-----|-----|----------|----------|
| base | 2.7279 | 6.6853 | 5% | 0.6424 |
| e50 | 1.1068 | 2.4054 | 0% | 0.6600 |
| e60 | 1.0890 | 2.1752 | 0% | 0.6609 |
| e70 | 1.0530 | 2.0113 | 0% | 0.6578 |

这说明：

1. **模型本身已经学会了使用 goal 条件**
2. **此前看到的大幅退化，主要来自评估方式错误，而不是方法本身无效**
3. **在正确 goal 条件下，goal-conditioned checkpoint 显著优于 base**

#### 对论文叙事的意义

这个结果直接改变了我们的判断：

- 现在不能再把问题表述成“goal conditioning 没有提升主指标”
- 更准确的说法应该是：

> Goal-conditioned Flow Matching 在 oracle goal 条件下显著优于 base，说明模型已经具备利用高层意图变量拆解多模态 future 的能力；当前真正的瓶颈不在生成器本身，而在于 **测试时如何在没有 GT 的情况下选择或预测合适的 goal**。

这会把后续论文故事自然推向：

1. **先证明 goal conditioning 可以缓解 mode ambiguity**
2. **再解决 non-oracle goal selection / prediction**
3. **最后把更强的多模态候选接入 DPO**

#### 初步 non-oracle baseline 结果

为了回答“没有 GT 时 goal 从哪里来”，我们继续做了两个无需额外训练的 baseline：

1. **route-nearest**
   - 只用 `route_lanes` 几何
   - 先从 route 上提取一个前向 anchor
   - 再从 `goal_vocab` 中检索最贴 route 的 goal

2. **self-nearest**
   - 先让同一个 goal-conditioned checkpoint 在 **无 goal** 下跑一遍
   - 用它自己预测的终点检索最近的 goal cluster
   - 再带着这个 goal 重跑一次

结果如下。

**route-nearest（100 scenes）**

| 模型 | ADE | FDE | CollRate | Progress |
|------|-----|-----|----------|----------|
| base | 2.5332 | 5.9611 | 5% | 0.6403 |
| e50 | 8.7242 | 22.7152 | 23% | 0.4426 |
| e60 | 8.4508 | 22.2266 | 24% | 0.4478 |
| e70 | 8.5094 | 22.2303 | 25% | 0.4450 |

**self-nearest（100 scenes）**

| 模型 | ADE | FDE | CollRate | Progress |
|------|-----|-----|----------|----------|
| base | 2.6493 | 6.3378 | 5% | 0.6427 |
| e50 | 6.1501 | 15.0788 | 18% | 0.4508 |
| e60 | 5.2266 | 12.9208 | 17% | 0.4841 |
| e70 | 4.3135 | 11.4244 | 15% | 0.5075 |

这两个 baseline 说明：

1. **纯几何 route 检索远远不够**
   - 只靠 route polyline 去匹配 goal prototype，会在岔路、长 route、交互场景里严重选偏

2. **self-bootstrap 比 route-only 稍好，但仍然明显落后于 oracle**
   - 说明 goal-conditioned model 在没有显式 goal 时的初始终点预测还不够稳定
   - “先无条件跑一遍，再映射到最近 goal” 不能充分恢复 oracle 条件下的优势

3. **当前最真实的结论是：生成器已经具备使用正确 goal 的能力，但 non-oracle goal selection 仍是主瓶颈**

这意味着后续真正值得做的，不是立刻切到方向 3，而是优先补一个更可靠的：

`scene-conditioned goal selector / predictor`

例如：

- route-aware goal classifier
- top-K goal proposal + reranking
- 使用 encoder 全局表征预测 goal cluster id

#### 当前最重要的研究结论

如果要把这段压缩成论文里一句最关键的话，可以写成：

> Our oracle-conditioned analysis shows that the apparent degradation of goal-conditioned Flow Matching is largely caused by evaluation mismatch rather than model failure. When supplied with a correct goal prototype, the goal-conditioned planner substantially outperforms the base model, suggesting that the remaining challenge lies in deployable goal selection rather than conditional trajectory generation itself.

---

## 五、推荐阅读顺序

| 优先级 | 论文 | 理由 |
|-------|------|------|
| ⭐⭐⭐ | DPO 原论文 (NeurIPS 2023) | 必读，理解 DPO 数学基础 |
| ⭐⭐⭐ | DriveDPO | 最贴近我们的 DPO + AD 实现 |
| ⭐⭐⭐ | Plan-R1 | 同样在 nuPlan 上做后训练对齐，直接竞争对手 |
| ⭐⭐ | DIVER | Diffusion + GRPO，技术最相近 |
| ⭐⭐ | READ | Diffusion + PPO，可对标证明 DPO 更优 |
| ⭐⭐ | CuriousVLA | Narrow Policy 问题定义 |
| ⭐ | DiffusionDriveV2 | GRPO 在 Diffusion 上的更多细节 |
| ⭐ | SafeDPO | 安全约束 DPO Loss 设计参考 |
| ⭐ | LoRD | LoRA 工程参考 |

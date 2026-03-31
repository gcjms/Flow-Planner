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

---

## 三、我们的定位：Flow-Planner + DPO

### 3.1 我们的独特优势

| 对比维度 | DriveDPO | READ | CuriousVLA | **Ours (Flow-DPO)** |
|---------|---------|------|------------|---------------------|
| 生成模型 | Transformer | Diffusion | VLM (自回归) | **Flow Matching** |
| 对齐方法 | DPO | PPO (RL) | RL + Reward | **DPO** |
| 参数效率 | 全量微调 | 全量微调 | 全量微调 | **LoRA (1~2%)** |
| 多样性来源 | 多次采样 | 噪声扰动 | FTE 扩展 | **高斯噪声天然多样** |
| 似然计算 | 自回归分解 | ELBO 近似 | 自回归分解 | **直接 MSE** |

### 3.2 论文故事线（如果要写论文）

1. **问题**：引用 CuriousVLA 的 Narrow Policy 概念，指出现有模型存在均值回归
2. **观察**：展示 Best-of-N 实验中 collision=0（5 条全撞）的案例，证明推理时选择无法自救
3. **方法**：提出 Flow-DPO —— 在 Flow Matching 生成器上直接做 DPO 偏好对齐
4. **优势**：
   - 比 READ (PPO) 更简洁稳定
   - 比 DriveDPO (Transformer) 在连续空间生成更平滑
   - 比 Best-of-N 推理时快 5 倍（N=1 vs N=5）
5. **实验**：nuPlan val14 闭环 NR-CLS 对比

---

## 四、推荐阅读顺序

1. **DPO 原论文**（必读）：Rafailov et al., "Direct Preference Optimization", NeurIPS 2023
2. **DriveDPO**（核心参考）：最贴近我们方案的 AD-specific DPO 实现
3. **READ**（技术对标）：Diffusion + RL，可对比证明 DPO 更优
4. **CuriousVLA**（问题定义）：Narrow Policy 概念可以直接引用
5. **LoRD**（工程参考）：LoRA 在 AD 中的具体实现细节

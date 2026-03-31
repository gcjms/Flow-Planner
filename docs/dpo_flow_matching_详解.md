# DPO + Flow Matching 数学原理详解

## 一、前置知识：Flow Matching 的训练过程

### 1.1 Flow Matching 在做什么？

Flow Matching 的目标：学习一个速度场 $v_\theta$，它能把一团随机噪声"推"成一条合理的驾驶轨迹。

想象一条直线公路：
- 起点（$t=0$）：一团随机噪声 $x_0 \sim \mathcal{N}(0, I)$
- 终点（$t=1$）：一条真实的驾驶轨迹 $y$（来自人类驾驶数据）

在中间任意时刻 $t$，我们可以用线性插值得到一个"半成品"：

$$x_t = (1 - t) \cdot x_0 + t \cdot y$$

其中 $x_0$ 是噪声，$y$ 是目标轨迹，$t \in [0, 1]$。

### 1.2 "真实速度场"是什么？

从 $x_0$ 到 $y$ 走的是一条直线，所以这条直线的"速度"就是：

$$v_{\text{true}} = \frac{dy}{dt} = y - x_0$$

**这就是真实速度场。它不是什么外部标注，而是直接用"终点 减 起点"算出来的。**

### 1.3 训练 Loss

模型 $v_\theta$ 的训练目标是：给定中间状态 $x_t$ 和时间 $t$，预测出的速度要尽量接近真实速度：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, y} \left[ \| v_\theta(x_t, t) - (y - x_0) \|^2 \right]$$

这就是你每天在 AutoDL 上训 120 个 epoch 的那个 Loss。

---

## 二、DPO 需要的关键概念：对数似然

### 2.1 什么是 $\log P(y|x)$？

DPO 要比较模型生成 chosen 和 rejected 的"概率"。但 Flow Matching 不是分类模型，怎么算概率？

答案是：**用训练 Loss 本身来近似概率。**

$$\log \pi_\theta(y|x) \approx -\| v_\theta(x_t, t) - (y - x_0) \|^2$$

直觉：
- 如果模型对轨迹 $y$ 的速度场预测非常准（MSE 很小） → 概率高 → "这很像我会生成的轨迹"
- 如果模型对轨迹 $y$ 的速度场预测很差（MSE 很大） → 概率低 → "这不像我的风格"

### 2.2 具体怎么算？

给定一条已知轨迹 $y$（不管是 chosen 还是 rejected）：

```
步骤 1：随机采样一个噪声 x_0 ~ N(0, I)
步骤 2：随机采样一个时间 t ~ Uniform(0, 1)
步骤 3：计算中间状态 x_t = (1-t) * x_0 + t * y
步骤 4：让模型预测速度 v_pred = model(x_t, t, scene_condition)
步骤 5：计算真实速度 v_true = y - x_0
步骤 6：log_prob = -||v_pred - v_true||²
```

**注意：步骤 5 中的 $v_{\text{true}} = y - x_0$ 完全由已知量决定，不需要任何外部数据。**

---

## 三、DPO Loss 的完整推导

### 3.1 DPO 公式

给定偏好对 $(y_w, y_l)$（$y_w$ = chosen 好轨迹，$y_l$ = rejected 坏轨迹），DPO Loss 为：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \cdot \Delta \right)$$

其中：

$$\Delta = \underbrace{\left( \log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x) \right)}_{\text{模型对好轨迹的偏好变化}} - \underbrace{\left( \log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x) \right)}_{\text{模型对坏轨迹的偏好变化}}$$

### 3.2 各符号含义

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 正在训练的模型（带 LoRA 的 Flow-Planner） |
| $\pi_{\text{ref}}$ | 冻结的原始模型（不带 LoRA，作为参照基准） |
| $y_w$ | Chosen 轨迹（VLM/打分器评价好的） |
| $y_l$ | Rejected 轨迹（评价差的） |
| $x$ | 场景条件（邻居、地图、自车状态） |
| $\beta$ | 温度超参数（控制偏好强度，通常 0.1~0.5） |
| $\sigma$ | Sigmoid 函数 |

### 3.3 Loss 的梯度方向是什么？

梯度会驱动模型权重发生以下变化：
- **$\log \pi_\theta(y_w|x)$ 增大** → 模型更容易生成好轨迹
- **$\log \pi_\theta(y_l|x)$ 减小** → 模型更不容易生成坏轨迹

而 $\pi_{\text{ref}}$ 是冻结的，起到一个"锚"的作用，防止模型跑偏太远。

---

## 四、伪代码汇总

```python
def dpo_loss(model, ref_model, chosen_traj, rejected_traj, condition, beta=0.1):
    """
    model:         带 LoRA 的 Flow-Planner（正在训练）
    ref_model:     冻结的原始 Flow-Planner（参考模型）
    chosen_traj:   (B, T, 2) 好轨迹
    rejected_traj: (B, T, 2) 坏轨迹
    condition:     场景编码（邻居 + 地图 + 自车）
    """
    # 采样噪声和时间
    x0 = torch.randn_like(chosen_traj)      # 随机噪声
    t = torch.rand(B, 1, 1)                 # 随机时间 [0,1]

    # ---- Chosen 轨迹的 log prob ----
    xt_w = (1 - t) * x0 + t * chosen_traj   # 中间状态
    v_true_w = chosen_traj - x0             # 真实速度 = 终点 - 起点

    v_pred_w = model(xt_w, t, condition)     # 模型预测速度
    v_ref_w  = ref_model(xt_w, t, condition) # 参考模型预测速度

    log_pi_w     = -(v_pred_w - v_true_w).pow(2).sum()  # 训练模型的 log prob
    log_pi_ref_w = -(v_ref_w  - v_true_w).pow(2).sum()  # 参考模型的 log prob

    # ---- Rejected 轨迹的 log prob ----
    xt_l = (1 - t) * x0 + t * rejected_traj
    v_true_l = rejected_traj - x0

    v_pred_l = model(xt_l, t, condition)
    v_ref_l  = ref_model(xt_l, t, condition)

    log_pi_l     = -(v_pred_l - v_true_l).pow(2).sum()
    log_pi_ref_l = -(v_ref_l  - v_true_l).pow(2).sum()

    # ---- DPO Loss ----
    delta = (log_pi_w - log_pi_ref_w) - (log_pi_l - log_pi_ref_l)
    loss = -torch.log(torch.sigmoid(beta * delta)).mean()

    return loss
```

## 五、和 Flow-Planner 现有代码的关系

| 现有代码 | DPO 中的角色 |
|---------|-------------|
| `flow_planner.py:forward_train` | 这就是在算 $\|v_\theta - v_{\text{true}}\|^2$ |
| `velocity_model.py` | 预测速度场 $v_\theta$ 的神经网络，LoRA 插在这里 |
| `flow_ode.py` | 推理时用 ODE 求解器把噪声积分成轨迹，DPO 训练时不用 |
| `model.pth` (120 epoch) | 冻结为 $\pi_{\text{ref}}$，复制一份加 LoRA 成为 $\pi_\theta$ |

# Flow Matching + DPO：连续轨迹空间的偏好对齐

> 本文档详细解释如何在 Flow Matching（流匹配）轨迹生成模型上实现 DPO（Direct Preference Optimization）。
> 核心难点：Flow Matching 输出的是连续轨迹，不像离散模型可以直接查 softmax 概率表，
> 需要通过 ODE 求解器计算对数似然。

---

## 一、背景：为什么 DPO 需要 log π(y|x)？

DPO 的核心 Loss 公式：

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \cdot \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

其中 $\log \pi_\theta(y|x)$ 表示：**模型 θ 认为，在场景 x 下，生成轨迹 y 的对数概率有多大。**

- 对于离散模型（DriveDPO）：直接查 softmax 概率表，$\log \pi = \log P[\text{模板}_k]$
- 对于自回归模型（Plan-R1）：每个 Token 的 $\log$ softmax 概率之和
- **对于 Flow Matching（我们）**：需要通过 ODE 变量替换公式计算

---

## 二、Flow Matching 的对数似然计算

### 2.1 Flow Matching 回顾

Flow Matching 学习一个**速度场** $v_\theta(z_t, t)$，把标准高斯噪声 $z_0 \sim \mathcal{N}(0, I)$ 通过 ODE 积分变换成目标轨迹 $y$：

$$\frac{dz_t}{dt} = v_\theta(z_t, t), \quad t \in [0, 1]$$
$$z_0 \sim \mathcal{N}(0, I) \xrightarrow{\text{ODE积分}} z_1 = y \text{ (生成的轨迹)}$$

### 2.2 连续归一化流 (CNF) 的对数似然公式

根据变量替换定理（Change of Variables），从噪声 $z_0$ 到轨迹 $y = z_1$ 的概率密度变化：

$$\log p_\theta(y) = \log p_0(z_0) - \int_0^1 \text{tr}\left(\frac{\partial v_\theta(z_t, t)}{\partial z_t}\right) dt$$

其中：
- $\log p_0(z_0)$：噪声空间的先验概率（标准高斯）= $-\frac{1}{2}\|z_0\|^2 - \frac{d}{2}\log(2\pi)$
- $\int_0^1 \text{tr}(\cdot) dt$：速度场散度的时间积分（**雅可比行列式的对数**）

### 2.3 计算步骤

```
给定：场景 x，轨迹 y（chosen 或 rejected）

Step 1：反向 ODE 积分（把轨迹"退回"噪声空间）
        z_1 = y
        z_0 = ODE_reverse(z_1, v_θ, t=1→0)
        
Step 2：计算噪声空间先验概率
        log_p_z0 = -0.5 * ||z_0||² - d/2 * log(2π)

Step 3：正向积分计算散度（雅可比修正项）
        log_det = ∫₀¹ tr(∂v_θ/∂z_t) dt
        （实际中用 Hutchinson 估计器近似）

Step 4：合并得到轨迹的对数似然
        log_π(y|x) = log_p_z0 - log_det
```

---

## 三、实现伪代码

### 3.1 计算单条轨迹的 log probability

```python
import torch
from torchdiffeq import odeint

def compute_log_prob(velocity_model, scenario_features, trajectory, device='cuda'):
    """
    计算 Flow Matching 模型对一条轨迹的对数概率。
    
    Args:
        velocity_model: 速度场网络 v_θ(z_t, t, context)
        scenario_features: 场景编码 (ego history + neighbors + map)
        trajectory: 目标轨迹 y, shape = (T, 2) 其中 T=10 个时间步, 2=(x,y)
    
    Returns:
        log_prob: 标量，log π_θ(y|x)
    """
    d = trajectory.numel()  # 轨迹总维度 = 10 * 2 = 20
    z_1 = trajectory.flatten()  # (20,)
    
    # === Step 1: 反向 ODE 积分，从 t=1 退回 t=0 ===
    # 同时计算散度（用 Hutchinson 估计器）
    
    # 随机向量用于 Hutchinson 迹估计
    epsilon = torch.randn_like(z_1)
    
    def augmented_dynamics(t, state):
        """增广 ODE：同时积分轨迹和散度"""
        z_t, _ = state[:-1], state[-1]  # z_t 和累积散度
        z_t = z_t.detach().requires_grad_(True)
        
        # 速度场前向计算
        v = velocity_model(z_t, t, scenario_features)
        
        # Hutchinson 迹估计: tr(∂v/∂z) ≈ εᵀ (∂v/∂z) ε
        vjp = torch.autograd.grad(v, z_t, epsilon, create_graph=False)[0]
        div_estimate = (vjp * epsilon).sum()
        
        return torch.cat([v, div_estimate.unsqueeze(0)])
    
    # 初始状态：[z_1, 累积散度=0]
    initial_state = torch.cat([z_1, torch.zeros(1, device=device)])
    
    # 反向积分 t: 1 → 0
    solution = odeint(augmented_dynamics, initial_state, 
                      torch.tensor([1.0, 0.0], device=device),
                      method='dopri5', atol=1e-5, rtol=1e-5)
    
    z_0 = solution[-1, :-1]          # 退回到的噪声
    neg_log_det = solution[-1, -1]   # 累积散度（负号因为是反向积分）
    
    # === Step 2: 噪声空间先验 ===
    log_p_z0 = -0.5 * (z_0 ** 2).sum() - 0.5 * d * torch.log(torch.tensor(2 * 3.14159))
    
    # === Step 3: 合并 ===
    log_prob = log_p_z0 + neg_log_det  # 注意符号，反向积分时散度取反
    
    return log_prob
```

### 3.2 DPO Loss 计算

```python
def dpo_loss(model, ref_model, scenario, chosen_traj, rejected_traj, beta=0.1):
    """
    计算 DPO Loss。
    
    Args:
        model: 当前正在训练的模型（带 LoRA）
        ref_model: 冻结的参考模型（原始基座）
        scenario: 场景编码特征
        chosen_traj: 被选中的"好"轨迹 (T, 2)
        rejected_traj: 被拒绝的"坏"轨迹 (T, 2)
        beta: DPO 温度系数（控制偏离参考模型的程度）
    
    Returns:
        loss: 标量
    """
    # 当前模型对两条轨迹的评价
    log_pi_w = compute_log_prob(model.velocity_model, scenario, chosen_traj)
    log_pi_l = compute_log_prob(model.velocity_model, scenario, rejected_traj)
    
    # 参考模型对两条轨迹的评价（不参与梯度计算）
    with torch.no_grad():
        log_ref_w = compute_log_prob(ref_model.velocity_model, scenario, chosen_traj)
        log_ref_l = compute_log_prob(ref_model.velocity_model, scenario, rejected_traj)
    
    # DPO 核心公式
    logit = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
    loss = -torch.nn.functional.logsigmoid(logit)
    
    return loss
```

### 3.3 完整训练循环

```python
def train_dpo(base_model, preference_dataset, config):
    """
    DPO 训练主循环。
    
    Args:
        base_model: 预训练好的 Flow-Planner 基座
        preference_dataset: [(scenario, chosen_traj, rejected_traj), ...]
        config: 超参数配置
    """
    # 1. 冻结基座作为参考模型
    ref_model = deepcopy(base_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # 2. 在基座上插入 LoRA
    model = insert_lora(base_model, 
                        target_modules=['velocity_model'],  # 只在速度场网络上插 LoRA
                        rank=config.lora_rank,               # rank=4~10
                        alpha=config.lora_alpha)              # alpha=16
    
    # 3. 只有 LoRA 参数可训练
    trainable_params = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)  # lr=5e-5
    
    # 4. 训练
    for epoch in range(config.num_epochs):  # 15 epochs
        total_loss = 0
        for batch in DataLoader(preference_dataset, batch_size=config.batch_size):
            scenario, chosen, rejected = batch
            
            loss = dpo_loss(model, ref_model, scenario, chosen, rejected, 
                           beta=config.beta)
            
            optimizer.zero_grad()
            loss.backward()  # 梯度只流过 LoRA 参数
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(preference_dataset)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
    
    # 5. 保存 LoRA 权重（仅几十 KB）
    save_lora_weights(model, 'checkpoints/dpo_lora.pth')
```

---

## 四、训练配置

```yaml
# DPO 超参数（参考 LoRD + DPO 原论文）
num_epochs: 15
batch_size: 16
learning_rate: 5e-5
beta: 0.1                    # DPO 温度系数，越大越保守
lora_rank: 4                 # LoRA 低秩维度，先从 4 开始
lora_alpha: 16               # LoRA 缩放因子
lora_dropout: 0.1            # LoRA dropout
ode_solver: dopri5            # ODE 求解器
ode_atol: 1e-5               # ODE 绝对容差
ode_rtol: 1e-5               # ODE 相对容差
hutchinson_samples: 1         # Hutchinson 迹估计的采样数
```

---

## 五、计算成本估算

| 步骤 | 耗时 | 说明 |
|------|------|------|
| 单次 log_prob 计算 | ~50ms | 需要一次完整的反向 ODE 积分 |
| 单个样本 DPO loss | ~200ms | 4 次 log_prob（model×2 + ref×2） |
| 1 epoch (5000 样本) | ~17 分钟 | batch_size=16，单卡 4090 |
| 完整训练 (15 epoch) | **~4 小时** | 可接受范围 |

---

## 六、与离散模型 DPO 的关键区别

| 维度 | 离散模型 (DriveDPO/Plan-R1) | 我们 (Flow Matching) |
|------|---|---|
| log π 计算 | 查 softmax 概率表 | ODE 反向积分 + 散度估计 |
| 计算成本 | O(1) | O(NFE)（ODE 步数） |
| 精度 | 精确 | 近似（Hutchinson 估计器引入方差） |
| 梯度流 | 简单的 softmax 反传 | 通过 ODE 求解器反传（adjoint method） |
| 优势 | 快 | 轨迹天然平滑，无量化误差 |

> **这是我们技术方案的核心创新点**：首次在 Flow Matching 连续轨迹生成模型上实现 DPO 偏好对齐。
> 此前所有的 DPO + 自动驾驶工作（DriveDPO, Plan-R1）都依赖离散化的轨迹表示。

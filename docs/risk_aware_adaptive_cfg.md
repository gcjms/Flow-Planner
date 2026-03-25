# Risk-Aware Adaptive CFG for Flow Matching Motion Planning

## 概述

本文档描述 FlowPlanner 的核心创新点：**基于驾驶风险评估的自适应 Classifier-Free Guidance（CFG）引导方法**。

核心思想：将 CFG 权重 `w` 从固定超参数升级为由风险评估网络动态预测的值，使模型在不同风险等级的场景中自动调整对周围交通参与者的感知敏感度。

---

## 1. 动机

### 1.1 固定 CFG 的局限性

Flow Planner 使用 Classifier-Free Guidance (CFG) 增强模型对周围车辆的感知能力：

```
v_guided = (1 - w) · v_uncond + w · v_cond
```

其中 `v_uncond` 是忽略邻居的预测，`v_cond` 是考虑邻居的预测，`w` 控制引导强度。

**问题**：当前使用固定 `w=1.8`，但不同场景的最优 `w` 不同：

| 场景类型 | 最优 w | 原因 |
|---------|--------|------|
| 空旷直道 | ~0.5-1.0 | 无风险，过度引导反而不自然 |
| 正常跟车 | ~1.5-2.0 | 中等风险，需要适度关注 |
| 密集路口 | ~2.5-3.0 | 高风险，需要强感知 |
| 紧急避让 | ~3.5-4.0 | 极高风险，最大化安全响应 |

我们的实验验证了这一点：CFG 权重从 `w=1.0` 到 `w=2.5`，NR Score 从 71.2% 提升至 80.2%（+9分），但 `w=1.8` 目前是整体最优的折中值。

### 1.2 风险视角的重新定义

**w 的本质是风险响应强度**：
- `w` 低 → 模型认为周围没有威胁，轻松驾驶
- `w` 高 → 模型高度关注周围车辆，积极避险

因此，预测最优 `w` 等价于评估当前场景的驾驶风险等级。

---

## 2. 技术方案

### 2.1 风险评估网络 (Risk Network)

设计一个轻量级 MLP 网络，从场景特征中预测风险分数，并映射为 CFG 权重：

```
场景特征 → Risk Network → 风险分数 r ∈ [0, 1] → w = w_min + r × (w_max - w_min)
```

#### 输入特征

| 特征 | 含义 | 物理意义 |
|------|------|---------|
| `TTC` | Time-to-Collision (最小碰撞时间) | 越小越危险 |
| `THW` | Time Headway (跟车时距) | 越小越危险 |
| `DRAC` | Deceleration Rate to Avoid Crash | 避撞所需减速度 |
| `min_dist` | 最近邻车辆距离 | 基础安全距离 |
| `n_neighbors` | 有效邻居数量 | 交互复杂度 |
| `ego_speed` | 自车速度 | 速度越高风险越大 |
| `max_delta_v` | 最大速度差 | 博弈强度 |
| `intersection_type` | 路口类型 (one-hot) | 场景类别 |

#### 网络结构

```python
class RiskNetwork(nn.Module):
    """风险评估网络：场景特征 → CFG 权重"""
    def __init__(self, input_dim=12, hidden_dim=64, w_min=0.5, w_max=4.0):
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 r ∈ [0, 1]
        )
    
    def forward(self, scene_features):
        risk_score = self.mlp(scene_features)  # (B, 1)
        w = self.w_min + risk_score * (self.w_max - self.w_min)
        return w, risk_score
```

参数量：~5K（相比主模型 14.28M 可忽略不计）

### 2.2 训练方式

#### Step 1: Grid Search（~1天 GPU）

冻结已训好的 Flow Planner 基线模型，对 Val14 每个场景，遍历不同 `w` 值：

```python
w_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
for scenario in val14_scenarios:
    for w in w_candidates:
        nr_score = evaluate(model, scenario, cfg_weight=w)
    best_w[scenario] = argmax(nr_scores)
```

得到约 1000 组 `(场景特征, 最优 w)` 训练数据。

#### Step 2: 训练 Risk Network（~几分钟）

```python
loss = MSE(risk_network(scene_features), optimal_w_from_grid_search)
```

独立训练，不动基线模型权重，零风险。

#### Step 3: 融入超让标签（博世独有数据优势）

已有的超让标签直接提供风险先验：

```
「超」场景 → 高风险 → 高 w (2.5-4.0)    # 需要激进博弈
「让」场景 → 中风险 → 中 w (1.5-2.5)    # 需要保守避让
「正常跟车」→ 低风险 → 低 w (0.5-1.5)   # 正常驾驶即可
```

超让标签扩充 Step 2 的训练数据集。

### 2.3 扩展：Risk-Aware 自适应 ODE 步数

风险分数还可以用于动态调整 ODE 求解步数：

```python
# 高风险 → 多步求解（精度高，安全性强）
# 低风险 → 少步求解（速度快，效率高）
risk_score = risk_network(scene_features)
ode_steps = min_steps + int(risk_score * (max_steps - min_steps))
# 例如: risk=0.2 → 2步, risk=0.8 → 6步
```

| 场景 | 风险 | ODE 步数 | 推理频率 |
|------|------|---------|---------|
| 空旷直道 | 低 | 2 步 | ~25Hz |
| 正常跟车 | 中 | 4 步 | ~12Hz |
| 密集路口 | 高 | 6 步 | ~8Hz |

平均推理速度提升约 30-50%（相比固定 4 步），同时高风险场景安全性不降。

---

## 3. 与现有工作的对比

| 方法 | 引导方式 | 自适应维度 | 训练成本 | 可解释性 |
|------|---------|-----------|---------|---------|
| Flow Planner (原始) | 固定 w=1.8 | 无 | 0 | 无 |
| β-CFG | 时间步自适应 | 时间 | 低 | 低 |
| GuideFlow 激进度参数 | 显式激进度标签 | 场景级 | 高（需联合训练） | 中 |
| SDD Planner | Style-guided | 风格级 | 高（Diffusion） | 中 |
| **Ours: Risk-Aware CFG** | **风险评估网络** | **场景级** | **极低（免重训）** | **高（TTC/THW可解释）** |

核心差异：
1. **免训练**：Risk Network 独立训练，不动基线模型
2. **可解释**：w 的值可以追溯到 TTC、THW 等物理风险指标
3. **双控制**：同时控制 CFG 权重和 ODE 步数
4. **超让标签**：博世独有的训练数据优势

---

## 4. 专利与论文规划

### 专利方向

**「一种基于驾驶风险评估的 Flow Matching 自适应引导方法及系统」**

权利要求：
1. 一种风险评估网络，输入包括 TTC、THW、DRAC、邻车数、ego 速度等驾驶安全特征
2. 该网络输出风险分数，映射为 Flow Matching 的 CFG 引导权重
3. 所述风险分数同时用于动态调整 ODE 求解步数
4. 在推理阶段即插即用，无需重新训练基线生成模型

### 论文方向

**"Risk-Aware Adaptive Inference for Flow Matching-based Motion Planning"**

核心贡献：
1. 提出 Risk-Aware CFG：首个将驾驶风险评估与 Flow Matching 引导强度关联的框架
2. 提出自适应 ODE 步数：根据风险等级动态调整推理精度/速度平衡
3. 在 nuPlan、InterPlan 基准上验证有效性
4. 可解释性分析：风险网络学到的 w 与 TTC/THW 的物理关系

---

## 5. 实现路径

```
第1-3月: 基线复现 + 全量训练 → Val14 NR ≥ 88-90
第4月:   Grid Search 找每个场景的最优 w（~1天）
第4月:   训练 Risk Network（~几分钟）+ 融入超让标签
第5月:   实现自适应 ODE 步数 + 消融实验
第6月:   InterPlan 验证 + 论文撰写
第7-10月: 公司数据迁移 + 实车验证
```

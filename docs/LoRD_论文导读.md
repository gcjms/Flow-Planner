# LoRD 论文导读：如何用 LoRA 微调自动驾驶策略

> 论文：LoRD: Adapting Differentiable Driving Policies to Distribution Shifts
> 作者：Christopher Diehl (TU Dortmund), Peter Karkus, Sushant Veer, Marco Pavone (NVIDIA + Stanford)
> 会议：ICRA 2025 | 代码：https://github.com/rst-tu-dortmund/LoRD

---

## 一、这篇论文解决什么问题？

一句话：**模型在 A 城市训好了，搬到 B 城市就拉胯了，怎么办？**

具体来说：
- 用美国（波士顿+匹兹堡）数据训练的自动驾驶模型
- 直接部署到新加坡 → 性能暴跌（交通规则不同、车辆靠左行驶等）
- 直接用新加坡数据微调 → 波士顿的能力又丢了（**灾难性遗忘**）

### 他们想要的效果：
在新加坡能开好，**同时不忘记**怎么在波士顿开。

---

## 二、他们用的基座模型是什么？

**DTPP** (Differentiable Trajectory Prediction and Planning)，架构如下：

```
输入：自车历史 + 邻居历史 + 地图（车道线/人行道）
  ↓
Encoder（Transformer 融合编码）
  ↓
Decoder（预测邻居轨迹 + 打分器选最优自车轨迹）
  ↓
输出：最优自车轨迹 + 邻居预测轨迹
```

### 和 Flow-Planner 的对应关系

| DTPP | Flow-Planner | 说明 |
|------|-------------|------|
| AgentEncoder (LSTM) | neighbor/ego 编码器 | 把历史轨迹编码成特征向量 |
| VectorMapEncoder | lane 编码器 | 把地图编码成特征向量 |
| Transformer 融合 | Transformer 融合 | Attention 融合所有特征 |
| AgentDecoder (MLP) | Flow Matching 解码器 | **这里不同！** DTPP 直接 MLP 出轨迹，我们用 ODE 积分出轨迹 |
| ScoreDecoder | TrajectoryScorer | 对候选轨迹打分 |

---

## 三、LoRA 是怎么插进去的？（核心部分）

### 3.1 标准 LoRA 原理

原始的 Linear 层做的运算：
```
y = W · x        # W 是 (out, in) 的大矩阵
```

LoRA 的做法：**冻结 W，加一个低秩旁路**
```
y = W · x + (B · A) · x · scaling

其中：
  W: (out, in)  ← 冻结，不更新
  A: (r, in)    ← 可训练，随机初始化
  B: (out, r)   ← 可训练，初始化为 0
  r: 秩，远小于 in 和 out（比如 r=10）
  scaling = alpha / r
```

**直觉**：A 把高维输入压缩到 r 维，B 再展开回去。等于学了一个"小补丁"叠加在原始 W 上。

### 3.2 LoRD 的代码实现

```python
# 文件：lora_layers.py（第 42-108 行）

class Linear(nn.Linear, LoRALayer):
    def __init__(self, in_features, out_features, r=0, ...):
        # 创建低秩矩阵
        self.lora_A = nn.Parameter(zeros((r, in_features)))    # 压缩
        self.lora_B = nn.Parameter(zeros((out_features, r)))   # 展开
        self.scaling = lora_alpha / r
        self.weight.requires_grad = False  # 冻结原始权重！

    def forward(self, x):
        # 只计算低秩旁路，不碰原始 W
        result = (dropout(x) @ A^T @ B^T) * scaling
        return result
```

### 3.3 LoRA 插在哪里？

LoRD 不是替换已有的层，而是**新增了一个 ResidualDecoder**：

```python
# 文件：scenario_tree_prediction.py（第 143-146 行）

# 原始轨迹预测
trajectory = self.agent_traj_decoder(decoding, ...)

# LoRA 残差补丁
res_trajectory = self.residual_decoder(decoding)  # ← 这就是 LoRA 层！
trajectory = trajectory + res_trajectory           # ← 原始 + 补丁
```

**翻译成人话**：
1. 原始 Decoder 正常预测轨迹（冻结，不动）
2. ResidualDecoder（只有一个 LoRA Linear 层）也预测一个"修正量"
3. 两者相加 = 最终轨迹

这个 ResidualDecoder 的结构极其简单：

```python
# ResidualDecoder 本质上就是一个 LoRA Linear 层
# 输入: 512 维特征
# 输出: 30 个参数 (3 x 10 = (x,y,yaw) × 10 个时间步)
# rank: 10

input (512) → LoRA_A (10, 512) → 压缩到 10 维
           → LoRA_B (30, 10)  → 展开到 30 维 → reshape 成轨迹
```

---

## 四、结果怎么读？

论文中的结果表：

| 方法 | OOD CL-NR | OOD CL-R | 平均(ID+OOD) CL-NR | 平均(ID+OOD) CL-R |
|------|-----------|----------|--------------------|--------------------|
| 直接微调 | 0.677 | 0.681 | 0.678 | 0.616 |
| **LoRD + CR** | **0.750** | **0.753** | **0.722** | **0.689** |

### 各指标含义：
- **OOD** = 新加坡（没见过的域），越高越好
- **ID+OOD 平均** = 波士顿 + 新加坡的综合得分，越高说明遗忘越少
- **CL-NR** = Closed-Loop Non-Reactive（闭环非反应式，邻居不受自车影响）
- **CL-R** = Closed-Loop Reactive（闭环反应式，邻居会躲避自车）
- **CR** = Closed-loop Regularization（闭环正则化，额外技巧防止遗忘）

### 对比着看：
- 直接微调：OOD 还行（0.677），但综合得分低（0.616）→ 说明**忘记了波士顿**
- LoRD + CR：OOD 更高（0.750），综合也高（0.689）→ 新加坡强了，波士顿也没忘

---

## 五、训练配置

```yaml
# LoRD 的超参数
train_epochs: 15                    # 只需 15 轮
batch_size: 16
learning_rate: 5e-5                 # 很小的学习率
lora_rank: 10                       # 秩 = 10
residual_decoder_dropout: 0.5       # 50% dropout 防止过拟合
weight_aux_collision_reward_loss: 2  # 碰撞惩罚权重
weight_aux_progress_reward_loss: 1  # 行驶进度奖励权重
```

---

## 六、对我们 Flow-DPO 的启发

### 6.1 能直接搬过来用的

| LoRD 做法 | 我们怎么用 |
|----------|----------|
| LoRA rank=10 | 我们用 rank=4~10，先试 4 |
| 新增 ResidualDecoder（不改原始网络） | 我们也可以在 Flow-Planner Decoder 外面加旁路 |
| 冻结整个基座，只训 LoRA | ✅ 完全一致 |
| 15 epoch + lr=5e-5 | DPO 训练也用类似超参 |

### 6.2 关键区别

| | LoRD | 我们 |
|--|------|------|
| 微调目标 | 域适应（学新交通规则） | 安全对齐（学避撞偏好） |
| Loss | Imitation Loss（模仿新域数据） | **DPO Loss**（偏好对比） |
| 数据需求 | 新域的专家轨迹 | (chosen, rejected) 偏好对 |
| LoRA 插在哪 | 邻居轨迹预测的 Decoder | Flow Matching 的速度场网络 |

### 6.3 我们的 LoRA 应该插在哪？

LoRD 把 LoRA 插在了邻居预测 Decoder 上。但我们的目标不同——我们要改变的是**自车轨迹的生成偏好**，所以应该插在 Flow-Planner 的 **velocity_model**（速度场预测网络）上：

```python
# 我们的做法（概念性伪代码）
class FlowPlannerWithLoRA:
    def __init__(self, base_model):
        self.base_model = base_model  # 冻结
        # 在 velocity_model 的 Linear 层上插入 LoRA
        for name, module in base_model.velocity_model.named_modules():
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(module, rank=4)
                # 替换原始层
```

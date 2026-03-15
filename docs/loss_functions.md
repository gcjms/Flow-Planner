# Flow-Planner 训练 Loss 详解

## 总 Loss 公式

```
Total Loss = 1.0 × ego_planning_loss + 0.5 × consistency_loss
```

权重定义在 `flow_planner/script/core/flow_matching.yaml`:
```yaml
ego_planning_loss: 1.0
consistency_loss: 0.5
```

---

## Loss 1: ego_planning_loss (权重 1.0)

**类型**: MSE Loss (均方误差)

**公式**: `loss = mean(Σ (prediction - target)²)`

**作用**: 让模型预测的去噪轨迹尽可能接近真实目标

**计算过程**:
```
1. GT轨迹 (B, 1, 80, 4) → 切成 7 个 action token → target_tokens (B, 7, 20, 4)
2. 加噪后的轨迹 → decoder 预测 → prediction (B, 7, 20, 4)
3. batch_loss = MSE(prediction, target_tokens)   # (B, 7, 20, 4)
4. ego_planning_loss = mean(sum(batch_loss, dim=-1))  # 先对4维求和，再取平均
```

**注意**: 这里的 target 取决于 `model_type`:
- `x_start`: target = 干净的 GT 轨迹
- `noise`: target = 加的噪声 ε
- `velocity`: target = 干净轨迹 - 噪声 (Flow Matching 的速度场)

当前配置用的是 `model_type: x_start`，即直接预测干净轨迹。

---

## Loss 2: consistency_loss (权重 0.5)

**类型**: MSE Loss

**公式**: `loss = mean(MSE(token_i 的最后 10 帧, token_{i+1} 的前 10 帧))`

**作用**: 强制相邻 action token 在重叠区域的预测保持一致

**计算过程**:
```
对每对相邻 token (i, i+1), 其中 i = 0, 1, ..., 4:
  overlap_前 = prediction[:, i,   -10:, :]    # token_i 的最后 10 帧
  overlap_后 = prediction[:, i+1, :10,  :]    # token_{i+1} 的前 10 帧
  loss_i = mean(sum(MSE(overlap_前, overlap_后), dim=-1))

consistency_loss = average(loss_0, loss_1, ..., loss_4)
```

**为什么需要**: 
- 模型对每个 token 是独立预测的
- 没有 consistency_loss，重叠帧的预测可能不一致
- 推理时拼接会导致轨迹跳变

**为什么权重是 0.5 而不是 1.0**:
- consistency_loss 是辅助 loss，不应喧宾夺主
- 主 loss (ego_planning_loss) 保证预测准确
- consistency_loss 保证预测一致，权重适当降低防止过度约束

---

## 代码位置

| 组件 | 文件 | 行号 |
|------|------|------|
| loss 计算 | `flow_planner/model/flow_planner_model/flow_planner.py` | L152-163 |
| loss 加权求和 | `flow_planner/core/flow_matching_core.py` | L29 |
| loss 权重配置 | `flow_planner/script/core/flow_matching.yaml` | L13-14 |
| MSE 定义 | `flow_planner/model/flow_planner_model/flow_planner.py` | L54 |

## 训练过程中 loss 的典型变化

```
Epoch 1:   total ≈ 0.38  (模型刚开始学)
Epoch 5:   total ≈ 0.09  (快速收敛)
Epoch 20:  total ≈ 0.05  (进入平台期)
Epoch 100: total ≈ 0.03  (收敛)
```

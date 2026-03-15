# CFG (Classifier-Free Guidance) 详解 & 自适应 w 研究方向

## 总览

```
CFG 的作用: 推理时放大"邻居交互"对规划轨迹的影响
关键公式:   v_guided = v_uncond + w × (v_cond - v_uncond)
当前实现:   w = cfg_weight = 1.8 (全局固定超参数)
问题:       固定 w 对所有场景一视同仁，缺少场景针对性
```

---

## CFG 完整工作流程

### 训练阶段

```
trainer.py → FlowMatchingCore.train_step()
  └─ FlowPlanner.forward_train(data)
       ├─ ① 掷骰子决定 cfg_flags
       │    roll_dice = rand(B,1)                     → (B, 1)
       │    cfg_flags = (roll_dice > 0.3).int()       → (B, 1)  
       │    # 30% 概率 = 0 (无条件), 70% = 1 (有条件)
       │
       ├─ ② mask 邻居数据 (cfg_type='neighbors' 时)
       │    data.neighbor_past *= cfg_flags             # flag=0 → 邻居全置零
       │
       ├─ ③ cfg_flags 传入 Decoder
       │    cfg_embedding = Embedding(cfg_flags)        → (B, 1, hidden_dim)
       │    # flag=0 → 嵌入向量 A ("我没有条件信息")
       │    # flag=1 → 嵌入向量 B ("我有条件信息")
       │    cfg_embedding 被加入 DiT 的条件向量 y
       │
       └─ ④ 同一个模型同时学会两种模式
            有条件 (flag=1): 正常看到邻居 → 学会交互规划
            无条件 (flag=0): 看不到邻居   → 学会只看路规划
```

**文件**: [flow_planner.py L121-167](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_planner.py#L121-L167)

---

### 推理阶段

```
FlowPlanner.forward_inference(data, use_cfg=True, cfg_weight=1.8)
  │
  ├─ ① 构造 2B 的 cfg_flags
  │    cfg_flags = [1,1,...,1, 0,0,...,0]    → (2B,)
  │                 前B个有条件  后B个无条件
  │    data = data.repeat(2)                 # 数据复制一份
  │
  ├─ ② Encoder: 同时编码 2B 个样本
  │    前B个: 正常邻居 → 有条件编码
  │    后B个: 邻居被 mask → 无条件编码
  │
  ├─ ③ ODE 积分时，每步做引导组合
  │    v_cond   = decoder(x[:B], t)   # 有条件预测
  │    v_uncond = decoder(x[B:], t)   # 无条件预测
  │    v = v_uncond + w × (v_cond - v_uncond)    ← 核心公式
  │
  └─ ④ 用 v 积分得到最终轨迹
```

**文件**: [flow_planner.py L169-190](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_planner.py#L169-L190), [flow_ode.py L44+](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_utils/flow_ode.py#L44)

---

### w 的含义

```
差值 = v_cond - v_uncond = "邻居对我规划的影响方向"

场景A (空旷道路):   v_cond ≈ v_uncond → 差值≈0 → w×0 ≈ 0 (w无所谓)
场景B (行人横穿):   v_cond ≠ v_uncond → 差值大 → w放大避让行为

w = 0.0  → 完全忽略邻居
w = 1.0  → 标准条件生成
w = 1.8  → 增强邻居影响 80% (当前默认)
w = 3.0  → 过度保守
```

---

## 固定 w 的问题

```
场景            理想w    固定w=1.8   结果
────────────    ─────    ────────    ──────────────
行人横穿        3.0+     1.8        避让力度不够
正常跟车        1.0      1.8        过度保守
空旷直行        0-1.0    1.8        浪费 2× 前向计算
密集路口        2.0-2.5  1.8        缺少针对性
```

核心矛盾: 固定 w 是场景级粗控制，真实驾驶需要实例级精细控制。

---

## 改进方向

### 方向 1: 时间步自适应 w

```
改动: flow_ode.py 中 1 行
当前:  v = v_uncond + 1.8 * (v_cond - v_uncond)
改进:  w(t) = w_max * (1 - t)
       v = v_uncond + w(t) * (v_cond - v_uncond)
       # ODE 早期(噪声大)强引导，后期(收敛)弱引导

可行性: ⭐⭐⭐⭐⭐ | 改动量: 1行 | 收益: ⭐⭐
```

### 方向 2: 场景自适应 w

```
改动: 加一个 2 层 MLP (~200 参数)

scene_feature = routes_cond                  # (B, hidden_dim)
w_pred = sigmoid(MLP(scene_feature)) * 3.0   # (B,) ∈ [0, 3]
v = v_uncond + w_pred * (v_cond - v_uncond)

难点: 训练信号 → 需要 closed-loop 评估做 reward (类似 RL)
可行性: ⭐⭐⭐⭐ | 改动量: 小 | 收益: ⭐⭐⭐
```

### 方向 3: 维度级引导向量

```
当前:  标量 w 对 (x, y, cos, sin) 同等放大
改进:  向量 w̃ ∈ R^4 分维度控制

w_vec = MLP(condition)                         # (B, state_dim)
v = v_uncond + w_vec ⊙ (v_cond - v_uncond)    # 逐元素
# 例: 只增强 y 方向(横向避让), 不影响 x 方向(纵向速度)

可行性: ⭐⭐⭐ | 改动量: 中 | 收益: ⭐⭐⭐
```

### 方向 4: 去掉 CFG (省一半推理计算)

```
当前:  推理需要 2× forward (有条件+无条件)
改进:  用 DiT 内部的 gate 机制替代外部 CFG

gate 本身就在控制每层 "关注邻居多少"
加强 gate 学习 → 模型自适应关注度 → 不需要外部 CFG

好处: 推理速度 ×2, 不需要数据复制
代价: 失去推理时可调的灵活性

可行性: ⭐⭐⭐ | 改动量: 中 | 收益: ⭐⭐⭐⭐
```

---

## 建议实验顺序

```
1. 时间步自适应 w    → 改 1 行，先试
2. 场景自适应 w      → 效果好就深入
3. 去掉 CFG         → gate 够强就省推理计算
```

## 代码位置

| 组件 | 文件 | 关键行 |
|------|------|--------|
| CFG flag 生成 | `flow_planner.py` | L134-135 |
| 邻居 mask | `flow_planner.py` | L62-67 |
| CFG 嵌入 | `decoder.py` | L108 |
| ODE 中 w 使用 | `flow_ode.py` | L44+ |
| cfg_weight 配置 | `flow_planner_standard.yaml` | cfg_weight: 1.8 |

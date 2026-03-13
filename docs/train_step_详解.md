# train_step 完整调用链详解

一次 `train_step` 的完整执行过程，从数据进入到 loss 输出。

---

## 总览调用链

```
trainer.py 训练循环
  └─ core.train_step(model, data)           # FlowMatchingCore
       ├─ input_aug(data)                    # StatePerturbation 数据增强
       └─ model(data, mode='train')          # FlowPlanner.forward_train()
            ├─ ① 生成 CFG 标志
            ├─ ② prepare_model_input()       # 预处理 + 归一化
            ├─ ③ Encoder                     # FlowPlannerEncoder
            ├─ ④ Flow ODE 采样              # FlowODE.sample()
            ├─ ⑤ Decoder (DiT)              # FlowPlannerDecoder
            └─ ⑥ 计算 Loss
```

---

## 详细步骤

### Step 0: 数据增强 — `StatePerturbation.__call__(data)`

**文件**: [state_aug.py](file:///home/gcjms/Flow-Planner/flow_planner/data/augmentation/state_aug.py#L130-L140)

```python
# FlowMatchingCore.train_step()
data = self.input_aug(data)  # StatePerturbation
```

| 子步骤 | 操作 | 说明 |
|--------|------|------|
| `augment()` | 对 ego_current 加均匀噪声 | 扰动 x,y,heading,vel,acc,steer,yaw_rate |
| `refine_future_trajectory()` | 五次多项式插值 | 扰动后起点 → GT 终点平滑衔接 |
| `centric_transform()` | 坐标系重新中心化 | 所有数据转到**扰动后的** ego 坐标系 |

**输入/输出**: `NuPlanDataSample` → `NuPlanDataSample`（same structure，值已变）

---

### Step ①: 生成 CFG 标志

**文件**: [flow_planner.py L133-135](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_planner.py#L133-L135)

```python
B = data.ego_current.shape[0]  # batch size
roll_dice = torch.rand((B, 1))
cfg_flags = (roll_dice > self.cfg_prob).int()  # 1=保留条件, 0=mask掉
```

- `cfg_prob ≈ 0.1` → 约 10% 的样本会 mask 掉邻居/车道信息
- 训练时学会"即使没有邻居信息也能预测"，推理时做 guided 采样

---

### Step ②: `prepare_model_input()` — 预处理 + 归一化

**文件**: [flow_planner.py L56-90](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_planner.py#L56-L90)

分两个子步骤：

#### ②-a: CFG Masking

```python
# cfg_type='neighbors' 时:
# 对 cfg_flags=0 的样本, mask 掉 neighbor_past (置零)
mask_flags = cfg_flags.view(B, 1, 1, 1).repeat(1, neighbor_num, 1, 1)
mask_flags[:, cfg_neighbor_num:, :] = 1  # 保留前几个邻居
data.neighbor_past *= mask_flags
```

#### ②-b: `ModelInputProcessor.sample_to_model_input()`

**文件**: [input_preprocess.py L32-97](file:///home/gcjms/Flow-Planner/flow_planner/model/model_utils/input_preprocess.py#L32-L97)

```
输入: NuPlanDataSample
│
├─ obs_normalizer(data)            # 对观测数据归一化
│
├─ 构建 model_inputs dict:
│   ├─ ego_past:          (B, 21, 14)
│   ├─ neighbor_past:     (B, 32, 21, 11)
│   ├─ lanes:             (B, 70, 20, 12)
│   ├─ lanes_speedlimit:  (B, 70, 1)
│   ├─ routes:            (B, 25, 20, 12)
│   ├─ map_objects:       (B, 5, 10)
│   └─ ego_current:       (B, 16)
│
├─ 构建 GT 轨迹:
│   ├─ ego_current → 提取 (x, y, heading)
│   ├─ ego_future  → (B, 1, 80, 3)
│   ├─ 拼接 current+future → gt_with_current: (B, 1, 81, 3)
│   │
│   └─ kinematic='waypoints' 时:
│       ├─ heading → (cos_h, sin_h) → gt: (B, 1, 81, 4)
│       └─ state_normalizer(gt[:,:,1:,:])  # 归一化未来帧
│
输出: model_inputs (dict), gt_with_current (B, 1, 81, 4)
```

> **gt_with_current 的 4 维**: `[x, y, cos(heading), sin(heading)]`，第 0 帧是 current state，第 1-80 帧是 GT future。

---

### Step ③: Encoder — `FlowPlannerEncoder.forward()`

**文件**: [encoder.py L76-117](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/encoder.py#L76-L117)

```python
encoder_inputs = {
    'neighbors': model_inputs['neighbor_past'],      # (B, 32, 21, 11)
    'lanes': model_inputs['lanes'],                   # (B, 70, 20, 12)
    'lanes_speed_limit': ...,
    'lanes_has_speed_limit': ...,
    'static': model_inputs['map_objects'],            # (B, 5, 10)
    'routes': model_inputs['routes']                  # (B, 25, 20, 12)
}
encoder_outputs = self.encoder(**encoder_inputs)
```

#### Encoder 内部 4 个子编码器:

| 子编码器 | 输入 | 处理方式 | 输出 |
|----------|------|----------|------|
| **AgentFusionEncoder** | `(B,32,21,11)` | MLP→MixerBlock×3→mean pool+type_emb | `(B,32,hidden_dim)` |
| **StaticFusionEncoder** | `(B,5,10)` | MLP 直接投影 | `(B,5,hidden_dim)` |
| **LaneFusionEncoder** | `(B,70,20,12)` | MLP→MixerBlock×3→mean pool+traffic+speed_emb | `(B,70,hidden_dim)` |
| **RouteEncoder** | `(B,25,20,12)` | 取前4维→MLP→MixerBlock→mean pool | `(B,hidden_dim)` — 全局向量 |

每个子编码器的详细流程（以 `AgentFusionEncoder` 为例）：

```
输入: (B, 32, 21, 11)  → 32个邻居, 21帧, 11维(x,y,cos,sin,vx,vy,w,l,type×3)
│
├─ 分离 type 信息: type = x[:, :, -1, 8:]     → (B, 32, 3)
├─ 取有效特征:      x = x[..., :8]              → (B, 32, 21, 8)
├─ 提取位置编码:    pos = x[:, :, -1, :7]        → (B, 32, 7) 最后一帧位置
├─ 生成 mask:       有效性检测，全零=padding
│
├─ channel_pre_project: Linear(9 → 128)          # 每帧特征投影  
├─ token_pre_project:   Linear(21 → 64)          # 时间维度压缩
├─ MixerBlock × 3:      token mixing + channel mixing
├─ mean pooling:         对时间维度取平均           → (valid_count, 128)
├─ + type_embedding:     类型嵌入
├─ emb_project:          Linear(128 → hidden_dim)
│
输出: encoding (B, 32, hidden_dim),  mask (B, 32),  pos (B, 32, 7)
```

#### Encoder 最终输出：

```python
encoder_outputs = {
    'encodings': (
        cat[neighbors_enc, static_enc],  # (B, 37, hidden_dim) agents部分
        lanes_enc                         # (B, 70, hidden_dim) lanes部分
    ),
    'masks': (
        cat[~neighbors_mask, ~static_mask],  # (B, 37) 有效性
        ~lanes_mask                           # (B, 70)
    ),
    'routes_cond': routes_enc,  # (B, hidden_dim) 全局路由条件
    'token_dist': token_dist    # (B, 118, 118) 所有token间的距离矩阵
}
```

> `token_dist` 用于 attention bias：距离越远的 token，attention 权重越低。

---

### Step ④: Flow ODE 采样 — `FlowODE.sample()`

**文件**: [flow_ode.py L29-42](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_utils/flow_ode.py#L29-L42)

```python
# gt shape: (B, 1, 81, 4), 取 gt[:, :, 1:, :] 即去掉 current → (B, 1, 80, 4)
noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], 'x_start')
```

内部过程：
```
1. 采样随机时间 t ~ Uniform(0, 1)       → (B,)
2. 采样纯噪声   x_0 ~ N(0, 1)          → (B, 1, 80, 4) 
3. CondOT 插值:  x_t = t * x_1 + (1-t) * x_0
   其中 x_1 = GT 未来轨迹（归一化后）
4. target = x_1  (因为 model_type='x_start')

输出:
  noised_traj: x_t  → (B, 1, 80, 4)   模型的输入（加噪轨迹）
  target:      x_1  → (B, 1, 80, 4)   模型应该预测的目标
  t:           时间  → (B,)
```

#### 轨迹分块 (Trajectory Chunking)

```python
# 把 80 帧轨迹切分为多个 action token（有重叠）
noised_traj_tokens = traj_chunking(noised_traj, action_len, action_overlap)
# 例: action_len=16, action_overlap=4 → 每个token 16帧, 重叠4帧
# 80帧 → ~7个token, 每个 (B, 1, 16, 4)
noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)  # (B, 7, 16, 4)
# target 同样分块
target_tokens = ...  # (B, 7, 16, 4)
```

---

### Step ⑤: Decoder (DiT) — `FlowPlannerDecoder.forward()`

**文件**: [decoder.py L98-160](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/decoder.py#L98-L160)

```python
prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
# noised_traj_tokens: (B, action_num, action_len, state_dim)
# t: (B,)
# decoder_model_extra: encodings, masks, routes_cond, token_dist, cfg_flags
```

Decoder 内部详细流程：

```
输入: x=(B, P, action_len, 4),  t=(B,)
│
├─ Step 5a: 轨迹预投影
│   x.reshape(B, P, -1)            → (B, P, action_len*4)
│   x = preproj(x)                  → (B, P, hidden_dim)    # MLP投影到隐空间
│
├─ Step 5b: 构建条件向量
│   time_cond    = t_embedder(t)     → (B, 1, hidden_dim)   # 正弦时间嵌入+MLP
│   routes_cond  → unsqueeze          → (B, 1, hidden_dim)   # 路由全局条件
│   action_pe    = learnable PE       → (B, P, hidden_dim)   # action位置编码
│   cfg_embedding = Embedding(flag)  → (B, 1, hidden_dim)   # CFG条件嵌入
│   y = time_cond + routes_cond + action_pe + cfg_embedding  # 综合条件
│
├─ Step 5c: 拼接多模态 tokens
│   encodings = [agents_enc, lanes_enc, x]  # 3个模态
│   masks     = [agents_mask, lanes_mask, None]
│
├─ Step 5d: FlowPlannerDiT (N层 DiTBlock)
│   每层 DiTBlock 内部:
│   │
│   ├─ 每个模态生成自己的条件:
│   │   agents/lanes: cond = time_cond + routes_cond
│   │   trajectory:   cond = time_cond + routes_cond + action_pe + cfg_embedding
│   │
│   ├─ AdaptiveLayerNorm (条件调制)
│   ├─ JointAttention: 所有模态 token 拼接做全局 attention
│   │   ├─ 每个模态各自做 Q,K,V 投影 → 拼接
│   │   ├─ attention + distance bias (可选)
│   │   └─ 各自 output 投影回原始维度
│   ├─ gate * attention_output + residual
│   │
│   ├─ AdaptiveLayerNorm
│   ├─ FeedForward (各模态独立)
│   └─ gate * ffn_output + residual
│
│   输出: (agents_token, lanes_token, x_token) 各自更新后的表示
│
├─ Step 5e: PostFusion (后融合)
│   ├─ agents_token, lanes_token 投影到统一维度
│   ├─ kv_token = cat[agents, lanes]
│   ├─ SelfAttention(cat[kv_token, x_token])  # 联合自注意力
│   ├─ mean pooling → MLP → 残差加到 x_token
│   输出: x_token (B, P, hidden_dim)
│
├─ Step 5f: FinalLayer
│   ├─ AdaLN: shift, scale = MLP(y)
│   ├─ x = LayerNorm(x) * (1+scale) + shift   # 条件调制
│   ├─ MLP: hidden_dim → hidden_dim*4 → output_dim
│   输出: (B, P, action_len * state_dim)
│
└─ reshape → (B, P, action_len, state_dim)     # 即 (B, 7, 16, 4)
```

---

### Step ⑥: 计算 Loss

**文件**: [flow_planner.py L152-167](file:///home/gcjms/Flow-Planner/flow_planner/model/flow_planner_model/flow_planner.py#L152-L167)

```python
# 1. 基础 MSE Loss
batch_loss = MSE(prediction, target_tokens)     # (B, P, action_len, 4)
ego_planning_loss = batch_loss.sum(dim=-1).mean()

# 2. 一致性 Loss (相邻 action token 重叠部分应一致)
# action_i 的最后 overlap 帧 ≈ action_{i+1} 的前 overlap 帧
consistency_loss = MSE(
    prediction[:, i, -overlap:, :],    # 第i个token尾部
    prediction[:, i+1, :overlap, :]    # 第i+1个token头部
)

# 3. 总 Loss (在 FlowMatchingCore 中加权)
total_loss = w1 * ego_planning_loss + w2 * consistency_loss
```

---

## 完整维度追踪表

假设 `B=32, action_len=16, action_overlap=4, state_dim=4, hidden_dim=256`:

| 阶段 | 张量 | 维度 |
|------|------|------|
| 输入 ego_current | `(B, 16)` | 32×16 |
| 输入 neighbor_past | `(B, 32, 21, 11)` | 32×32×21×11 |
| 输入 lanes | `(B, 70, 20, 12)` | 32×70×20×12 |
| GT future | `gt_with_current` | (32, 1, 81, 4) |
| Flow 采样 | `noised_traj` | (32, 1, 80, 4) |
| 分块后 | `noised_traj_tokens` | (32, 7, 16, 4) |
| Encoder agents | `agents_encoding` | (32, 37, 256) |
| Encoder lanes | `lanes_encoding` | (32, 70, 256) |
| Encoder routes | `routes_cond` | (32, 256) |
| Decoder 预投影后 | `x` | (32, 7, 256) |
| DiT 输出 | `x_token` | (32, 7, 256) |
| FinalLayer 输出 | `prediction` | (32, 7, 16, 4) |
| Loss | scalar | — |

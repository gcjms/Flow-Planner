# Goal Conditioning 设计笔记

本文总结当前 Flow-Planner 中 `goal` 条件的实现现状、实验中观察到的问题、相关工作参考，以及下一步的改造方案。

---

## 1. 现状

### 1.1 Goal vocabulary

- `goal/cluster_goals.py`
  - 从训练集 GT 轨迹中取 `goal_frame` 这一帧（默认 `39`，即 `4s@10Hz`）。
  - 对该帧的 `(x, y)` 做 KMeans，`n_clusters=64`。
  - 输出 `goal_vocab.npy`，形状 `(K, 2)`。

### 1.2 训练时 goal 获取

- `model/flow_planner_model/flow_planner.py::_get_goal_for_gt`
  - 取 GT 在 `goal_frame` 的 `(x, y)`。
  - 通过 `find_nearest_goal_torch` 映射到 vocab 最近的 anchor。
  - 返回 `(B, 2)` 的 goal point 喂给 decoder。

### 1.3 推理时 goal 来源

- `goal/goal_predictor.py::GoalPredictor`
  - 在 backbone 场景特征上做分类，输出 `num_goals` 个 logits。
  - `predict_topk` 返回 top-k 的 `indices` 和对应 `goal_points`。
- `model/flow_planner_model/flow_planner.py::forward_inference`
  - 接受外部传入的 `goal_point=(B, 2)`。
  - 和 CFG 无条件半边拼接后，喂给 decoder 的 `model_extra['goal_point']`。

### 1.4 Decoder 中的接入

- `model/flow_planner_model/decoder.py`
  - `goal_dim=2`，`goal_proj: Linear(2, hidden) → GELU → Linear(hidden, hidden)`（最后一层零初始化）。
  - `goal_embedding` 形状 `(B, 1, hidden)`，在 `cfg_flags=0` 时被置零。
  - 被加到全部 `P` 个 action token 的 trajectory 条件里：
    `y = time_cond + routes_cond + action_pe + cfg_embedding + goal_embedding`
  - 同时也作为 `FlowPlannerDiT` 中 trajectory 模态每一层的条件。

---

## 2. 实验观察到的问题

- 多模态分桶成功：不同 `goal` 生成的轨迹在横向位置上能区分出左 / 右 mode。
- 但 mode 内部形状退化：
  - 即使在直行场景，轨迹也是“直冲 goal”。
  - 没有“先横向偏一下再回正”的形状。
  - yaw 变化幅度很小。

---

## 3. 相关工作参考

把同类方法按“与当前实现的相似度”从高到低排：

1. **GoalFlow (CVPR 2025)**  
   对训练集轨迹终点 `(x, y, θ)` 做 KMeans，vocab 规模 `~4096/8192`；配合 scene-aware scorer 选最优 goal；主干用 Flow Matching（`Rectified Flow`）。结构和本仓库最像，但 goal 表达更强、vocab 更密、还有 scorer。

2. **MTR (NeurIPS 2022)**  
   论文强调 `learnable motion query pairs`，但官方 Waymo 配置里实际也加载了离线聚类的 `cluster_64_center_dict.pkl`，64 个 intention points 作为空间先验。

3. **MultiPath / CoverNet**  
   更传统的 anchor 家族。通常不是聚单个 endpoint，而是准备一组完整轨迹 anchor，模型做“选 anchor + 回归 residual”。mode 用“整条轨迹形状”表达，而不是一个终点。

4. **Plan-R1**  
   和当前思路不同。把轨迹离散成 motion tokens 做自回归生成，再用 GRPO + rule-based reward 做对齐。不属于 goal-cluster 这一路。

结论：用聚类终点做多模态先验本身没问题，属于主流路线之一；当前实现接近`简化版 GoalFlow`。

---

## 4. 问题分析

### 4.1 Goal 表达太弱

只聚 `(x, y)` 时，一个 `(x, y)` anchor 下会混入完全不同的运动 mode。举例：

- 场景 A：变道  
  终点 `(30, 5)`，最终 heading ≈ `0°`；中间 yaw 先偏后回。
- 场景 B：缓左转  
  终点 `(30, 5)`，最终 heading ≈ `+30°`；yaw 单调增加，不回正。

这两个样本会被映射到**同一个 anchor**，训练时 decoder 看到完全相同的 `goal_point`，却要同时拟合两种截然不同的 yaw 轨迹。

MSE 对这种冲突最安全的解是取“平均”：

- 终点 `(x, y)` 学得准。
- 终点 heading 变成折中值，幅度被压扁。
- 中段选曲率最小的形状，因为“回正”和“不回正”互相抵消。

这就是观察到的“直冲 goal、yaw 不变”的直接成因。  
**不是模型不会绕，而是该分开的 mode 被压在同一个 anchor 下互相抵消。**

### 4.2 Goal 注入太强

`goal_embedding` 被广播到所有 `P` 个 action token，并且在每一层 DiT block 的 trajectory 条件里都注入一次。结果是从 token 0 到 token P-1 都被 goal 直接拉着，中段没有依 `scene / route` 自由发挥的空间。

### 4.3 两个角色应该拆开

理想情况下：

- `goal` 负责“把多模态拆开”（分 mode）。
- `scene / route / interaction` 负责“决定这条 mode 具体怎么走”。

当前实现让 `goal` 同时承担两个角色，于是轨迹形状被 goal 定死。

---

## 5. 改造方案

按优先级从高到低，分四步。前两步是核心修复，第 3 步独立收益，第 4 步精修。

### Step 1: goal 表达升级到 pose `(x, y, cos, sin)`

**动机**：直接解 4.1 的冲突。同一个 `(x, y)` 但方向不同的样本会落到不同 anchor，训练目标不再互相打架。

**改动点**：

- `goal/cluster_goals.py`
  - 采样时连 heading 一起取。
  - KMeans 在 `(x, y, cos, sin)` 上聚类；heading 维度建议乘权重（比如 `~10m` 换算量级），避免被 xy 吃掉。
  - 输出 vocab 从 `(K, 2)` 变成 `(K, 4)`。
- `goal/goal_utils.py`
  - `find_nearest_goal_torch` / `select_goal_from_route` / `select_diverse_goals` 改为支持 4D。
  - 建议用前 2 维找最近 top-m，再用方向一致性做 tiebreak。
- `model/flow_planner_model/flow_planner.py`
  - `_get_goal_for_gt` 返回 `(B, 4)`，同时取 heading。
- `model/flow_planner_model/decoder.py`
  - `goal_dim=2 → goal_dim=4`，`goal_proj` 输入维度同步。
- `goal/goal_predictor.py`
  - 分类头结构不变；`predict_topk` 返回的 `goal_points` 变成 `(K_top, 4)`。

**预期收益**：

- Mode 语义边界在 label 空间里先分开，从源头消除折中解。
- 能区分“变道后回正” vs “转弯末端仍在偏”两类终态。
- 严格超集改动，不破坏已验证的多模态能力。

### Step 2: 把 goal 从“全程条件”改成“末端偏重”

**动机**：解 4.2。给中段 token 松绑，让 `routes_cond / scene` 重新主导轨迹形状。

**推荐方案：per-token ramp**

- 构造 `(P,)` 权重 `w`，从 `~0.1` 单调升到 `1.0`，越靠后 goal 权重越大。
- `goal_embedding * w[None, :, None]` 后再参与加和。
- 可以选择只在 `FinalLayer` 的 AdaLN 条件和后几层 DiT block 的 traj 条件里注入 goal，而不是每层都注入。

**改动点**：

- `model/flow_planner_model/decoder.py`
  - 在 `forward` 里给 `goal_embedding` 加 per-token 权重。
  - （可选）在 `FlowPlannerDiT.forward` 里把 `goal_embedding` 从“每层都加”改成“后半段层再加”。

**预期收益**：

- 早期 token 依赖 scene / route，恢复中段形状自由度。
- 末端 token 强约束保持 mode 分离。
- 轨迹更容易出现“先偏再回”等非单调形状。
- 只改一个文件，可灰度开启。

### Step 3: goal scorer + top-k 推理候选

**动机**：提高候选集的 mode recall，解 “top-1 goal 选错 → 整条 rollout 作废”。

**改动点**：

- `goal/goal_predictor.py`
  - 由单分类扩成双 head（参考 GoalFlow）：
    - `dis head`：与 GT endpoint 对齐（类似现状）。
    - `dac head`：goal 是否在 drivable / route 附近。`routes` 已在 encoder 输入里，不依赖额外 HD map。
  - 推理时选 top-k goal 作为候选输入。
- 推理入口（如 `script` 下的评测或 `dpo/generate_candidates_goal.py`）
  - 遍历 top-k goals，各生成一条候选；用现有 `TrajectoryScorer` / DPO 流程选最优。

**预期收益**：

- 候选集质量显著提升，DPO pair 质量上限也提升。
- 与 `dpo/generate_candidates_goal.py` 天然兼容。
- 独立于 Step 1/2，可并行推进。

### Step 4: vocab 加密

**动机**：前两步做完后，goal 才真正承担“精细 mode 标签”角色，加密才有意义。

**改动点**：

- `goal/cluster_goals.py`：`n_clusters: 64 → 256 / 512`。
- `goal/goal_predictor.py`：最后一层 `out_features` 跟着变。

**预期收益**：

- Mode 粒度更细。
- 在 Step 1/2 之前单独加密，边际收益小，所以放在最后。

---

## 6. 执行顺序与收益归因

| 步骤 | 解决的问题 | 主要收益 |
|------|------------|----------|
| Step 1 | goal 信息量不足（§4.1） | mode 语义边界清晰化，训练目标不冲突 |
| Step 2 | goal 注入过强（§4.2） | 中段形状自由，yaw 变化恢复 |
| Step 3 | 推理候选集质量 | top-k 提升 mode recall |
| Step 4 | mode 粒度 | 精修 |

推荐顺序：`Step 1 → Step 2`（核心修复） → `Step 3`（可并行） → `Step 4`（精修）。

---

## 7. 与现有模块的关系

- 所有改动保持向后兼容：`goal_dim=0` / `goal_vocab_path=None` 时路径不变。
- `dpo/generate_candidates_goal.py`、`dpo/eval_goal_diversity.py` 对 vocab 形状敏感，需要配合 Step 1 更新。
- `risk/trajectory_scorer.py` 与 goal 无耦合，Step 3 的候选流程可直接复用。

---

## 8. Trajectory Anchor 路线（强烈推荐，解决根本问题）

### 8.1 当前 Goal 的核心缺陷回顾

你最初的目标是：**原始 Flow Planner（不带 goal）就能在同一 scene 下输出明显不同的多模态轨迹**，然后再用这些轨迹去做 DPO preference。

但实际发现：
- 原始 FM 严重 mode collapse，采样出来的轨迹高度相似。
- 加入 endpoint goal 后，虽然能在一定程度上分离粗模态，但因为 goal 只包含 `(x,y)`（信息太少），同一个 goal 下会混入多种冲突行为（早变道 vs 晚变道、激进 vs 保守），导致训练时互相抵消，最终轨迹“直冲 goal、yaw 不变”。

这就是你现在遇到的根本矛盾。

### 8.2 Anchor vs 当前 Goal 的本质区别

**当前 Goal** = 一个终点 `(x, y)`（或 `(x, y, θ)`）
- 信息量少
- 多个不同驾驶风格容易被映射到同一个 anchor
- 训练时同一个条件对应多种 target → 模型学“平均解”

**Trajectory Anchor** = 一整条多秒的参考轨迹（例如 4 秒 = 40 帧的 `(x, y, heading)`）
- 信息量丰富（携带形状、曲率、最终朝向、驾驶风格）
- 不同行为风格天然被聚到不同 cluster
- 每个 anchor 代表一种清晰的“驾驶意图模板”

**MTR 的 Motion Query** 则是 Anchor 的可学习版本：
- 不是离线 KMeans 聚类，而是模型内部放 64~128 个可学习 query
- 训练过程中这些 query 自己去竞争、收敛成不同 mode
- 比静态聚类更灵活（MTR 的核心思想）

### 8.3 Anchor 的具体好处（针对你的痛点）

1. **Mode 分离能力大幅提升**  
   同一个终点可能对应“早超 vs 晚超 vs 跟车”，现在会被分到不同 Anchor，训练冲突大大减少，不再容易学成平均解。

2. **为 FM 提供清晰的 mode 先验**  
   不再让 FM 从纯噪声里“自己长出 mode”，而是“先选 Anchor → FM 在 Anchor 附近 refine”。这比纯采样稳定得多。

3. **DPO pair 构造更合理**  
   - 可以做 **同 Anchor 内** 的细粒度 preference（同一模态下的不同风格）
   - 也可以做 **跨 Anchor** 的粗粒度 preference（不同驾驶意图之间的优选）
   - 避免了你现在遇到的严重 cross-goal 问题。

4. **更接近人类驾驶意图**  
   人类想的是“我要走哪种风格”，而不是“我终点要去哪个 (x,y)”。

### 8.4 推荐实现路径（最小改动版 → MTR 风格）

**第一阶段（推荐立即开始，改动最小）—— 静态 Trajectory Anchor**

1. 修改 `cluster_goals.py` → `cluster_trajectories.py`
   - 不再只取终点，改成取未来 40 帧 `(x, y, heading)`
   - 对展平后的轨迹做 KMeans（建议先 PCA 降维再聚类，K=128~256）
   - 输出 `trajectory_anchors.npy`（形状 `(K, 40, 3)` 或展平形式）

2. 将 `GoalPredictor` 升级为 `AnchorPredictor`
   - 输入 scene feature，输出每个 anchor 的 imitation score + multi-metric scores（collision, progress, comfort, route 等）
   - 复用你现有的 `TrajectoryScorer` 做 offline teacher labeling

3. 训练时同时优化：
   - Imitation loss（学像 GT 的 anchor）
   - Multi-target distillation loss（学每个 anchor 在当前 scene 下的合理性）

4. 推理时：Predictor 输出 top-k anchors → 每个 anchor 让 FM refine 几次 → 得到真正多模态候选

**第二阶段**：把静态 Anchor 升级为可学习的 Motion Queries（MTR 风格），进一步提升灵活性。

### 8.5 与 DPO 的结合方式

- **粗粒度**：不同 Anchor 之间的 preference（激进超车 anchor vs 保守跟车 anchor）
- **细粒度**：同一个 Anchor 下，用不同 seed / noise 生成多个变体，再做 DPO pair
- 这样 pair 的 condition 一致性远高于现在的 cross-goal 做法

---

**总结推荐**

当前最优路径不是继续在 endpoint goal 上打补丁，而是**切换到 Trajectory Anchor 路线**。

这既能解决你原始 FM 多模态不足的问题，又能让后续 DPO 建立在更稳固的 mode 基础上，是目前最匹配你真实需求的方案。

下一阶段具体行动计划：
1. 先实现静态 Trajectory Anchor（1-2 周可见效）
2. 验证多模态是否显著提升
3. 再决定是否往 MTR-style learnable queries 迁移

（本节完）


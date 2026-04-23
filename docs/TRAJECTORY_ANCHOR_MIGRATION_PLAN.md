# Trajectory Anchor 迁移计划 (Flow Planner)

> 分支：`feature/anchor`（从 `feature/goal` tip `6356744` 拉出）
> Goal 线相关 md 只保留在 `feature/goal`，此处专注 anchor 路线。

---

## 1. 背景与目标

### 1.1 当前系统症状（`feature/goal` 已验证）

1. **原始 FM（no goal）mode collapse**：同一场景下采样轨迹高度相似，SDE 实验（`measure_sde_diversity.py`）确认噪声只能抖动不能换 mode。
2. **加入 2D endpoint goal 后出现新问题**：
   - goal 信息量太少，同一 (x,y) 下混入多种驾驶风格 → MSE 训练取平均解 → "直冲 goal、yaw 不变"。
   - goal_embedding 全程注入每个 action token，把中段形状也定死了。
3. **DPO cross-goal pair 训练失败**：preference 对的 chosen/rejected 来自不同 goal，训练信号冲突。
4. `oracle_goal ≈ none` 的评测结果说明 decoder 没有真正把 goal 用起来。

### 1.2 本分支目标

把"条件接口"从 **2D endpoint goal** 换成 **trajectory anchor**（完整多秒轨迹模板），并引入 **multi-target distillation**（Hydra-MDP 风格），让同一场景下多条合理轨迹同时参与监督。

可量化的验收指标（以 `none` baseline 为参照）：

| 指标 | baseline (no goal) | feature/goal 现状 | feature/anchor 目标 |
|---|---|---|---|
| 同场景 top-5 轨迹 minFDE 分散度 | 低 | 中 | **> baseline 2 倍** |
| 碰撞率 | 18-20% | ~ | **≤ baseline** |
| `oracle_anchor` vs `none` 的 closed-loop 差距 | 0 | 0 | **显著（>30% 相对）** |
| DPO chosen/rejected gap 学得动 | 否 | 否 | 是 |

### 1.3 非目标（本分支不做）

- 不引入 VLM 或 VLA 范式（那是另一条线）
- 不做端到端感知（保留 nuPlan 结构化输入）
- 不迁移到 Diffusion Policy（仍用 Flow Matching）
- 不在第一阶段引入可学习 motion queries（MTR-style，放到 Phase 3 之后）

---

## 2. 架构草图

### 2.1 Flow 数据路径（目标态）

```text
Scene (agents+lanes+ego) ──┐
                           ├──▶ Encoder ──▶ scene_feat
                           │                   │
                           │                   ├──▶ AnchorPredictor ──▶ top-k anchor ids  ─┐
                           │                   │    (+ metric heads, Hydra-MDP 风格)        │
                           │                   └──▶ scene cond ──────────────────────────┐  │
                           │                                                             │  │
                  anchor vocab (K, T, 3) ──▶ index ──▶ anchor trajectory ──▶ TrajEncoder │  │
                                                                                 │       │  │
                                                                        anchor_embedding │  │
                                                                                 │       │  │
                                                         x_t (traj noise) ──▶ DiT decoder ┴──┘
                                                                                 │
                                                                            vector field
                                                                                 │
                                                                  Flow ODE/SDE integrate
                                                                                 │
                                                                       refined trajectory
```

### 2.2 和现在的 goal 路径的核心区别

| 维度 | 现在 (goal) | 目标 (anchor) |
|---|---|---|
| 条件载体 | `(x,y)` 2D 点 | `(T, 3)` 完整轨迹 |
| vocab 规模 | 64 个点 | 128~256 个轨迹（先 128） |
| 注入方式 | 广播到所有 P 个 token，每层都加 | TrajEncoder 出 anchor token，cross-attn 到 trajectory query |
| 监督信号 | 单 GT → nearest goal 的 imitation | 多 anchor 同时打分（Hydra-MDP 风格 soft distillation） |
| DPO pair | same-goal / cross-goal | same-anchor（细粒度风格）+ cross-anchor（粗粒度意图） |

---

## 3. 分阶段计划（估时以全职 1 人计）

### Phase 0 — 准备 & 离线产物（1-2 天）

- [ ] P0-1 `cluster_goals.py` → `cluster_trajectories.py`（新建，老文件保留）
  - 从训练集取完整 `(T=40, 3)` GT 轨迹
  - PCA 降维（可选）→ KMeans (K=128)
  - 产出 `anchor_vocab.npy`, shape `(128, 40, 3)`
- [ ] P0-2 **离线 teacher 打分**：用现有 `risk/trajectory_scorer.py` 给每个 (scene, anchor) 打 multi-metric score，缓存成 `(scene_id, anchor_id) → score_dict`
- [ ] P0-3 Anchor 可视化脚本 `visualize_anchors.py`（看 128 条 anchor 在 ego frame 下长什么样）
- [ ] P0-4 `goal_vocab.npy` 标记 deprecated（加 README 注释），不删

**Exit criteria**：`anchor_vocab.npy` 生成，可视化通过（覆盖直行/左转/右转/变道/停车），teacher 打分 cache 生成在一个训练子集上。

### Phase 1 — Static Anchor + 基础 FM（3-5 天）

目标：跑通最小可用版本，先 work 再优化。

- [ ] P1-1 **AnchorPredictor**：`goal_predictor.py` 原地改成 `anchor_predictor.py`，输入 scene_feat，输出 K=128 logits + imitation probability（第一版保持单头，metric heads 放到 Phase 2）
- [ ] P1-2 **TrajEncoder**（新模块）：把 `(T, 3)` anchor 编码成 1~8 个 token，简单 MLP+PE 起步
- [ ] P1-3 **decoder.py 改造**：
  - 去掉 `goal_dim=2, goal_proj(Linear(2,hidden))` 分支
  - 加 `anchor_tokens` 作为 cross-attn key/value（不再广播加到每个 token）
  - 保留零初始化策略，`cfg_flags=0` 时 zero-out anchor tokens
- [ ] P1-4 **flow_planner.py 改造**：
  - `_get_goal_for_gt` → `_get_anchor_for_gt`（nearest anchor lookup，距离用轨迹 L2）
  - `forward_inference` 接受 `anchor_id` 或 `anchor_traj`
- [ ] P1-5 **训练入口**：`train_anchor_predictor.py`（对标 `train_goal_predictor.py`），loss 只含 imitation CE
- [ ] P1-6 **最小 eval**：在 `eval_multidim_utils.py` 里加 `choose_anchor(mode="none"/"predicted"/"oracle")`，跑一轮 smoke test

**Exit criteria**：训练不崩、`oracle_anchor` 显著好于 `none`（closed-loop 碰撞率相对下降 > 20%）。如果 `oracle_anchor ≈ none`，先停下来 debug，不往后推。

### Phase 2 — Multi-Target Distillation（Hydra-MDP 风格，3-5 天）

核心贡献。Phase 1 只让模型学"最像 GT 的 anchor"，Phase 2 让模型同时学"所有合理 anchor"。

- [ ] P2-1 AnchorPredictor 加 metric heads：`dis_head / dac_head / safety_head / progress_head / comfort_head`（参考 Hydra-MDP），每头输出 K 维 logit
- [ ] P2-2 训练 loss：`L = L_imi + λ_soft * KL(teacher_soft_target || multi_head_soft_pred)`，teacher 来自 Phase 0 的打分 cache
- [ ] P2-3 改造 `train_soft_pref.py`：goal_labels 路径改 anchor_labels；scene-level soft distillation 逻辑全部保留（这个文件是 Hydra-MDP 思想的现成原型，最小改动）
- [ ] P2-4 推理：top-k anchor → 对每个 anchor 让 FM refine → 得到 k 条候选轨迹

**Exit criteria**：同场景 top-5 轨迹的 minFDE 分散度较 Phase 1 翻倍；候选集 recall@5 显著提升。

### Phase 3 — Anchor-Conditioned DPO（3-5 天）

- [ ] P3-1 `build_multi_pairs.py` 改造：pair 组里按 `anchor_id` 分桶
  - same-anchor pair：同风格内的细粒度偏好（沿用 DriveDPO hard-neg 策略）
  - cross-anchor pair：跨风格的粗粒度偏好（给 chosen/rejected 打明显不同的 anchor 标签）
- [ ] P3-2 `train_dpo.py` 的 `attach_goal_to_decoder_inputs` → `attach_anchor_to_decoder_inputs`，loss 与循环全部复用
- [ ] P3-3 评测：cross-anchor pair 的 chosen preference 能被模型学到（policy preference prob 提升）

**Exit criteria**：DPO 在 anchor 条件下 loss 正常收敛，offline preference accuracy > 0.65；closed-loop 指标优于 Phase 2 baseline。

### Phase 4（可选）— Dynamic/Learnable Anchors（MTR 风格，2-3 周）

- 引入 64~128 个可学习 motion query，竞争 + hungarian 匹配。此时 static anchor vocab 变成初始化先验，query 自己在训练中调整。
- Phase 1-3 效果如果不够好再考虑，否则先不动。

---

## 4. 文件级别分类表

### 4.1 图例

- ♻️ **REUSE** — 不改
- 🔧 **REFACTOR** — 原地改造，保留文件
- 🔁 **REWRITE** — 新文件替代，老文件 deprecate
- 🗑️ **DEPRECATE** — 不再用，暂保留，Phase 3 末删除

### 4.2 `flow_planner/goal/`

| 文件 | 行数 | 分类 | 改动要点 | 估时 |
|---|---|---|---|---|
| `cluster_goals.py` | 104 | 🔁 | 新建 `cluster_trajectories.py`；KMeans 目标从 `(N,2)` 变 `(N, T*3)` | 0.5d |
| `goal_predictor.py` | 92 | 🔧 | rename → `anchor_predictor.py`，Phase 2 加 metric heads | 0.5d (P1) + 1d (P2) |
| `goal_utils.py` | 150 | 🔧 | `find_nearest_goal_torch` → `find_nearest_anchor_torch`（L2 over trajectory），`select_diverse_goals` → anchor 版 | 1d |
| `__init__.py` | 0 | ♻️ | — | — |

### 4.3 `flow_planner/model/flow_planner_model/`

| 文件 | 行数 | 分类 | 改动要点 | 估时 |
|---|---|---|---|---|
| `decoder.py` | 449 | 🔧 | 删 `goal_proj(Linear(2,h))`；加 `TrajEncoder` + cross-attn pathway；保留零初始化 | 1.5d |
| `flow_planner.py` | 350 | 🔧 | `_get_goal_for_gt` → `_get_anchor_for_gt`；`forward_inference` 签名变 | 0.5d |
| `encoder.py` | 94 | ♻️ | — | — |
| `global_attention.py` | 111 | ♻️ | — | — |
| `flow_utils/*` | — | ♻️ | — | — |

### 4.4 `flow_planner/dpo/` — DPO 核心（与条件接口正交）

| 文件 | 行数 | 分类 | 备注 |
|---|---|---|---|
| `dpo_loss.py` | 216 | ♻️ | — |
| `train_dpo.py` | 729 | 🔧 | 仅改 `attach_goal_*` 辅助函数 (~50 行) |
| `lora.py` | 277 | ♻️ | — |
| `score_hybrid.py` | 816 | ♻️ | — |
| `analyze_candidate_modes.py` | 376 | ♻️ | — |
| `audit_preferences.py` | 93 | ♻️ | — |
| `bev_renderer.py` | 309 | ♻️ | — |
| `config_utils.py` | 59 | ♻️ | — |
| `data_mining.py` | 214 | ♻️ | — |
| `render_pair_spotcheck.py` | 258 | ♻️ | — |
| `measure_sde_diversity.py` | 359 | ♻️ | 诊断工具 |
| `vlm_score_candidates.py` | 273 | ♻️ | — |
| `generate_preferences.py` | 214 | ♻️ | — |
| `generate_candidates.py` | 153 | ♻️ | 无条件路径 |
| `generate_multiobjective_pairs.py` | 491 | ♻️ | — |
| `generate_onpolicy_pairs.py` | 317 | ♻️ | — |

DPO 核心估时：`train_dpo.py` 小改 0.5d；其余 0d。

### 4.5 `flow_planner/dpo/` — goal 相关（需要改造）

| 文件 | 行数 | 分类 | 改动要点 | 估时 |
|---|---|---|---|---|
| `build_multi_pairs.py` | 514 | 🔧 | pair 分桶 key 从 goal_id 换 anchor_id；DriveDPO hard-neg 保留 | 1d |
| `train_soft_pref.py` | 661 | 🔧 | goal_labels → anchor_labels；scene-level soft KL 保留；**重要：这是 Hydra-MDP 思想的现成实现** | 1d |
| `eval_multidim_utils.py` | 556 | 🔧 | `choose_goal_point` → `choose_anchor`；mode 枚举改名 | 1d |
| `eval_multidim.py` | 129 | 🔧 | 调用点同步 | 0.3d |
| `eval_multidim_goal_ablation.py` | 175 | 🔁 | 新建 `eval_multidim_anchor_ablation.py`，老的标 deprecated | 0.5d |
| `eval_goal_diversity.py` | 191 | 🔁 | 新建 `eval_anchor_diversity.py` | 0.5d |
| `generate_candidates_goal.py` | 184 | 🔁 | 新建 `generate_candidates_anchor.py` | 0.5d |
| `generate_oracle_pairs.py` | 304 | 🔧 | oracle 概念天然映射 oracle anchor | 0.5d |
| `__init__.py` | 31 | 🔧 | 导出更新 | 0.1d |

### 4.6 根目录脚本/数据

| 文件 | 分类 | 改动要点 | 估时 |
|---|---|---|---|
| `train_goal_predictor.py` | 🔁 | 新建 `train_anchor_predictor.py` | 0.5d |
| `eval_goal_predictor.py` | 🔁 | 新建 `eval_anchor_predictor.py` | 0.3d |
| `eval_goal_checkpoint_table.py` | 🔧 | 原地改 | 0.3d |
| `eval_goal_diversity.py` | 🔁 | 新建 `eval_anchor_diversity.py`（dpo/ 下已有，此处为根目录 CLI 入口） | 0.3d |
| `visualize_goals.py` | 🔁 | 新建 `visualize_anchors.py` | 0.3d |
| `goal_vocab.npy` | 🗑️ | 保留归档，不删；用 `anchor_vocab.npy` 替代 | — |
| `goal_pipeline.sh` | 🔁 | 新建 `anchor_pipeline.sh` | 0.2d |
| `auto_goal_dpo_pipeline.sh` | 🔁 | 新建 `auto_anchor_dpo_pipeline.sh` | 0.2d |
| `run_goal_predictor_run1.sh` | 🔁 | 新建对应 anchor 版 | 0.1d |
| `resume_goal_finetune_plus20.sh` | 🗑️ | 不再需要 | — |

### 4.7 `flow_planner/dpo/dpo/`（嵌套重复目录）

历史债务，本分支不动；独立工单清理。

---

## 5. 总估时

| Phase | 估时 | 累计 |
|---|---|---|
| Phase 0 准备 | 1-2 d | 1-2 d |
| Phase 1 static anchor 打通 | 3-5 d | 4-7 d |
| Phase 2 multi-target distillation | 3-5 d | 7-12 d |
| Phase 3 anchor DPO | 3-5 d | 10-17 d |
| **基础版本完成** | — | **约 2-3.5 周全职** |
| Phase 4 learnable queries（可选） | 2-3 周 | 4-6 周 |

主要不确定性来自训练时间（不是开发时间），上表只算 coding。实验时长另算。

---

## 6. 风险与未知

| # | 风险 | 概率 | 应对 |
|---|---|---|---|
| R1 | Anchor vocab 太稀疏，有些场景没有合理 anchor | 中 | Phase 0 可视化验证；必要时 K 调到 256 |
| R2 | AnchorPredictor 过拟合 imitation，ignore metric heads | 中 | Phase 2 引入 `λ_soft` 调参；冻结 predictor 主干 |
| R3 | cross-attn 注入太弱，`oracle_anchor ≈ none` 重演 | 中 | Phase 1 出口严格 gate；真出现就先做 injection 消融 |
| R4 | Multi-target distillation 和单 GT imitation 冲突 | 低 | `λ_soft` warmup；先 imitation 预训练再加 soft |
| R5 | DPO same-anchor pair 挖不够 | 中 | Phase 3 复用 build_multi_pairs 的 K=8 候选 |
| R6 | feature/anchor 与 feature/goal 后续合并/对比成本 | 低 | 保持 DPO infra 改动最小化；接口命名显式区分 goal/anchor |

---

## 7. 参考工作

按"对本分支直接可借鉴度"从高到低：

### 7.1 主要参考（核心思想直接套用）

1. **Hydra-MDP (CVPR 2024)** — 多 teacher / 多 target distillation
   - 直接对应 Phase 2 的 metric heads + soft teacher 打分
   - 解决"单场景单 GT，但多条合理"这个根本问题，最贴我们现状
2. **MTR (NeurIPS 2022)** — 64 intention points 聚类 + motion query
   - 直接对应 Phase 0/1 的 static anchor 聚类
   - Phase 4 的 learnable motion query 路线
3. **MultiPath / CoverNet** — 整条轨迹 anchor + classification + residual
   - 直接对应 Phase 1 的"选 anchor + FM refine residual"范式
   - 给 anchor vs endpoint 的表达力差异提供了历史证据

### 7.2 次要参考（范式相似但细节不通用）

4. **GoalFlow (CVPR 2025)** — Flow Matching + goal KMeans + scorer
   - 条件是 `(x,y,θ)` 点而非轨迹，与我们 Phase 1 后的升级方向一致但表达力弱
   - scorer 的设计思路可参考
5. **AnchDrive (2025)** — 混合 trajectory anchors + Diffusion Policy
   - **仅作为同期旁证，不作为实现参考**
   - E2E 感知、camera/LiDAR 输入、Diffusion（非 FM）、无 DPO，方法栈差异大
   - 唯一借鉴点："不从纯噪声生成，而是从 anchor refine" 的直觉——但这本来就是 MultiPath 范式

### 7.3 仅作为思路背景

6. **pi_0 / pi_0.5** — 机器人 VLA，高层意图隐式/显式注入
   - 范式不直接迁移；提供了"高层意图应与低层控制解耦"的架构直觉
7. **Plan-R1** — motion token 自回归 + GRPO
   - 不同范式，但对"同场景多合理解"的问题意识相同

---

## 8. 执行跟踪

进入 Phase N 前先把本文件对应 section 的 checkbox 勾完，commit 信息 prefix 用 `[anchor-PN-X]`。

代码上线顺序建议：
1. 先 Phase 0 产物（数据 + 可视化），工程量小，早得反馈
2. Phase 1 最小可用路径全打通再做 Phase 2
3. Phase 3 的 DPO 一定要等 Phase 2 soft distillation 稳定后再做，否则 pair 信号和 distillation 信号会互相干扰

---

## 9. 接下来的动作

等待审核通过后：
1. 按 Phase 0 checklist 开始（`cluster_trajectories.py` + `anchor_vocab.npy` 先出来）
2. 每个 Phase 结束在本文件打勾 + 写一段实测数据
3. 阶段性结果（anchor 可视化、首次 eval 表）贴到本文件末尾作为实验日志


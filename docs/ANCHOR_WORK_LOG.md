# Anchor 分支工作日志

> 分支：`feature/anchor`（从 `feature/goal` tip `6356744` 拉出）
> 用途：组内汇报 + 论文素材 + AI 跨对话记忆
> 更新规则：见 `.cursor/rules/anchor-worklog.mdc`（**追加不覆盖**）

---

## 0. TL;DR（给自己看的一段话）

- **做什么**：把 Flow Planner 的条件接口从 **2D goal 点** 换成 **完整轨迹 anchor**（Hydra-MDP / MultiPath 风格），解决 goal 线 `oracle_goal ≈ none`、mode collapse、cross-goal DPO 学不动这三个问题。
- **当前状态**：Phase 0/1 工程链路已打通，`oracle_anchor` 证明架构本身可用；但同 manifest/同 seed 的 full suite 显示 **raw baseline = `2.4% / 0.3282 / 0.8513`**，当前 `predicted_anchor_top1 / rerank_a = 6.2%`，还没有超过原始 FlowPlanner。
- **当前瓶颈**：`oracle_anchor=2.0%` 明显好于 `predicted_anchor=6.2%`；且 `oracle_anchor_rerank=2.8%` 说明即使不经过 predictor，planner 对 near-oracle 但非 top-1 anchor 也会退化，瓶颈优先指向 planner robustness / exposure bias。
- **下一步**：已实现 **scheduled anchor sampling** 代码与 AutoDL 脚本；先跑 `p_max=0.3` 和 `p_max=0.5` 两组，观察 predicted collision 是否从 `6.2%` 明显下降、`oracle_anchor_rerank` 是否更接近 `oracle_anchor`。

---

## 1. 背景与动机（可直接摘去论文/slide）

### 1.1 前置实验（`feature/goal` 线已验证失败的东西）

| 做法 | 结果 | 归因 |
|---|---|---|
| 原始 FM（no goal） | mode collapse，同场景 top-5 轨迹高度相似 | 噪声只能抖动，无法切 mode |
| 加 2D endpoint goal（AdaLN additive） | `oracle_goal ≈ none` | goal 信息量太少 + additive shift 被模型学会 ignore（γ→0/β→0） |
| Cross-goal DPO（chosen/rejected 跨 goal） | loss 不动，policy preference prob 不升 | pair 同时耦合"意图差异"和"质量差异"，DPO 分不清信号 |

### 1.2 本分支的核心假设

1. **条件表达力不够** → 用完整轨迹（`T=80, 3`）替代 2D 点
2. **注入方式太弱** → 用 **cross-attention**（trajectory token = Q，anchor token = K/V）替代 AdaLN additive
3. **单 GT 监督信号噪声大** → 借鉴 Hydra-MDP，让多条合理 anchor 同时参与监督（Phase 2）
4. **DPO cross-condition 信号冲突** → 只做 same-anchor pair（细粒度风格），cross-anchor 归 Phase 3.5 消融

### 1.3 验收指标（相对 `none` baseline）

| 指标 | 目标 | 当前（2026-04） |
|---|---|---|
| `oracle_anchor` vs `none` closed-loop collision 相对下降 | > 30% | **78%**（7.2% → 1.6%）✅ |
| `predicted_anchor` vs `none` collision 相对下降 | > 10% | **19%**（7.2% → 5.8%）✅ |
| 同场景 top-5 轨迹 minFDE 分散度 | > 2× baseline | 未测 |
| DPO chosen/rejected gap 学得动 | 是 | Phase 3 未开始 |

> 注：上表里的 `none = 7.2% / 0.35 / 0.85` 是 **`planner_ft_run1/planner_anchor_best.pth` 在 `anchor_mode none`** 下的内部对照。
> 2026-04-25 新补跑的 **官方原始 `flowplanner_no_goal.pth + anchor_mode none`** 为 `2.0% / 0.3275 / 0.8547`。
> 因此对外汇报时必须区分：
> - **内部对照**：anchor finetune 后关掉 anchor
> - **外部基线**：原始 FlowPlanner（更公平）

---

## 2. 架构关键决策记录

### 2.1 为什么用 Cross-Attention（不用 AdaLN additive）

- additive shift/scale 喂进 LayerNorm，信号弱到模型能学会关掉（γ→0）
- cross-attention 让 trajectory token 主动 query anchor，softmax 逼模型至少 attend 到某处
- **输出投影零初始化**：确保训练开始时 anchor 贡献严格为零，从 pretrained backbone 平滑"点亮"

### 2.2 AnchorTokenEncoder（把轨迹压成 token）

- 输入：`(B, T=80, 3)` anchor 轨迹（x, y, yaw）
- 机制：learnable query + 单层 cross-attn
- 输出：`(B, L=4, H)` anchor tokens
- 位置：DiT decoder 之前

### 2.3 AnchorCrossAttention（注入轨迹）

- Q = trajectory token
- K/V = anchor tokens
- `out_proj.weight` **零初始化** → 训练开始 anchor 贡献为 0
- DiT 内部 block 完全看不到 anchor（anchor 不再参与 AdaLN 的 `y` 加法）

### 2.4 Teacher Forcing（训练时用 oracle anchor）

- 训练时 anchor = **GT 轨迹的最近邻 anchor**（nearest neighbor in anchor vocab）
- nearest lookup 对齐的是模型实际预测的那段：`gt_future[:, -T_anchor:, :3]`（最后 T_anchor 帧）
- 推理时 anchor = predictor 输出 top-1 / top-k（Phase 1 单头，Phase 2 metric heads）

### 2.5 Finetune 策略（差分学习率）

`finetune_anchor_planner.py`：

| 参数组 | LR | 理由 |
|---|---|---|
| `anchor_new`（AnchorTokenEncoder + AnchorCrossAttention） | `lr` = 2e-5 | 从零开始，需要正常 LR |
| `decoder_other`（decoder 其它） | `lr * 0.1` = 2e-6 | 保护 pretrained DiT，只微调 |
| `model_encoder`（scene encoder） | 冻结 | pretrained FlowPlanner 已稳定 |

起点：`flowplanner_no_goal.pth`（官方 FlowPlanner，无 goal 条件版本）
加载：`strict=False`（新模块走 zero-init）

### 2.6 Anchor 词表

- K = 128 条轨迹
- 每条 shape = `(T=80, 3)` = `(x, y, yaw)`
- 来源：从训练集抽 GT 轨迹做 KMeans 聚类（`cluster_trajectories.py`）
- 文件：`anchor_vocab.npy`

---

## 3. 关键代码文件索引

### 3.1 模型
| 文件 | 作用 | 关键点 |
|---|---|---|
| `flow_planner/model/flow_planner_model/decoder.py` | DiT decoder + anchor 注入 | `AnchorTokenEncoder`, `AnchorCrossAttention`（out_proj 零初始化） |
| `flow_planner/model/flow_planner_model/flow_planner.py` | 主模型 | `_get_anchor_index_for_gt` / `_get_anchor_for_gt` 对齐到末 T_anchor 帧 |

### 3.2 配置
| 文件 | 作用 |
|---|---|
| `flow_planner/script/model/flow_planner_anchor.yaml` | model-level 配置（anchor 维度/token 数/head 数/vocab 路径） |
| `flow_planner/script/anchor_finetune.yaml` | top-level Hydra finetune 配置（必须用这个，别用 model-level） |

### 3.3 训练
| 文件 | 作用 |
|---|---|
| `train_anchor_predictor.py` | 训练 AnchorPredictor；`load_planner` 自动 patch decoder 的 anchor 字段 |
| `finetune_anchor_planner.py` | Planner 级 finetune，teacher forcing + 差分 LR；监控 `anchor_out_proj_abs_mean_end` |
| `cluster_trajectories.py` | 聚类生成 `anchor_vocab.npy` |
| `visualize_anchors.py` | anchor 可视化（ego frame） |

### 3.4 评测
| 文件 | 作用 |
|---|---|
| `flow_planner/dpo/eval_multidim.py` | 主评测入口；支持 `none / route_anchor / predicted_anchor / oracle_anchor / predicted_anchor_rerank / oracle_anchor_rerank` |
| `flow_planner/dpo/eval_multidim_utils.py` | 模型加载 + anchor 选择 + rerank 上下文构建；**已修 loader bug**，并已支持 `oracle_anchor_rerank`；日志输出统一改为 `conditioning`，避免 `goal_mode:none` 误导 |
| `run_anchor_eval_common.sh` | AutoDL 部署共用 helper：统一路径、manifest 复用、summary 打印 |
| `run_anchor_raw_no_goal_eval.sh` | AutoDL 一键补跑 **官方 `flowplanner_no_goal.pth` + `anchor_mode none`** |
| `run_anchor_eval_suite.sh` | AutoDL 总控脚本：按同一 manifest 跑 `raw_no_goal / planner_ft_none / predicted_anchor / rerank A / oracle / oracle_rerank`，也支持只跑子集 |
| `run_anchor_scheduled_sampling.sh` | AutoDL 一键训练 scheduled anchor sampling，并自动跑关键 eval suite |

### 3.5 文档
| 文件 | 作用 |
|---|---|
| `docs/TRAJECTORY_ANCHOR_MIGRATION_PLAN.md` | 完整迁移计划（phase 划分、文件改造表、风险） |
| `docs/ANCHOR_DEPLOYMENT_AND_VERIFICATION.md` | 部署与验证 runbook |
| `docs/TONIGHT_AUTODL_TASKS.md` | AutoDL 实验任务单（4 个任务 + 命令 + gate） |
| `docs/ANCHOR_WORK_LOG.md` | **本文件** |

---

## 4. 实验记录（append-only，按时间倒序）

### 2026-04-25 — Scheduled sampling 部署脚本修正（自动生成 data list）

**背景**：用户 AutoDL 上当前数据形态是 `train_dataset` / `val_dataset` 目录，昨晚 eval suite 能跑是因为 eval 只需要 `scene_dir` + manifest；scheduled sampling 属于训练流程，继承 `finetune_anchor_planner.py` 的 `--train-data-list / --val-data-list` 输入。

**修正**：
- `run_anchor_scheduled_sampling.sh` 默认优先识别 `/root/autodl-tmp/train_dataset` 和 `/root/autodl-tmp/val_dataset`
- 如果没有显式设置 `TRAIN_DATA_LIST / VAL_DATA_LIST`，脚本会扫描目录下 `.npz`，自动生成 `/root/autodl-tmp/anchor_runs/generated_lists/train_list.json` 和 `val_list.json`
- 若默认 eval 的 `/root/autodl-tmp/nuplan_npz` 不存在，训练后的 eval 会自动使用 `VAL_DATA_DIR` 作为 `SCENE_DIR`

**结论**：`train_list.json / val_list.json` 不是额外标注文件，只是训练 loader 要的文件名清单；eval 不需要它们，训练需要但可以自动生成。

### 2026-04-25 — Full eval suite（同一 manifest + `BON_SEED=3402`）

**设置**：
- 500 scenes
- 同一份 `eval_manifest.json`
- 固定 `BON_SEED=3402`
- 脚本：`run_anchor_eval_suite.sh`

| case | collision | progress | route |
|---|---|---|---|
| `raw_no_goal_baseline` | **2.4%** | 0.3282 | 0.8513 |
| `planner_ft_none` | **6.4%** | 0.3549 | 0.8583 |
| `predicted_anchor_top1` | **6.2%** | 0.3715 | 0.8417 |
| `predicted_anchor_rerank_a` | **6.2%** | 0.3818 | 0.8595 |
| `oracle_anchor` | **2.0%** | 0.3459 | 0.8396 |
| `oracle_anchor_rerank` | **2.8%** | 0.3634 | 0.8595 |

**正式解读**：
1. **anchor 架构本身有效**：`oracle_anchor` 相比 `planner_ft_none` 从 `6.4% → 2.0%`，说明 finetuned planner 确实在利用 anchor 信息。
2. **当前 predicted pipeline 还没超过原始 FlowPlanner**：`predicted_anchor_top1 / rerank_a` 都在 `6.2%`，明显差于 raw baseline `2.4%`。
3. **P1 已给出方向性结论**：`oracle_anchor_rerank` 从 `2.0%` 退到 `2.8%`，而这一步**完全不经过 predictor**，说明 planner 对“非 top-1 但仍接近 GT 的 anchor”并不鲁棒，问题不只是 predictor 排序。
4. **rerank 的作用是二级指标整理，不是关 gap 主武器**：`predicted_anchor_rerank_a` 没改善 collision（仍 `6.2%`），但把 progress/route 提到 `0.3818 / 0.8595`。
5. **当前最可信的下一步优先级**：先做 **planner robustness / exposure bias**（scheduled anchor sampling），再做 predictor metric heads。

### 2026-04-25 — Raw no-goal baseline（`flowplanner_no_goal.pth + anchor_mode none`）

**Checkpoint**：`/root/autodl-tmp/ckpts/flowplanner_no_goal.pth`
**脚本**：`run_anchor_raw_no_goal_eval.sh`
**数据集**：500 scenes
**结果来源**：用户回传。若按脚本默认执行，则优先复用已有 `eval_manifest.json`。

| 模式 | collision | progress | route |
|---|---|---|---|
| `raw_no_goal_baseline` | **2.0%** | **0.3275** | **0.8547** |

**关键解读**：
1. **这是目前最重要的新基线**：原始 FlowPlanner 的 collision `2.0%`，远好于 `planner_ft_run1 + anchor_mode none` 的 `7.2%`。
2. **当前 predicted 线还没超过原始模型**：
   - `predicted_anchor top1` = `5.8% / 0.3710 / 0.8408`
   - `predicted_anchor_rerank A` = `5.8% / 0.3833 / 0.8595`
   - `predicted_anchor_rerank C` = `5.4% / 0.3771 / 0.8372`
   在 collision 维度都明显差于 raw baseline `2.0%`。
3. **`oracle_anchor` 只比 raw baseline 略好一点**：`1.6%` vs `2.0%`，说明 anchor 架构上限是有增益的，但当前 predicted pipeline 还没有把这个增益兑现出来。
4. **新的研究问题出现了**：除了 predictor/planner 协同外，还要解释为什么 `planner_ft_run1 + none` 会相对 raw baseline 明显退化。

### 2026-04-25 — P1 诊断工具落地（`oracle_anchor_rerank`）

- 已在 `eval_multidim.py` / `eval_multidim_utils.py` 加入 `oracle_anchor_rerank`
- 实现方式：从 GT future 在 anchor vocab 中取 **top-k 最近邻 anchor** 作为候选池；planner 分别生成轨迹后，复用现有 `TrajectoryScorer` rerank
- 作用：**排除 predictor 因素**，只测 planner 对“非 top-1 但仍接近 GT 的 anchor”是否鲁棒
- 状态：**代码已就绪，结果待跑**

### 2026-04-25 — Top-3 rerank 权重扫描（A / B / C）

**Checkpoint**：`planner_ft_run1`
**数据集**：500 scenes
**前提**：`predicted_anchor` top1 baseline = collision `5.8%`, progress `0.3710`, route `0.8408`
**备注**：本节以及下文反复引用的 `none = 7.2% / 0.35 / 0.85`，目前指的是 **`planner_ft_run1/planner_anchor_best.pth` 在 `anchor_mode none` 下的结果**，不是已经单独确认过的“官方原始 `flowplanner_no_goal.pth` baseline”。如果论文/汇报需要“原始 FlowPlanner”严格数值，需要额外单跑一次 `flowplanner_no_goal.pth + anchor_mode none`。

| 组别 | 权重 `(collision, ttc, route, comfort, progress)` | collision | progress | route |
|---|---|---|---|---|
| A | `(80, 15, 5, 0, 0)` | `5.8%` | `0.3833` | `0.8595` |
| B | `(40, 20, 15, 25, 0)` | `6.6%` | `0.3780` | `0.8634` |
| C | `(100, 0, 0, 0, 0)` | `5.4%` | `0.3771` | `0.8372` |

**重新解读（和前两轮口头判断对齐后的正式版）**：
1. **A 是当前推荐默认方案**：相对 top1，collision **不变**（`5.8% → 5.8%`），但 progress 和 route 同时上升（`0.3710 → 0.3833`, `0.8408 → 0.8595`），是最稳妥的多指标折中。
2. **C 不是默认方案，而是 safety-only 诊断上限**：它给出当前 top-3 候选池里按纯 collision 排序能达到的最好安全值（`5.4%`），但 route 退到 `0.8372`，说明它是在用少量路线一致性换碰撞率。
3. **B 可以排除**：相比默认 top3 rerank（`6.4 / 0.3808 / 0.8660`），B 在 collision / progress / route 三轴都更差，说明高 comfort 权重会压掉本该保留的避撞动作。
4. **P0 的核心结论不变**：不管 A 还是 C，都还远高于 oracle `1.6%`；rerank 只能做小修，关不上 oracle-vs-predicted 的大 gap。

### 2026-04-24 — Loader bug 修复 + rerank ablation

**Checkpoint**：`planner_ft_run1`（finetune 完成，`anchor_out_proj_abs_mean_end` 从 epoch1 `4.1e-4` → epoch10 `1.9e-3`）
**数据集**：500 scenes

| 模式 | collision | progress | route |
|---|---|---|---|
| `none` | 7.2% | 0.35 | 0.85 |
| `predicted_anchor` (top1) | **5.8%** | **0.371** | **0.841** |
| `predicted_anchor_rerank` (top3, 默认权重) | **6.4%** | **0.381** | **0.866** |
| `oracle_anchor` | **1.6%** | 0.347 | 0.837 |

**AnchorPredictor accuracy**：top1 ≈ 30%+，top3 ≈ 99%

**关键解读**：
1. **架构 work**：oracle 1.6% 远好于 none 7.2%（−78% 相对）→ planner 真的在听 anchor
2. **predicted top1 超过 none**：collision −19%，progress +6%，route 持平 → Phase 1 exit criteria ✅
3. **rerank 反直觉**：top3 rerank 相比 top1，route/progress 好了但 collision 反而略升（+0.6pp）
   - 推测：默认 rerank 权重偏 route/progress，选了"更贴路线但稍近旁车"的候选
4. **真 gap 在 oracle vs predicted**：1.6% vs 5.8% = 3.6×，属于 exposure bias

**已修 bug**：见 §5 的 loader bug 和 state-num mismatch。

### 2026-04-23 — Loader bug 被发现前的错误数据

**Checkpoint**：同上
**数据集**：500 scenes

| 模式 | collision | progress | route |
|---|---|---|---|
| `none` | 7.2% | 0.35 | 0.85 |
| `predicted_anchor` (top1) | ❌ **58%** | 0.8079 | 0.4196 |
| `oracle_anchor` | 1.6% | 0.347 | 0.837 |

**解读（事后）**：predicted_anchor 58% 是 predictor head 权重没加载导致 anchor 信号纯噪声 → **不是架构问题**。修 loader bug 后见 2026-04-24 数据。

**教训**：evaluator 加载 predictor ckpt 时，一定要检查 state_dict 的 "missing keys / unexpected keys" 警告，static shape 可能让错加载静默通过。

### 2026-04-22 — AnchorPredictor 首次训练完成

- 训练脚本：`train_anchor_predictor.py`
- top1 accuracy 30%+，top3 accuracy 99%
- 结论：predictor 能力够强，**不是 predictor 在拖后腿**

### 2026-04-21 — Planner finetune 完成

- 脚本：`finetune_anchor_planner.py`
- 起点 ckpt：`flowplanner_no_goal.pth`
- 监控指标 `anchor_out_proj_abs_mean_end`：
  - epoch 1：`4.1e-4`
  - epoch 10：`1.9e-3`（约 5× 增长）
- 判定：**符合预期**。从零初始化开始平滑点亮，数量级 1e-3 属于"开始起作用但还没饱和"的典型范围

### 2026-04-20 — Phase 0 产物

- `anchor_vocab.npy` 生成完成，shape `(128, 80, 3)`
- `visualize_anchors.py` 可视化通过（覆盖直行 / 左右转 / 变道 / 停车）

---

## 5. 已修复的重大 Bug 清单

### B1. Anchor 注入是 additive 不是 cross-attention
- **症状**：初版实现把 anchor embedding 加到 AdaLN condition，重演了 goal 线的 `oracle ≈ none`
- **修复**：重写 `decoder.py`，实装 `AnchorTokenEncoder` + `AnchorCrossAttention`，out_proj 零初始化，DiT 内部完全不看 anchor
- **Commit**：`c48b4d2`

### B2. 配置参数缺失导致 anchor 模块没被实例化
- **症状**：训练不报错但 anchor 路径是死的
- **根因**：`train_anchor_predictor.py` 没把 `anchor_state_dim / anchor_len / anchor_token_num / anchor_attn_heads` patch 到 cfg
- **修复**：在 `train_anchor_predictor.py::load_planner` 和 `eval_multidim_utils.py::load_planner_model` 都加 patching

### B3. 指错了 top-level 配置
- **症状**：启动直接 `model.future_len missing` 崩掉
- **根因**：早期文档让用户 `--planner-config flow_planner_anchor.yaml`（model-level），但 Hydra 需要 top-level
- **修复**：新建 `flow_planner/script/anchor_finetune.yaml` 作为 top-level，统一到这个

### B4. Planner finetune 脚本缺失（P0 #1 blocker）
- **症状**：AnchorCrossAttention zero-init 后如果不训练，`oracle_anchor` 和 `none` 完全一样
- **修复**：新建 `finetune_anchor_planner.py`，差分 LR，teacher forcing

### B5. AnchorPredictor 权重在评测时没加载 ★最严重
- **症状**：`predicted_anchor` collision 58%，比 `none` 7.2% 差 8 倍
- **根因**：`train_anchor_predictor.py` 把权重存在 payload 的 `"model"` key，但 `_unwrap_state_dict` 不认 `"model"` 这个 key，静默返回整个 payload 当 state_dict → 导致实际加载失败
- **修复**：
  1. `_unwrap_state_dict` 新增认 `"model"` 和 `"anchor_predictor_state_dict"`
  2. 新增 `_extract_predictor_head_state_dict`，只保留 `head.*` 权重（backbone 由 load_planner 负责）
- **影响**：修完 collision 58% → 5.8%，**证明架构本身完全 work**
- **重要备注**：commit `a047873` 的 message 写成了 `fix state num match bug`，但实际 diff 修的是 **predictor checkpoint loader**，不是模型 `state_dim/state_num` 维度错误；而且这套 loader 修复同时覆盖 `load_goal_predictor_model()` 和 `load_anchor_predictor_model()`，因此**旧 goal predictor 线理论上也可能中招过同类错加载**。

### B6. `_get_anchor_for_gt` shape alignment 脆弱
- **症状**：某些场景 anchor 对齐错帧
- **修复**：对齐到末 `T_anchor` 帧（`gt_future[:, -T_anchor:, :3]`），而不是头 T_anchor 帧

### B7. AnchorPredictor.get_anchor_labels 冗余 NN 查找
- **修复**：refactor 避免重复计算

### B8. State num mismatch
- **Commit**：`a047873`
- **修复**：对齐 state 维度

---

## 6. 当前瓶颈分析

### 6.1 原始 FlowPlanner baseline 改变了参照系

| 模式 | collision | progress | route |
|---|---|---|---|
| raw `flowplanner_no_goal.pth` + `none` | **2.4%** | 0.3282 | 0.8513 |
| `planner_ft_run1` + `none` | 6.4% | 0.3549 | 0.8583 |
| `predicted_anchor` top1 | 6.2% | 0.3715 | 0.8417 |
| `predicted_anchor_rerank` A | 6.2% | 0.3818 | 0.8595 |
| `oracle_anchor` | **2.0%** | 0.3459 | 0.8396 |

这说明：
1. **过去把 `planner_ft none` 当 baseline 的说法只能做内部对照**，不能再当“原始 FlowPlanner baseline”。
2. **当前 predicted 系统在 collision 上仍输给原始模型**；当前同 manifest/同 seed 最好也只是 `6.2%`。
3. **anchor 架构不是没用**，因为 `oracle_anchor=2.0%` 仍优于 raw baseline `2.4%`；但目前训练/推理链路还没把这个潜力稳定转化成 deployable 改善。

### 6.2 oracle vs predicted 的 gap 依然显著

| 模式 | collision |
|---|---|
| oracle_anchor | 2.0% |
| predicted_anchor top1 | 6.2% |

这个 gap **不是因为** predictor 没训好（top3 accuracy 已经 99%）。
这个 gap **来自**：
1. **Predictor ↔ Planner 协同不完美**：top-1 ≠ 最近邻 anchor，会差几名
2. **Exposure bias**：planner 训练时只见过完美 anchor，推理时稍有偏差就容易出问题
3. **P1 新证据偏向 planner 侧**：`oracle_anchor_rerank` = `2.8%`，明显差于 `oracle_anchor` = `2.0%`，即使 predictor 完全被移除，planner 对近邻但非 top-1 的 anchor 仍会退化

### 6.3 为什么 rerank 当前没完全解决

- `TrajectoryScorer` **不做权重归一化**，而是直接 `sum(weight_i * score_i)`；因此看的是**相对比例**，不是绝对数值。
- 当前 top-3 sweep 表明：**A (`80,15,5,0,0`) 是最好的默认折中点**，它相对 top1 不伤 collision，却能同时抬高 route 和 progress。
- **C (`100,0,0,0,0`) 只能当纯安全诊断**：它说明 top-3 候选池里按 collision 选，最好也就到 `5.4%`，仍然远高于 oracle `1.6%`。
- **B (`40,20,15,25,0`) 证明 comfort 不能当安全代理**：高 comfort 权重会偏好平顺动作，但闭环里急刹/急避有时正是正确避撞动作。

### 6.4 关键诊断实验（待做）

**Oracle top-k rerank**：从 GT 邻居 top-k anchor 里选（oracle candidate pool），看 rerank 会不会比 oracle top-1 差。
- 如果 oracle top-k rerank ≈ oracle top-1 → planner 对不同 anchor 都鲁棒 → 瓶颈在 predictor 排序
- 如果 oracle top-k rerank 明显退化 → planner 对 "非最近 anchor" 脆弱 → 瓶颈在 planner 鲁棒性

---

## 7. 下一步 TODO（按优先级）

### P0 — 立即可做（1 小时级别）
- [x] Rerank 权重扫描（top-3 only）：A=`80/15/5/0/0`，B=`40/20/15/25/0`，C=`100/0/0/0/0`
- [x] **P0 结论**：默认部署/汇报配置采用 **A**；C 仅保留作 safety-only 诊断；B 排除不用
- [x] **P0 进一步结论**：rerank 只能带来小幅修正，无法关闭 `predicted` vs `oracle` 的主要 gap，所以下一步必须做 P1 诊断

### P1 — 诊断（半天）
- [x] 在 `eval_multidim.py` 增加 `oracle_anchor_rerank` 模式（从 oracle 邻居 top-k 里 rerank）
- [x] 跑 oracle top-k rerank，结果：`oracle_anchor 2.0%` vs `oracle_anchor_rerank 2.8%`，说明 planner 对 near-oracle 但非 top-1 anchor 仍有明显退化
- [x] 单独补跑 **官方原始 `flowplanner_no_goal.pth` + `anchor_mode none`**，结果：`2.0% / 0.3275 / 0.8547`
- [x] 已补 AutoDL 脚本：`run_anchor_raw_no_goal_eval.sh`
- [ ] 分析为什么 `planner_ft_run1 + none` 在同 manifest/同 seed 下仍从 raw baseline `2.4%` 退化到 `6.4%`

### P2 — Phase 2 开工（3–5 天）
- [x] **先做 Scheduled anchor sampling（当前唯一最高优先级）**：在 `finetune_anchor_planner.py` 中，不再永远喂 oracle anchor；而是以概率 `p` 喂 predictor top-1，`p` 从 0 线性 ramp 到 0.3~0.5
- [x] Scheduled sampling AutoDL 部署脚本：`run_anchor_scheduled_sampling.sh`
- [ ] Scheduled sampling 第一轮实验设计：只做 2 个短跑版本
  - run S1：`p_max = 0.3`
  - run S2：`p_max = 0.5`
  - 两个 run 都复用当前 `planner_ft_run1` 配置与数据规模，先验证方向，不追求一次性训满
- [ ] Scheduled sampling 第一轮验收：
  - `predicted_anchor_top1` collision 相比当前 `6.2%` 明显下降
  - `oracle_anchor_rerank` 更接近 `oracle_anchor`
  - 若 `planner_ft_none` 进一步恶化，则需要同时重新平衡 unconditional path
- [ ] `train_soft_pref.py` 改造：goal_labels → anchor_labels，scene-level soft KL 保留
- [ ] AnchorPredictor 加 metric heads（dis / dac / safety / progress / comfort）
- [ ] Phase 0 teacher 打分 cache 生成（`risk/trajectory_scorer.py`）

### P3 — Phase 3（DPO）
- [ ] `build_multi_pairs.py`：only same-anchor pair（bucket by anchor_id，DriveDPO hard-neg）
- [ ] `train_dpo.py`：`attach_goal_*` → `attach_anchor_*`

---

## 8. 命令速查（cheat sheet）

### 生成 anchor vocab
```bash
python cluster_trajectories.py \
    --train_data_path /path/to/train \
    --num_anchors 128 \
    --traj_len 80 \
    --output anchor_vocab.npy
```

### 训练 AnchorPredictor
```bash
python train_anchor_predictor.py \
    --planner-config flow_planner/script/anchor_finetune.yaml \
    --anchor-vocab anchor_vocab.npy \
    --planner-ckpt flowplanner_no_goal.pth
```

### Finetune Planner（用 oracle anchor）
```bash
python finetune_anchor_planner.py \
    --planner-config flow_planner/script/anchor_finetune.yaml \
    --planner-ckpt flowplanner_no_goal.pth \
    --anchor-vocab anchor_vocab.npy \
    --lr 2e-5 --decoder-lr-mult 0.1 --encoder-lr-mult 0.0
```

### Eval：预测 anchor + top-k rerank（当前主要 ablation）
```bash
python -m flow_planner.dpo.eval_multidim \
    --planner-config flow_planner/script/anchor_finetune.yaml \
    --planner-ckpt planner_anchor_best.pth \
    --anchor-predictor-ckpt anchor_predictor_best.pth \
    --anchor_vocab anchor_vocab.npy \
    --anchor_mode predicted_anchor_rerank \
    --predicted_anchor_top_k 3 \
    --rerank_collision_weight 80 \
    --rerank_ttc_weight 15 \
    --rerank_route_weight 5 \
    --rerank_comfort_weight 0 \
    --rerank_progress_weight 0.0
```

### Eval：P1 `oracle_anchor_rerank`
```bash
python -m flow_planner.dpo.eval_multidim \
    --config_path flow_planner/script/anchor_finetune.yaml \
    --ckpt_path /root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --scene_dir /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode oracle_anchor_rerank \
    --predicted_anchor_top_k 3 \
    --rerank_collision_weight 80 \
    --rerank_ttc_weight 15 \
    --rerank_route_weight 5 \
    --rerank_comfort_weight 0 \
    --rerank_progress_weight 0.0 \
    --output_json /root/autodl-tmp/anchor_runs/eval_oracle_rerank.json
```

### 对照组
- `--anchor_mode none` → 无条件 baseline
- `--anchor_mode oracle_anchor` → 用 GT 最近邻 anchor（上限）
- `--anchor_mode predicted_anchor` → predictor top-1，无 rerank

### 补跑原始 no-goal baseline（AutoDL 一键脚本）
```bash
bash run_anchor_raw_no_goal_eval.sh
```

默认输出：
- json: `/root/autodl-tmp/anchor_runs/raw_no_goal_eval/eval_raw_no_goal.json`
- log: `/root/autodl-tmp/anchor_runs/raw_no_goal_eval/eval_raw_no_goal.log`

如果 AutoDL 路径不同，可用环境变量覆盖，例如：
```bash
RAW_CKPT=/root/autodl-tmp/ckpts/flowplanner_no_goal.pth \
ANCHOR_VOCAB_PATH=/root/autodl-tmp/anchor_runs/anchor_vocab.npy \
MANIFEST_PATH=/root/autodl-tmp/anchor_runs/eval_manifest.json \
bash run_anchor_raw_no_goal_eval.sh
```

### 一次跑完整 anchor 对照组（推荐部署入口）
```bash
bash run_anchor_eval_suite.sh
```

只跑子集也行，例如：
```bash
bash run_anchor_eval_suite.sh raw_no_goal planner_ft_none oracle_anchor_rerank
```

### Phase 2: Scheduled Anchor Sampling
```bash
# S1: 温和混入 predictor top1 anchor
bash run_anchor_scheduled_sampling.sh 0.3

# S2: 更强混入 predictor top1 anchor
bash run_anchor_scheduled_sampling.sh 0.5
```

每个 run 会先训练，再自动用新 ckpt 跑：
- `planner_ft_none`
- `predicted_anchor_top1`
- `predicted_anchor_rerank_a`
- `oracle_anchor`
- `oracle_anchor_rerank`

---

## 9. 论文/汇报可用素材备忘

### 9.1 Story line
1. goal 线失败 → 证明 2D endpoint + AdaLN 信号太弱
2. anchor 线（本分支）→ 证明完整轨迹 + cross-attention 有效（oracle 1.6% vs none 7.2%）
3. 但 predicted_anchor 离 oracle 还有 3.6× 差距 → 引出 Phase 2 多目标蒸馏和 Phase 3 same-anchor DPO

### 9.2 可以画的图
- Anchor vocab 可视化（ego frame 下 128 条，按聚类簇上色）
- `anchor_out_proj_abs_mean_end` 训练曲线（4.1e-4 → 1.9e-3，证明"从零激活"）
- oracle vs predicted vs none 的 collision/progress/route 三轴条形图
- top3 rerank 的候选 trajectory 叠加图（展示"同场景多合理解"）

### 9.3 Ablation 矩阵建议
| anchor 注入 | anchor 来源 | collision | progress | route |
|---|---|---|---|---|
| 无 | — | ... | ... | ... |
| AdaLN additive | oracle | ... | ... | ... |
| cross-attn | oracle | 1.6% | 0.347 | 0.837 |
| cross-attn | predicted top1 | 5.8% | 0.371 | 0.841 |
| cross-attn | predicted top3 rerank (default `40/20/25/10/0`) | 6.4% | 0.381 | 0.866 |
| cross-attn | predicted top3 rerank A (`80/15/5/0/0`) | 5.8% | 0.3833 | 0.8595 |
| cross-attn | predicted top3 rerank B (`40/20/15/25/0`) | 6.6% | 0.3780 | 0.8634 |
| cross-attn | predicted top3 rerank C (`100/0/0/0/0`) | 5.4% | 0.3771 | 0.8372 |

（additive 那行还没跑，是未来要补的 ablation）

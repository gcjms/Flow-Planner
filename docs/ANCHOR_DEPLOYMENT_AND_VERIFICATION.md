# Anchor 分支部署 & 验证手册 (Phase 0 + Phase 1)

> 目标读者：把 `feature/anchor` 分支完整跑通、拿到 Phase 0 / Phase 1 结论的人。
> 作者假设本文件与所有引用代码均在 `feature/anchor` 分支下。
> **本手册只覆盖 Phase 0（anchor vocab 产物）和 Phase 1（static-anchor baseline）**。
> Phase 2/3 文档等 Phase 1 跑完拿到真实数据再补，避免凭空设计。

---

## 0. 前置条件

| 项目 | 要求 |
|---|---|
| 分支 | `feature/anchor`，相对 `feature/goal` 多出 `anchor_*` 代码 |
| 原 planner 权重 | 一个 `no-goal` 或 `goal` 时代训出来的 ckpt (对应 `future_len=80`) |
| 训练数据 | nuPlan NPZ + `train_list.json` / `val_list.json` |
| Python 环境 | 和旧 DPO 流水线一致（torch、sklearn、tqdm、hydra-core、omegaconf、matplotlib） |
| 硬件 | Phase 0 聚类 CPU 即可；Phase 1 训练单 GPU (RTX 3090/4090/A10/A100) |

下文所有路径按用户习惯的 AutoDL 布局写：
- 代码根：`/root/Flow-Planner`
- 数据：`/root/autodl-tmp/nuplan_npz`
- 输出根：`/root/autodl-tmp/anchor_runs`

请根据你的环境替换。

---

## 1. Phase 0：Anchor Vocabulary

### 1.1 生成 `anchor_vocab.npy`

**输入**：训练 NPZ + 文件清单
**输出**：`anchor_vocab.npy` (K=128, T=80, 3) + `anchor_vocab_meta.json`

```bash
cd /root/Flow-Planner
python -m flow_planner.goal.cluster_trajectories \
    --data_dir  /root/autodl-tmp/nuplan_npz \
    --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
    --output_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --n_anchors 128 \
    --traj_len 80 \
    --heading_weight 5.0 \
    --use_pca \
    --pca_dim 32 \
    --n_init 10
```

关键参数：
- `--traj_len 80` **必须** 等于 FlowPlanner 配置里的 `future_len`（默认 80）。Phase 1 的 `_get_anchor_for_gt` 会断言 `T_anchor == ego_future.shape[1]`。
- `--n_anchors 128`：计划默认；如果 Phase 1 发现 `oracle_anchor` 效果一般再试 192 / 256。
- `--heading_weight 5.0`：让朝向 (cos,sin ∈ [-1,1]) 在距离度量里不被几十米量级的 xy 淹没。
- `--use_pca --pca_dim 32`：T*4 = 320 维对 KMeans 偏重，PCA 到 32 提速且大多数方差都在前 16 维。

**Gate 检查**（直接看 stdout）：
1. `kept` 占比 > 95%，否则 `--max_endpoint_norm` 不合理或数据有问题。
2. `empty_clusters=0` 或 ≤ 2。非零时记下，后面可视化也要看。
3. PCA `explained variance` ≥ 0.90 说明 16-32 维覆盖够。

### 1.2 可视化 anchor

```bash
python visualize_anchors.py \
    --vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --meta_path  /root/autodl-tmp/anchor_runs/anchor_vocab_meta.json \
    --output_dir /root/autodl-tmp/anchor_runs/anchor_viz
```

产出 4 张图：`overlay.png`, `overlay_by_size.png`, `small_multiples.png`, `stats.png`。

**Phase 0 Exit Criteria**（把这张图发给我 code review）：
- [ ] `overlay.png`：能肉眼辨认 5 大类——直行 / 左转 / 右转 / 左变道 / 右变道 / 停车。
- [ ] `overlay_by_size.png`：主方向（直行）的簇 alpha 最深，符合真实分布直觉。
- [ ] `small_multiples.png`：随机抽 5-10 个子图，轨迹光滑、heading 连续，没有「折线鬼画符」。
- [ ] `stats.png`：endpoint 距离分布覆盖 0-60m（取决于 8s 水平），heading 变化分布有明显的弯道尾巴。

任一项不达标：
1. 优先调 `--n_anchors`（调到 192 / 256）。
2. 调 `--heading_weight`（如果方向簇不分明 → 调大到 8；如果弯道被拆太碎 → 调小到 3）。
3. 关掉 `--use_pca` 看是否是降维造成的。

---

## 2. Phase 1：Static Anchor + FlowPlanner

### 2.1 架构改动回顾（已完成的代码）

1. `flow_planner/goal/anchor_utils.py`
   - `load_anchor_vocab`, `find_nearest_anchor(_torch)`, `lookup_anchor_traj`
   - `select_diverse_anchors`（farthest-point sampling），`select_anchor_from_route`
2. `flow_planner/goal/anchor_predictor.py`
   - `AnchorPredictor`：复用 `GoalPredictor` 的 scene feature 抽取，只换 label/classification 维度 = K
3. `flow_planner/model/flow_planner_model/decoder.py`
   - 新增 `AnchorTrajEncoder`：把 `(B, T, 3)` anchor 切成 P 个与 action token 对齐的 window，MLP → `(B, P, hidden)`
   - `FlowPlannerDecoder.__init__` 新增 `anchor_state_dim`, `anchor_len` 两个参数（和 `goal_dim` 互斥）
   - `forward` 里用 `anchor_traj` 替代 `goal_point`；注入方式改为 **逐 action token** 而非 broadcast
4. `flow_planner/model/flow_planner_model/flow_planner.py`
   - `__init__` 新增 `anchor_vocab_path`，加载 `(K, T, 3)` 到 `_anchor_vocab_tensor`
   - 新增 `_get_anchor_for_gt` → 供 AnchorPredictor 和 `forward_train` 取 label / 注入条件
   - `forward_train` 同时支持 goal / anchor（互斥）
   - `forward_inference` 新增 `anchor_traj=None` 参数 + CFG 复制逻辑
5. `flow_planner/dpo/eval_multidim_utils.py`
   - 新增 `choose_anchor(anchor_mode, ...)` — mode ∈ {none, route_anchor, predicted_anchor, oracle_anchor}
   - `run_multidim_evaluation` 新增 `anchor_mode` / `anchor_vocab` / `anchor_predictor`，与 goal 路径互斥
   - 新增 `load_anchor_predictor_model`, `resolve_anchor_vocab`
6. `train_anchor_predictor.py`（根目录入口）— 镜像 `train_goal_predictor.py`

**向后兼容保留**：goal 路径完全没删，`feature/goal` 现有的 goal 相关命令继续能用，方便做 A/B 对比。

### 2.2 Planner 配置调整

原 `goal_vocab_path` 配置保持不变。你需要在 model 配置里加：

```yaml
# conf/planner.yaml (示意，按实际字段名补)
model:
  _target_: flow_planner.model.flow_planner_model.flow_planner.FlowPlanner
  # ...既有字段...
  anchor_vocab_path: /root/autodl-tmp/anchor_runs/anchor_vocab.npy   # 新增
  goal_vocab_path: null                                               # 必须置空，二选一
  model_decoder:
    _target_: flow_planner.model.flow_planner_model.decoder.FlowPlannerDecoder
    # ...既有字段...
    goal_dim: 0                     # 关闭老 goal 通路
    anchor_state_dim: 3             # 开启 anchor (x, y, heading)
    anchor_len: 80                  # 必须 = future_len
```

> 代码里设置了三重 guard：`(goal_vocab_path, anchor_vocab_path)` 互斥、`(goal_dim, anchor_state_dim)` 互斥、
> `anchor_state_dim>0` 时 `anchor_len>0` 必填。配错的话启动就会 raise，不会默默跑错。

### 2.3 冷启动 smoke test（5 分钟内跑完）

**目的**：在真正训之前，验证前向能通、loss 可算、CFG 不崩。

```bash
cd /root/Flow-Planner
python - <<'PY'
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flow_planner.dpo.config_utils import load_composed_config

cfg_path = "conf/planner.yaml"
cfg = load_composed_config(cfg_path)
OmegaConf.update(cfg, "model.anchor_vocab_path",
                 "/root/autodl-tmp/anchor_runs/anchor_vocab.npy",
                 force_add=True)
OmegaConf.update(cfg, "model.goal_vocab_path", None, force_add=True)
OmegaConf.update(cfg, "model.model_decoder.goal_dim", 0, force_add=True)
OmegaConf.update(cfg, "model.model_decoder.anchor_state_dim", 3, force_add=True)
OmegaConf.update(cfg, "model.model_decoder.anchor_len", 80, force_add=True)
model = instantiate(cfg.model).cuda().eval()
print("model built:", type(model).__name__,
      "anchor vocab shape:", model._anchor_vocab_tensor.shape)

# Fake batch with a single scene
# (replace with a real data sample if you have one; smoke test can stop here)
print("OK")
PY
```

预期 stdout：
```
model built: FlowPlanner anchor vocab shape: torch.Size([128, 80, 3])
OK
```

如果 `model_decoder` 抱怨 `anchor_encoder` 找不到模块 — 检查你更新了 `decoder.py`，以及 config 里 `_target_` 指向仍是同一个类。

### 2.4 AnchorPredictor 训练

```bash
python train_anchor_predictor.py \
    --planner-config conf/planner.yaml \
    --planner-ckpt   /root/autodl-tmp/ckpts/flowplanner_no_goal.pth \
    --anchor-vocab-path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --train-data-dir  /root/autodl-tmp/nuplan_npz \
    --train-data-list /root/autodl-tmp/nuplan_npz/train_list.json \
    --val-data-dir    /root/autodl-tmp/nuplan_npz \
    --val-data-list   /root/autodl-tmp/nuplan_npz/val_list.json \
    --save-dir        /root/autodl-tmp/anchor_runs/anchor_predictor_run1 \
    --epochs 10 --batch-size 64 --lr 1e-3 --hidden-dim 256 \
    --max-train-samples 50000   # 先跑 small，确认可用再全量
```

**gate**（看 `history.json` 最后一行）：
- [ ] val `top1` ≥ 0.25（128 类 random = 0.78%，25% 说明 scene feature 对 anchor 有区分力）
- [ ] val `top5` ≥ 0.60
- [ ] train/val gap < 10% 绝对（没严重过拟合）

达不到 top1 ≥ 0.25 的常见原因：
1. `anchor_vocab` 里有"同一方向但速度不同"的近重复簇，CE 分不清 → 调小 K 或提高 `heading_weight`。
2. Planner backbone 冻结导致 scene feature 不适配 anchor label → 加 `--unfreeze-backbone`。

### 2.5 联合训练 FlowPlanner + anchor 注入（可选，Phase 1 的硬核环节）

Phase 1 的 Exit Criteria 允许直接从"已有 no-goal planner ckpt + 冻结 anchor predictor"走开路评估（见 §2.6），
如果走到这一步说明你想看"Flow 生成时真正用 anchor 去 refine"的效果。

**方案 A：从 scratch 训**（最干净，推荐）
- 用标准训练脚本 `train.py`（老的），config 按 §2.2 设置为 anchor 模式即可。
- 学习率、batch size 同 no-goal baseline；条件 zero-init 保证早期稳定。

**方案 B：finetune no-goal ckpt**
- `strict=False` 加载 no-goal 权重。`anchor_encoder` 的权重是新的（missing keys），会被 zero-init。
- 建议 10-20 epoch 小 lr (2e-5) fine-tune。

训练期间关键 metric：
| Metric | 期望 |
|---|---|
| `ego_planning_loss` | 和 no-goal baseline 相当；anchor 不应该让 loss 变高 |
| `consistency_loss` | 不变（与 anchor 无关） |
| anchor zero-init 的 L2 | 前 1-2 个 epoch 应持续增长（模型开始使用 anchor） |

### 2.6 Open-loop 多维评估（Phase 1 核心 gate）

不联合训，直接用 `no-goal` ckpt + anchor 注入（冷启动）是最快的路径，以及 finetune 后再跑同样命令。

`eval_multidim.py` 的 CLI 用下划线风格（与仓库既有约定一致），注意区别于
`train_anchor_predictor.py` 用连字符的风格：

```bash
cd /root/Flow-Planner

# 先保存一个公用 manifest，确保三个 run 看同一批场景
python -m flow_planner.dpo.eval_multidim \
    --config_path conf/planner.yaml \
    --ckpt_path   /root/autodl-tmp/ckpts/flowplanner_anchor_ft.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --write_scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --max_scenes 500 \
    --anchor_mode none \
    --output_json /root/autodl-tmp/anchor_runs/eval_none.json

# predicted_anchor (同一 manifest，同一 ckpt)
python -m flow_planner.dpo.eval_multidim \
    --config_path conf/planner.yaml \
    --ckpt_path   /root/autodl-tmp/ckpts/flowplanner_anchor_ft.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --anchor_predictor_ckpt /root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode predicted_anchor \
    --output_json /root/autodl-tmp/anchor_runs/eval_predicted.json

# oracle_anchor (同一 manifest，同一 ckpt)
python -m flow_planner.dpo.eval_multidim \
    --config_path conf/planner.yaml \
    --ckpt_path   /root/autodl-tmp/ckpts/flowplanner_anchor_ft.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode oracle_anchor \
    --output_json /root/autodl-tmp/anchor_runs/eval_oracle.json
```

CLI 已经在本分支加好 `--anchor_vocab_path`, `--anchor_mode`, `--anchor_predictor_ckpt`
这 3 个参数，不需要再改 argparse。

**Phase 1 Exit Criteria**（严格 gate，达不到就停下来 debug，不推 Phase 2）：

| 指标 | `none` baseline | `predicted_anchor` | `oracle_anchor` | 通过条件 |
|---|---|---|---|---|
| collision_rate | X | ~X | < X × 0.8 | **oracle 比 none 相对下降 > 20%** |
| route_score | Y | > Y | > Y | oracle 不能劣化 route |
| comfort_score | Z | ~Z | ~Z | 舒适度不能显著劣化 |

如果 `oracle_anchor ≈ none`：
1. 确认 `anchor_traj` 进到了 decoder：在 `forward_train` 打印 `decoder_model_extra.keys()` 看有没有 `anchor_traj`。
2. 确认 `AnchorTrajEncoder` 的 last-layer 权重非零：训练若干 step 后 `decoder.anchor_encoder.window_mlp[-1].weight.abs().mean()` 应 > 0。
3. CFG 概率太高：unconditioned 占比 50% 可能过高，试 `cfg_prob=0.2`。
4. 本质问题：anchor 没有足够区分力，回到 §1.1 调 K。

---

## 3. 自查清单（按顺序打勾）

### 3.1 Phase 0
- [ ] `anchor_vocab.npy` 文件存在，`shape == (K, 80, 3)`
- [ ] `anchor_vocab_meta.json` 存在，`empty_clusters ≤ 2`
- [ ] `overlay.png` 能辨认 5 大驾驶模态
- [ ] `stats.png` endpoint 距离分布合理

### 3.2 Phase 1 Smoke
- [ ] `flow_planner.model.flow_planner_model.decoder.AnchorTrajEncoder` 可 import
- [ ] `FlowPlanner.__init__` 接受 `anchor_vocab_path` 不报错
- [ ] `model._anchor_vocab_tensor.shape == (K, 80, 3)`
- [ ] `model(data, mode='train')` 前向不崩
- [ ] `model(data, mode='inference', anchor_traj=anchor)` 前向不崩
- [ ] CFG replication 正确：`use_cfg=True` 时 `anchor_traj` 被拼成 `(2B, T, 3)`

### 3.3 Phase 1 Predictor
- [ ] `AnchorPredictor.get_anchor_labels(data).shape == (B,)`，值 ∈ `[0, K)`
- [ ] 训练 loss 单调下降，10 epoch 内 val top1 ≥ 0.25
- [ ] `predict_topk(data, 5)["anchor_trajs"].shape == (B, 5, 80, 3)`

### 3.4 Phase 1 Eval
- [ ] `eval_none`, `eval_predicted`, `eval_oracle` 三份 json 都生成
- [ ] `oracle_anchor` 碰撞率比 `none` 相对下降 ≥ 20%
- [ ] `predicted_anchor` 碰撞率 ≥ `none` 至少不差（差也不超过 5% 绝对）

只有 **3.1 / 3.2 / 3.3 / 3.4 全绿** 才推 Phase 2（multi-target distillation）。任何一个红灯先在这一章解决。

---

## 4. 常见故障速查

| 症状 | 最可能原因 | 修复 |
|---|---|---|
| `Anchor horizon T=40 must match data.ego_future T=80` | Phase 0 用了 `--traj_len 40` | 重跑 Phase 0 `--traj_len 80` |
| `goal_dim and anchor_state_dim are mutually exclusive` | config 里两个都设了 | 老 `goal_dim` 改 `0`，anchor 改 `3` |
| 训练 loss NaN | `anchor_traj` 里有 NaN / 无穷大 | 看 `anchor_vocab_meta.json` 确认 `cluster_counts` 里没 0 簇(NaN mean) |
| predictor top1 卡在 3% | scene feature 没区分度 | `--unfreeze-backbone` + 更小 lr (3e-4) |
| `eval_multidim.py: unrecognized arguments: --anchor-mode` | CLI 还没加 anchor flag | 参考 §2.6 注释，在 `eval_multidim.py` argparse 里加 3 行 |
| `oracle_anchor ≈ none` | anchor 没被 decoder 读到 | §2.6 debug 清单 1-4 |

---

## 5. Phase 1 完成之后再告诉我

把以下 3 组数据贴给我（或直接 commit 到 `feature/anchor` 下的 `experiments/phase1_report.md`）：

1. `anchor_vocab_meta.json` 的关键字段（K, empty_clusters, inertia）
2. `anchor_predictor_run1/history.json` 最后 3 个 epoch
3. `eval_none.json` / `eval_predicted.json` / `eval_oracle.json` 的 summary 段

有了这三份我就能判断是否真的 gate through 到 Phase 2，以及下一步要不要调 K、要不要改 anchor injection 位置（currently additive per-token；候选升级：加一路 cross-attn）。

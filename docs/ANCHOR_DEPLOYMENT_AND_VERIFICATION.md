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
   - 新增 `AnchorTokenEncoder`：`(B, T, 3)` → L 个 learnable query + 单层 cross-attn 得到 `(B, L, hidden)` anchor tokens（L=4）
   - 新增 `AnchorCrossAttention`：trajectory token 作 Q，anchor token 作 K/V，output_proj 零初始化
   - `FlowPlannerDecoder.__init__` 新增 `anchor_state_dim` / `anchor_len` / `anchor_token_num` / `anchor_attn_heads`（和 `goal_dim` 互斥）
   - `forward` 在 DiT 之前做 **一次 cross-attention** 把 anchor 信息写入 trajectory token；anchor 不再进入 AdaLN 的 `y` 加法路径（goal 的 AdaLN additive 仅作为 legacy 保留）
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

提供了两层 config：

1. **model-level**：`flow_planner/script/model/flow_planner_anchor.yaml`
   — 只负责 FlowPlanner 的模型结构（encoder / decoder / anchor 相关字段），供其他 top-level config 用 `defaults: - model: flow_planner_anchor` 引用。
2. **top-level**：`flow_planner/script/anchor_finetune.yaml`
   — 把 `model / data / core / optimizer / scheduler / ema` 全部组合好，**这是** `train_anchor_predictor.py`、
   `finetune_anchor_planner.py`、`flow_planner.dpo.eval_multidim` 都要传入的那个 `--planner-config`。
   它的 `pretrained_checkpoint / save_dir / project_root` 按需 override。

关键字段（都由 `flow_planner_anchor.yaml` 写死）：

```yaml
model:
  _target_: flow_planner.model.flow_planner_model.flow_planner.FlowPlanner
  anchor_vocab_path: ${project_root}/anchor_vocab.npy   # 新增
  # goal_vocab_path 不设 → 为 null
  model_decoder:
    goal_dim: 0                     # 关闭老 goal 通路
    anchor_state_dim: 3             # 开启 anchor (x, y, heading)
    anchor_len: ${..future_len}     # 必须 = future_len
    anchor_token_num: 4             # 压缩成 4 个 anchor summary token
    anchor_attn_heads: 8            # trajectory -> anchor cross-attn 的 heads
```

> 代码里设置了多重 guard：`(goal_vocab_path, anchor_vocab_path)` 互斥、`(goal_dim, anchor_state_dim)` 互斥、
> `anchor_state_dim>0` 时 `anchor_len>0` 必填。配错的话启动就会 raise，不会默默跑错。
>
> **重要**：`train_anchor_predictor.py`、`finetune_anchor_planner.py` 和
> `flow_planner/dpo/eval_multidim_utils.py::load_planner_model` 都会**自动**把 `model.model_decoder` 下的
> `anchor_state_dim / anchor_len / anchor_token_num / anchor_attn_heads` 给 patch 上（基于 `anchor_vocab_path`
> + `future_len`）。即使 `--planner-config` 指向一个不含这些字段的旧 top-level config（如 `goal_finetune.yaml`），
> 也不会因为缺字段而静默失效。但首选还是 `flow_planner/script/anchor_finetune.yaml`。

### 2.3 冷启动 smoke test（5 分钟内跑完）

**目的**：在真正训之前，验证前向能通、loss 可算、CFG 不崩。

```bash
cd /root/Flow-Planner
python - <<'PY'
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flow_planner.dpo.config_utils import load_composed_config

cfg_path = "flow_planner/script/anchor_finetune.yaml"
cfg = load_composed_config(cfg_path)
# Override the vocab path to where you actually saved it (optional if your
# anchor_finetune.yaml's project_root already resolves to the right place).
OmegaConf.update(cfg, "model.anchor_vocab_path",
                 "/root/autodl-tmp/anchor_runs/anchor_vocab.npy",
                 force_add=True)

model = instantiate(cfg.model).cuda().eval()
print("model built:", type(model).__name__,
      "anchor vocab:", tuple(model._anchor_vocab_tensor.shape),
      "has_anchor_encoder:", hasattr(model.model_decoder, "anchor_encoder"),
      "has_anchor_cross_attn:", hasattr(model.model_decoder, "anchor_cross_attn"))
print("OK")
PY
```

预期 stdout（关键是 `has_anchor_encoder: True` 和 `has_anchor_cross_attn: True`）：
```
model built: FlowPlanner anchor vocab: (128, 80, 3) has_anchor_encoder: True has_anchor_cross_attn: True
OK
```

若 `has_anchor_encoder: False` — 你传入的 top-level config 走的是 `flow_planner` / `flow_planner_goal` 的 model，
没走 `flow_planner_anchor`。要么换成 `flow_planner/script/anchor_finetune.yaml`，要么在 cfg override 里显式设
`model.model_decoder.anchor_state_dim=3`、`model.model_decoder.anchor_len=${model.future_len}` 等字段。

### 2.4 AnchorPredictor 训练

```bash
python train_anchor_predictor.py \
    --planner-config flow_planner/script/anchor_finetune.yaml \
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

### 2.5 联合训练 FlowPlanner + anchor 注入（Phase 1 硬核环节，**必跑**）

**为什么必跑**：我们新加的 `AnchorTokenEncoder` + `AnchorCrossAttention` 在 no-goal ckpt 里完全不存在权重，
加载时是随机初始化的 + `anchor_cross_attn.out_proj` 被**零初始化**。这意味着在 finetune 之前，
anchor 对模型输出的贡献精确等于 0；你直接跑 §2.6 的 `oracle_anchor` 和 `none` 指标会一模一样。
跑完这一步以后，`oracle_anchor` 才会把 anchor 真正用起来。

直接用仓库里的专用脚本（teacher-force oracle anchor）：

```bash
python finetune_anchor_planner.py \
    --planner-config flow_planner/script/anchor_finetune.yaml \
    --planner-ckpt   /root/autodl-tmp/ckpts/flowplanner_no_goal.pth \
    --anchor-vocab-path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --train-data-dir  /root/autodl-tmp/nuplan_npz \
    --train-data-list /root/autodl-tmp/nuplan_npz/train_list.json \
    --val-data-dir    /root/autodl-tmp/nuplan_npz \
    --val-data-list   /root/autodl-tmp/nuplan_npz/val_list.json \
    --save-dir        /root/autodl-tmp/anchor_runs/planner_ft_run1 \
    --epochs 10 --batch-size 32 --lr 2e-5 --decoder-lr-mult 0.1 \
    --max-train-samples 80000
```

脚本的默认 lr 分组（可通过 CLI 调）：
- `anchor_encoder + anchor_cross_attn`：`--lr`（默认 2e-5）
- 其余 `model_decoder`：`--lr * --decoder-lr-mult`（默认 0.1x）
- `model_encoder`：**冻结**（`--encoder-lr-mult` 默认 0；若强行解冻会让 AnchorPredictor 的 scene feature 漂走，慎用）

训练期间关键 diagnostic（脚本每个 epoch 会打印）：

| Metric | 期望 |
|---|---|
| `ego_planning_loss` | 和 no-goal baseline 相当；anchor 不应该让 loss 变高 |
| `consistency_loss` | 基本不变（与 anchor 无关） |
| `anchor_out_proj_abs_mean_end` | **必须从 0 开始持续增长**；如果跑完 10 epoch 还接近 0，说明 anchor 路径没收到梯度 |

**跑完这一步产出的 ckpt**（`planner_anchor_best.pth`）就是 §2.6 里各个 `--ckpt_path` 要指向的那个。

### 2.6 Open-loop 多维评估（Phase 1 核心 gate）

不联合训，直接用 `no-goal` ckpt + anchor 注入（冷启动）是最快的路径，以及 finetune 后再跑同样命令。

`eval_multidim.py` 的 CLI 用下划线风格（与仓库既有约定一致），注意区别于
`train_anchor_predictor.py` 用连字符的风格：

```bash
cd /root/Flow-Planner

# 先保存一个公用 manifest，确保三个 run 看同一批场景
python -m flow_planner.dpo.eval_multidim \
    --config_path flow_planner/script/anchor_finetune.yaml \
    --ckpt_path   /root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --write_scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --max_scenes 500 \
    --anchor_mode none \
    --output_json /root/autodl-tmp/anchor_runs/eval_none.json

# predicted_anchor (同一 manifest，同一 ckpt)
python -m flow_planner.dpo.eval_multidim \
    --config_path flow_planner/script/anchor_finetune.yaml \
    --ckpt_path   /root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --anchor_predictor_ckpt /root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode predicted_anchor \
    --output_json /root/autodl-tmp/anchor_runs/eval_predicted.json

# predicted_anchor_rerank (top3 候选 + 在线启发式 rerank；不使用 GT 未来参与选择)
python -m flow_planner.dpo.eval_multidim \
    --config_path flow_planner/script/anchor_finetune.yaml \
    --ckpt_path   /root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --anchor_predictor_ckpt /root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode predicted_anchor_rerank \
    --predicted_anchor_top_k 3 \
    --rerank_collision_weight 40 \
    --rerank_ttc_weight 20 \
    --rerank_route_weight 25 \
    --rerank_comfort_weight 10 \
    --rerank_progress_weight 0 \
    --output_json /root/autodl-tmp/anchor_runs/eval_predicted_rerank_top3.json

# oracle_anchor (同一 manifest，同一 ckpt)
python -m flow_planner.dpo.eval_multidim \
    --config_path flow_planner/script/anchor_finetune.yaml \
    --ckpt_path   /root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth \
    --anchor_vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --scene_dir   /root/autodl-tmp/nuplan_npz \
    --scene_manifest /root/autodl-tmp/anchor_runs/eval_manifest.json \
    --anchor_mode oracle_anchor \
    --output_json /root/autodl-tmp/anchor_runs/eval_oracle.json
```

CLI 已经在本分支加好 `--anchor_vocab_path`, `--anchor_mode`, `--anchor_predictor_ckpt`，
以及 `predicted_anchor_rerank` 所需的 `--predicted_anchor_top_k` / `--rerank_*` 参数。

> 兼容性说明：`load_anchor_predictor_model()` 现在会正确读取
> `train_anchor_predictor.py` / `train_goal_predictor.py` 保存的 `payload["model"]`
> 格式 checkpoint，并且只加载 predictor 的 `head.*` 权重；如果之前看到
> `Anchor predictor missing 6 keys / unexpected 9 keys`，请更新到当前 commit 后重跑。

**Phase 1 Exit Criteria**（严格 gate，达不到就停下来 debug，不推 Phase 2）：

| 指标 | `none` baseline | `predicted_anchor` | `oracle_anchor` | 通过条件 |
|---|---|---|---|---|
| collision_rate | X | ~X | < X × 0.8 | **oracle 比 none 相对下降 > 20%** |
| route_score | Y | > Y | > Y | oracle 不能劣化 route |
| comfort_score | Z | ~Z | ~Z | 舒适度不能显著劣化 |

如果 `oracle_anchor ≈ none`：
1. **没跑 §2.5 finetune** — 最常见。预训 ckpt 里 `anchor_cross_attn.out_proj` 零初始化，不 finetune 永远是 0 贡献。先跑 §2.5。
2. finetune 跑了但 `anchor_out_proj_abs_mean_end` 仍接近 0：检查 `decoder.anchor_cross_attn.out_proj.weight.abs().mean()` 以及 `decoder.anchor_encoder` 参数的 `grad` 有没有被生成；很可能是 param group 没把它收进去（看 finetune 脚本启动时打印的 `[param-group] ... n_params=...` 是否合理）。
3. 确认 `anchor_traj` 进到了 decoder：在 `forward_train` 打印 `decoder_model_extra.keys()` 看有没有 `anchor_traj`。
4. CFG 概率太高：unconditioned 占比 30% 在 finetune 早期可能过高，试 `cfg_prob=0.1` 跑 1 个 epoch 看 out_proj 是否更快脱离 0。
5. 本质问题：anchor 没有足够区分力，回到 §1.1 调 K。

---

## 3. 自查清单（按顺序打勾）

### 3.1 Phase 0
- [ ] `anchor_vocab.npy` 文件存在，`shape == (K, 80, 3)`
- [ ] `anchor_vocab_meta.json` 存在，`empty_clusters ≤ 2`
- [ ] `overlay.png` 能辨认 5 大驾驶模态
- [ ] `stats.png` endpoint 距离分布合理

### 3.2 Phase 1 Smoke
- [ ] `flow_planner.model.flow_planner_model.decoder.AnchorTokenEncoder` 可 import
- [ ] `flow_planner.model.flow_planner_model.decoder.AnchorCrossAttention` 可 import
- [ ] `FlowPlanner.__init__` 接受 `anchor_vocab_path` 不报错
- [ ] `model._anchor_vocab_tensor.shape == (K, 80, 3)`
- [ ] `model.model_decoder.anchor_encoder` 和 `model.model_decoder.anchor_cross_attn` 均存在
- [ ] `model.model_decoder.anchor_cross_attn.out_proj.weight.abs().max() == 0`（零初始化）
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

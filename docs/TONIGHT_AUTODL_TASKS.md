# 今晚（AutoDL）试验任务清单 - 2026.04.23

**目标**：完成 Phase 0（anchor vocabulary）+ Phase 1 核心（AnchorPredictor + Planner Finetune），为 Phase 1 Exit Criteria 收集数据。

预计总耗时 **4~8 小时**（取决于数据量和 GPU 速度）。按顺序执行，不要跳过。

---

## 任务 1: Phase 0 — 生成 Anchor Vocabulary（30~90 分钟） ★ **必须先跑**

**命令**：

```bash
cd /root/Flow-Planner

# 1. 聚类生成 vocab
python -m flow_planner.goal.cluster_trajectories \
    --data_dir  /root/autodl-tmp/nuplan_npz \
    --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
    --output_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --n_anchors 128 --traj_len 80 --heading_weight 5.0 \
    --use_pca --pca_dim 32 --n_init 10

# 2. 可视化
python visualize_anchors.py \
    --vocab_path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
    --meta_path  /root/autodl-tmp/anchor_runs/anchor_vocab_meta.json \
    --output_dir /root/autodl-tmp/anchor_runs/anchor_viz
```

**预期结果**：
- `anchor_vocab.npy` (shape `(128, 80, 3)`)
- `anchor_vocab_meta.json`：`empty_clusters ≤ 2`，`kept > 95%`，PCA explained variance ≥ 0.90
- `anchor_viz/` 下 4 张图：
  - `overlay.png`：清晰分辨**直行、左转、右转、左变道、右变道、停车**等 5~6 大类
  - `small_multiples.png`：轨迹光滑、heading 连续
  - `stats.png`：endpoint 分布合理（0-60m 覆盖）

**Gate**：把 `overlay.png`、`stats.png` 和 `anchor_vocab_meta.json` 关键字段发给我 review。如果聚类质量差，立即调 `--n_anchors` 或 `--heading_weight` 重跑。

---

## 任务 2: Smoke Test（5 分钟） ★ 验证架构是否正确加载

**命令**：

```bash
python - <<'PY'
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flow_planner.dpo.config_utils import load_composed_config

cfg = load_composed_config("flow_planner/script/anchor_finetune.yaml")
OmegaConf.update(cfg, "model.anchor_vocab_path", 
                 "/root/autodl-tmp/anchor_runs/anchor_vocab.npy", force_add=True)

model = instantiate(cfg.model).cuda().eval()
print("model built:", type(model).__name__)
print("anchor vocab shape:", tuple(model._anchor_vocab_tensor.shape))
print("has_anchor_encoder:", hasattr(model.model_decoder, "anchor_encoder"))
print("has_anchor_cross_attn:", hasattr(model.model_decoder, "anchor_cross_attn"))
print("out_proj zero-init check:", 
      model.model_decoder.anchor_cross_attn.out_proj.weight.abs().max().item() == 0)
print("✅ Smoke test passed")
PY
```

**预期结果**：输出中 `has_anchor_encoder: True`、`has_anchor_cross_attn: True`、`out_proj zero-init check: True`。

**Gate**：如果任何一个 False，停止并把报错贴给我。

---

## 任务 3: 训练 AnchorPredictor（1~3 小时）

**命令**：

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
    --max-train-samples 50000
```

**预期结果**（看 `anchor_predictor_run1/history.json` 最后几行）：
- val **top1 ≥ 0.25**（随机 baseline ≈ 0.78%）
- val **top5 ≥ 0.60**
- train/val gap < 10%

**Gate**：top1 < 0.20 时，尝试加 `--unfreeze-backbone` 或调小 K 重跑 Phase 0。

---

## 任务 4: Planner Finetune（2~4 小时） ★ **今晚最重要任务**

**命令**：

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

**预期结果**（每 epoch 打印）：
- `anchor_out_proj_abs_mean_end` **必须从接近 0 开始明显增长**（这是判断新 cross-attention 模块是否在学习的**最关键指标**）
- `ego_planning_loss` 与 no-goal baseline 接近
- 最终生成 `planner_anchor_best.pth`

**Gate**：如果 `out_proj` 一直接近 0，立即告诉我 debug。

---

## 整体今晚目标 & 明天汇报内容

**最佳完成度**：
- Phase 0 视觉质量过关
- AnchorPredictor top1 ≥ 0.25
- Planner finetune 后 out_proj 有明显增长
- （如果时间允许）跑完三路 eval（none / predicted_anchor / oracle_anchor）

**明天请把以下内容发给我**：
1. `anchor_vocab_meta.json` 关键字段（K, empty_clusters, inertia）
2. `anchor_predictor_run1/history.json` 最后 3 个 epoch
3. `planner_ft_run1/history.json`（重点看 out_proj 增长趋势）
4. （如果跑了）三个 eval json 的 summary 部分

---

**执行建议**：按 1→2→3→4 顺序跑。如果时间不够，先确保完成 **任务 1+2+3**，任务 4 可以明天白天继续。

文件路径已全部更新为 `flow_planner/script/anchor_finetune.yaml`（顶层 config），不再使用旧的 `conf/planner.yaml` 或 model-level yaml 作为主入口。

**参考文档**：`docs/ANCHOR_DEPLOYMENT_AND_VERIFICATION.md`（已同步更新）

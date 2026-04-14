# Goal Predictor Guide

## 目的

这套模块解决的问题是：

`没有 GT 时，goal-conditioned FlowPlanner 应该喂哪个 goal？`

当前实现采用最小可行方案：

- 复用已经训练好的 `goal-conditioned FlowPlanner` backbone
- 冻结 backbone 的 encoder / preprocessing
- 单独训练一个小型 `goal classifier`
- 输出 `goal_vocab` 上的分类分布

目标使用方式：

```text
scene -> top-K goal clusters -> K goal-conditioned trajectories -> rerank
```

## 新增文件

- `flow_planner/goal/goal_predictor.py`
  - 轻量 goal classifier
  - 复用 planner backbone 的 encoder 特征

- `train_goal_predictor.py`
  - 训练脚本

- `eval_goal_predictor.py`
  - top-1 / top-3 / top-5 goal accuracy 评估脚本

## 训练标签

无需新标注。

训练标签直接来自现有 goal conditioning 训练流程里的同一个定义：

```text
GT endpoint -> nearest goal cluster id
```

也就是说，goal predictor 学的是：

```text
scene/context -> which goal cluster should be activated
```

## 输入特征

当前版本使用的 scene-level feature 由以下几部分拼接：

- `routes_cond`
- pooled agent tokens
- pooled lane tokens
- `ego_current`

这些特征全部来自现有 FlowPlanner backbone。

## 推荐训练方式

第一版建议：

- `freeze_backbone = True`
- 只训练 MLP 分类头

理由：

- 风险最小
- 训练快
- 更容易判断“scene feature 本身是否足够预测 goal”

如果第一版 top-K 命中率不够，再考虑部分解冻 route encoder 或 scene encoder。

## 示例命令

```bash
python train_goal_predictor.py \
  --planner-config /root/autodl-tmp/Flow-Planner/outputs/goal_finetune/2026-04-11_21-32-21/.hydra/config.yaml \
  --planner-ckpt /root/autodl-tmp/Flow-Planner/outputs/goal_finetune/2026-04-11_21-32-21/model_epoch_70_trainloss_0.0064.pth \
  --train-data-dir /root/autodl-tmp/nuplan_npz \
  --train-data-list /root/autodl-tmp/nuplan_npz/train_list.json \
  --val-data-dir /root/autodl-tmp/hard_scenarios_v2 \
  --val-data-list /root/autodl-tmp/hard_scenarios_v2/train_list.json \
  --save-dir /root/autodl-tmp/Flow-Planner/outputs/goal_predictor_run1 \
  --device cuda \
  --batch-size 64 \
  --epochs 10
```

评估：

```bash
python eval_goal_predictor.py \
  --planner-config /root/autodl-tmp/Flow-Planner/outputs/goal_finetune/2026-04-11_21-32-21/.hydra/config.yaml \
  --planner-ckpt /root/autodl-tmp/Flow-Planner/outputs/goal_finetune/2026-04-11_21-32-21/model_epoch_70_trainloss_0.0064.pth \
  --predictor-ckpt /root/autodl-tmp/Flow-Planner/outputs/goal_predictor_run1/goal_predictor_best.pth \
  --data-dir /root/autodl-tmp/hard_scenarios_v2 \
  --data-list /root/autodl-tmp/hard_scenarios_v2/train_list.json \
  --device cuda
```

## 下一步

如果 top-K goal accuracy 足够高，下一步就是把 predictor 接入：

1. 提 `top-K` goals
2. 每个 goal 跑一条 goal-conditioned trajectory
3. 用 `TrajectoryScorer` 或 DPO scorer rerank

如果 top-K accuracy 不够，再考虑：

- 部分解冻 backbone
- route-aware auxiliary loss
- top-K proposal + learned reranker

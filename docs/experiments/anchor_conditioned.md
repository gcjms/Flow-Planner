# Anchor Conditioned Experiments

## Experiment: anchor_eval_suite_clean

- Goal: 验证 anchor conditioning 在部署评测中的上限与当前 predictor 瓶颈。
- Setup: 使用 `planner_ft_run_clean/planner_anchor_best.pth`，比较 `planner_ft_none`、`predicted_anchor_top1`、`predicted_anchor_rerank_a`、`oracle_anchor`、`oracle_anchor_rerank`。
- Artifacts:
  - Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_latest`
  - Predictor ckpt: `/root/autodl-tmp/anchor_runs/anchor_predictor_run_clean/anchor_predictor_best.pth`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_run_clean/planner_anchor_best.pth`
- Data:
  - Eval manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_clean.json`
  - Eval scenes: 500
- Results:
  - `planner_ft_none`: collision_rate 6.4
  - `predicted_anchor_top1`: collision_rate 6.2
  - `predicted_anchor_rerank_a`: collision_rate 6.2
  - `oracle_anchor`: collision_rate 2.0
  - `oracle_anchor_rerank`: collision_rate 2.8
- Conclusion:
  - `oracle_anchor` 明确证明 anchor 信息本身有价值。
  - `predicted_anchor` 仍然没有把这种价值传到部署端。
- Decision:
  - 尝试 scheduled sampling，缓解 train/inference mismatch。

## Experiment: anchor_sched_p0p3_20260426

- Goal: 验证 scheduled sampling `rho=0.3` 是否能改善 predicted anchor 的部署表现。
- Setup:
  - Script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_scheduled_sampling.sh`
  - `p_max=0.3`
  - `epochs=10`
  - `batch_size=32`
  - `max_train_samples=80000`（未触发，实际 train 集更小）
- Artifacts:
  - Train output: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p3_20260426_1506`
  - Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p3_20260426_1506`
  - Predictor ckpt: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p3_20260426_1506/planner_anchor_best.pth`
- Data:
  - Train dir: `/root/autodl-tmp/train_dataset`
  - Val dir: `/root/autodl-tmp/val_dataset`
  - Train samples: 40079
  - Val samples: 17224
  - Effective train samples/epoch: 40064 (`batch_size=32`, `drop_last=True`)
  - Eval manifest: `/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - Eval scenes: 500
- Results:
  - `planner_ft_none`: collision_rate 6.6
  - `predicted_anchor_top1`: collision_rate 3.6
  - `predicted_anchor_rerank_a`: collision_rate 5.0
  - `oracle_anchor`: collision_rate 1.8
  - `oracle_anchor_rerank`: collision_rate 3.6
- Conclusion:
  - `rho=0.3` 对 `predicted_anchor_top1` 有明显帮助，collision_rate 从上一轮约 6.2 降到 3.6。
  - 但 predicted anchor 与 oracle anchor 之间仍有明显差距，predictor 质量仍是主瓶颈。
  - rerank 方案当前不稳定，不应作为主结论。
- Decision:
  - 继续尝试 `rho=0.5`。

## Experiment: anchor_sched_p0p5_20260426

- Goal: 在 `rho=0.3` 已出现正向信号后，继续测试更强的 scheduled sampling 是否进一步改善 predicted anchor 部署表现。
- Setup:
  - Script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_scheduled_sampling.sh`
  - `p_max=0.5`
- Artifacts:
  - Launch log: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1608.launch.log`
  - Train output: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1608`
  - Planned eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_20260426_1608`
- Status:
  - 已启动，待训练与评测完成后补结果。

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
  - `epochs=10`
  - `batch_size=32`
  - `max_train_samples=80000`（未触发，实际 train 集更小）
- Artifacts:
  - Launch log: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612.launch.log`
  - Train output: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612`
  - Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_20260426_1612`
  - Predictor ckpt: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
- Data:
  - Train dir: `/root/autodl-tmp/train_dataset`
  - Val dir: `/root/autodl-tmp/val_dataset`
  - Train samples: 40079
  - Val samples: 17224
  - Effective train samples/epoch: 40064 (`batch_size=32`, `drop_last=True`)
  - Eval manifest: `/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - Eval scenes: 500
- Results:
  - `planner_ft_none`: collision_rate 7.4
  - `predicted_anchor_top1`: collision_rate 3.2
  - `predicted_anchor_rerank_a`: collision_rate 4.6
  - `oracle_anchor`: collision_rate 2.2
  - `oracle_anchor_rerank`: collision_rate 3.4
- Conclusion:
  - 相比 `rho=0.3`，`rho=0.5` 继续改善了 predicted anchor 部署表现，但提升幅度有限。
  - `predicted_anchor_top1` 从 3.6 进一步降到 3.2，说明提高 scheduled sampling 比例仍有帮助。
  - 但 `oracle_anchor` 与 `predicted_anchor` 之间仍有明显 gap，说明 predictor 质量仍是主要瓶颈。
  - 收益已经开始趋于饱和，继续单纯提高 rho 未必是下一步最优方向。
- Decision:
  - 保留 `rho=0.3` 和 `rho=0.5` 作为正结果。
  - 后续优先考虑提升 predictor / anchor selection 质量，而不是继续单独上调 rho。

## Next-stage plan: anchor to DPO readiness

- Goal: 判断 anchor 是否已经适合作为后续 DPO 的轨迹 mode / candidate organization 单元，并避免重复 goal-conditioned DPO 中 cross-goal pair 失败的问题。
- Why 500 scenes were used:
  - 当前部署评测脚本默认 MAX_SCENES=500，默认复用 /root/autodl-tmp/anchor_runs/eval_manifest.json。
  - eval_manifest.json 目前指向 eval_manifest_clean.json，其中包含 500 scenes。
  - 这批 500 scenes 已用于 clean suite、rho=0.3、rho=0.5 的 planner_ft_none、predicted_anchor_top1、predicted_anchor_rerank_a、oracle_anchor、oracle_anchor_rerank 对比。
  - 500 scenes 适合快速判断方向和发现大问题，但不适合作为最终论文主表的唯一依据。
- Current evidence:
  - oracle_anchor 明显优于 no-anchor / predicted-anchor，说明 anchor 表示本身有价值。
  - scheduled sampling 已经把 predicted_anchor_top1 collision_rate 从约 6.2 降到 3.6 / 3.2，说明 planner 可以学习部署时的 predicted-anchor 噪声。
  - rho=0.5 相比 rho=0.3 继续提升有限，说明继续单独扫更高 rho 的信息量下降。
  - 当前 predictor formal run 的验证集指标为 top1=0.403、top3=0.775、top5=0.902，说明 top1 selection 仍弱，但正确 mode 经常已经在 top-k candidate pool 中。
- Key constraint:
  - 旧 goal-DPO 记录显示，任意 cross-goal pair + goal-aware DPO 路线不稳定。
  - anchor-DPO 不能直接复刻任意 cross-anchor pair；否则 preference 可能混入条件变量差异，而不是只表达轨迹质量差异。
- Working hypothesis:
  - anchor 的下一步价值不应只看 top1 是否完美，而应看 top-k anchor mode pool 是否能稳定产生高质量候选。
  - DPO pair construction 应优先保证 condition-clean：同一 scene、同一 anchor，或语义非常接近的 near-anchor。
  - 任意 far cross-anchor pair 暂不作为主路线。
- Default planner setting:
  - 目前优先使用 rho=0.5 的 planner checkpoint 作为 candidate generator。
  - 理由：它是当前 predicted-anchor 部署指标最好的 scheduled-sampling planner，但收益已趋于饱和，不建议继续优先扫 rho=0.7/1.0。
- Immediate experiments:
  - E1 predictor diagnosis: 统计 top1/top3/top5 accuracy、confusion matrix、oracle anchor rank distribution、错误是否集中在相近 anchors。
  - E2 top-k candidate quality: 对每个 scene 用 predicted top-3/top-5 anchors 生成候选，评估候选池中是否存在低 collision / 高 route / 高 progress 轨迹。
  - E3 condition-clean preference mining: 先构造 same-anchor pairs；如果需要 near-anchor pairs，必须先定义 anchor 距离阈值和 mode similarity 规则。
  - E4 rerank sanity check: 如果 top-k pool 中经常有好候选但 top1 选错，优先做 reranker / selector；如果 top-k pool 本身质量不足，先改 predictor。
- Pair construction policy:
  - Strong preference only: collision-free 优先于 collision，route/progress 明显更好优先于更差，模糊 pair 丢弃。
  - Same-anchor first: 同一 scene、同一 anchor 条件下比较多条轨迹候选。
  - Near-anchor optional: 只在 anchors 轨迹形态接近且共享局部驾驶意图时比较。
  - Far cross-anchor avoided: 不把明显不同驾驶 mode 的 pair 直接喂给 DPO。
- Manifest policy:
  - 500-scene manifest 继续作为 quick dev / smoke eval。
  - 下一阶段 readiness 诊断建议建立更大的固定 manifest，例如 2k scenes；如果运行时间可接受，再扩到 5k scenes。
  - 论文主表不应只依赖 500 scenes，至少需要一个更大的固定 manifest 复核核心结论。
- Decision criteria:
  - 如果 top-k pool coverage 高，且 same-anchor / near-anchor 能挖出足够强偏好 pair，则进入 anchor-DPO data construction。
  - 如果 top-k coverage 高但 top1/rerank 差，则先做 anchor selector / reranker。
  - 如果 top-k coverage 也差，则先重训或改造 anchor predictor，不急着做 DPO。
- Expected output:
  - dpo_data/anchor_conditioned/candidates/ 保存 anchor candidate pool。
  - dpo_data/anchor_conditioned/preferences/ 保存 condition-clean preference pairs。
  - docs/experiments/anchor_conditioned.md 持续记录诊断结果、pair mining 规则、DPO 是否启动的判定。

## Update: larger eval manifests

- Created /root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json with 2000 scenes from /root/autodl-tmp/val_dataset.
- Created /root/autodl-tmp/anchor_runs/eval_manifest_5k_seed3402.json with 5000 scenes from /root/autodl-tmp/val_dataset.
- Kept /root/autodl-tmp/anchor_runs/eval_manifest.json unchanged as the 500-scene quick-dev baseline used by the previous clean/rho=0.3/rho=0.5 comparisons.
- Fixed /root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_eval_common.sh so the default SCENE_DIR is /root/autodl-tmp/val_dataset instead of the missing /root/autodl-tmp/nuplan_npz.
- Next readiness eval should set MANIFEST_PATH to the 2k manifest first; the 5k manifest is reserved for stronger confirmation or paper-facing tables.

## Experiment: anchor_sched_p0p5_eval_2k_20260426

- Goal: 用更大的固定 manifest 复核 rho=0.5 planner 在 predicted/oracle anchor 部署评测上的核心趋势，降低 500-scene 快速评测的方差风险。
- Setup:
  - Script: /root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_eval_suite.sh
  - Planner ckpt: /root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth
  - Scene dir: /root/autodl-tmp/val_dataset
  - Manifest: /root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
  - Eval scenes: 2000
  - Cases: planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor, oracle_anchor_rerank
- Artifacts:
  - Eval output: /root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_2k_20260426_1743
- Status:
  - Started on 2026-04-26 17:43 CST.
  - Results pending.

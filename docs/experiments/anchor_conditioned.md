# Anchor Conditioned Experiments

## 0. 20260426 下午 anchor preference 实验总览（可读版）

这一节是给人直接看的，不是原始日志。后面的 `Experiment:` 小节保留完整路径、参数和细节。

### 0.1 今天下午到底在验证什么

我们的最终目标不是做一个手写 reranker，而是把未来轨迹先离散成 anchor / trajectory modes，再在这些离散候选上学习 preference。换句话说，planner 负责生成可行轨迹，preference 模块负责学会更偏向安全、合理、符合路线的候选。

今天下午的实验按问题拆成四层：

1. `rho=0.5` 的 anchor planner 到底稳不稳。
2. same-anchor hard DPO 能不能直接训练 planner。
3. safe-vs-safe 不适合硬分好坏时，soft preference 能不能解决。
4. 如果 planner-level DPO 太弱，是否应该改成离散的 learned anchor selector。

### 0.2 第一层：anchor 本身有没有价值

2k val 上，`rho=0.5` 的关键结果是：

| 方法 | collision | progress | route | 解释 |
|---|---:|---:|---:|---|
| no anchor | 5.45% | 0.3393 | 0.8592 | 不使用 anchor 条件 |
| predicted_anchor_top1 | 4.20% | 0.3253 | 0.8548 | predictor 只选 top1 anchor |
| predicted_anchor_rerank_a | 3.15% | 0.3293 | 0.8738 | top3 anchor 生成后用手写规则挑一个，诊断用 |
| oracle_anchor | 2.20% | 0.3149 | 0.8580 | 用 GT 最近 anchor，上限参考 |
| oracle_anchor_rerank | 2.80% | 0.3309 | 0.8748 | oracle top-k + 手写 rerank |

结论：anchor 空间本身有价值，因为 oracle anchor 明显降低 collision。当前瓶颈不是 “anchor 没用”，而是 predictor / selector 还没把 oracle 的价值完全传出来。

### 0.3 第二层：为什么 same-anchor DPO 没直接成功

这里的 DPO pair 都尽量限制在同一个 scene、同一个 anchor 下面，避免跨 goal / 跨 anchor 乱比较。

`mixed v2 DPO` 表示两类 pair 混合：

- `same_anchor_collision`：chosen 是 safe，rejected 是 collided。这个标签最干净。
- `same_anchor_quality`：两条都 safe，但一条 structured score 更高。这个标签更软，因为 rejected 也可能是可接受轨迹。

`collision-only DPO` 表示只保留 safe-vs-collided pair，去掉 safe-vs-safe quality pair。它的目的不是最终方案，而是排查：如果 collision-only 也学不动，就说明问题不只是 safe-vs-safe 标签噪声。

结果：

| 实验 | 结果 | 判断 |
|---|---:|---|
| mixed v2 DPO | epoch2 pair acc 44.85% | 弱，甚至低于随机附近 |
| collision-only DPO | epoch2 pair acc 51.61% | 接近随机，信号仍弱 |

结论：当前 planner-level hard DPO 不能作为正结果。即使用最干净的 safe-vs-collided pair，模型也没有稳定学会偏好 chosen。

### 0.4 第三层：为什么 continuous flow-matching log-prob 是瓶颈

DPO 需要比较 `chosen` 和 `rejected` 哪个更像模型会生成的轨迹，也就是需要类似 `log pi_theta(trajectory | scene, anchor)` 的分数。

但 Flow Planner 不是离散分类器，它是连续轨迹生成模型，训练方式接近 flow matching / diffusion 去噪。我们只能用 flow-matching loss 近似某条 candidate trajectory 的 likelihood。这个近似在候选很相似、同属一个 anchor 时非常不灵敏：好轨迹和坏轨迹的分数差经常接近 0。

实验表现就是：

- chosen/rejected DPO margin 很小。
- pair acc 接近随机。
- soft preference 训练后，模型给 safe candidates 的总概率几乎没变。
- top1 candidate 也没有稳定转向 teacher 认为最好的轨迹。

结论：soft preference 数据生成方向是对的，但把它直接压到 planner 的连续 log-prob 上，目前不是可靠路径。

### 0.5 第四层：为什么转向 learned anchor selector


#### 0.5.1 当前 v1：anchor-level soft selector

当前已经实现和评测的是 `anchor-level soft selector`，不是最终 DPO。模型仍然是 `AnchorPredictor`：输入 scene，输出整个 anchor vocab 上的 logits / scores。

形式上是：

```text
scene -> selector -> score(anchor_1), score(anchor_2), ...
```

数据生成过程：

1. 对每个 scene，用当前 predictor 提出 top3 anchors。
2. 每个 anchor 让 planner 采样 3 条轨迹，所以每个 scene 有 9 条 candidate trajectories。
3. 用 safety / route / progress 等 structured metrics 给 9 条 candidate 打分。
4. 把同一个 anchor 下的 candidate 分数聚合成一个 anchor score，例如 mean / max。
5. 把 anchor scores 转成 soft target：

```text
q(anchor | scene) = softmax(anchor_score / temperature)
```

训练目标是 soft cross entropy，不是 DPO：

```text
loss = - sum_anchor q(anchor | scene) * log p_selector(anchor | scene)
```

含义：这版 selector 不再只模仿 GT 最近 anchor，而是尝试偏向那些能让 planner 生成更安全、更合理轨迹的 anchor。

#### 0.5.2 下一步：selector-DPO，不再用 planner continuous log-prob

下一步真正的 selector-DPO 不应该再使用 planner 的连续轨迹 likelihood：

```text
log pi_planner(trajectory | scene, anchor)
```

而应该把 DPO 放在 selector 自己的离散 score 上：

```text
s_theta(scene, anchor)
```

构造 pair 的方式是：

```text
chosen anchor = 生成候选中更安全 / structured score 更高的 anchor
rejected anchor = 更危险 / structured score 更低的 anchor
```

DPO loss 写在 selector score 差上：

```text
loss_dpo = -log sigmoid(
  beta * (
    (s_theta(scene, chosen) - s_theta(scene, rejected))
    - (s_ref(scene, chosen) - s_ref(scene, rejected))
  )
)
```

这里 `s_ref` 是原始 predictor / frozen selector 的 score，用于限制模型不要偏离太远。这样学到的是：在同一个 scene 下，selector 应该给 chosen anchor 更高分，给 rejected anchor 更低分。

这个设计的关键点是：DPO 比较的是离散 anchor/candidate score，而不是连续 planner trajectory log-prob。这样可以避开当前 flow-matching log-prob 对候选排序不敏感的问题。

#### 0.5.3 再下一层：candidate-level selector

`anchor-level selector` 只看：

```text
scene -> anchor score
```

`candidate-level selector` 会进一步看：

```text
scene + anchor + generated candidate trajectory -> preference score
```

这更接近最终目标：同一个 scene 下有多条候选轨迹，模型自己判断哪条更好。手写 `predicted_anchor_rerank_a` 只作为 teacher / diagnostic baseline / ablation，不作为最终方法或创新点。最终部署希望用 learned selector / learned preference scorer 给候选打分，而不是靠手写规则硬选。

因为 planner-level likelihood 太弱，我转向离散 selector：让模型先学 `这个 scene 应该选哪个 anchor / candidate mode`。

这不是手写 reranker。手写 `predicted_anchor_rerank_a` 只作为诊断工具，证明 top-k anchor pool 里确实有更好的候选。真正可能作为论文方法的是 learned selector / preference policy。

当前 selector 做法：

1. 每个 scene 用 predictor 取 top3 anchors。
2. 每个 anchor 采样 3 条轨迹，共 9 条 candidate。
3. 用 structured score 给 candidate 打分。
4. 按 anchor 聚合成 soft target。
5. 训练 AnchorPredictor head，让它输出更偏向高分 anchor 的概率。

train500 selector 的 2k val 结果：

| 方法 | collision | progress | route | 判断 |
|---|---:|---:|---:|---|
| original predicted_anchor_top1 | 4.20% | 0.3253 | 0.8548 | 原 predictor top1 |
| selector predicted_anchor_top1 | 3.70% | 0.3185 | 0.8574 | collision 有改善，但 progress 降低 |
| original predicted_anchor_rerank_a | 3.15% | 0.3293 | 0.8738 | 当前最稳部署 baseline |
| selector predicted_anchor_rerank_a | 3.35% | 0.3248 | 0.8768 | route 高，但 collision 没赢 |

结论：learned selector 已经出现正信号，尤其是 top1 collision 从 4.20% 降到 3.70%。但它还没有超过当前最稳的 original `predicted_anchor_rerank_a`。

### 0.6 当前最清楚的判断

- anchor 方向是可行的，oracle 和 top-k 结果都支持这一点。
- 直接 planner-level DPO 现在不行，主要卡在 continuous flow-matching log-prob 对候选排序不敏感。
- safe-vs-safe 硬 pair 的担心是合理的，所以后面不应把它当成同等强度的 hard rejected。
- 手写 rerank 不作为创新点，只作为诊断 baseline / teacher signal。
- 真正应该推进的创新线是 learned anchor/candidate selector：在离散 anchor / candidate 层面做 preference learning。
- 当前部署最好结果仍是 `rho=0.5 + original predicted_anchor_rerank_a`。
- 当前最有研究价值的 learned signal 是 selector top1 的 collision 改善。

### 0.7 记录规范更新

后续每个 anchor preference 实验都按这个顺序写：

1. 目的：这个实验要回答什么问题。
2. 数据：用了多少 train / val scenes，manifest 是哪一个。
3. 方法：candidate 怎么生成，pair / soft target 怎么定义。
4. 路径：输出目录、log、checkpoint。
5. 结果：关键指标表。
6. 解释：这个结果说明什么，不说明什么。
7. 下一步：继续、停止、还是改方向。


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
  - Completed on 2026-04-26 18:18 CST.
  - Results summarized below.

## Anchor-DPO implementation note

- Existing train_dpo.py supports generic chosen/rejected pairs and optional goal fields, but it does not yet consume anchor-specific fields.
- For real anchor-DPO, preference data must carry the anchor condition used for each pair, e.g. anchor_trajs or chosen_anchor_trajs/rejected_anchor_trajs.
- Same-anchor DPO should use the same anchor_traj for chosen and rejected, then pass that anchor_traj into decoder inputs during DPO loss computation.
- Without this change, training with anchor-generated pairs would become ordinary DPO under scene context, not anchor-conditioned DPO.
- Immediate coding task: add anchor fields to PreferenceDataset/collate_preferences and add attach_anchor_to_decoder_inputs mirroring the existing goal path.

## Results: anchor_sched_p0p5_eval_2k_20260426

- Status: completed on 2026-04-26 18:18 CST.
- Eval output: /root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_2k_20260426_1743
- Manifest: /root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
- Eval scenes: 2000
- Results:
  - planner_ft_none: collision_rate 5.45, avg_progress 0.3393, avg_route 0.8592
  - predicted_anchor_top1: collision_rate 4.20, avg_progress 0.3253, avg_route 0.8548
  - predicted_anchor_rerank_a: collision_rate 3.15, avg_progress 0.3293, avg_route 0.8738
  - oracle_anchor: collision_rate 2.20, avg_progress 0.3149, avg_route 0.8580
  - oracle_anchor_rerank: collision_rate 2.80, avg_progress 0.3309, avg_route 0.8748
- Conclusion:
  - 2k eval confirms predicted anchors help over no-anchor, and the top-k reranker is useful on a larger fixed validation subset.
  - The 500-scene result had noticeable variance, especially for planner_ft_none and rerank.
  - predicted_anchor_rerank_a is now the best predicted-anchor deployment setting, but it still trails oracle_anchor.
  - oracle_anchor remains the upper-bound signal; predictor/selection quality is still the bottleneck.
- Decision:
  - Use rho=0.5 planner as the current candidate generator.
  - Continue anchor selection/rerank work in parallel with anchor-DPO readiness.

## Experiment: anchor_dpo_readiness_smoke_20260426

- Goal: 验证 anchor-DPO 的最小数据/训练链路是否可行，且 preference pair 保持 condition-clean。
- Code review finding:
  - Existing train_dpo.py supported ordinary DPO and optional goal fields, but did not consume anchor-specific fields.
  - Runtime train_dpo.py was patched to accept anchor_vocab_path, load anchor-enabled planner config, read anchor_trajs / chosen_anchor_trajs / rejected_anchor_trajs, and attach anchor_traj to decoder inputs.
  - Added runtime script flow_planner/dpo/generate_anchor_same_anchor_pairs.py for same-scene + same-anchor pair mining.
  - Runtime patches are preserved at /root/autodl-tmp/anchor_runs/patches/anchor_dpo_train_dpo_runtime.patch and /root/autodl-tmp/anchor_runs/patches/anchor_same_anchor_pair_generator.patch.
  - These runtime code changes still need formal migration to the anchor branch before a large run is treated as canonical.
- Pair mining smoke setup:
  - Script: /root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/generate_anchor_same_anchor_pairs.py
  - Planner ckpt: /root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth
  - Predictor ckpt: /root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth
  - Manifest: /root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
  - max_scenes: 20, top_k: 3, samples_per_anchor: 3
- Pair mining smoke results:
  - Output: /root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_smoke_20260426_1815.npz
  - Pairs: 28
  - Failures: 0
  - Pair labels: same_anchor_collision 5, same_anchor_quality 23
  - Shapes: chosen/rejected (28, 80, 4), anchor_trajs (28, 80, 3)
- DPO train smoke setup:
  - Preference path: /root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_smoke_20260426_1815.npz
  - Output: /root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_smoke_20260426_1817
  - max_pairs: 8, epochs: 1, batch_size: 2, num_t_samples: 1, lora_rank: 2
- DPO train smoke result:
  - Training completed successfully.
  - Best accuracy: 87.50%
  - This is only a pipeline sanity check, not a model-quality result.
- Conclusion:
  - same-anchor preference mining is feasible: even 20 scenes produced nonzero clean pairs.
  - anchor-conditioned DPO training path is now technically viable in runtime.
  - Next step should be pair-yield scaling and quality audit before any full anchor-DPO training.
- Decision:
  - Scale pair mining to a larger fixed subset next, starting from 500 scenes or 2k scenes depending on runtime budget.
  - Do not treat smoke LoRA output as a deployable checkpoint.

## Experiment: anchor_dpo_train500_gap0p15_pilot_20260426

- Goal: 验证 same-anchor preference pair 能否真正接到 anchor-conditioned DPO 训练，并快速判断小规模 DPO 是否可能改善 anchor 部署评测。
- Setup:
  - Pair generator: `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/generate_anchor_same_anchor_pairs.py`
  - Base planner: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Predictor: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Pair source split: train, `/root/autodl-tmp/train_dataset`
  - Train manifest: `/root/autodl-tmp/anchor_runs/generated_lists/train_list.json`
  - Pair mining subset: 500 train scenes
  - Pair mining config: `top_k=3`, `samples_per_anchor=3`, same scene + same predicted anchor only
- Pair mining artifacts:
  - Raw preference file: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_20260426_1830.npz`
  - Filtered preference file: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_gap0p15_20260426_1830.npz`
- Pair mining results:
  - Raw pairs: 924 pairs from 437 / 500 scenes, 0 failures
  - Raw labels: `same_anchor_quality` 800, `same_anchor_collision` 124
  - Anchor ranks: rank0 290, rank1 308, rank2 326
  - Filter used for pilot: keep all collision pairs plus quality pairs with `score_gap >= 0.15`
  - Filtered pairs: 321 pairs, including `same_anchor_quality` 197 and `same_anchor_collision` 124
- DPO train setup:
  - Script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_dpo.py`
  - Preference path: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_gap0p15_20260426_1830.npz`
  - Output dir: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840`
  - Epochs: 2
  - Batch size: 8
  - `num_t_samples=4`, `beta=0.1`, `sft_weight=0.05`, `lr=1e-5`, `lora_rank=4`, `lora_alpha=16`
- DPO train result:
  - Epoch 1: loss 0.7464, accuracy 46.56%, delta -0.0076
  - Epoch 2: loss 0.7460, accuracy 51.88%, delta -0.0010
  - Best train accuracy: 51.88%
  - Saved merged model: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged.pth`
  - Saved clean merged model with LoRA side keys stripped: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged_stripped.pth`
- Code review notes during pilot:
  - Original merged checkpoint retained 150 LoRA side keys, causing eval `unexpected keys` warnings. A stripped checkpoint was created for follow-up eval; this is a checkpoint-format issue, not evidence that DPO weights were ignored.
  - Weight diff check against rho=0.5 base planner found 72 changed floating-point tensors; the largest relative change was in `model_decoder.anchor_cross_attn.out_proj.weight`.
  - Runtime `train_dpo.py` currently exposes `--min_score_gap`, but the dataset loader does not actually apply it. This did not affect the pilot because filtering was done into a separate `.npz`, but it must be fixed before relying on CLI filtering.
  - Runtime `dpo_loss.py` currently samples independent flow-matching noise/timesteps for policy and reference log-probs. This is valid as a noisy estimator but unnecessarily high variance; next DPO training should share sampled noise/t between policy/reference for the same trajectory.
  - Runtime code changes are still in the anchor runtime snapshot and patch artifacts, not yet formally migrated into the `anchor` branch.
- 500-scene quick eval setup:
  - Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_500_20260426_1848`
  - Scene dir: `/root/autodl-tmp/val_dataset`
  - Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - Eval scenes: 500
  - Cases: planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor
- 500-scene quick eval results:
  - planner_ft_none: collision_rate 5.60, avg_progress 0.3512, avg_route 0.8536
  - predicted_anchor_top1: collision_rate 3.40, avg_progress 0.3350, avg_route 0.8516
  - predicted_anchor_rerank_a: collision_rate 3.20, avg_progress 0.3400, avg_route 0.8653
  - oracle_anchor: collision_rate 2.00, avg_progress 0.3256, avg_route 0.8472
- 500-scene comparison against rho=0.5 non-DPO planner on same manifest:
  - planner_ft_none improved from 7.40 to 5.60 collision_rate.
  - predicted_anchor_top1 regressed slightly from 3.20 to 3.40 collision_rate.
  - predicted_anchor_rerank_a improved from 4.60 to 3.20 collision_rate.
  - oracle_anchor improved from 2.20 to 2.00 collision_rate.
- Interim conclusion:
  - The pilot is technically successful and shows a small positive signal for rerank/oracle, but training accuracy is near random and top1 did not improve.
  - This is not enough to justify a large DPO run yet.
  - It is enough to justify a 2k fixed-manifest replication before deciding whether to scale pair mining/training.
- 2k replication status:
  - Started on 2026-04-26 18:50 CST.
  - Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_2k_20260426_1852`
  - Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Checkpoint: `model_dpo_merged_stripped.pth`
  - Cases: planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor
  - Status at record time: running.

## Results: anchor_dpo_train500_gap0p15_eval_2k_20260426

- Status: completed on 2026-04-26 19:11 CST.
- Eval output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_2k_20260426_1852`
- Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
- Checkpoint: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged_stripped.pth`
- Eval scenes: 2000
- Results:
  - planner_ft_none: collision_rate 5.15, avg_progress 0.3419, avg_route 0.8609
  - predicted_anchor_top1: collision_rate 4.10, avg_progress 0.3238, avg_route 0.8545
  - predicted_anchor_rerank_a: collision_rate 3.10, avg_progress 0.3282, avg_route 0.8738
  - oracle_anchor: collision_rate 2.20, avg_progress 0.3138, avg_route 0.8559
- Comparison against rho=0.5 non-DPO 2k eval:
  - planner_ft_none: 5.45 -> 5.15 collision_rate, small improvement.
  - predicted_anchor_top1: 4.20 -> 4.10 collision_rate, negligible improvement.
  - predicted_anchor_rerank_a: 3.15 -> 3.10 collision_rate, essentially tied.
  - oracle_anchor: 2.20 -> 2.20 collision_rate, tied.
- Conclusion:
  - The anchor-DPO path is technically working and did not break anchor-conditioned inference.
  - On 2k fixed validation scenes, this 321-pair pilot only gives marginal changes; the apparent 500-scene rerank gain was mostly variance.
  - This pilot is not strong enough to justify scaling DPO training as-is.
  - Current best deployment candidate remains rho=0.5 + predicted_anchor_rerank_a.
  - The oracle gap remains: predicted_anchor_rerank_a 3.10 vs oracle_anchor 2.20 collision_rate.
- Decision:
  - Do not launch a large anchor-DPO training run from the current 321-pair setup.
  - Before the next DPO run, fix the DPO implementation issues found during review: apply `min_score_gap` filtering correctly, strip LoRA side keys when saving merged checkpoints, and reduce DPO loss variance by sharing sampled flow-matching noise/timesteps between policy and reference for each trajectory.
  - After the code fix, mine a larger train-split same-anchor preference set, likely 2k train scenes first, then train/eval a second pilot.
  - Continue treating rerank/selector as a high-priority path, because top-k predicted anchors are already more useful than top1 alone.

## Code review/fix: anchor DPO runtime v2 20260426

- Goal: 修掉下一轮 anchor-DPO 前会影响结论可信度的实现问题。
- Runtime files changed:
  - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_dpo.py`
  - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/dpo_loss.py`
- Fixes:
  - `PreferenceDataset` now actually applies `min_score_gap` filtering and keeps chosen/rejected, scenario ids, labels, goals, and anchor trajectories aligned.
  - CLI default `--min_score_gap` changed from 2.0 to 0.0 so existing files are not silently over-filtered unless explicitly requested.
  - Merged DPO checkpoint saving now strips `.lora_A` / `.lora_B` side keys after merge, avoiding eval-time `unexpected keys` warnings.
  - DPO loss now evaluates policy and reference with the same sampled flow-matching noise/timestep for each trajectory, reducing DPO delta variance.
- Validation:
  - `python -m py_compile` passed for runtime `train_dpo.py` and `dpo_loss.py`.
  - Dataset smoke: raw train500 preference file with `min_score_gap=0.15` keeps 321 / 924 pairs.
  - Anchor field smoke: dataset item contains `chosen_anchor_traj` with shape `(80, 3)`.
  - DPO loss smoke: dummy policy/reference forward returns finite DPO/SFT losses.
- Patch artifacts:
  - `/root/autodl-tmp/anchor_runs/patches/anchor_dpo_train_dpo_runtime_v2.patch`
  - `/root/autodl-tmp/anchor_runs/patches/anchor_dpo_loss_runtime_v2.patch`
- Decision:
  - Next anchor-DPO run should use this v2 runtime behavior or a formal migration of these changes into the `anchor` branch.
  - Do not compare future DPO runs against the v1 pilot without noting that v1 had higher-variance loss estimation and dirty merged checkpoint format.

## Experiment: anchor_dpo_pair_mining_train2k_v2_20260426

- Goal: 在修复 DPO runtime v2 后，先扩大 same-anchor preference mining 到 2k train scenes，判断是否有足够干净 pair 支撑第二轮 DPO pilot。
- Setup:
  - Runtime: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Script: `flow_planner/dpo/generate_anchor_same_anchor_pairs.py`
  - Base planner: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Predictor: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Scene dir: `/root/autodl-tmp/train_dataset`
  - Scene manifest: `/root/autodl-tmp/anchor_runs/generated_lists/train_list.json`
  - max_scenes: 2000
  - top_k: 3
  - samples_per_anchor: 3
  - min_quality_gap: 0.05
- Artifacts:
  - Output preference path: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_v2_20260426_1921.npz`
  - Log: `/root/autodl-tmp/anchor_runs/same_anchor_train2k_v2_20260426_1921.log`
  - PID at launch: 22951
- Status:
  - Started on 2026-04-26 19:21 CST.
  - Running at record time.
- Decision rule:
  - If pair yield and score-gap distribution look healthy, create an explicit filtered `.npz` and run a v2 DPO pilot.
  - If yield is weak or mostly ambiguous quality pairs, improve pair selection before training.

## Experiment: anchor_dpo_train2k_gap0p15_v2_e2_20260426

- Goal: 在 runtime v2 修复后，用更大的 train2k same-anchor preference set 跑第二版 DPO pilot，验证更稳定的 DPO loss 和更多 pair 是否能产生更可信的提升。
- Pair source:
  - Raw preference file: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_v2_20260426_1921.npz`
  - Raw pairs: 3777 from 2000 train scenes, 0 failures
  - Labels: `same_anchor_collision` 567, `same_anchor_quality` 3210
  - Anchor ranks: rank0 1192, rank1 1263, rank2 1322
  - Score gap: min 0.0500, median 0.1143, mean 15.1240, p75 0.2126, p90 99.9807, max 100.5446
- Filter for training:
  - Use `--min_score_gap 0.15` in runtime v2 `PreferenceDataset`.
  - Effective pairs: 1367, including collision 567 and quality 800.
- Train setup:
  - Runtime: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Script: `python -m flow_planner.dpo.train_dpo`
  - Base planner: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Anchor vocab: `/root/autodl-tmp/anchor_runs/anchor_vocab.npy`
  - Scene dir: `/root/autodl-tmp/train_dataset`
  - Output dir: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958.log`
  - epochs: 2, batch_size: 8, lr: 1e-5, beta: 0.1, sft_weight: 0.05, num_t_samples: 4, lora_rank: 4, lora_alpha: 16
- Status:
  - Started on 2026-04-26 19:58 CST.
  - PID at launch: 24227.
  - Running at record time.
- Decision rule:
  - If train accuracy/delta becomes meaningfully better than v1 and 500 quick eval improves predicted_anchor_rerank without hurting oracle, then run 2k eval.
  - If still near-random or hurts quick eval, stop DPO scaling and focus on selector/reranker/predictor quality.

## Correction: anchor_dpo_train2k_gap0p15_v2 launch 20260426

- The first launch record for `anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958` is invalid because the non-interactive shell did not activate conda; log showed `nohup: failed to run command python: No such file or directory`.
- Corrected launch:
  - Output dir: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001.log`
  - PID at corrected launch: 24390
  - Status: running; reached Epoch 1/2 with 170 steps per epoch.
- Interpretation:
  - Treat the `1958` record as a failed launch attempt only.
  - Treat the `2001` run as the valid v2 DPO pilot.

## Update: anchor-DPO pair semantics concern 20260426

- User concern: safe-vs-safe pairs may be noisy because both trajectories can be acceptable. Hard DPO on these pairs may incorrectly push a usable safe trajectory as `rejected`.
- Current generator behavior confirmed:
  - If same-anchor candidates contain safe and collided trajectories: choose best safe vs worst collided, label `same_anchor_collision`.
  - If all candidates are safe: choose max-quality vs min-quality when `quality_gap >= min_quality_gap`, label `same_anchor_quality`.
  - Quality uses `safe_bonus = 100 * (1 - collided)`, so collision pairs have score gaps around 100 and naturally pass `min_score_gap=0.15`; the 0.15 threshold mainly filters safe-vs-safe quality pairs.
- Train2k v2 score-gap distribution:
  - `same_anchor_collision`: 567 pairs, gap min 99.5767, median 100.0292, max 100.5446.
  - `same_anchor_quality`: 3210 pairs, gap min 0.0500, median 0.1005, p75 0.1498, p90 0.2245, max 0.8520.
  - With `min_score_gap=0.15`, kept 1367 pairs = 567 collision + 800 quality.
- Valid v2 mixed-pair training result:
  - Run: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001.log`
  - Epoch 1: loss 0.7459, acc 47.79%, delta -0.0003.
  - Epoch 2: loss 0.7458, acc 44.85%, delta -0.0008.
  - Per-label epoch 2: collision acc 44.86%, quality acc 44.85%.
  - Interpretation: weak/near-random; do not treat this as a positive anchor-DPO result.
- Collision-only ablation:
  - Filtered preference file: `/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_collision_only_v2_20260426_2010.npz`
  - Filter: `dim_labels == same_anchor_collision`, 567 pairs.
  - Invalid launch: `anchor_dpo_train2k_collision_only_v2_e2_20260426_2010` imported non-anchor model code because `PYTHONPATH=/root/autodl-tmp/Flow-Planner-anchor-runtime` was missing. It has `INVALID.txt`; do not use it.
  - Valid run: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_collision_only_v2_e2_20260426_2015`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_collision_only_v2_e2_20260426_2015.log`
  - Correct runtime check: model loaded 15.187M params; LoRA includes anchor encoder/cross-attn modules; 75 LoRA layers.
  - Epoch 1: loss 0.7458, acc 50.00%, delta -0.0002.
  - Epoch 2: loss 0.7458, acc 51.61%, delta 0.0001.
  - Interpretation: cleaner labels alone did not produce a strong DPO signal; current DPO objective/log-prob estimate is barely distinguishing chosen vs rejected under same anchor.
- Decision:
  - Safe-vs-safe hard pairs should not be used as equal-strength formal DPO labels yet.
  - Treat current anchor-DPO as diagnostic, not deployment-ready.
  - Next priority is to inspect why DPO log-prob deltas are almost zero under same-anchor collision pairs before scaling more data.
  - Candidate fixes: stronger/less noisy DPO signal, more samples per anchor, larger collision-pair set, gap/confidence-weighted or soft preference for safe-vs-safe pairs, and explicit eval only after train deltas become meaningful.

## Design note: soft preference distillation from goal branch 20260426

- Goal branch reference commit: `6356744 Wrap up goal-line: soft preference distill + DriveDPO-style hard negatives + anchor notes in GOAL_DESIGN`.
- Relevant files in `origin/feature/goal`:
  - `flow_planner/dpo/SOFT_PREF_DISTILL.md`
  - `flow_planner/dpo/train_soft_pref.py`
  - `flow_planner/dpo/build_multi_pairs.py`
  - `flow_planner/goal/GOAL_DESIGN.md`
- Key idea: replace pure `best vs worst` hard-pair learning with scene-level candidate distribution learning.
  - Generate K candidates per scene.
  - Score every candidate with structured metrics.
  - Convert candidate scores into a teacher soft target distribution `q_i = softmax(u_i / T)`.
  - Train policy candidate probabilities `p_theta(i) = softmax(log pi_theta(tau_i | condition_i))` with `KL(q || p_theta)` / cross entropy.
  - Optional terms: teacher top-1 log-prob anchor and reference KL to control drift.
- Goal implementation details:
  - Teacher logit uses z-scored GT similarity plus z-scored structured scorer value: `u_i = gt_weight * z(gt_sim_i) + score_weight * z(score_i)`.
  - `train_soft_pref.py` currently attaches `goal_labels` via `attach_goal_to_decoder_inputs`.
  - `build_multi_pairs.py` also improves later hard DPO by selecting `strict_same_group`, `gt_near_unsafe`, `chosen_near_unsafe`, `same_group_soft`, `cross_group_soft`, and fallback hard failures.
- Anchor adaptation required:
  - Current anchor runtime `train_soft_pref.py` exists but is still goal-oriented; it does not yet consume `anchor_trajs` or call `attach_anchor_to_decoder_inputs`.
  - Do not run it as-is for anchor soft preference.
  - Need an anchor candidate artifact that preserves all candidates per scene, not only mined chosen/rejected pairs: trajectories, per-candidate anchor trajectory, anchor index/rank, metrics, and teacher score.
  - Then implement anchor soft-pref loss over candidates under their own anchor condition.
- Interpretation for current anchor-DPO issue:
  - This is the right direction for safe-vs-safe ambiguity: do not force every acceptable safe trajectory to be a hard rejected sample.
  - Use soft distribution / confidence weighting for quality differences, while keeping clear safe-vs-collided pairs as hard negatives later.

## Experiment: anchor soft preference distillation smoke 20260426

- Goal: address the user concern that safe-vs-safe hard DPO pairs can incorrectly mark acceptable trajectories as `rejected`. Instead of hard best-vs-worst pairs, test a scene-level soft ranking objective over all anchor-conditioned candidates.
- Runtime code added in `/root/autodl-tmp/Flow-Planner-anchor-runtime`:
  - `flow_planner/dpo/generate_anchor_softpref_candidates.py`
  - `flow_planner/dpo/train_anchor_soft_pref.py`
- Method:
  - For each scene, AnchorPredictor proposes top-3 anchors.
  - For each anchor, planner samples 3 trajectories, giving 9 candidates per scene.
  - Each candidate is scored with a moderate safety-first teacher score, not the earlier 100-point hard safety bonus.
  - Candidate score distribution is converted into a soft teacher target over 9 candidates.
  - Train decoder LoRA with `KL(q_teacher || p_policy)` where `p_policy = softmax(log_prob(candidate_i | scene, anchor_i))`.
- Smoke candidate generation, 20 train scenes:
  - Output: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_smoke20_20260426_2047`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_smoke20_20260426_2047.log`
  - Result: 20 / 20 scenes written, 180 candidates, 0 failures.
  - Candidate NPZ contains `candidates (9,80,4)` and `anchor_trajs (9,80,3)`.
  - Average score stats over scenes: min 5.375, mean 5.892, max 6.294, std 0.366.
  - Average collided candidates per scene: 0.55.
  - Average soft target entropy: 1.744; average top probability: 0.375.
- Smoke training, 20 train scenes:
  - Output: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_smoke20_e1_20260426_2048`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_smoke20_e1_20260426_2048.log`
  - Correct anchor runtime check: LoRA includes `anchor_encoder` and `anchor_cross_attn`; merged checkpoint removes 150 LoRA side keys.
  - Epoch 1: loss 2.2457, top1 match 20.00%.
  - Interpretation: pipeline works; sample too small for a method conclusion.
- Candidate generation, 100 train scenes:
  - Output: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050.log`
  - Result: 100 / 100 scenes written, 900 candidates, 0 failures.
  - Average score stats: min 5.328, mean 5.999, max 6.404, std 0.423.
  - Score std distribution: p10 0.031, median 0.059, p90 2.103.
  - Average collided candidates per scene: 0.56; only 18 / 100 scenes have any collision candidate.
  - Average soft target entropy: 1.810; average top probability: 0.328.
  - Teacher top candidate safe rate: 100%.
- Training, 100 train scenes:
  - Output: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_train100_e2_20260426_2052`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_train100_e2_20260426_2052.log`
  - Epoch 1: loss 2.2545, top1 match 11.00%.
  - Epoch 2: loss 2.2531, top1 match 9.00%.
  - Offline diagnostic on same 100 scenes, `num_t_samples=2`:
    - Base: CE 2.2016, top1 9.0%, policy safe mass 0.9370, target safe mass 0.9891.
    - Soft100: CE 2.1988, top1 10.0%, policy safe mass 0.9381, target safe mass 0.9891.
  - Interpretation: full 100-scene soft distillation gives only a tiny CE/safe-mass improvement, not enough for eval.
- Informative-scene filter:
  - Added runtime filter args: `--min_score_std` and `--min_top_prob`.
  - On train100 candidates, `min_score_std=1.0` keeps 18 / 100 scenes, essentially the collision-rich scenes.
- Filtered training, 18 informative scenes:
  - Output: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_train100_std1_e5_20260426_2055`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_train100_std1_e5_20260426_2055.log`
  - Setup: `min_score_std=1.0`, `target_temp=0.7`, `gt_weight=0.0`, `top1_weight=0.0`, `lr=2e-5`, epochs 5.
  - Best top1 match: 16.67%.
  - Offline diagnostic on the same 18 scenes:
    - Base: CE 2.1988, top1 11.11%, policy safe mass 0.6539, target safe mass 0.9669.
    - Filtered softpref: CE 2.1992, top1 27.78%, policy safe mass 0.6545, target safe mass 0.9669.
  - Interpretation: top1 can move a little, but probability mass does not shift toward safe candidates.
- Overfit diagnostic, 18 informative scenes:
  - Output: `/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_overfit_std1_e10_20260426_2100`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_overfit_std1_e10_20260426_2100.log`
  - Setup: `min_score_std=1.0`, `target_temp=0.7`, `gt_weight=0.0`, `top1_weight=1.0`, `lr=1e-4`, epochs 10.
  - Best top1 match: 22.22%.
  - Offline diagnostic:
    - Base: CE 2.1934, top1 0.00%, policy safe mass 0.6573, target safe mass 0.9669.
    - Overfit run: CE 2.1948, top1 22.22%, policy safe mass 0.6554, target safe mass 0.9669.
  - Checkpoint diff confirms weights changed, especially in anchor encoder and decoder LoRA-merged weights, so the weak result is not just a save/load failure.
- Decision:
  - Anchor soft preference data pipeline is now available and valid.
  - However, using the current continuous flow-matching log-prob as a candidate-ranking objective is still too weak/noisy to justify deployment eval.
  - Do not run 500/2k eval for the softpref checkpoints above.
  - Next technical priority: diagnose/rework the probability objective, or move preference learning one level up to an explicit anchor selector/reranker where probabilities are discrete and trainable, closer to DriveDPO-style soft distribution over anchors/candidates.

## Experiment: discrete anchor selector soft preference 20260426

- Motivation:
  - Continuous planner-DPO / soft preference uses flow-matching log-prob over full trajectories, but same-anchor chosen/rejected deltas were nearly zero.
  - Existing `predicted_anchor_rerank_a` works because it changes the problem into discrete candidate selection after seeing planner-generated trajectories.
  - This experiment tests a lighter middle ground: fine-tune the discrete `AnchorPredictor` head with soft preference targets aggregated from generated top-k anchor candidates.
- Runtime code added in `/root/autodl-tmp/Flow-Planner-anchor-runtime`:
  - `flow_planner/dpo/train_anchor_selector_softpref.py`
  - Patch artifact: `/root/autodl-tmp/anchor_runs/patches/anchor_selector_softpref_train_runtime.patch`
- Method:
  - Reuse softpref candidate artifacts from `generate_anchor_softpref_candidates.py`.
  - For each scene, aggregate candidate scores by `anchor_index` using `mean` or `max`.
  - Convert anchor scores into sparse full-vocab teacher distribution with `softmax(score / T)`.
  - Fine-tune only the `AnchorPredictor` head on `CE(q_anchor, p_predictor)` with small optional GT-nearest CE regularization.
  - This is not planner-DPO; it is an anchor-level discrete selector/reranker diagnostic.
- Train100 diagnostics:
  - Candidate artifact: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050`
  - `lr=0` baseline on 80/20 split, `mean`, `T=0.5`: val top1_match 34.4%, target_prob_on_pred 0.339.
  - High-lr selector: `/root/autodl-tmp/anchor_runs/anchor_selector_train100_mean_t0p5_e8_20260426_2108`; best early val top1_match 25.0%, later 9.4%.
  - Low-lr selector: `/root/autodl-tmp/anchor_runs/anchor_selector_train100_mean_t0p5_lr3e5_e8_20260426_2111`; best val top1_match stayed at baseline 34.4%, then dropped to 31.2%.
  - Interpretation: 100 scenes is too small/noisy for this selector objective; high lr overfits quickly.
- Train500 candidate generation:
  - Output: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train500_20260426_2113`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train500_20260426_2113.log`
  - Result: 500 / 500 scenes written, 4500 candidates, 0 failures.
  - Mean aggregation target stats: 500 records, 400 train / 100 val split, score_std_mean 0.3125, score_gap_mean 0.2383, top_prob_mean 0.4485, collision_scene_count 134.
- Train500 selector diagnostics:
  - `lr=0` baseline, `mean`, `T=0.5`: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_baseline_lr0_20260426_2129`; val top1_match 28.1%, target_prob_on_pred 0.324.
  - Low-lr selector, `mean`, `T=0.5`: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_mean_t0p5_lr3e5_e10_20260426_2130`; best val top1_match 26.6%, target_prob_on_pred about 0.330.
  - `max` aggregation baseline: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_max_t0p5_baseline_lr0_20260426_2132`; val top1_match 25.8%, target_prob_on_pred 0.313.
  - Internal target-match metrics do not improve over the original predictor, but this metric is not perfectly aligned with deployment safety.
- Direct 500-val deployment smoke, same manifest as rho=0.5 500 eval:
  - Output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train500_mean_500_20260426_2134`
  - Original `predicted_anchor_top1`: collision 3.2%, progress 0.3361, route 0.8468, collision_score 0.1289.
  - Selector `predicted_anchor_top1`: collision 2.0%, progress 0.3332, route 0.8480, collision_score 0.1256.
  - Original `predicted_anchor_rerank_a`: collision 4.6%, progress 0.3434, route 0.8657, collision_score 0.1266.
  - Selector `predicted_anchor_rerank_a`: collision 2.4%, progress 0.3387, route 0.8665, collision_score 0.1224.
- Direct 2k-val deployment smoke, manifest `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`:
  - Output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train500_mean_2k_20260426_2137`
  - Baseline planner, no anchor: collision 5.45%, progress 0.3393, route 0.8592.
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Selector `predicted_anchor_top1`: collision 3.70%, progress 0.3185, route 0.8574.
  - Original `predicted_anchor_rerank_a`: collision 3.15%, progress 0.3293, route 0.8738.
  - Selector `predicted_anchor_rerank_a`: collision 3.35%, progress 0.3248, route 0.8768.
  - Oracle anchor: collision 2.20%, progress 0.3149, route 0.8580.
  - Oracle anchor rerank: collision 2.80%, progress 0.3309, route 0.8748.
- Interpretation:
  - Selector top1 has a real safety signal: on 2k val it improves predicted top1 collision from 4.20% to 3.70%, but pays progress cost.
  - Selector plus existing hand rerank is not yet better than original hand rerank on 2k: 3.35% vs 3.15% collision, though route is slightly higher.
  - Internal target-match metrics were pessimistic; deployment eval is necessary for selector variants.
  - Best current deployment choice remains rho=0.5 + original `predicted_anchor_rerank_a` for safety/progress balance.
  - Best learned preference signal so far is selector top1, not planner-DPO; this suggests future anchor-DPO should operate over discrete anchors/candidates or train a candidate-aware reranker rather than relying only on continuous planner log-prob.
- Next recommended experiment:
  - Generate a larger selector dataset, preferably 2k train scenes, and train a candidate-aware reranker that sees scene features plus candidate anchor/trajectory features.
  - Keep the learned selector top1 as a positive but not final result.
  - Do not replace the current production rerank_a with selector+rerank until it beats 3.15% collision on the 2k manifest.

## Glossary note: DPO pair types and flow-matching log-prob 20260426

- `mixed v2 DPO` means the preference dataset contains two kinds of same-anchor pairs at the same time:
  - `same_anchor_collision`: chosen is a safe trajectory, rejected is a collided trajectory under the same scene and same anchor. This is the clearest preference label.
  - `same_anchor_quality`: both candidates are safe, but one has a higher structured score than the other. This is weaker because the rejected trajectory may still be acceptable; it should not be treated as equally strong as a collision failure.
  - Purpose: test whether combining clear safety pairs and softer safe-vs-safe quality pairs can train a useful planner preference signal.
- `collision-only DPO` means we throw away all safe-vs-safe quality pairs and train only on `same_anchor_collision` pairs.
  - Purpose: isolate the cleanest signal. If this works but mixed fails, the safe-vs-safe labels are probably noisy. If this also fails, the bottleneck is more likely the DPO objective / log-prob estimate itself.
  - Result: collision-only reached only about random pair accuracy, so cleaner labels alone were not enough.
- `DPO acc` here is not driving accuracy or trajectory accuracy. It is pairwise preference accuracy:
  - For each pair, compute whether the model assigns a larger relative likelihood / DPO margin to `chosen` than to `rejected`.
  - Around 50% means the model is basically not distinguishing the preferred trajectory from the rejected trajectory.
  - Below 50% can happen with noisy estimates or if updates push in the wrong/noisy direction.
- `continuous flow-matching log-prob` explanation:
  - Flow Planner is a continuous trajectory generator trained with flow matching / diffusion-like denoising, not a normal classifier over a small discrete set.
  - DPO needs a number like `log pi_theta(trajectory | scene, condition)` so it can say whether chosen is more likely than rejected.
  - For a continuous flow-matching model, this log-prob is only approximated through the denoising/flow-matching loss on a candidate trajectory, often with sampled noise/timesteps.
  - That estimate can be noisy and very close for two similar trajectories, especially when both use the same anchor.
  - In our experiments, chosen/rejected margins stayed near zero, so the training objective could not reliably tell which candidate should win.
- Practical interpretation:
  - The weak result does not mean anchor conditioning is useless. Oracle anchor and predicted-anchor rerank show the anchor candidate space has value.
  - It means planner-level DPO using the current continuous trajectory log-prob estimate is not yet a reliable preference learner.
  - This is why the later selector experiment moves preference learning to a discrete anchor/candidate selection problem, where probabilities are cleaner and easier to train.

## Experiment: anchor selector train2k mean soft target 20260426

- Goal: verify whether scaling the learned anchor selector from 500 train scenes to 2000 train scenes improves deployment collision on the fixed 2k val manifest.
- Candidate generation:
  - Output: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153`
  - Log: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153.log`
  - Result: 2000 / 2000 train scenes, 18000 candidates, 0 failures.
  - Candidate structure: top3 predicted anchors per scene, 3 planner samples per anchor.
- Selector training:
  - Baseline run: `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_baseline_lr0_20260426_2226`
  - Train run: `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227`
  - Target stats: 2000 records, 1600 train / 400 val, score_std_mean 0.2936, score_gap_mean 0.2528, top_prob_mean 0.4400, collision_scene_count 482.
  - Baseline internal val: top1_match 30.4%, target_prob_on_pred 0.364.
  - Train internal val best: top1_match about 31.5% at epoch 3; final epoch 10 top1_match 30.1%, target_prob_on_pred 0.346.
- Deployment eval:
  - Output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train2k_mean_2k_20260426_2235`
  - Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Case: `predicted_anchor_top1_selector_train2k`
  - Result: collision 3.55%, progress 0.3187, route 0.8545, collision_score 0.1062, scenes 2000 / 2000.
- Comparison on the same 2k manifest:
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Selector train500 top1: collision 3.70%, progress 0.3185, route 0.8574.
  - Selector train2k top1: collision 3.55%, progress 0.3187, route 0.8545.
- Interpretation:
  - Scaling selector data from 500 to 2000 train scenes gives a small additional safety gain: 3.70% -> 3.55% collision.
  - Compared with the original predictor top1, learned selector reduces collision by 0.65 percentage points absolute, but still trades off progress.
  - It still does not beat original `predicted_anchor_rerank_a` collision 3.15%, so train2k selector is a positive learned-signal result, not the current best deployment method.

## Experiment: selector-DPO all-pairs pilot 20260426

- Goal: test whether putting DPO directly on discrete selector scores improves learned anchor selection beyond soft-CE selector.
- Script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_anchor_selector_dpo.py`
- Runtime code patch artifact should be saved before branch sync: `/root/autodl-tmp/anchor_runs/patches/anchor_selector_dpo_train_runtime.patch`
- Pair construction:
  - Source candidates: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir`
  - Aggregation: mean score per anchor.
  - Pair mode: all ordered anchor pairs with `score_gap >= 0.05`.
  - Policy init: soft selector train2k `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227/anchor_selector_best.pth`
  - Reference: original anchor predictor `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
- Training output:
  - `/root/autodl-tmp/anchor_runs/anchor_selector_dpo_train2k_all_gap0p05_e10_20260426_2256`
  - Pairs: 3020 total, 2416 train / 604 val.
  - Labels: 2024 `anchor_quality`, 746 `anchor_collision`, 250 `anchor_collision_rate`.
  - Training did learn the pair objective: val pair acc improved to about 61%, val margin about 0.29.
- Deployment eval:
  - Output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_train2k_all_2k_20260426_2300`
  - Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Result: collision 5.65%, progress 0.3516, route 0.8508.
- Interpretation:
  - This is a negative result for all-pairs selector-DPO.
  - The model learned the offline pair labels, but deployment safety got much worse than original top1 4.20% and soft selector train2k 3.55%.
  - Likely issue: all-pairs DPO over-weights `anchor_quality` / progress-like differences and treats weak safe-vs-safe preferences as hard pair constraints.
  - Next action: switch to collision-only selector-DPO, using only the clearest safety pairs.

## Experiment: selector-DPO collision-only pilot 20260426

- Goal: after all-pairs selector-DPO failed in deployment safety, test whether DPO on only clean safety pairs can improve learned selector top1.
- Script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_anchor_selector_dpo.py`
- Patch artifact: `/root/autodl-tmp/anchor_runs/patches/anchor_selector_dpo_train_runtime.patch`
- Pair construction:
  - Source candidates: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir`
  - Aggregation: mean score per anchor.
  - Pair mode: all ordered anchor pairs with `score_gap >= 0.05`.
  - Filter: `--require-collision-pair`, keeping only `anchor_collision` and `anchor_collision_rate` pairs.
  - Policy init: soft selector train2k `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227/anchor_selector_best.pth`
  - Reference: original anchor predictor `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
- Training output:
  - `/root/autodl-tmp/anchor_runs/anchor_selector_dpo_train2k_collision_gap0p05_e10_20260426_2302`
  - Pairs: 996 total, 797 train / 199 val.
  - Labels: 746 `anchor_collision`, 250 `anchor_collision_rate`.
  - Internal val pair acc: about 67%; val margin grows to about 0.62.
- Deployment eval:
  - Output: `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_collision_train2k_2k_20260426_2305`
  - Manifest: `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Case: `predicted_anchor_top1_selector_dpo_collision_train2k`
  - Result: collision 3.15%, progress 0.3150, route 0.8549, collision_score 0.1080, scenes 2000 / 2000.
- Comparison on the same 2k manifest:
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Soft selector train2k top1: collision 3.55%, progress 0.3187, route 0.8545.
  - Selector-DPO all-pairs top1: collision 5.65%, progress 0.3516, route 0.8508.
  - Selector-DPO collision-only top1: collision 3.15%, progress 0.3150, route 0.8549.
  - Original hand `predicted_anchor_rerank_a`: collision 3.15%, progress 0.3293, route 0.8738.
- Interpretation:
  - This is the first strong selector-DPO result: learned top1 selector matches the hand-rerank collision rate without using hand rerank at inference.
  - The tradeoff is lower progress and route than hand-rerank, so it is not yet a strictly better deployment method.
  - Important lesson: selector-DPO should use clean safety preference pairs first; all-pairs DPO with many weak quality pairs can hurt safety even if offline pair accuracy improves.
  - Next direction: improve selector-DPO objective to recover progress/route, or move to candidate-level learned selector that can replace hand-rerank more directly.

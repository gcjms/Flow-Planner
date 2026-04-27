# Anchor 条件化实验记录

## 0. 20260426 下午 anchor 偏好实验总览（可读版）

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

### 0.3 第二层：为什么同 anchor DPO 没直接成功

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

### 0.4 第三层：为什么连续 flow-matching log-prob 是瓶颈

DPO 需要比较 `chosen` 和 `rejected` 哪个更像模型会生成的轨迹，也就是需要类似 `log pi_theta(trajectory | scene, anchor)` 的分数。

但 Flow Planner 不是离散分类器，它是连续轨迹生成模型，训练方式接近 flow matching / diffusion 去噪。我们只能用 flow-matching loss 近似某条 candidate trajectory 的 likelihood。这个近似在候选很相似、同属一个 anchor 时非常不灵敏：好轨迹和坏轨迹的分数差经常接近 0。

实验表现就是：

- chosen/rejected DPO margin 很小。
- pair acc 接近随机。
- soft preference 训练后，模型给 safe candidates 的总概率几乎没变。
- top1 candidate 也没有稳定转向 teacher 认为最好的轨迹。

结论：soft preference 数据生成方向是对的，但把它直接压到 planner 的连续 log-prob 上，目前不是可靠路径。

### 0.5 第四层：为什么转向可学习 anchor selector


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


## 实验：anchor_eval_suite_clean

- 目的：验证 anchor conditioning 在部署评测中的上限与当前 predictor 瓶颈。
- 设置：使用 `planner_ft_run_clean/planner_anchor_best.pth`，比较 `planner_ft_none`、`predicted_anchor_top1`、`predicted_anchor_rerank_a`、`oracle_anchor`、`oracle_anchor_rerank`。
- 产物：
  - 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_latest`
  - Predictor ckpt：`/root/autodl-tmp/anchor_runs/anchor_predictor_run_clean/anchor_predictor_best.pth`
  - Planner ckpt：`/root/autodl-tmp/anchor_runs/planner_ft_run_clean/planner_anchor_best.pth`
- 数据：
  - 评测 manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_clean.json`
  - 评测场景数：500
- 结果：
  - `planner_ft_none`: collision_rate 6.4
  - `predicted_anchor_top1`: collision_rate 6.2
  - `predicted_anchor_rerank_a`: collision_rate 6.2
  - `oracle_anchor`: collision_rate 2.0
  - `oracle_anchor_rerank`: collision_rate 2.8
- 结论：
  - `oracle_anchor` 明确证明 anchor 信息本身有价值。
  - `predicted_anchor` 仍然没有把这种价值传到部署端。
- 决策：
  - 尝试 scheduled sampling，缓解 train/inference mismatch。

## 实验：anchor_sched_p0p3_20260426

- 目的：验证 scheduled sampling `rho=0.3` 是否能改善 predicted anchor 的部署表现。
- 设置：
  - 脚本：`/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_scheduled_sampling.sh`
  - `p_max=0.3`
  - `epochs=10`
  - `batch_size=32`
  - `max_train_samples=80000`（未触发，实际 train 集更小）
- 产物：
  - 训练输出：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p3_20260426_1506`
  - 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p3_20260426_1506`
  - Predictor ckpt：`/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Planner ckpt：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p3_20260426_1506/planner_anchor_best.pth`
- 数据：
  - 训练目录：`/root/autodl-tmp/train_dataset`
  - 验证目录：`/root/autodl-tmp/val_dataset`
  - 训练样本数：40079
  - 验证样本数：17224
  - 每个 epoch 的有效训练样本数：40064（`batch_size=32`，`drop_last=True`）
  - 评测 manifest：`/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - 评测场景数：500
- 结果：
  - `planner_ft_none`: collision_rate 6.6
  - `predicted_anchor_top1`: collision_rate 3.6
  - `predicted_anchor_rerank_a`: collision_rate 5.0
  - `oracle_anchor`: collision_rate 1.8
  - `oracle_anchor_rerank`: collision_rate 3.6
- 结论：
  - `rho=0.3` 对 `predicted_anchor_top1` 有明显帮助，collision_rate 从上一轮约 6.2 降到 3.6。
  - 但 predicted anchor 与 oracle anchor 之间仍有明显差距，predictor 质量仍是主瓶颈。
  - rerank 方案当前不稳定，不应作为主结论。
- 决策：
  - 继续尝试 `rho=0.5`。

## 实验：anchor_sched_p0p5_20260426

- 目的：在 `rho=0.3` 已出现正向信号后，继续测试更强的 scheduled sampling 是否进一步改善 predicted anchor 部署表现。
- 设置：
  - 脚本：`/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_scheduled_sampling.sh`
  - `p_max=0.5`
  - `epochs=10`
  - `batch_size=32`
  - `max_train_samples=80000`（未触发，实际 train 集更小）
- 产物：
  - 启动日志：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612.launch.log`
  - 训练输出：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612`
  - 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_20260426_1612`
  - Predictor ckpt：`/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Planner ckpt：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
- 数据：
  - 训练目录：`/root/autodl-tmp/train_dataset`
  - 验证目录：`/root/autodl-tmp/val_dataset`
  - 训练样本数：40079
  - 验证样本数：17224
  - 每个 epoch 的有效训练样本数：40064（`batch_size=32`，`drop_last=True`）
  - 评测 manifest：`/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - 评测场景数：500
- 结果：
  - `planner_ft_none`: collision_rate 7.4
  - `predicted_anchor_top1`: collision_rate 3.2
  - `predicted_anchor_rerank_a`: collision_rate 4.6
  - `oracle_anchor`: collision_rate 2.2
  - `oracle_anchor_rerank`: collision_rate 3.4
- 结论：
  - 相比 `rho=0.3`，`rho=0.5` 继续改善了 predicted anchor 部署表现，但提升幅度有限。
  - `predicted_anchor_top1` 从 3.6 进一步降到 3.2，说明提高 scheduled sampling 比例仍有帮助。
  - 但 `oracle_anchor` 与 `predicted_anchor` 之间仍有明显 gap，说明 predictor 质量仍是主要瓶颈。
  - 收益已经开始趋于饱和，继续单纯提高 rho 未必是下一步最优方向。
- 决策：
  - 保留 `rho=0.3` 和 `rho=0.5` 作为正结果。
  - 后续优先考虑提升 predictor / anchor selection 质量，而不是继续单独上调 rho。

## 下一阶段计划：从 anchor 到 DPO readiness

- 目的：判断 anchor 是否已经适合作为后续 DPO 的轨迹 mode / candidate organization 单元，并避免重复 goal-conditioned DPO 中 cross-goal pair 失败的问题。
- 为什么使用 500 scenes：
  - 当前部署评测脚本默认 MAX_SCENES=500，默认复用 /root/autodl-tmp/anchor_runs/eval_manifest.json。
  - eval_manifest.json 目前指向 eval_manifest_clean.json，其中包含 500 scenes。
  - 这批 500 scenes 已用于 clean suite、rho=0.3、rho=0.5 的 planner_ft_none、predicted_anchor_top1、predicted_anchor_rerank_a、oracle_anchor、oracle_anchor_rerank 对比。
  - 500 scenes 适合快速判断方向和发现大问题，但不适合作为最终论文主表的唯一依据。
- 当前证据：
  - oracle_anchor 明显优于 no-anchor / predicted-anchor，说明 anchor 表示本身有价值。
  - scheduled sampling 已经把 predicted_anchor_top1 collision_rate 从约 6.2 降到 3.6 / 3.2，说明 planner 可以学习部署时的 predicted-anchor 噪声。
  - rho=0.5 相比 rho=0.3 继续提升有限，说明继续单独扫更高 rho 的信息量下降。
  - 当前 predictor formal run 的验证集指标为 top1=0.403、top3=0.775、top5=0.902，说明 top1 selection 仍弱，但正确 mode 经常已经在 top-k candidate pool 中。
- 关键约束：
  - 旧 goal-DPO 记录显示，任意 cross-goal pair + goal-aware DPO 路线不稳定。
  - anchor-DPO 不能直接复刻任意 cross-anchor pair；否则 preference 可能混入条件变量差异，而不是只表达轨迹质量差异。
- 工作假设：
  - anchor 的下一步价值不应只看 top1 是否完美，而应看 top-k anchor mode pool 是否能稳定产生高质量候选。
  - DPO pair construction 应优先保证 condition-clean：同一 scene、同一 anchor，或语义非常接近的 near-anchor。
  - 任意 far cross-anchor pair 暂不作为主路线。
- 默认 planner 设置：
  - 目前优先使用 rho=0.5 的 planner checkpoint 作为 candidate generator。
  - 理由：它是当前 predicted-anchor 部署指标最好的 scheduled-sampling planner，但收益已趋于饱和，不建议继续优先扫 rho=0.7/1.0。
- 近期实验：
  - E1 predictor 诊断：统计 top1/top3/top5 accuracy、confusion matrix、oracle anchor rank distribution、错误是否集中在相近 anchors。
  - E2 top-k 候选质量：对每个 scene 用 predicted top-3/top-5 anchors 生成候选，评估候选池中是否存在低 collision / 高 route / 高 progress 轨迹。
  - E3 condition-clean preference mining：先构造 same-anchor pairs；如果需要 near-anchor pairs，必须先定义 anchor 距离阈值和 mode similarity 规则。
  - E4 rerank sanity check：如果 top-k pool 中经常有好候选但 top1 选错，优先做 reranker / selector；如果 top-k pool 本身质量不足，先改 predictor。
- Pair 构造策略：
  - 只保留强偏好：collision-free 优先于 collision，route/progress 明显更好优先于更差，模糊 pair 丢弃。
  - same-anchor 优先：同一 scene、同一 anchor 条件下比较多条轨迹候选。
  - near-anchor 可选：只在 anchors 轨迹形态接近且共享局部驾驶意图时比较。
  - 避免 far cross-anchor：不把明显不同驾驶 mode 的 pair 直接喂给 DPO。
- Manifest 策略：
  - 500-scene manifest 继续作为 quick dev / smoke eval。
  - 下一阶段 readiness 诊断建议建立更大的固定 manifest，例如 2k scenes；如果运行时间可接受，再扩到 5k scenes。
  - 论文主表不应只依赖 500 scenes，至少需要一个更大的固定 manifest 复核核心结论。
- 决策标准：
  - 如果 top-k pool coverage 高，且 same-anchor / near-anchor 能挖出足够强偏好 pair，则进入 anchor-DPO data construction。
  - 如果 top-k coverage 高但 top1/rerank 差，则先做 anchor selector / reranker。
  - 如果 top-k coverage 也差，则先重训或改造 anchor predictor，不急着做 DPO。
- 预期输出：
  - dpo_data/anchor_conditioned/candidates/ 保存 anchor candidate pool。
  - dpo_data/anchor_conditioned/preferences/ 保存 condition-clean preference pairs。
  - docs/experiments/anchor_conditioned.md 持续记录诊断结果、pair mining 规则、DPO 是否启动的判定。

## 更新：更大的 eval manifest

- 已从 `/root/autodl-tmp/val_dataset` 创建包含 2000 scenes 的 `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`。
- 已从 `/root/autodl-tmp/val_dataset` 创建包含 5000 scenes 的 `/root/autodl-tmp/anchor_runs/eval_manifest_5k_seed3402.json`。
- 保持 `/root/autodl-tmp/anchor_runs/eval_manifest.json` 不变，继续作为前面 clean / rho=0.3 / rho=0.5 对比使用的 500-scene quick-dev baseline。
- 修复 `/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_eval_common.sh`，让默认 `SCENE_DIR` 指向 `/root/autodl-tmp/val_dataset`，而不是缺失的 `/root/autodl-tmp/nuplan_npz`。
- 下一次 readiness eval 应优先把 `MANIFEST_PATH` 设为 2k manifest；5k manifest 留给更强确认或论文主表。

## 实验：anchor_sched_p0p5_eval_2k_20260426

- 目的：用更大的固定 manifest 复核 rho=0.5 planner 在 predicted/oracle anchor 部署评测上的核心趋势，降低 500-scene 快速评测的方差风险。
- 设置：
  - 脚本：/root/autodl-tmp/Flow-Planner-anchor-runtime/run_anchor_eval_suite.sh
  - Planner ckpt：/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth
  - 场景目录：/root/autodl-tmp/val_dataset
  - Manifest：/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
  - 评测场景数：2000
  - 评测 case：planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor, oracle_anchor_rerank
- 产物：
  - 评测输出：/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_2k_20260426_1743
- 状态：
  - 2026-04-26 17:43 CST 开始。
  - 2026-04-26 18:18 CST 完成。
  - 结果见下方总结。

## Anchor-DPO 实现说明

- 现有 `train_dpo.py` 支持通用 chosen/rejected pair 和可选 goal 字段，但还没有消费 anchor 专用字段。
- 真正的 anchor-DPO 中，preference 数据必须携带每个 pair 使用的 anchor 条件，例如 `anchor_trajs` 或 `chosen_anchor_trajs` / `rejected_anchor_trajs`。
- Same-anchor DPO 应该让 chosen 和 rejected 使用同一个 `anchor_traj`，然后在 DPO loss 计算时把这个 `anchor_traj` 传入 decoder inputs。
- 如果没有这个改动，用 anchor 生成的 pair 训练时会退化成 scene context 下的普通 DPO，而不是 anchor-conditioned DPO。
- 近期编码任务：给 `PreferenceDataset` / `collate_preferences` 增加 anchor 字段，并仿照现有 goal 路径添加 `attach_anchor_to_decoder_inputs`。

## 结果：anchor_sched_p0p5_eval_2k_20260426

- 状态：2026-04-26 18:18 CST 完成。
- 评测输出：/root/autodl-tmp/anchor_runs/deploy_eval_sched_p0p5_2k_20260426_1743
- Manifest：/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
- 评测场景数：2000
- 结果：
  - planner_ft_none: collision_rate 5.45, avg_progress 0.3393, avg_route 0.8592
  - predicted_anchor_top1: collision_rate 4.20, avg_progress 0.3253, avg_route 0.8548
  - predicted_anchor_rerank_a: collision_rate 3.15, avg_progress 0.3293, avg_route 0.8738
  - oracle_anchor: collision_rate 2.20, avg_progress 0.3149, avg_route 0.8580
  - oracle_anchor_rerank: collision_rate 2.80, avg_progress 0.3309, avg_route 0.8748
- 结论：
  - 2k eval 确认 predicted anchors 相比 no-anchor 有帮助，top-k reranker 在更大的固定验证子集上也有价值。
  - 500-scene 结果存在明显方差，尤其是 `planner_ft_none` 和 rerank。
  - `predicted_anchor_rerank_a` 当前是最好的 predicted-anchor 部署设置，但仍落后于 `oracle_anchor`。
  - `oracle_anchor` 仍然是上限信号；predictor / selection 质量仍是瓶颈。
- 决策：
  - 当前 candidate generator 使用 rho=0.5 planner。
  - 继续并行推进 anchor selection / rerank 和 anchor-DPO readiness。

## 实验：anchor_dpo_readiness_smoke_20260426

- 目的：验证 anchor-DPO 的最小数据/训练链路是否可行，且 preference pair 保持 condition-clean。
- 代码审查发现：
  - 现有 `train_dpo.py` 支持普通 DPO 和可选 goal 字段，但没有消费 anchor 专用字段。
  - 运行时 `train_dpo.py` 已 patch，可接受 `anchor_vocab_path`、加载 anchor-enabled planner config、读取 `anchor_trajs` / `chosen_anchor_trajs` / `rejected_anchor_trajs`，并把 `anchor_traj` 传入 decoder inputs。
  - 新增运行时脚本 `flow_planner/dpo/generate_anchor_same_anchor_pairs.py`，用于 same-scene + same-anchor pair mining。
  - 运行时 patch 保存在 `/root/autodl-tmp/anchor_runs/patches/anchor_dpo_train_dpo_runtime.patch` 和 `/root/autodl-tmp/anchor_runs/patches/anchor_same_anchor_pair_generator.patch`。
  - 这些运行时代码变更在大规模实验被视为 canonical 之前，还需要正式迁移到 anchor 分支。
- Pair mining smoke 设置：
  - 脚本：/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/generate_anchor_same_anchor_pairs.py
  - Planner ckpt：/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth
  - Predictor ckpt：/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth
  - Manifest：/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json
  - max_scenes: 20, top_k: 3, samples_per_anchor: 3
- Pair mining smoke 结果：
  - 输出：/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_smoke_20260426_1815.npz
  - Pair 数：28
  - 失败数：0
  - Pair 标签：same_anchor_collision 5, same_anchor_quality 23
  - Shape：chosen/rejected (28, 80, 4), anchor_trajs (28, 80, 3)
- DPO train smoke 设置：
  - Preference 路径：/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_smoke_20260426_1815.npz
  - 输出：/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_smoke_20260426_1817
  - max_pairs: 8, epochs: 1, batch_size: 2, num_t_samples: 1, lora_rank: 2
- DPO train smoke 结果：
  - 训练成功完成。
  - 最佳 accuracy：87.50%
  - 这只是 pipeline sanity check，不是模型质量结果。
- 结论：
  - same-anchor preference mining is feasible: even 20 scenes produced nonzero clean pairs.
  - anchor-conditioned DPO training path is now technically viable in runtime.
  - Next step should be pair-yield scaling and quality audit before any full anchor-DPO training.
- 决策：
  - 下一步把 pair mining 扩大到更大的固定子集，根据 runtime budget 从 500 scenes 或 2k scenes 开始。
  - 不把 smoke LoRA 输出当作可部署 checkpoint。

## 实验：anchor_dpo_train500_gap0p15_pilot_20260426

- 目的：验证 same-anchor preference pair 能否真正接到 anchor-conditioned DPO 训练，并快速判断小规模 DPO 是否可能改善 anchor 部署评测。
- 设置：
  - Pair 生成器：`/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/generate_anchor_same_anchor_pairs.py`
  - Base planner：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Predictor：`/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Pair 来源 split：train，`/root/autodl-tmp/train_dataset`
  - 训练 manifest：`/root/autodl-tmp/anchor_runs/generated_lists/train_list.json`
  - Pair mining 子集：500 train scenes
  - Pair mining 配置：`top_k=3`，`samples_per_anchor=3`，只使用同一 scene + 同一 predicted anchor
- Pair mining 产物：
  - 原始 preference 文件：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_20260426_1830.npz`
  - 过滤后 preference 文件：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_gap0p15_20260426_1830.npz`
- Pair mining 结果：
  - 原始 pairs：来自 437 / 500 scenes 的 924 pairs，0 failures
  - 原始标签：`same_anchor_quality` 800，`same_anchor_collision` 124
  - Anchor ranks：rank0 290，rank1 308，rank2 326
  - Pilot 使用的过滤：保留所有 collision pairs，以及 `score_gap >= 0.15` 的 quality pairs
  - 过滤后 pairs：321 pairs，其中 `same_anchor_quality` 197，`same_anchor_collision` 124
- DPO 训练设置：
  - 脚本：`/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_dpo.py`
  - Preference 路径：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train500_gap0p15_20260426_1830.npz`
  - 输出目录：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840`
  - Epochs：2
  - Batch size：8
  - `num_t_samples=4`, `beta=0.1`, `sft_weight=0.05`, `lr=1e-5`, `lora_rank=4`, `lora_alpha=16`
- DPO 训练结果：
  - Epoch 1: loss 0.7464, accuracy 46.56%, delta -0.0076
  - Epoch 2: loss 0.7460, accuracy 51.88%, delta -0.0010
  - 最佳 train accuracy：51.88%
  - 保存的 merged model：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged.pth`
  - 保存的干净 merged model（已去掉 LoRA side keys）：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged_stripped.pth`
- Pilot 期间的代码审查记录：
  - 原始 merged checkpoint 保留了 150 个 LoRA side keys，导致 eval 出现 `unexpected keys` warning。后续 eval 创建了 stripped checkpoint；这是 checkpoint 格式问题，不代表 DPO 权重被忽略。
  - 相比 rho=0.5 base planner 的权重 diff 检查发现 72 个浮点 tensor 发生变化；最大相对变化在 `model_decoder.anchor_cross_attn.out_proj.weight`。
  - 运行时 `train_dpo.py` 暴露了 `--min_score_gap`，但 dataset loader 实际没有应用它。本次 pilot 不受影响，因为过滤已经写入单独 `.npz`；但依赖 CLI filtering 前必须修复。
  - 运行时 `dpo_loss.py` 当前为 policy 和 reference log-probs 独立采样 flow-matching noise/timesteps。这作为 noisy estimator 是有效的，但方差不必要地高；下一轮 DPO 训练应让同一条轨迹的 policy/reference 共享 sampled noise/t。
  - 运行时代码变更仍在 anchor runtime snapshot 和 patch artifacts 中，还没有正式迁移到 `anchor` 分支。
- 500-scene quick eval 设置：
  - 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_500_20260426_1848`
  - 场景目录：`/root/autodl-tmp/val_dataset`
  - Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest.json`
  - 评测场景数：500
  - 评测 case：planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor
- 500-scene quick eval 结果：
  - planner_ft_none: collision_rate 5.60, avg_progress 0.3512, avg_route 0.8536
  - predicted_anchor_top1: collision_rate 3.40, avg_progress 0.3350, avg_route 0.8516
  - predicted_anchor_rerank_a: collision_rate 3.20, avg_progress 0.3400, avg_route 0.8653
  - oracle_anchor: collision_rate 2.00, avg_progress 0.3256, avg_route 0.8472
- 500-scene 同 manifest 上与 rho=0.5 non-DPO planner 的对比：
  - planner_ft_none collision_rate 从 7.40 改善到 5.60。
  - predicted_anchor_top1 collision_rate 从 3.20 轻微退化到 3.40。
  - predicted_anchor_rerank_a collision_rate 从 4.60 改善到 3.20。
  - oracle_anchor collision_rate 从 2.20 改善到 2.00。
- 阶段性结论：
  - 这个 pilot 在技术链路上成功，并对 rerank/oracle 显示小的正信号，但训练 accuracy 接近随机，top1 没有改善。
  - 这还不足以支撑大规模 DPO run。
  - 但足以支撑先做一次 2k fixed-manifest 复核，再决定是否扩大 pair mining/training。
- 2k 复核状态：
  - 2026-04-26 18:50 CST 开始。
  - 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_2k_20260426_1852`
  - Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Checkpoint：`model_dpo_merged_stripped.pth`
  - 评测 case：planner_ft_none, predicted_anchor_top1, predicted_anchor_rerank_a, oracle_anchor
  - 记录时状态：running。

## 结果：anchor_dpo_train500_gap0p15_eval_2k_20260426

- 状态：2026-04-26 19:11 CST 完成。
- 评测输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_dpo_train500_gap0p15_e2_2k_20260426_1852`
- Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
- Checkpoint：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train500_gap0p15_e2_20260426_1840/model_dpo_merged_stripped.pth`
- 评测场景数：2000
- 结果：
  - planner_ft_none: collision_rate 5.15, avg_progress 0.3419, avg_route 0.8609
  - predicted_anchor_top1: collision_rate 4.10, avg_progress 0.3238, avg_route 0.8545
  - predicted_anchor_rerank_a: collision_rate 3.10, avg_progress 0.3282, avg_route 0.8738
  - oracle_anchor: collision_rate 2.20, avg_progress 0.3138, avg_route 0.8559
- 与 rho=0.5 non-DPO 2k eval 的对比：
  - planner_ft_none：5.45 -> 5.15 collision_rate，小幅改善。
  - predicted_anchor_top1：4.20 -> 4.10 collision_rate，几乎可以忽略。
  - predicted_anchor_rerank_a：3.15 -> 3.10 collision_rate，基本持平。
  - oracle_anchor：2.20 -> 2.20 collision_rate，持平。
- 结论：
  - anchor-DPO 链路在技术上可运行，且没有破坏 anchor-conditioned inference。
  - 在 2k fixed validation scenes 上，这个 321-pair pilot 只带来边际变化；500-scene 上看起来的 rerank 收益主要是方差。
  - 这个 pilot 还不够强，不能按当前设置直接扩大 DPO training。
  - 当前最佳部署候选仍是 rho=0.5 + predicted_anchor_rerank_a。
  - oracle gap 仍然存在：predicted_anchor_rerank_a 3.10 vs oracle_anchor 2.20 collision_rate。
- 决策：
  - 不从当前 321-pair setup 启动大规模 anchor-DPO training run。
  - 下一次 DPO run 前，先修复 review 中发现的 DPO 实现问题：正确应用 `min_score_gap` 过滤、保存 merged checkpoint 时去掉 LoRA side keys，并通过让同一轨迹的 policy/reference 共享 sampled flow-matching noise/timesteps 来降低 DPO loss 方差。
  - 修复代码后，先在 train split 上挖更大的 same-anchor preference set，大概率先做 2k train scenes，然后再训练/评测第二个 pilot。
  - 继续把 rerank/selector 作为高优先级方向，因为 top-k predicted anchors 已经比单纯 top1 更有价值。

## 代码审查/修复：anchor DPO runtime v2 20260426

- 目的：修掉下一轮 anchor-DPO 前会影响结论可信度的实现问题。
- 运行时变更文件：
  - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_dpo.py`
  - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/dpo_loss.py`
- 修复内容：
  - `PreferenceDataset` now actually applies `min_score_gap` filtering and keeps chosen/rejected, scenario ids, labels, goals, and anchor trajectories aligned.
  - CLI default `--min_score_gap` changed from 2.0 to 0.0 so existing files are not silently over-filtered unless explicitly requested.
  - Merged DPO checkpoint saving now strips `.lora_A` / `.lora_B` side keys after merge, avoiding eval-time `unexpected keys` warnings.
  - DPO loss now evaluates policy and reference with the same sampled flow-matching noise/timestep for each trajectory, reducing DPO delta variance.
- 验证：
  - `python -m py_compile` passed for runtime `train_dpo.py` and `dpo_loss.py`.
  - Dataset smoke: raw train500 preference file with `min_score_gap=0.15` keeps 321 / 924 pairs.
  - Anchor field smoke: dataset item contains `chosen_anchor_traj` with shape `(80, 3)`.
  - DPO loss smoke: dummy policy/reference forward returns finite DPO/SFT losses.
- Patch artifacts:
  - `/root/autodl-tmp/anchor_runs/patches/anchor_dpo_train_dpo_runtime_v2.patch`
  - `/root/autodl-tmp/anchor_runs/patches/anchor_dpo_loss_runtime_v2.patch`
- 决策：
  - Next anchor-DPO run should use this v2 runtime behavior or a formal migration of these changes into the `anchor` branch.
  - Do not compare future DPO runs against the v1 pilot without noting that v1 had higher-variance loss estimation and dirty merged checkpoint format.

## 实验：anchor_dpo_pair_mining_train2k_v2_20260426

- 目的：在修复 DPO runtime v2 后，先扩大 same-anchor preference mining 到 2k train scenes，判断是否有足够干净 pair 支撑第二轮 DPO pilot。
- 设置：
  - Runtime：`/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - 脚本：`flow_planner/dpo/generate_anchor_same_anchor_pairs.py`
  - Base planner：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Predictor：`/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - 场景目录：`/root/autodl-tmp/train_dataset`
  - 场景 manifest：`/root/autodl-tmp/anchor_runs/generated_lists/train_list.json`
  - max_scenes: 2000
  - top_k: 3
  - samples_per_anchor: 3
  - min_quality_gap: 0.05
- 产物：
  - 输出 preference 路径：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_v2_20260426_1921.npz`
  - 日志：`/root/autodl-tmp/anchor_runs/same_anchor_train2k_v2_20260426_1921.log`
  - 启动 PID：22951
- 状态：
  - 2026-04-26 19:21 CST 开始。
  - 记录时仍在运行。
- 决策规则：
  - 如果 pair yield 和 score-gap 分布健康，则创建明确过滤后的 `.npz` 并跑 v2 DPO pilot。
  - 如果 yield 弱或大多是模糊 quality pairs，则训练前先改进 pair selection。

## 实验：anchor_dpo_train2k_gap0p15_v2_e2_20260426

- 目的：在 runtime v2 修复后，用更大的 train2k same-anchor preference set 跑第二版 DPO pilot，验证更稳定的 DPO loss 和更多 pair 是否能产生更可信的提升。
- Pair 来源：
  - 原始 preference 文件：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_v2_20260426_1921.npz`
  - 原始 pairs：来自 2000 train scenes 的 3777 pairs，0 failures
  - 标签：`same_anchor_collision` 567，`same_anchor_quality` 3210
  - Anchor ranks：rank0 1192，rank1 1263，rank2 1322
  - Score gap：min 0.0500，median 0.1143，mean 15.1240，p75 0.2126，p90 99.9807，max 100.5446
- 训练过滤：
  - 在 runtime v2 `PreferenceDataset` 中使用 `--min_score_gap 0.15`。
  - 有效 pairs：1367，其中 collision 567、quality 800。
- 训练设置：
  - Runtime：`/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - 脚本：`python -m flow_planner.dpo.train_dpo`
  - Base planner：`/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Anchor vocab：`/root/autodl-tmp/anchor_runs/anchor_vocab.npy`
  - 场景目录：`/root/autodl-tmp/train_dataset`
  - 输出目录：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958.log`
  - epochs: 2, batch_size: 8, lr: 1e-5, beta: 0.1, sft_weight: 0.05, num_t_samples: 4, lora_rank: 4, lora_alpha: 16
- 状态：
  - 2026-04-26 19:58 CST 开始。
  - 启动 PID：24227。
  - 记录时仍在运行。
- 决策规则：
  - 如果 train accuracy/delta 明显优于 v1，并且 500 quick eval 改善 predicted_anchor_rerank 且不伤 oracle，则跑 2k eval。
  - 如果仍接近随机或伤害 quick eval，则停止扩大 DPO，转向 selector/reranker/predictor quality。

## 更正：anchor_dpo_train2k_gap0p15_v2 launch 20260426

- `anchor_dpo_train2k_gap0p15_v2_e2_20260426_1958` 的第一次 launch 记录无效，因为 non-interactive shell 没有激活 conda；日志显示 `nohup: failed to run command python: No such file or directory`。
- 修正后的 launch：
  - 输出目录：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001.log`
  - 修正后启动 PID：24390
  - 状态：running；已到 Epoch 1/2，每个 epoch 170 steps。
- 解释：
  - `1958` 记录仅作为失败 launch attempt。
  - `2001` run 作为有效的 v2 DPO pilot。

## 更新：anchor-DPO pair 语义问题 20260426

- 用户担心：safe-vs-safe pairs 可能有噪声，因为两条轨迹都可能可接受。对这些 pair 做 hard DPO 可能会错误地把可用的安全轨迹推成 `rejected`。
- 已确认当前 generator 行为：
  - 如果 same-anchor candidates 同时包含 safe 和 collided trajectories：选择 best safe vs worst collided，标签为 `same_anchor_collision`。
  - 如果所有 candidates 都 safe：当 `quality_gap >= min_quality_gap` 时，选择 max-quality vs min-quality，标签为 `same_anchor_quality`。
  - Quality 使用 `safe_bonus = 100 * (1 - collided)`，所以 collision pairs 的 score gap 大约为 100，自然通过 `min_score_gap=0.15`；0.15 阈值主要过滤 safe-vs-safe quality pairs。
- Train2k v2 score-gap 分布：
  - `same_anchor_collision`: 567 pairs, gap min 99.5767, median 100.0292, max 100.5446.
  - `same_anchor_quality`: 3210 pairs, gap min 0.0500, median 0.1005, p75 0.1498, p90 0.2245, max 0.8520.
  - With `min_score_gap=0.15`, kept 1367 pairs = 567 collision + 800 quality.
- 有效 v2 mixed-pair 训练结果：
  - Run：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_gap0p15_v2_e2_20260426_2001.log`
  - Epoch 1: loss 0.7459, acc 47.79%, delta -0.0003.
  - Epoch 2: loss 0.7458, acc 44.85%, delta -0.0008.
  - Epoch 2 按标签统计：collision acc 44.86%，quality acc 44.85%。
  - 解释：信号弱/接近随机；不要把它当作正向 anchor-DPO 结果。
- Collision-only ablation：
  - 过滤后 preference 文件：`/root/autodl-tmp/Flow-Planner/dpo_data/anchor_conditioned/preferences/same_anchor_train2k_collision_only_v2_20260426_2010.npz`
  - 过滤条件：`dim_labels == same_anchor_collision`，567 pairs。
  - 无效 launch：`anchor_dpo_train2k_collision_only_v2_e2_20260426_2010` 因为缺少 `PYTHONPATH=/root/autodl-tmp/Flow-Planner-anchor-runtime`，导入了非 anchor model code。它有 `INVALID.txt`，不要使用。
  - 有效 run：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_dpo_train2k_collision_only_v2_e2_20260426_2015`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_dpo_train2k_collision_only_v2_e2_20260426_2015.log`
  - 正确 runtime 检查：model loaded 15.187M params；LoRA 包含 anchor encoder/cross-attn modules；75 LoRA layers。
  - Epoch 1: loss 0.7458, acc 50.00%, delta -0.0002.
  - Epoch 2: loss 0.7458, acc 51.61%, delta 0.0001.
  - 解释：仅靠更干净的标签没有产生强 DPO 信号；当前 DPO objective/log-prob estimate 在 same anchor 下几乎无法区分 chosen 和 rejected。
- 决策：
  - safe-vs-safe hard pairs 暂时不应作为同等强度的正式 DPO labels。
  - 当前 anchor-DPO 视为诊断实验，不是 deployment-ready。
  - 下一优先级是在扩大数据前，检查为什么 same-anchor collision pairs 下 DPO log-prob deltas 几乎为 0。
  - 候选修复方向：更强/更低噪声的 DPO signal、每个 anchor 采更多样本、更大的 collision-pair set、对 safe-vs-safe pairs 做 gap/confidence-weighted 或 soft preference，并且只有 train deltas 有意义后才做正式 eval。

## 设计记录：来自 goal 分支的 soft preference distillation 20260426

- Goal 分支参考 commit：`6356744 Wrap up goal-line: soft preference distill + DriveDPO-style hard negatives + anchor notes in GOAL_DESIGN`。
- `origin/feature/goal` 中的相关文件：
  - `flow_planner/dpo/SOFT_PREF_DISTILL.md`
  - `flow_planner/dpo/train_soft_pref.py`
  - `flow_planner/dpo/build_multi_pairs.py`
  - `flow_planner/goal/GOAL_DESIGN.md`
- 核心思路：用 scene-level candidate distribution learning 替代纯 `best vs worst` hard-pair learning。
  - 每个 scene 生成 K 个 candidates。
  - 用 structured metrics 给每个 candidate 打分。
  - 将 candidate scores 转成 teacher soft target distribution：`q_i = softmax(u_i / T)`。
  - 用 `KL(q || p_theta)` / cross entropy 训练 policy candidate probabilities：`p_theta(i) = softmax(log pi_theta(tau_i | condition_i))`。
  - 可选项：teacher top-1 log-prob anchor 和 reference KL，用来控制 drift。
- Goal 实现细节：
  - Teacher logit 使用 z-scored GT similarity 加 z-scored structured scorer value：`u_i = gt_weight * z(gt_sim_i) + score_weight * z(score_i)`。
  - `train_soft_pref.py` 当前通过 `attach_goal_to_decoder_inputs` 附加 `goal_labels`。
  - `build_multi_pairs.py` 也通过选择 `strict_same_group`、`gt_near_unsafe`、`chosen_near_unsafe`、`same_group_soft`、`cross_group_soft` 和 fallback hard failures 来改进后续 hard DPO。
- Anchor 适配需求：
  - 当前 anchor runtime 中有 `train_soft_pref.py`，但仍是 goal-oriented；它尚未消费 `anchor_trajs`，也没有调用 `attach_anchor_to_decoder_inputs`。
  - 不要直接把它用于 anchor soft preference。
  - 需要一个 anchor candidate artifact，保留每个 scene 的所有 candidates，而不只是 mined chosen/rejected pairs：trajectories、per-candidate anchor trajectory、anchor index/rank、metrics 和 teacher score。
  - 然后在每个 candidate 自己的 anchor condition 下实现 anchor soft-pref loss。
- 对当前 anchor-DPO 问题的解释：
  - 这是处理 safe-vs-safe ambiguity 的正确方向：不要强行把每条可接受的安全轨迹作为 hard rejected sample。
  - 对 quality differences 使用 soft distribution / confidence weighting，同时把明确的 safe-vs-collided pairs 保留为后续 hard negatives。

## 实验：anchor soft preference distillation smoke 20260426

- 目的：回应 safe-vs-safe hard DPO pairs 可能把可接受轨迹错误标为 `rejected` 的担心。这里不再使用 hard best-vs-worst pairs，而是在所有 anchor-conditioned candidates 上测试 scene-level soft ranking objective。
- 在 `/root/autodl-tmp/Flow-Planner-anchor-runtime` 中新增的运行时代码：
  - `flow_planner/dpo/generate_anchor_softpref_candidates.py`
  - `flow_planner/dpo/train_anchor_soft_pref.py`
- 方法：
  - 每个 scene 中，AnchorPredictor 提出 top-3 anchors。
  - 每个 anchor 下，planner 采样 3 条轨迹，因此每个 scene 有 9 个 candidates。
  - 每个 candidate 使用中等强度、safety-first 的 teacher score 打分，而不是早期的 100-point hard safety bonus。
  - Candidate score distribution 被转换成 9 个 candidates 上的 soft teacher target。
  - 用 `KL(q_teacher || p_policy)` 训练 decoder LoRA，其中 `p_policy = softmax(log_prob(candidate_i | scene, anchor_i))`。
- Smoke candidate generation，20 train scenes：
  - 输出：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_smoke20_20260426_2047`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_smoke20_20260426_2047.log`
  - 结果：20 / 20 scenes 写入成功，180 candidates，0 failures。
  - Candidate NPZ 包含 `candidates (9,80,4)` 和 `anchor_trajs (9,80,3)`。
  - Scene 平均 score stats：min 5.375，mean 5.892，max 6.294，std 0.366。
  - 每个 scene 的平均 collided candidates：0.55。
  - 平均 soft target entropy：1.744；平均 top probability：0.375。
- Smoke training，20 train scenes：
  - 输出：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_smoke20_e1_20260426_2048`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_smoke20_e1_20260426_2048.log`
  - 正确 anchor runtime 检查：LoRA 包含 `anchor_encoder` 和 `anchor_cross_attn`；merged checkpoint 移除了 150 个 LoRA side keys。
  - Epoch 1: loss 2.2457, top1 match 20.00%.
  - 解释：pipeline 可运行；样本太小，不能形成方法结论。
- Candidate generation，100 train scenes：
  - 输出：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050.log`
  - 结果：100 / 100 scenes 写入成功，900 candidates，0 failures。
  - 平均 score stats：min 5.328，mean 5.999，max 6.404，std 0.423。
  - Score std 分布：p10 0.031，median 0.059，p90 2.103。
  - 每个 scene 的平均 collided candidates：0.56；只有 18 / 100 scenes 有 collision candidate。
  - 平均 soft target entropy：1.810；平均 top probability：0.328。
  - Teacher top candidate safe rate：100%。
- Training，100 train scenes：
  - 输出：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_train100_e2_20260426_2052`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_train100_e2_20260426_2052.log`
  - Epoch 1: loss 2.2545, top1 match 11.00%.
  - Epoch 2: loss 2.2531, top1 match 9.00%.
  - 同 100 scenes 上的 offline diagnostic，`num_t_samples=2`：
    - Base: CE 2.2016, top1 9.0%, policy safe mass 0.9370, target safe mass 0.9891.
    - Soft100: CE 2.1988, top1 10.0%, policy safe mass 0.9381, target safe mass 0.9891.
  - 解释：完整 100-scene soft distillation 只带来很小的 CE/safe-mass 改善，不足以进入 eval。
- Informative-scene filter：
  - 新增 runtime filter args：`--min_score_std` 和 `--min_top_prob`。
  - 在 train100 candidates 上，`min_score_std=1.0` 保留 18 / 100 scenes，基本就是 collision-rich scenes。
- 过滤后训练，18 informative scenes：
  - 输出：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_train100_std1_e5_20260426_2055`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_train100_std1_e5_20260426_2055.log`
  - 设置：`min_score_std=1.0`，`target_temp=0.7`，`gt_weight=0.0`，`top1_weight=0.0`，`lr=2e-5`，epochs 5。
  - 最佳 top1 match：16.67%。
  - 同 18 scenes 上的 offline diagnostic：
    - Base: CE 2.1988, top1 11.11%, policy safe mass 0.6539, target safe mass 0.9669.
    - Filtered softpref: CE 2.1992, top1 27.78%, policy safe mass 0.6545, target safe mass 0.9669.
  - 解释：top1 能动一点，但 probability mass 没有转向 safe candidates。
- Overfit diagnostic，18 informative scenes：
  - 输出：`/root/autodl-tmp/Flow-Planner/checkpoints/dpo_outputs/anchor_conditioned/anchor_softpref_overfit_std1_e10_20260426_2100`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_overfit_std1_e10_20260426_2100.log`
  - 设置：`min_score_std=1.0`，`target_temp=0.7`，`gt_weight=0.0`，`top1_weight=1.0`，`lr=1e-4`，epochs 10。
  - 最佳 top1 match：22.22%。
  - Offline diagnostic：
    - Base: CE 2.1934, top1 0.00%, policy safe mass 0.6573, target safe mass 0.9669.
    - Overfit run: CE 2.1948, top1 22.22%, policy safe mass 0.6554, target safe mass 0.9669.
  - Checkpoint diff 确认权重发生变化，尤其是 anchor encoder 和 decoder LoRA-merged weights，因此弱结果不是简单的 save/load failure。
- 决策：
  - Anchor soft preference data pipeline 现在可用且有效。
  - 但是，用当前连续 flow-matching log-prob 作为 candidate-ranking objective 仍然太弱/太噪，不足以支撑 deployment eval。
  - 不对上面的 softpref checkpoints 跑 500/2k eval。
  - 下一技术优先级：诊断/重做 probability objective，或把 preference learning 上移到显式 anchor selector/reranker，让概率变成离散且可训练，更接近 DriveDPO-style soft distribution over anchors/candidates。

## 实验：discrete anchor selector soft preference 20260426

- 动机：
  - Continuous planner-DPO / soft preference 使用完整轨迹上的 flow-matching log-prob，但 same-anchor chosen/rejected deltas 几乎为 0。
  - 现有 `predicted_anchor_rerank_a` 有效，是因为它把问题转成了看过 planner-generated trajectories 后的离散 candidate selection。
  - 这个实验测试一个更轻量的中间方案：用从 generated top-k anchor candidates 聚合出来的 soft preference targets，fine-tune 离散 `AnchorPredictor` head。
- 在 `/root/autodl-tmp/Flow-Planner-anchor-runtime` 中新增的运行时代码：
  - `flow_planner/dpo/train_anchor_selector_softpref.py`
  - Patch 产物：`/root/autodl-tmp/anchor_runs/patches/anchor_selector_softpref_train_runtime.patch`
- 方法：
  - 复用 `generate_anchor_softpref_candidates.py` 生成的 softpref candidate artifacts。
  - 每个 scene 中，按 `anchor_index` 用 `mean` 或 `max` 聚合 candidate scores。
  - 用 `softmax(score / T)` 把 anchor scores 转成 sparse full-vocab teacher distribution。
  - 只在 `AnchorPredictor` head 上用 `CE(q_anchor, p_predictor)` 进行 fine-tune，并可选加入小权重 GT-nearest CE regularization。
  - 这不是 planner-DPO，而是 anchor-level 离散 selector/reranker 诊断。
- Train100 diagnostics：
  - Candidate artifact：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train100_20260426_2050`
  - `lr=0` baseline on 80/20 split, `mean`, `T=0.5`: val top1_match 34.4%, target_prob_on_pred 0.339.
  - High-lr selector: `/root/autodl-tmp/anchor_runs/anchor_selector_train100_mean_t0p5_e8_20260426_2108`; best early val top1_match 25.0%, later 9.4%.
  - Low-lr selector: `/root/autodl-tmp/anchor_runs/anchor_selector_train100_mean_t0p5_lr3e5_e8_20260426_2111`; best val top1_match stayed at baseline 34.4%, then dropped to 31.2%.
  - 解释：100 scenes 对这个 selector objective 来说太小/太噪；high lr 很快 overfit。
- Train500 candidate generation：
  - 输出：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train500_20260426_2113`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train500_20260426_2113.log`
  - 结果：500 / 500 scenes 写入成功，4500 candidates，0 failures。
  - Mean aggregation target stats：500 records，400 train / 100 val split，score_std_mean 0.3125，score_gap_mean 0.2383，top_prob_mean 0.4485，collision_scene_count 134。
- Train500 selector diagnostics：
  - `lr=0` baseline, `mean`, `T=0.5`: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_baseline_lr0_20260426_2129`; val top1_match 28.1%, target_prob_on_pred 0.324.
  - Low-lr selector, `mean`, `T=0.5`: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_mean_t0p5_lr3e5_e10_20260426_2130`; best val top1_match 26.6%, target_prob_on_pred about 0.330.
  - `max` aggregation baseline: `/root/autodl-tmp/anchor_runs/anchor_selector_train500_max_t0p5_baseline_lr0_20260426_2132`; val top1_match 25.8%, target_prob_on_pred 0.313.
  - Internal target-match metrics 没有优于原始 predictor，但这个指标并不完全等价于 deployment safety。
- Direct 500-val deployment smoke，使用与 rho=0.5 500 eval 相同的 manifest：
  - 输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train500_mean_500_20260426_2134`
  - Original `predicted_anchor_top1`: collision 3.2%, progress 0.3361, route 0.8468, collision_score 0.1289.
  - Selector `predicted_anchor_top1`: collision 2.0%, progress 0.3332, route 0.8480, collision_score 0.1256.
  - Original `predicted_anchor_rerank_a`: collision 4.6%, progress 0.3434, route 0.8657, collision_score 0.1266.
  - Selector `predicted_anchor_rerank_a`: collision 2.4%, progress 0.3387, route 0.8665, collision_score 0.1224.
- Direct 2k-val deployment smoke，manifest `/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`：
  - 输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train500_mean_2k_20260426_2137`
  - Baseline planner, no anchor: collision 5.45%, progress 0.3393, route 0.8592.
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Selector `predicted_anchor_top1`: collision 3.70%, progress 0.3185, route 0.8574.
  - Original `predicted_anchor_rerank_a`: collision 3.15%, progress 0.3293, route 0.8738.
  - Selector `predicted_anchor_rerank_a`: collision 3.35%, progress 0.3248, route 0.8768.
  - Oracle anchor: collision 2.20%, progress 0.3149, route 0.8580.
  - Oracle anchor rerank: collision 2.80%, progress 0.3309, route 0.8748.
- 解释：
  - Selector top1 有真实 safety signal：在 2k val 上把 predicted top1 collision 从 4.20% 改善到 3.70%，但牺牲了 progress。
  - Selector 加现有 hand rerank 在 2k 上还没有超过 original hand rerank：collision 3.35% vs 3.15%，虽然 route 略高。
  - Internal target-match metrics 偏悲观；selector variants 必须做 deployment eval。
  - 当前最佳部署选择仍是 rho=0.5 + original `predicted_anchor_rerank_a`，因为 safety/progress balance 最稳。
  - 到目前为止，最好的 learned preference signal 是 selector top1，而不是 planner-DPO；这说明后续 anchor-DPO 应该在 discrete anchors/candidates 上做，或训练 candidate-aware reranker，而不是只依赖 continuous planner log-prob。
- 下一步推荐实验：
  - 生成更大的 selector dataset，最好是 2k train scenes，并训练能看到 scene features + candidate anchor/trajectory features 的 candidate-aware reranker。
  - 保留 learned selector top1 作为正向但非最终结果。
  - 在 selector+rerank 在 2k manifest 上超过 3.15% collision 前，不替换当前 production rerank_a。

## 术语说明：DPO pair 类型与 flow-matching log-prob 20260426

- `mixed v2 DPO` 表示 preference dataset 同时包含两类 same-anchor pairs：
  - `same_anchor_collision`：chosen 是安全轨迹，rejected 是同一 scene、同一 anchor 下发生碰撞的轨迹。这是最清晰的 preference label。
  - `same_anchor_quality`：两个 candidates 都安全，但其中一个 structured score 更高。这个信号更弱，因为 rejected trajectory 可能仍然可接受；它不应被当作与 collision failure 同等强度的负样本。
  - 目的：测试把清晰 safety pairs 和较软的 safe-vs-safe quality pairs 结合起来，是否能训练出有用的 planner preference signal。
- `collision-only DPO` 表示丢弃所有 safe-vs-safe quality pairs，只在 `same_anchor_collision` pairs 上训练。
  - 目的：隔离最干净的信号。如果它有效但 mixed 失败，safe-vs-safe labels 可能有噪声；如果它也失败，瓶颈更可能是 DPO objective / log-prob estimate 本身。
  - 结果：collision-only 也只达到接近随机的 pair accuracy，说明仅靠更干净的标签还不够。
- 这里的 `DPO acc` 不是驾驶准确率或轨迹准确率，而是 pairwise preference accuracy：
  - 对每个 pair，计算模型是否给 `chosen` 比 `rejected` 更大的相对 likelihood / DPO margin。
  - 大约 50% 表示模型基本无法区分 preferred trajectory 和 rejected trajectory。
  - 低于 50% 可能来自估计噪声，也可能是更新方向被噪声带偏。
- `continuous flow-matching log-prob` 解释：
  - Flow Planner 是连续轨迹生成模型，通过 flow matching / diffusion-like denoising 训练，不是小离散集合上的普通分类器。
  - DPO 需要类似 `log pi_theta(trajectory | scene, condition)` 的数值，才能判断 chosen 是否比 rejected 更可能。
  - 对连续 flow-matching model 来说，这个 log-prob 只能通过 candidate trajectory 上的 denoising/flow-matching loss 近似，且经常要采样 noise/timesteps。
  - 这个估计可能很噪，而且对两条相似轨迹给出的差异非常小，尤其当二者使用同一个 anchor 时。
  - 在我们的实验里，chosen/rejected margins 基本接近 0，所以训练目标不能可靠判断哪个 candidate 应该赢。
- 实际解释：
  - 弱结果并不说明 anchor conditioning 没用。Oracle anchor 和 predicted-anchor rerank 证明 anchor candidate space 有价值。
  - 它说明：使用当前 continuous trajectory log-prob estimate 的 planner-level DPO 还不是可靠的 preference learner。
  - 因此后续 selector 实验把 preference learning 移到 discrete anchor/candidate selection problem 上，那里的概率更干净，也更容易训练。

## 实验：anchor selector train2k mean soft target 20260426

- 目的：验证 learned anchor selector 从 500 train scenes 扩大到 2000 train scenes 后，是否能改善固定 2k val manifest 上的 deployment collision。
- Candidate generation：
  - 输出：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153`
  - 日志：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153.log`
  - 结果：2000 / 2000 train scenes，18000 candidates，0 failures。
  - Candidate structure：每个 scene top3 predicted anchors，每个 anchor 3 planner samples。
- Selector training：
  - Baseline run：`/root/autodl-tmp/anchor_runs/anchor_selector_train2k_baseline_lr0_20260426_2226`
  - Train run：`/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227`
  - Target stats：2000 records，1600 train / 400 val，score_std_mean 0.2936，score_gap_mean 0.2528，top_prob_mean 0.4400，collision_scene_count 482。
  - Baseline internal val：top1_match 30.4%，target_prob_on_pred 0.364。
  - Train internal val best：epoch 3 约 top1_match 31.5%；epoch 10 最终 top1_match 30.1%，target_prob_on_pred 0.346。
- Deployment eval：
  - 输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_train2k_mean_2k_20260426_2235`
  - Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Case：`predicted_anchor_top1_selector_train2k`
  - 结果：collision 3.55%，progress 0.3187，route 0.8545，collision_score 0.1062，scenes 2000 / 2000。
- 同一 2k manifest 上的对比：
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Selector train500 top1: collision 3.70%, progress 0.3185, route 0.8574.
  - Selector train2k top1: collision 3.55%, progress 0.3187, route 0.8545.
- 解释：
  - selector data 从 500 扩到 2000 train scenes 后，带来小幅额外 safety gain：collision 3.70% -> 3.55%。
  - 相比原始 predictor top1，learned selector 绝对降低 collision 0.65 个百分点，但仍牺牲 progress。
  - 它仍然没有超过 original `predicted_anchor_rerank_a` 的 3.15% collision，所以 train2k selector 是正向 learned-signal 结果，但不是当前最佳部署方法。

## 实验：selector-DPO all-pairs pilot 20260426

- 目的：测试直接在 discrete selector scores 上做 DPO，是否能比 soft-CE selector 进一步改善 learned anchor selection。
- 脚本：`/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_anchor_selector_dpo.py`
- 分支同步前应保存的 runtime code patch artifact：`/root/autodl-tmp/anchor_runs/patches/anchor_selector_dpo_train_runtime.patch`
- Pair 构造：
  - Source candidates：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir`
  - 聚合方式：每个 anchor 的 mean score。
  - Pair mode：所有 `score_gap >= 0.05` 的 ordered anchor pairs。
  - Policy init：soft selector train2k `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227/anchor_selector_best.pth`
  - Reference：original anchor predictor `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
- 训练输出：
  - `/root/autodl-tmp/anchor_runs/anchor_selector_dpo_train2k_all_gap0p05_e10_20260426_2256`
  - Pairs: 3020 total, 2416 train / 604 val.
  - Labels: 2024 `anchor_quality`, 746 `anchor_collision`, 250 `anchor_collision_rate`.
  - 训练确实学到了 pair objective：val pair acc 提升到约 61%，val margin 约 0.29。
- Deployment eval：
  - 输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_train2k_all_2k_20260426_2300`
  - Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - 结果：collision 5.65%，progress 0.3516，route 0.8508。
- 解释：
  - 这是 all-pairs selector-DPO 的负结果。
  - 模型学到了 offline pair labels，但 deployment safety 明显差于 original top1 4.20% 和 soft selector train2k 3.55%。
  - 可能问题：all-pairs DPO 过度强调 `anchor_quality` / progress-like differences，并把弱 safe-vs-safe preferences 当作 hard pair constraints。
  - 下一步：切到 collision-only selector-DPO，只使用最清晰的 safety pairs。

## 实验：selector-DPO collision-only pilot 20260426

- 目的：在 all-pairs selector-DPO 的 deployment safety 失败后，测试只在 clean safety pairs 上做 DPO 是否能改善 learned selector top1。
- 脚本：`/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/train_anchor_selector_dpo.py`
- Patch 产物：`/root/autodl-tmp/anchor_runs/patches/anchor_selector_dpo_train_runtime.patch`
- Pair 构造：
  - Source candidates：`/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir`
  - 聚合方式：每个 anchor 的 mean score。
  - Pair mode：所有 `score_gap >= 0.05` 的 ordered anchor pairs。
  - 过滤：`--require-collision-pair`，只保留 `anchor_collision` 和 `anchor_collision_rate` pairs。
  - Policy init：soft selector train2k `/root/autodl-tmp/anchor_runs/anchor_selector_train2k_mean_t0p5_lr3e5_e10_20260426_2227/anchor_selector_best.pth`
  - Reference：original anchor predictor `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
- 训练输出：
  - `/root/autodl-tmp/anchor_runs/anchor_selector_dpo_train2k_collision_gap0p05_e10_20260426_2302`
  - Pairs: 996 total, 797 train / 199 val.
  - Labels: 746 `anchor_collision`, 250 `anchor_collision_rate`.
  - Internal val pair acc 约 67%；val margin 增长到约 0.62。
- Deployment eval：
  - 输出：`/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_collision_train2k_2k_20260426_2305`
  - Manifest：`/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`
  - Case：`predicted_anchor_top1_selector_dpo_collision_train2k`
  - 结果：collision 3.15%，progress 0.3150，route 0.8549，collision_score 0.1080，scenes 2000 / 2000。
- 同一 2k manifest 上的对比：
  - Original `predicted_anchor_top1`: collision 4.20%, progress 0.3253, route 0.8548.
  - Soft selector train2k top1: collision 3.55%, progress 0.3187, route 0.8545.
  - Selector-DPO all-pairs top1: collision 5.65%, progress 0.3516, route 0.8508.
  - Selector-DPO collision-only top1: collision 3.15%, progress 0.3150, route 0.8549.
  - Original hand `predicted_anchor_rerank_a`: collision 3.15%, progress 0.3293, route 0.8738.
- 解释：
  - 这是第一个强 selector-DPO 结果：learned top1 selector 在 inference 不使用 hand rerank 的情况下，达到了 hand-rerank 的 collision rate。
  - 代价是 progress 和 route 低于 hand-rerank，因此还不是严格更好的部署方法。
  - 重要经验：selector-DPO 应优先使用 clean safety preference pairs；包含大量弱 quality pairs 的 all-pairs DPO 即使提升 offline pair accuracy，也可能伤害 safety。
  - 下一方向：改进 selector-DPO objective 来恢复 progress/route，或转向 candidate-level learned selector，更直接地替代 hand-rerank。

## 实验：experiment record utility 20260426

- 状态：completed
- 目的：
  - 不再只依赖手动 markdown edits 记录 anchor experiments。
- 方法：
  - 添加 dependency-free append helper，字段包括：目的、数据、方法、产物、结果、Eval JSON 结果、解释、下一步。
- 产物：
  - runtime script：/root/autodl-tmp/Flow-Planner-anchor-runtime/scripts/anchor/record_anchor_experiment.py
  - branch target：origin/feature/anchor 上的 scripts/anchor/record_anchor_experiment.py
- 结果：
  - local 和 remote dry-run 通过。
- 解释：
  - 后续 anchor experiments 应通过这个 helper 记录，或手动遵循相同字段顺序。
- 下一步：
  - 每次 training/eval run 后立即使用这个脚本追加可复用到论文的记录。

## 实验：20260426 selector-DPO ablation A：collision-only beta0.2 sft0.10

- 目的：
  - 验证更保守的 collision-only anchor-selector-DPO 是否能在保持安全性的同时恢复 progress/route。
- 数据：
  - 使用 train2k softpref candidate scored_dir 构造 collision-only anchor pairs；固定 2k val manifest eval_manifest_2k_seed3402.json。
- 方法：
  - 以 train2k soft-CE selector 为 init，以原始 anchor predictor 为 reference；DPO beta=0.2，SFT weight=0.10，epochs=10；部署时 predicted_anchor top1，不使用手写 rerank。
- 产物：
  - /root/autodl-tmp/anchor_runs/anchor_selector_dpo_collision_b0p2_sft0p10_e10_20260426_2315
- Eval JSON 结果：
  - `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_b0p2_sft0p10_2k_20260426_2320/predicted_anchor_top1_selector_dpo_b0p2_sft0p10/predicted_anchor_top1_selector_dpo_b0p2_sft0p10.json` -> collision 3.7500, progress 0.3133, route 0.8552, collision_score 0.1066, scenes 2000, failed 0
- 解释：
  - 负结果：collision=3.75%，不如前一版 collision-only selector-DPO 的 3.15%，progress=0.3133 也未恢复；说明这个保守项组合没有带来预期 tradeoff。
- 下一步：
  - 继续跑 beta/sft ablation，优先尝试更小 beta 或不同 SFT 权重，寻找安全与 progress/route 的折中。

## 实验：20260426 selector-DPO ablation B：collision-only beta0.1 sft0.10

- 目的：
  - 验证更小 beta 的 collision-only anchor-selector-DPO 是否能接近 3.15% safety，同时恢复 progress/route。
- 数据：
  - 使用 train2k softpref candidate scored_dir 构造 collision-only anchor pairs；固定 2k val manifest eval_manifest_2k_seed3402.json。
- 方法：
  - 以 train2k soft-CE selector 为 init，以原始 anchor predictor 为 reference；DPO beta=0.1，SFT weight=0.10，epochs=10；部署时 predicted_anchor top1，不使用手写 rerank。
- 产物：
  - /root/autodl-tmp/anchor_runs/anchor_selector_dpo_collision_b0p1_sft0p10_e10_20260426_2325
- Eval JSON 结果：
  - `/root/autodl-tmp/anchor_runs/deploy_eval_anchor_selector_dpo_b0p1_sft0p10_2k_20260426_2330/predicted_anchor_top1_selector_dpo_b0p1_sft0p10/predicted_anchor_top1_selector_dpo_b0p1_sft0p10.json` -> collision 3.2500, progress 0.3166, route 0.8563, collision_score 0.1085, scenes 2000, failed 0
- 解释：
  - 折中结果：collision=3.25%，略差于原 collision-only selector-DPO 的 3.15%，但 progress=0.3166、route=0.8563 略有恢复；说明降低 beta 可以缓和安全/route-progress tradeoff，但提升幅度很小。
- 下一步：
  - 保留原 collision-only selector-DPO 作为当前 safety best；beta0.1+sft0.10 作为保守折中点。下一步优先做可视化和 candidate-level selector 设计，而不是继续盲目扫小超参。

## 实验：20260426 anchor candidate visualization：complex mixed-collision scene

- 目的：
  - 给 anchor preference/selector-DPO 找一个可解释的复杂场景可视化，用于理解训练信号和后续论文图。
- 数据：
  - 场景 us-ma-boston_0d5ec00a9c025a5f；来自 train2k softpref candidates；32 个有效邻车、70 条 lane；top3 anchors x 每个 anchor 3 samples，共 9 条 candidate。
- 方法：
  - 从 scored candidates 中筛选 mixed-collision case：同一 scene 内既有 safe candidate 又有 collided candidates；选择最高分 safe 作为 chosen，最低分 collided 作为 rejected；使用 BEVRenderer 渲染 all-candidates 和 chosen-vs-rejected 两版。
- 产物：
  - /root/autodl-tmp/anchor_runs/visualizations/anchor_selector_case_20260426_2335
- 结果：
  - 该 scene 中 9 条 candidates 有 8 条 collided、1 条 safe；chosen 为原始 candidate idx=5，score=6.67；rejected 为原始 candidate idx=3，score=1.34。
- 解释：
  - 这个图直观说明 anchor-level selector 只是在 scene 级选择更可能产生安全轨迹的 anchor；同一个 anchor 下仍可能有好/坏采样差异，因此后续 candidate-level selector 有必要。
- 下一步：
  - 保留该图作为 anchor-DPO 动机图候选；下一步设计 candidate-level selector，直接对 scene+anchor+trajectory 打分。

## 实验：20260426 night closeout：sample-size diagnostic and tomorrow plan

- 目的：
  - 睡前确认 anchor-DPO 下一步是否继续扫 anchor-selector 超参，还是转向 candidate-level selector；同时客观检查当前样本量是否太小。
- 数据：
  - 使用 train2k softpref candidates: 2000 train scenes，每个 scene 为 top3 anchors x 每个 anchor 3 samples，共 18000 candidate trajectories。
- 方法：
  - 不训练新模型，只统计 candidate pool 的 safety upper bound / oracle bound；统计 scene 内 safe candidate 覆盖、rank0 anchor safe 覆盖、best-score candidate collision，以及 anchor group all-safe/all-collide 比例。
- 结果：
  - raw candidate collision=8.44%; scene split=1518 all-safe / 458 mixed / 24 all-collide; 98.8% scenes have at least one safe candidate; candidate oracle best-score collision=1.2%; best-mean-anchor+best-sample collision=1.2%; rank0 any-safe=97.0%, rank0 all-collide=3.0%; 6000 anchor groups: any-safe=96.32%, all-safe=85.75%, all-collide=3.68%; score gap mean=1.317, median=0.215; mixed safe-vs-best-collision gap mean=5.005, median=4.984.
- 解释：
  - 样本量确实偏小：2000 scenes / 18000 candidates 只够方向判断和 pilot，不足以支撑 paper-facing robust conclusion。但候选池有明显潜力：candidate oracle 1.2% collision 明显低于 anchor-selector-DPO 和 hand rerank 的 3.15%，说明瓶颈主要在 selector 没有充分利用同一 scene 下多条 candidate，而不是 planner 完全生成不出安全轨迹。
- 下一步：
  - 明天 TODO: 1) anchor 细节只维护在 feature/anchor，不再进 main；2) 实现 candidate-level dataset，读取 scored_dir + candidate npz，构造 same-scene candidate pairs，优先 collision/mixed pairs，避免 safe-vs-safe 小 gap 强行偏好；3) 实现 candidate-level selector v0: scene feature + anchor id/embedding + trajectory feature -> candidate score；4) 先跑 100-200 scenes smoke；5) 再跑 train2k pilot 并评估 2k manifest，对比 original top1 / soft selector / collision-only anchor-selector-DPO / hand rerank_a；6) 只有 pilot 有正信号才扩大 candidate generation。

## 架构补充：GoalFlow 对齐后的 scorer 与 pipeline 决策 20260427

- 背景：
  - GoalFlow 的架构启发不是要回到 2D goal，而是学习它把 scorer/selector 和 flow planner 解耦的工程组织方式。
  - 我们已经完成 anchor-level selector、selector-DPO、candidate pool、candidate oracle diagnostic 和可视化；原有 candidate-level TODO 仍然保留，不在本节覆盖或改写。
- 决策 1：把 scorer 正式模块化。
  - 当前 scoring 信号分散在 hand `rerank_a`、softpref candidate score、selector-DPO pair construction、candidate oracle diagnostic 中。
  - 后续应整理成统一的 `AnchorCandidateScorer` / `TrajectoryCandidateScorer` 风格模块，至少显式输出：
    - `collision_score`
    - `route_score`
    - `progress_score`
    - `comfort_score`
    - `final_score`
  - `final_score` 只作为 teacher / selector target / ablation 依据，不直接包装成创新点；创新点仍是 learned anchor/candidate selector。
  - 这样做不会改变现有 hand `predicted_anchor_rerank_a` 结论，只是把它背后的 scoring 逻辑标准化，方便复现和替换。
- 决策 3：把两阶段流程正式化。
  - 当前已经有 candidate generation、scored candidates、selector training、deployment eval，但入口还偏实验脚本化。
  - 后续建议按 GoalFlow 的 generate-then-score 思路固定为四步：

```text
generate_anchor_candidates
  -> score_anchor_candidates
  -> train_anchor_or_candidate_selector
  -> eval_selector_deployment
```

  - 推荐数据目录保持为：

```text
dpo_data/anchor_conditioned/candidates/
dpo_data/anchor_conditioned/scored_candidates/
dpo_data/anchor_conditioned/preferences/
```

  - 这只是把已有实验链路规范化，不改变原 TODO 中的 candidate-level dataset / selector v0 任务；相反，它们会作为 `train_anchor_or_candidate_selector` 这一步的具体实现。
- 对原 TODO 的影响：
  - 原 TODO 第 2/3 项（candidate-level dataset 与 candidate-level selector v0）仍然是下一步主任务。
  - 本节新增的 scorer 模块化与两阶段 pipeline 是支撑工程项，目的是让 candidate-level selector 的训练数据、teacher score、eval 入口更清晰。
  - 不应因为本节新增内容而重新开启 planner-level DPO 大规模训练；planner-level DPO 仍暂停，selector/candidate-level preference 仍是主线。

## 代码落地：scorer 模块化与 score stage 20260427

- 目的：
  - 将已经分散在 softpref generation、anchor selector soft-CE、selector-DPO pair mining 中的 candidate score 逻辑抽成可复用模块。
  - 支持不重新跑 planner 的情况下，对已有 anchor candidate NPZ 重新打分，形成更清晰的 generate-then-score pipeline。
- 代码变更：
  - 新增 `flow_planner/dpo/anchor_candidate_scorer.py`
    - `AnchorCandidateScoreWeights`
    - `score_components`
    - `build_candidate_record`
    - `aggregate_anchor_scores`
    - `summarize_anchor_groups`
    - `summarize_scene`
    - `pair_label`
  - 新增 `flow_planner/dpo/score_anchor_candidates.py`
    - 输入已有 `*_candidates.npz`
    - 读取对应 scene npz
    - 输出 scored candidate JSON 与 `score_meta.json`
  - 更新 `generate_anchor_softpref_candidates.py`
    - 复用公共 scorer
    - 保留旧字段 `total_score`
    - 新增 `score_components`、`scene_stats`、`anchor_group_stats`
  - 更新 `train_anchor_selector_softpref.py`
    - 复用公共 `aggregate_anchor_scores`
  - 更新 `train_anchor_selector_dpo.py`
    - 复用公共 `summarize_anchor_groups` 和 `pair_label`
- 行为兼容性：
  - `total_score` 的公式保持与原 `_teacher_score` 一致，因此旧的 soft selector / selector-DPO 训练语义不应改变。
  - 新增字段是为了 diagnostics 和后续 candidate-level selector，不替换原有 JSON 字段。
- 验证：
  - Cursor lints: no errors.
  - 本机 Windows 只有 WindowsApps `python.exe` stub，`python` / `py` 不能执行真实 `py_compile`；语法级运行检查需要在 AutoDL/conda 环境补跑。
- 对原 TODO 的影响：
  - 不改变 candidate-level dataset / selector v0 的优先级。
  - 这次代码只是让 candidate-level selector 后续能直接读取标准化 score components、scene stats 和 anchor group stats。

## Experiment: 20260427 candidate-level selector smoke (launch record)

- Status: running
- Goal:
  - 验证 learned candidate-level selector 是否能替代手写 rerank，在 top-k anchor x multi-sample candidates 中直接选出更安全的轨迹。
- Data:
  - 使用已有 anchor softpref scored candidates 作为训练输入；优先复用 train2k 对应 scored_dir。
  - 部署评测固定使用 2k val manifest eval_manifest_2k_seed3402.json。
- Method:
  - 新增 scene + anchor_traj + candidate_traj -> score 的 candidate selector。
  - 先跑小规模 smoke / pilot，直接学习 scene 内 soft target，不先上 candidate-level DPO。
  - 部署时使用 predicted_anchor_candidate_selector，从 top-k anchors x samples_per_anchor 候选中由 learned selector 选轨迹。
- Artifacts:
  - Record repo/worktree: /root/autodl-tmp/Flow-Planner
  - Runtime repo/worktree: /root/autodl-tmp/Flow-Planner-anchor-runtime
  - Planned train output: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_smoke_20260427
  - Planned eval output: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_smoke_20260427
- Artifacts Update:
  - Train ckpt: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_smoke_20260427/anchor_candidate_selector_best.pth
  - Train log: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_smoke_20260427/train.log
  - 500-scene eval JSON: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_smoke_20260427/predicted_anchor_candidate_selector_smoke.json
- Partial Results:
  - 500-scene smoke deployment eval: collision 3.60%, progress 0.3427, route 0.8338, scenes 500 / 500, failed 0.
- Interim Interpretation:
  - candidate-level selector 没有出现明显 safety 崩盘，500-scene smoke 已经能跑通闭环；但 route 明显偏低，是否值得扩到正式训练还要看同配置 2k eval。
- Final Smoke Results:
  - 2k-manifest deployment eval: collision 4.45%, progress 0.3384, route 0.8425, scenes 2000 / 2000, failed 0.
- Comparison Target:
  - same-manifest references: original predicted_anchor_top1 4.20% / 0.3253 / 0.8548; soft selector train2k top1 3.55% / 0.3187 / 0.8545; collision-only selector-DPO 3.15% / 0.3150 / 0.8549; hand rerank 3.15% / 0.3293 / 0.8738.
- Final Smoke Interpretation:
  - 2k smoke result => collision 4.45%, progress 0.3384, route 0.8425.
  - auto decision: hold 2k candidate-selector training and inspect before scaling.
- Pending:
  - 若自动起了 2k training，待补 train2k 运行结果与后续部署 eval。
- Next:
  - 依据 2k smoke 结果决定继续扩大训练还是先修 selector 行为。


## Experiment: 20260427 candidate-level selector sharp target T0p05 (launch record)

- Status: running
- Goal:
  - 验证 candidate-level selector smoke 失败是否主要来自 soft target 太平，而不是模型链路本身不可用。
- Diagnosis From Previous Smoke:
  - T=0.5 soft target 的 top candidate 平均概率只有 0.156，接近 9 candidates 均匀随机的 0.111。
  - train/val loss 贴近 log(9)=2.197，val top1_match 约 0.107，基本等于随机排序。
  - 2k eval 结果为 collision 4.45%, progress 0.3384, route 0.8425；弱于 original predicted_anchor_top1 4.20% 和 soft selector train2k 3.55%。
- Data:
  - train scored_dir: /root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir
  - train subset: max_scenes=500
  - eval subset: eval_manifest_2k_seed3402.json 的前 500 scenes
- Method:
  - 仍用 candidate-level soft CE，但把 target_temp 从 0.5 降到 0.05，让 top target 平均概率提升到约 0.414。
  - 如果 sharp target 仍学不动，下一步应转向 hard pair / candidate-level DPO 或加入 online structured features，而不是继续单纯扩大训练样本。
- Artifacts:
  - Runtime repo/worktree: /root/autodl-tmp/Flow-Planner-anchor-runtime
  - Train output: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_sharpT0p05_train500_20260427
  - Eval output: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_sharpT0p05_train500_20260427
- Results:
  - 500-scene eval JSON: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_sharpT0p05_train500_20260427/predicted_anchor_candidate_selector_sharpT0p05_train500_500.json
  - 500-scene sharp-target deployment eval: collision 13.20%, progress 0.3985, route 0.8278, scenes 500 / 500, failed 0.
- Conclusion:
  - 把 target 从 T=0.5 压到 T=0.05 虽然显著提升了 offline target match，但部署 safety 明显崩坏；说明当前 teacher score 的排序噪声会被 sharp target 放大。
- Decision:
  - 停止继续 sharp-target soft CE 路线，转向更干净的 candidate-level hard pair supervision。

## Experiment: 20260427 candidate-level pairwise selector same-anchor train2k pilot (launch record)

- Status: running
- Goal:
  - 验证 candidate-level hard pair supervision 是否能比 soft CE 更稳定地学会安全排序。
- Data:
  - source scored_dir: /root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir
  - pair scope: same_anchor only
  - pair reduce: one best-safe vs worst-collided pair per mixed anchor
  - pair yield: 634 compressed same-anchor safety pairs from 436 mixed scenes within 2000 scored scenes
- Method:
  - candidate selector 输入仍为 scene + anchor_traj + candidate_traj
  - supervision 改为 pairwise logistic ranking loss，不再模仿软 teacher 分布
  - 当前 pilot 先做 train2k pairwise 训练，再看 offline pair_acc 和后续部署 eval 是否优于 soft CE 路线
- Artifacts:
  - Runtime repo/worktree: /root/autodl-tmp/Flow-Planner-anchor-runtime
  - Train output: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_train2kpilot_20260427
- Partial Results:
  - pairwise train2k pilot offline: 634 same-anchor pairs, 507 train / 127 val, best val pair_acc 0.654.
  - 500-scene pairwise deployment eval: collision 3.40%, progress 0.3378, route 0.8379, scenes 500 / 500, failed 0.
- Results:
  - 2k eval JSON: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_pairwise_sameanchor_train2kpilot_2k_20260427/predicted_anchor_candidate_selector_pairwise_sameanchor_2k.json
  - 2k deployment eval: collision 3.60%, progress 0.3343, route 0.8514, scenes 2000 / 2000, failed 0.
- Conclusion:
  - same-anchor hard pair supervision 明显优于 soft CE candidate selector（4.45% -> 3.60%），并且优于原始 predicted_anchor_top1（4.20%）。
  - 这条线已经接近旧的 anchor-level soft selector train2k（3.55%），说明 candidate-level scorer 的主要问题确实在监督，而不在链路本身。
- Decision:
  - 继续扩大 clean same-anchor pair 数，优先从 634 压缩版扩到 1268 all-pairs 版本；之后再看是否值得做 4-3-2 / 5-2-2 的 candidate budget allocation。

## Experiment: 20260428 candidate-level pairwise selector same-anchor all-pairs train2k pilot (launch record)

- Status: running
- Goal:
  - 在压缩版 same-anchor hard pair 已优于 soft CE 的前提下，把监督从 634 对扩到完整 same-anchor safety pairs，验证更多 clean pairs 是否能进一步改善 2k manifest collision。
- Data:
  - source scored_dir: /root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir
  - pair scope: same_anchor only
  - pair reduce: all safe-vs-collided pairs within the same anchor
  - expected pair pool: about 1268 pairs from 2000 scored scenes
- Method:
  - 保持 candidate-level pairwise logistic ranking loss 不变，只扩大 clean pair 数量。
  - 训练后直接评测同一 2k manifest，和 634-pair pilot / soft CE / anchor-selector baselines 做 apples-to-apples 比较。
- Artifacts:
  - Runtime repo/worktree: /root/autodl-tmp/Flow-Planner-anchor-runtime
  - Train output: /root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_20260428
  - Eval output: /root/autodl-tmp/anchor_runs/deploy_eval_anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_20260428
- Pending:
  - 待补 train history、2k eval JSON、结论。


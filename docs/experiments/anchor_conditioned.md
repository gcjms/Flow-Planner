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

---

## TODO 总清单与发表路线图（20260428 同步）

> 这一节是对话整理结果，整合三件事：
> (1) work log §7 还在 open 的旧 TODO；
> (2) 与 Diffusion-Planner / Hyper-Diffusion-Planner 比较后新冒出的工程项；
> (3) 投稿前必须补齐的实验/写作 gap。
> 后续每完成一项，请在该条目后追加 `done @<日期>` 或独立小节。
> 失败/弃坑也要写明，不要悄悄删。

### 关键事实校正（先记下来，避免下次被自己误导）

- 之前对话/汇报里出现的 "原始 FlowPlanner = 2.0%" 这个数字，**含义是 500-scene 独立跑（`run_anchor_raw_no_goal_eval.sh`）**，不是 2k manifest。
- 在**同 manifest 同 seed 的 500-scene full eval suite** 上：原始 FP = **2.4%**、`oracle_anchor` = **2.0%**、`predicted_anchor_rerank_a` = **6.2%**。
- 在 **2k manifest（`eval_manifest_2k_seed3402.json`）** 上：当前最佳部署 `rho=0.5 + predicted_anchor_rerank_a` = **3.15%**，但**原始 FP 在 2k 上从未跑过**。
- 因此当前所有"我们方法在 2k 上是否赢过原始 FP"的判断**没有数据支撑**，必须先补这一格。

### P0 — 必须做（不做则 main result 不成立）

- [ ] **P0-0**：把 anchor pipeline 接进官方 nuPlan closed-loop planner。
  - 关键事实：当前 `flow_planner/planner.py` 只加载 `ckpt_path` 并调用 `core.inference(...)`，没有加载 `anchor_vocab.npy`、`AnchorPredictor`、`CandidateSelector`，也没有向模型传 `anchor_traj`。
  - 直接后果：现在用 `launch_sim_nuplan.sh` 跑 anchor finetuned ckpt 时，等价于 `anchor_mode=none`，不是 `predicted_anchor_top1` / `predicted_anchor_rerank_a` / candidate selector。
  - 最小验收：在官方 `nuplan-devkit run_simulation.py` 上跑通 1 个 `val14` scenario 的 `predicted_anchor_top1`。
  - 完整验收：在官方 closed-loop 上跑 `raw_no_goal`、`anchor_ft_none`、`predicted_anchor_top1`、`predicted_anchor_rerank_a`、candidate selector，并拿到 NR-CLS / R-CLS 及官方子指标。
  - 备注：在 P0-0 完成前，`eval_multidim.py` 的 collision / route / progress 只能作为内部 surrogate ablation，不能当作论文主表数字。
- [ ] **P0-1**：在 `eval_manifest_2k_seed3402.json` 上跑 `flowplanner_no_goal.pth + anchor_mode none`。
  - 脚本：`run_anchor_raw_no_goal_eval.sh`，`MANIFEST_PATH=/root/autodl-tmp/anchor_runs/eval_manifest_2k_seed3402.json`。
  - 输出建议：`/root/autodl-tmp/anchor_runs/raw_no_goal_eval_2k_<日期>`。
  - 决策规则：
    - 如果 raw FP 2k > 3.15% → 当前最佳部署赢过原始 FP，main result 成立，进入 P1。
    - 如果 raw FP 2k ≤ 3.15% → main result 不成立，必须先解决"为什么 anchor-finetuned planner 在 anchor 关掉后会退化"才能投稿。
- [ ] **P0-2**：解释 `planner_ft_run1 + anchor_mode none = 6.4%` vs `flowplanner_no_goal.pth + none = 2.4%` 的退化（500-scene 同 manifest）。
  - 假设候选：差分 LR 把 decoder backbone 推偏；anchor cross-attn out_proj 即使 zero-init 在 finetune 后产生残余偏移；optimizer state / 学习率调度差异。
  - 诊断手段：在 anchor-finetuned ckpt 上把 `anchor_cross_attn` 的 KV 完全屏蔽，看 collision 是否回到 raw FP 水平；若仍退化，说明问题在 decoder 主干 finetune。
- [ ] **P0-3**：等 candidate-level pairwise selector all-pairs train2k（2026-04-28 运行中）跑完，记录 2k eval。
  - 决策规则：
    - 若 2k collision < 3.15% → 接管 `predicted_anchor_rerank_a` 作为新部署 baseline，准备做 paper main 表。
    - 若 ≥ 3.15% → candidate-level 暂停扩量，回头先解决 P0-2。

### P1 — 强烈建议（论文 ablation 必交）

- [ ] **P1-1**：CFG weight sweep on anchor checkpoint。
  - 设置：`predicted_anchor_rerank_a` + 2k manifest，`cfg_weight ∈ {1.0, 1.25, 1.5, 1.75, 2.0}`。
  - 动机：原版 `cfg_weight=1.5` 是 no-anchor 时调出来的；anchor cross-attn 接入后 `(u_cond − u_uncond)` 的含义变了，最优 w 可能漂移。
- [ ] **P1-2**：训练时 CFG dropout 一致化。
  - 现状：训练 cfg_flags=0 只 mask neighbors（或 lanes，按 `cfg_type`），不动 anchor；推理 cfg_flags=0 同时 zero anchor。uncond 分支推理期 OOD。
  - 修法：在 `prepare_model_input` 里 cfg_flags=0 时把 anchor_traj 也置零，重训一个 short epoch 验证 collision 是否变化。
- [ ] **P1-3**：Hybrid waypoint+velocity loss（HDP 启发）。
  - 现状：`model_type=x_start`、`kinematic=waypoints`、`loss = MSE(prediction, target_tokens)`，**只监督 waypoint**。
  - 改动：在 loss 里加一项 `MSE(diff(prediction[..., :2]), gt_velocity)`，权重 sweep `{0.1, 0.5, 1.0}`。
  - 数学已被 HDP 证明：hybrid loss 不改变 optimal solution，只是降低高频抖动；工作量小、风险低。
- [ ] **P1-4**：多 seed 重跑当前最佳部署设置。
  - 至少 3 个 seed × `predicted_anchor_rerank_a` × 2k manifest，得到 mean ± std。
  - 现在所有数字单 seed，paper 主表必须报方差。
- [ ] **P1-5**：5k manifest 主表。
  - manifest 已存在：`/root/autodl-tmp/anchor_runs/eval_manifest_5k_seed3402.json`。
  - 至少跑 `raw_no_goal / planner_ft_none / predicted_anchor_top1 / predicted_anchor_rerank_a / oracle_anchor / oracle_anchor_rerank` 六格。
  - 论文主表用 5k 比 2k 更稳。

### P2 — 论文加分项（不做能投，做了更稳）

- [ ] **P2-1**：anchor 注入方式 ablation（`cross-attn` vs `AdaLN additive` vs `concat at token level`）。
  - 直接对应 §2.1 架构决策，是审稿人最爱问的"为什么这样接而不那样接"。
- [ ] **P2-2**：anchor vocab 大小 K sweep（K ∈ {32, 64, 128, 256}）。
  - 观察 `oracle_anchor` collision 与 K 的关系，验证 K=128 的合理性。
- [ ] **P2-3**：AnchorPredictor metric heads（Hydra-MDP style：dac / safety / progress / comfort 多头）。
  - 即旧 §7 P2 的 metric heads 项，扩展到候选级 multi-target distillation。
- [ ] **P2-4**：失败 case 分析 + 可视化。
  - oracle vs predicted 在 same scene 的轨迹差；selector picked vs hand-rerank picked 的 disagreement case。
  - paper figure 候选。
- [ ] **P2-5**：延迟 / 吞吐量分析。
  - 单帧推理时间、K candidate 生成开销、与原版 FlowPlanner 对比。
  - 工业审稿人会问的"你这一套比原版慢多少"。
- [ ] **P2-6**：跨城市 / 跨 split 泛化诊断（如果 nuPlan split 允许）。
- [ ] **P2-7**：timestep-adaptive anchor conditioning。
  - 当前实现中，anchor cross-attn 注入发生在 `time_cond = self.t_embedder(t)` 之前；anchor token / anchor 注入强度本身不显式随 flow timestep `t` 自适应，只由后续 DiT / FinalLayer 间接吸收时间条件。
  - 改进方向：在 `AnchorCrossAttention` 外加 timestep-conditioned gate / modulation，例如让 `time_cond` 预测 `anchor_delta` 的 gate，使高噪声阶段更依赖 anchor 全局形状、低噪声阶段允许 planner 保留细节修正。
  - 影响：这是模型结构变化，旧 checkpoint 会缺新参数；需要重新 fine-tune anchor planner，并至少对比 `oracle_anchor`、`predicted_anchor_top1`、`predicted_anchor_candidate_selector`。

### P3 — 收尾 / 长尾（不影响投稿）

- [ ] DPO chosen/rejected 路径同 noise（已修 policy/reference 同 noise，pair 内同 noise 还可再改一层）。
- [ ] Phase 0 teacher score cache（旧 §7 P2 项，目前由 candidate scoring 部分覆盖）。
- [ ] Code release / README / 复现指南整理。
- [ ] runtime patches 正式回收到 `feature/anchor` 分支（当前还有几份 patch 仅存于 runtime snapshot）。

### 发表路线图（保守目标：CoRL 2026 / ICRA 2027）

- **目标会议档次定调**：
  - 顶会 oral 不现实（HDP 已发，diffusion planner 这个生态位被占）。
  - 顶会 poster 风险高，仅当 P0-3（candidate-level all-pairs）出现明显胜过 hand rerank 的结果，再考虑冲。
  - 主推：CoRL / ICRA / IROS。
- **核心 contribution 候选**（按强度排）：
  - C1（强）："continuous flow-matching log-prob 在 same-anchor pair 上 margin ≈ 0，因此 trajectory-level DPO 失效；需要 reformulate 到 discrete anchor / candidate space"。已有 mixed-DPO / collision-only DPO / softpref distillation 三组失败实验作为支撑。
  - C2（中）："collision-only selector-DPO 优于 all-pairs selector-DPO，更多数据反而伤 safety"——反直觉发现，配 ablation 站得住。
  - C3（弱）：scheduled anchor sampling 缩小 train/inference gap（已被 DAgger / scheduled sampling 文献覆盖，新颖度低）。
- **短路线（约 3–4 周纯投入）**：
  - Week 1：P0-1 / P0-2 / P0-3 收口；P1-1 cfg sweep；P1-2 训练一致化重训。
  - Week 2：P1-3 hybrid loss；P1-4 多 seed；P1-5 5k 主表；P2-1 注入方式 ablation。
  - Week 3：写作（thesis statement + 主图 + 主表 + related work：HDP / Diffusion-Planner / Hydra-MDP / GoalFlow 定位）。
  - Week 4：buffer / polish / 内审。
- **stop conditions**（哪些情况就别硬投）：
  - P0-1 表明 raw FP 2k ≤ 3.15% 且 P0-2 解释不掉退化 → 暂停投稿。
  - P1-4 多 seed 显示当前最佳部署的 std > 0.5 pp → 数字不稳，先解决方差。

### 跟 Diffusion-Planner / Hyper-Diffusion-Planner 的相对位置（写 related work 用）

- **本工作 vs Diffusion-Planner（ICLR 2025 oral, arXiv 2501.15564）**：
  - DP 用 classifier guidance（无需训分类器，复用 diffusion model 估能量）做推理期偏好控制；本工作用 CFG + 离散 anchor selector + post-hoc rerank。
  - 两者完全正交，不冲突；本工作不在 ODE 步内改 score。
  - DP 的输入也是结构化（vector），与本工作可比；DP 报告的是 nuPlan benchmark closed-loop SOTA。
- **本工作 vs Hyper-Diffusion-Planner / HDP（arXiv 2602.22801, 真车 200km）**：
  - HDP 主张 τ₀-pred + τ₀-loss 是轨迹生成最优组合；**本工作（继承 FlowPlanner）已在 `model_type=x_start` 上，等价于 HDP 推荐组合**，无需修改。
  - HDP 用 hybrid waypoint+velocity loss + RL post-training；本工作目前**只 waypoint loss**，对应 P1-3。
  - HDP 明确表态"不依赖 anchor / goal point"，与本工作 anchor-conditioned 路线相反。审稿时审稿人会问"为什么 anchor 必要"，需要用 oracle gap（`oracle_anchor` 2.20% vs `planner_ft_none` 5.45% 在 2k 上）+ candidate oracle 1.2% 的诊断作为支撑回应。
  - 本工作与 HDP 是**正交贡献**：HDP 关注 backbone loss space + RL 后训；本工作关注 conditioning prior + discrete preference learning。

---

## 代码审查记录：recent anchor selector / scorer / eval code（20260428）

- 目的：
  - 用户要求重新检查最近几轮 candidate-level selector / scorer / eval 代码，判断实验结论是否可能被实现问题污染。
- 检查文件：
  - `flow_planner/planner.py`
  - `flow_planner/nuplan_simulation/planner/flow_planner.yaml`
  - `flow_planner/dpo/eval_multidim_utils.py`
  - `flow_planner/dpo/anchor_candidate_scorer.py`
  - `flow_planner/dpo/generate_anchor_softpref_candidates.py`
  - `flow_planner/dpo/score_anchor_candidates.py`
  - `flow_planner/dpo/train_anchor_candidate_selector_softpref.py`
  - `flow_planner/dpo/train_anchor_candidate_selector_pairwise.py`
  - `flow_planner/goal/candidate_selector.py`
- 发现 1（P0）：官方 closed-loop planner 没接 anchor pipeline。
  - `planner.py::compute_planner_trajectory` 当前只调用 `self.core.inference(self._planner, inputs, use_cfg, cfg_weight, num_candidates, bon_seed)`。
  - 没有加载 anchor vocab / anchor predictor / candidate selector，也没有传 `anchor_traj`。
  - 因此官方 nuPlan closed-loop 目前不能评估 anchor 方法；只能评估 raw FlowPlanner 或 anchor finetuned ckpt 的 `anchor_mode=none` 退化版本。
- 发现 2（P0）：`eval_multidim.py` 不是 nuPlan 官方 metric。
  - collision 是 ego/neighbor 中心点距离 `< collision_dist`，不是官方 vehicle footprint polygon intersection。
  - route/progress/comfort 也是自定义 surrogate，不是 nuPlan official NR-CLS/R-CLS metric。
  - 当前 3.15% / 3.55% / 4.20% 等数字只能做内部 ablation，不能直接与 FlowPlanner / Diffusion-Planner 官方分数比较。
- 发现 3（P1 bug）：candidate scorer 的 `collision_score` 符号疑似反了。
  - `TrajectoryScorer._collision_score` 中 `collision_score` 越大表示越安全（最小距离越大）。
  - 但 `anchor_candidate_scorer.score_components` 当前使用 `- weights.collision_weight * collision_score`，等价于轻微惩罚更安全的距离。
  - 影响：soft candidate selector / selector-DPO all-pairs / score-gap 排序会被轻微污染；same-anchor safe-vs-collided pairwise 受影响较小，因为 `safety_weight * (1-collided)` 主导。
  - 下一步：修成 `+ weights.collision_weight * collision_score` 或改名为 collision_penalty 并明确公式；修后必须重新 score candidates，再重训受影响的 selector 版本。
- 发现 4（P1 bug）：`CandidateSelector` 冻结 backbone 后，训练时仍会被 `model.train(True)` 递归切回 train mode。
  - `CandidateSelector.__init__` 里对 `backbone.eval()` 和 `requires_grad_(False)` 做了冻结。
  - 但 `train_anchor_candidate_selector_*::run_epoch` 调用 `model.train(train)`，会把 `CandidateSelector.backbone` 递归设置为 train mode；即使 no grad，dropout / train-time behavior 仍可能打开。
  - 影响：selector 训练和部署时的 scene features 分布不一致，可能解释 soft CE / sharp target / pairwise 训练不稳定。
  - 下一步：在 `CandidateSelector.train()` 中覆盖逻辑，或在 `extract_scene_features()` 内部若 `freeze_backbone=True` 则强制 `self.backbone.eval()`；`AnchorPredictor` 也有相同模式，建议一起修。
- 发现 5（P1 bug / validation risk）：candidate pairwise all-pairs 的 train/val split 是按 pair 随机切，不是按 scene 切。
  - `train_anchor_candidate_selector_pairwise.py::split_records` 当前直接 shuffle records。
  - 当 `pair_reduce=all` 时，同一个 scene 的多个 pair 可能同时出现在 train 和 val，offline val pair_acc 会偏乐观。
  - 部署 eval 不会直接泄漏，但会影响 best checkpoint selection 和对 offline pair_acc 的解读。
  - 下一步：按 `scenario_id` group split；历史 all-pairs pilot 的 val pair_acc 只作参考，不能作为强结论。
- 发现 6（P1 risk）：oracle anchor eval 与训练标签对齐逻辑不一致。
  - 训练中 `FlowPlanner._get_anchor_index_for_gt` 使用 `gt_future[:, -T_anchor:, :3]`。
  - `eval_multidim_utils.resolve_anchor_condition(oracle_anchor)` 与 `_get_oracle_topk_anchor_trajs()` 当前使用 `ego_future_arr[:T, :3]`。
  - 如果 NPZ 的 `ego_agent_future` 长度大于 `T_anchor`，oracle eval 会选错时间段的 anchor；如果恰好等于 80，则没有影响。
  - 下一步：统一改成 `ego_future_arr[-T:, :3]`，并在日志中记录 NPZ future length。
- 结论：
  - 近期 selector/pairwise 实验不是完全无效，但 paper-facing 结论必须降级：只能说明在当前自定义 surrogate eval 下有方向性信号。
  - 在修复 scorer 符号、frozen-backbone mode、scene-level split、oracle alignment，并接入官方 closed-loop 前，不应把这些结果包装成最终主结果。

## Experiment: 20260502 val20 strict_gate trace per-scene summary + focus scene analysis

- Status: completed / alignment caveat added 2026-05-05
- Goal:
  - 把 `strict_gate` 的 `val20` trace 落成 per-scene summary，确认新增碰撞 scene `71e4ce1d08e85a3c` 到底是 rare unsafe gate release，还是更长时程的 anchor-induced state-distribution shift。
- Data:
  - trace JSONL: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/candidate_trace.jsonl`
  - scene delta CSV: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_strict_gate_w2_20260501/scene_delta_vs_none.csv`
  - eval subset: official `val20_clean`
- Method:
  - Original 2026-05-02 summary used the first-seen order of `planner_instance_id` in trace JSONL and aligned that order to `scene_delta_vs_none.csv`.
  - 2026-05-05 follow-up found this implicit order-based alignment is risky.
  - A safer timestamp-based remap was added in `scripts/anchor/summarize_candidate_trace.py`; it maps trace `iteration_time_us` to official metric `time_series_timestamps`.
  - 对每个 scene 汇总：
    - raw selector 选择 anchor 的 tick 比例
    - strict gate 最终放行 anchor 的 tick 比例
    - fallback 比例
    - final anchor 的连续段长度
    - gate reason 计数
  - 对唯一 collision regression scene `71e4ce1d08e85a3c` 单独导出末段 tick 级 case study。
- Artifacts:
  - per-scene JSON: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/val20_trace_per_scene_summary.json`
  - per-scene CSV: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/val20_trace_per_scene_summary.csv`
  - focus case study: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/focus_scene_case_study.json`
  - timestamp-remapped per-scene CSV: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/trace_summary_repro_20260505/candidate_trace_per_scene_summary.csv`
  - timestamp-remapped focus case: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_20260502/trace_summary_repro_20260505/focus_scene_71e4ce1d08e85a3c.json`
- Results:
  - 全局 trace 结论保持不变：raw selector 在 `94.2%` ticks 想选 anchor，但 strict gate 最终只放行 `12.6%`。
  - `71e4ce1d08e85a3c` 是唯一 `delta.no_ego_at_fault_collisions = -1.0` 的 scene。
  - Order-aligned original focus numbers are superseded for tick-level details.
  - Timestamp-remapped `71e4ce1d08e85a3c` summary:
    - `149` ticks.
    - raw selector chose anchor on `137/149` ticks, raw anchor rate = `91.95%`.
    - strict gate final anchor on `8/149` ticks, final anchor rate = `5.37%`.
    - fallback ticks = `129/149`, fallback rate = `86.58%`.
    - final anchor iters: `73`, `81`, `84`, `91`, `92`, `102`, `103`, `145`; longest final-anchor run = `2`.
    - top gate reasons: `selected_final_x_lt_fallback_minus_2m=86`, `selected_path_lt_0p75_fallback=69`, `large_lateral_delta_without_progress=50`, `rule_score_margin_lt_1p0=42`, `rule_score_lt_unconditioned=33`.
- Conclusion:
  - 目前证据更支持“稀疏 anchor 执行造成的长时程状态分布偏移”而不是“某一个明显的 unsafe gate release 直接导致新增碰撞”。
  - 时间戳复核后，focus scene 仍然是低 final-anchor-rate、稀疏放行；但旧文档里具体 tick id / gate reason 计数不应继续引用。
  - 后续所有 candidate trace per-scene summary 应使用 timestamp-based mapping，而不是 `planner_instance_id` 首次出现顺序。
- Decision:
  - 不继续跑更大 budget sweep，也不继续把当前 `strict_gate` 当 deployment path 扩到 `val100`。
  - 如果继续这条线，优先级应放在：
    - 更保守的 learned override gate / confidence gate
    - 针对 focus scene 这类 sparse-anchor-distribution-shift case 的 scene-level gate features
    - 更便宜的 `unconditioned + top1x1` 或 `unconditioned + top1x2` 探针
  - 在没有达到“collision 不差于 `none` 且 anchor 使用率不塌到近零”之前，不建议继续大规模 official closed-loop 扩展。

## Experiment: 20260505 val20 strict_gate top1x1 official probe

- Status: aborted / discarded
- Goal:
  - 以最低成本测试“baseline/unconditioned + top1x1 anchor candidate + strict gate”是否能避免 `5-2-2` 的 sparse-anchor-distribution-shift 问题。
  - 如果 `top1x1` 仍然对 `anchor_mode=none` 出现 collision regression，则说明问题不主要来自多 anchor / 多 sample budget，而是更基本的 selector-vs-closed-loop misalignment。
- Setup:
  - Runtime repo: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Anchor predictor ckpt: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Candidate selector ckpt: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_rescorefix_scenegroup_20260429_2306/anchor_candidate_selector_pairwise_best.pth`
  - Anchor mode: `predicted_anchor_candidate_selector_strict_gate`
  - Candidate budget: `anchor_top_k=1`, `candidate_samples_per_anchor_list=[1]`
  - Gate policy: strict gate with baseline/unconditioned fallback candidate included
  - Worker setup:
    - smoke: `debug_2`
    - official probe: `val20_clean`, `worker.max_workers=2`
- Artifacts:
  - Discarded smoke output: `/root/autodl-tmp/anchor_runs/official_planner_anchor_top1x1_strict_gate_smoke_20260505` (deleted)
  - Discarded official output: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_top1x1_strict_gate_w2_20260505` (deleted)
  - Discarded script: `/root/autodl-tmp/Flow-Planner-anchor-runtime/scripts_anchor_run_closed_loop_top1x1_strict_gate.sh` (deleted)
- What happened:
  - `debug_2` smoke completed `2/2` successfully.
  - `val20_clean` was started, then manually stopped before completion.
  - All top1x1-specific temporary outputs and launch logs were removed.
- Reason:
  - This probe reused the old selector checkpoint, so its result would still be conditioned on historical training-signal contamination.
  - It could only answer a narrow budget question, not the main question: whether a clean selector trained after scorer/split/oracle fixes works in official closed-loop.
- Decision:
  - Treat this probe as discarded, not as an experiment result.
  - Do not start more budget probes until the root fixes are applied and a clean selector is retrained.

## Code fix: 20260505 clean selector root-fix pass

- Status: implemented / local verification passed
- Goal:
  - Remove known training/evaluation contamination before any new selector rescore/retrain run.
- Changes:
  - `flow_planner/dpo/anchor_candidate_scorer.py`
    - Fixed `collision_score` sign in candidate teacher score.
    - `collision_score` is now treated as a safety-margin score, so larger values increase `final_score`.
  - `flow_planner/goal/anchor_predictor.py`
    - Added `train()` override matching `CandidateSelector` / `GoalPredictor`, so a frozen backbone remains in eval mode after recursive `model.train(...)`.
  - `flow_planner/dpo/eval_multidim_utils.py`
    - Changed oracle anchor lookup from `ego_future_arr[:T, :3]` to `ego_future_arr[-T:, :3]`, matching the training label horizon semantics.
  - `flow_planner/dpo/train_anchor_candidate_selector_pairwise.py`
  - `flow_planner/dpo/train_anchor_candidate_selector_softpref.py`
  - `flow_planner/dpo/train_anchor_selector_dpo.py`
  - `flow_planner/dpo/train_anchor_selector_softpref.py`
    - Replaced record-level random split with scene-grouped split by `scenario_id`.
    - Stats now include `split_strategy=scene_grouped`, `num_train_scenes`, `num_val_scenes`, and `train_val_scene_overlap`.
- Verification:
  - `python -m py_compile` passed for all changed Python files.
  - grep check found no remaining target occurrences of:
    - `- weights.collision_weight * collision_score`
    - `ego_future_arr[:T`
    - `rng.shuffle(records)` in the touched split helpers
  - grep check confirmed `train_val_scene_overlap` is written by all four updated selector-training scripts.
- Decision:
  - Next valid selector result must be produced by:
    - rescoring existing candidate cache with the corrected teacher score,
    - retraining a clean selector with scene-grouped split,
    - then running official closed-loop validation.
  - Older selector checkpoints remain useful only as historical baselines, not as clean evidence.

## Experiment: 20260505 clean scorefix rescore + scene-grouped pairwise selector retrain

- Status: offline retrain completed
- Goal:
  - Produce the first clean candidate selector checkpoint after the known selector-line contamination fixes.
  - Separate low-cost teacher-score correction from expensive candidate resampling by reusing the existing train2k candidate NPZ cache.
- Code baseline:
  - Branch: `feature/anchor`
  - Commit: `070a534`
  - Required fixes included:
    - positive `collision_score` teacher sign,
    - frozen-backbone train/eval protection,
    - scene-grouped selector train/val split,
    - oracle anchor horizon alignment,
    - scene split clamp so small runs cannot put every scene into val.
- Data:
  - Source candidate root: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153`
  - Source scored_dir: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_20260426_2153/scored_dir`
  - Train scenes: `/root/autodl-tmp/train_dataset`
  - Candidate cache size: 2000 scored scenes, `top_k=3`, `samples_per_anchor=3`
- Planned setup:
  - Clean AutoDL worktree: `/root/autodl-tmp/Flow-Planner-sync-anchor`
  - Rescore output: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_clean_scorefix_20260505`
  - Train output: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Anchor vocab: `/root/autodl-tmp/anchor_runs/anchor_vocab.npy`
  - Pair mining: `pair_scope=same_anchor`, `pair_reduce=all`, `min_score_gap=0.0`
  - Split: scene-grouped, `val_fraction=0.2`, `seed=3402`
  - Training: `epochs=5`, `batch_size=32`, `lr=2e-4`, `weight_decay=1e-4`
- Required checks:
  - `pair_stats.json` must report `split_strategy=scene_grouped`.
  - `pair_stats.json` must report `train_val_scene_overlap=0`.
  - Training should use the clean `CandidateSelector.train()` behavior so the frozen planner backbone stays in eval mode.
- Results:
  - Rescore completed: `written_scenes=2000`, `rescored_candidates=3179`.
  - Rescore consistency spot-check passed: `total_score` matches `score_components.final_score`, and scene/anchor summaries were regenerated.
  - Pair mining produced `1268` same-anchor all-pairs from `436` scenes with usable pairs.
  - Scene-grouped split: `num_train_pairs=1010`, `num_val_pairs=258`, `num_train_scenes=349`, `num_val_scenes=87`, `train_val_scene_overlap=0`.
  - Best offline checkpoint: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505/anchor_candidate_selector_pairwise_best.pth`
  - Final/best offline validation: epoch 5, `val_pair_acc=0.7292`, `val_pair_loss=0.6270`, `val_chosen_prob=0.1457`.
- Artifacts:
  - Rescore meta: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_clean_scorefix_20260505/meta.json`
  - Rescore log: `/root/autodl-tmp/anchor_runs/anchor_softpref_candidates_train2k_clean_scorefix_20260505.rescore.log`
  - Train output: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505`
  - Train log: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505/train.log`
  - Pair stats: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505/pair_stats.json`
- Decision rule:
  - If offline clean training is sane, use this checkpoint for a new official closed-loop `val20_clean` probe.
  - If offline clean training is not sane, stop before spending official closed-loop budget and inspect labels/features first.
- Decision:
  - Offline clean training is sane and the required no-leakage split checks passed.
  - Launch a clean-selector official `val20_clean` probe with the same `5-2-2` candidate budget before any new budget sweep.

## Experiment: 20260505 clean selector 5-2-2 official val20_clean probe

- Status: completed
- Goal:
  - Test whether the clean selector checkpoint from the root-fix pass improves or still regresses in official closed-loop evaluation.
  - Keep the candidate budget comparable to the previous `5-2-2` official probe; do not start new budget sweeps yet.
- Setup:
  - Runtime repo: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Planner ckpt: `/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth`
  - Anchor predictor ckpt: `/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth`
  - Clean candidate selector ckpt: `/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505/anchor_candidate_selector_pairwise_best.pth`
  - Anchor mode: `predicted_anchor_candidate_selector`
  - Candidate budget: `anchor_top_k=3`, `candidate_samples_per_anchor_list=[5,2,2]`
  - Eval subset: official `val20_clean`
  - AutoDL visible resources checked on 2026-05-05:
    - visible GPU: `1 x NVIDIA GeForce RTX 4090 D`
    - CPU cores: `128`
    - memory: `503GiB`
  - Valid metric run worker setup: `single_machine_thread_pool`, `worker.max_workers=20`
  - Parallel trace run worker setup: `single_machine_thread_pool`, `worker.max_workers=8`
- Artifacts:
  - Valid metric output root: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505`
  - Valid metric experiment output: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/anchor_selector_522_clean_rootfix_val20_clean_w20_final`
  - Metric launch meta: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/launch_meta.txt`
  - Clean trace output root: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505`
  - Clean trace experiment output: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/anchor_selector_522_clean_rootfix_trace_val20_w8`
  - Clean trace JSONL: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/candidate_trace.jsonl`
  - Summary tool: `scripts/anchor/summarize_official_eval.py`
  - Trace summary tool: `scripts/anchor/summarize_candidate_trace.py`
- Running notes:
  - Earlier `w2`, `w16`, and first `w20` launches were stopped/superseded before completion to avoid wasting time on too-conservative worker settings.
  - Sharded attempts improved instantaneous GPU utilization but were invalid because chunk filters produced empty shards; future sharding must use verified scenario tokens or timestamp/metric-backed mapping.
  - The current valid metric run keeps full `val20_clean` coverage and avoids empty-shard risk.
  - A parallel clean trace run was launched because official closed-loop left the single visible GPU mostly idle; this run is for per-scene/tick diagnosis, not a new budget sweep.
- Current status:
  - As of `2026-05-05 20:00 CST`, both metric and trace runs had completed.
  - Both runs finished with `20/20` simulations succeeded and `0` failed.
- Results:
  - Official metric summary artifact:
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/summary_vs_none_20260505/official_eval_summary.txt`
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/summary_vs_none_20260505/official_eval_scene_delta.csv`
  - Clean trace summary artifact:
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/trace_summary_20260505/candidate_trace_per_scene_summary.csv`
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/trace_summary_20260505/focus_scene_71e4ce1d08e85a3c.json`
  - `anchor_none` vs clean root-fix raw selector:
    - weighted score: `94.7543 -> 84.6520` (`-10.1023`)
    - product score: `65.7745 -> 25.9733`
    - `no_ego_at_fault_collisions`: `19/20 -> 18/20`
    - `drivable_area_compliance`: `19/20 -> 16/20`
    - `ego_is_comfortable`: `19/20 -> 16/20`
    - `ego_is_making_progress`: `20/20 -> 18/20`
    - `ego_progress_along_expert_route` mean: `0.9215 -> 0.6523`
    - `time_to_collision_within_bound`: `17/20 -> 16/20`
    - `speed_limit_compliance` mean: `0.9794 -> 0.9602`
    - `compute_trajectory_runtimes_mean`: `5.94 -> 66.60`
  - Collision regressions remained concentrated in two scenes:
    - `0002182ea6cd5afd`
    - `bf01e524555c556e`
  - Strong progress collapses remained concentrated in the same family of scenes already seen in the older raw selector run:
    - `26023c247e8251e3`: `delta progress = -1.0`, `delta making_progress = -1.0`
    - `d92e020065eb5d9e`: `delta progress = -1.0`, `delta making_progress = -1.0`
    - `71e4ce1d08e85a3c`: no new collision, but progress `0.9042 -> 0.3266`
  - Compared against the older dirty raw selector run (`anchor_selector_522_val20_clean`), the clean root-fix raw selector is only moderately worse, not a new order-of-magnitude collapse:
    - weighted score: `86.3908 -> 84.6520` (`-1.7388`)
    - the main failing scenes overlap heavily; the root failure mode did not qualitatively change.
  - Important mode clarification:
    - This run used `anchor_mode=predicted_anchor_candidate_selector`, not `strict_gate`.
    - In [planner.py](/home/gcjms/Flow-Planner/flow_planner/planner.py), `predicted_anchor_candidate_selector` does not include the unconditioned baseline candidate, so there is no per-tick fallback path in this mode.
    - The clean trace confirms that behavior directly: all `20/20` scenes had `raw_anchor_rate=1.0`, `final_anchor_rate=1.0`, `fallback_rate=0.0`.
    - Therefore this result should be compared first to the older raw selector probe, not to `strict_gate` / `hybrid_gate`, which are different deployment modes with baseline fallback.
- Conclusion:
  - The root-fix pass corrected known contamination issues, but it did not solve the main closed-loop mismatch.
  - After removing leakage / score-sign / oracle-horizon / frozen-backbone contamination, the raw selector line still underperforms badly against `anchor_none` in official closed-loop.
  - The fact that the clean raw selector stays close to the older raw selector failure pattern suggests the dominant problem is not the fixed bugs alone; it is the lack of a closed-loop-consistent acceptance mechanism.
  - Put differently: the fixes cleaned the measurement and supervision path, but they did not change the fact that a raw open-loop-trained selector is too aggressive when deployed every tick without fallback.
- Decision:
  - Stop treating raw `predicted_anchor_candidate_selector` as a viable deployment mode.
  - Do not continue budget sweeps on raw selector.
  - If the selector line is continued, it should continue only behind an explicit gate / accept-reject mechanism or another closed-loop-consistent override path.
  - The next apples-to-apples follow-up, if we keep this line alive, should be a clean `strict_gate` or cleaner learned override gate using the same fixed selector checkpoint.

## Code / Experiment Prep: 20260505 closed-loop selector data collector v1

- Status: implementation started
- Goal:
  - Replace the open-loop selector supervision path with a closed-loop data path.
  - First collect official closed-loop labels for "baseline except this specific tick/candidate intervention" rollouts.
  - Use those labels later to train an accept/reject closed-loop selector, instead of trusting open-loop teacher scores.
- Setup:
  - Working branch: `anchor`
  - Official runtime target: `/root/autodl-tmp/Flow-Planner`
  - Artifact root: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505`
  - Source raw trace for first manifests:
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/candidate_trace.jsonl`
  - Source metric run for timestamp-to-scene mapping:
    - `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/anchor_selector_522_clean_rootfix_val20_clean_w20_final`
- Implementation plan:
  - Merge the official anchor execution path from the runtime snapshot back into the formal `anchor` branch.
  - Add `anchor_mode=predicted_anchor_candidate_selector_intervention`.
  - Add `planner.flow_planner.candidate_intervention_manifest_path`.
  - In intervention mode, non-listed ticks use baseline / unconditioned planning; listed ticks generate the same candidate pool and force the manifest-selected candidate.
  - Keep `candidate_trace_path` writing candidate metadata, raw selector choice, forced final choice, and intervention details.
  - Add `scripts/anchor/build_closed_loop_intervention_manifest.py` to build first manifests from existing selector trace plus official timestamp mapping.
- Data policy:
  - Do not train on `val20_clean`.
  - Use the `val20_clean` interventions only as probe / debugging evidence.
  - For actual closed-loop selector training data, create separate train-scene intervention manifests after the collector smoke passes.
- Initial decision:
  - Stop polishing raw open-loop selector as the main method.
  - Build the closed-loop label collector first; train the accept/reject selector only after the collector produces valid official rollouts.
- Smoke result:
  - Runtime snapshot remains the execution environment for official experiments because it already has the validated scenario filters and nuPlan wiring.
  - Formal code and records are still synced back to the `anchor` branch.
  - Code commit: `3f7b6f0 anchor: resolve interventions by candidate metadata`
  - Synced runtime files:
    - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/planner.py`
    - `/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/dpo/eval_multidim_utils.py`
    - `/root/autodl-tmp/Flow-Planner-anchor-runtime/scripts/anchor/build_closed_loop_intervention_manifest.py`
  - Smoke manifest:
    - `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/manifest_71e4_raw_first1.json`
  - Valid smoke run:
    - `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/intervention_71e4_raw_first1_exact_runtime_v2`
    - trace: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/intervention_71e4_raw_first1_exact_runtime_v2_trace.jsonl`
    - runner: `1/1` succeeded, scenario `71e4ce1d08e85a3c`
    - forced trace row: iteration `0`, final `anchor_rank=1`, `sample_i=1`, `gate_reasons=["forced_candidate"]`
    - official single-scene metrics: collision `0.5`, drivable `1.0`, making_progress `1.0`, progress `0.8763`, TTC `0.0`, comfort `1.0`
  - Invalid / diagnostic-only smoke runs:
    - `intervention_71e4_raw_first1_smoke_runtime` selected another scene from the same log because the command did not constrain `scenario_tokens`; do not use as data.
    - `intervention_71e4_raw_first1_exact_runtime` used the old forced-candidate index path before the metadata lookup fix; do not use as data.
- Next:
  - Build the first small batch of intervention manifests from regression scenes and raw-selector high-confidence ticks.
  - Run those official rollouts from the runtime snapshot.
  - Convert each rollout result into accept/reject labels versus `anchor_none` before training any closed-loop selector.

## Experiment: 20260505 closed-loop intervention batch1 val20_clean probe

- Status: completed
- Goal:
  - Verify that the new closed-loop intervention collector can produce usable per-tick accept/reject labels.
  - Keep this as a debugging/probe batch only; do not use `val20_clean` labels for training.
- Setup:
  - Runtime repo: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Formal branch: `anchor`
  - Artifact root: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505`
  - Source trace: `/root/autodl-tmp/anchor_runs/official_planner_anchor_trace_val20_clean_rootfix_20260505/candidate_trace.jsonl`
  - Timestamp mapping run: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_rootfix_w20_final_20260505/anchor_selector_522_clean_rootfix_val20_clean_w20_final`
  - Baseline comparison run: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val20_clean_20260501/anchor_none_val20_clean`
  - Intervention mode: `predicted_anchor_candidate_selector_intervention`
  - Semantics: all non-listed ticks use baseline/unconditioned planning; exactly one listed tick forces one raw-selector anchor candidate.
  - Scenes: `0002182ea6cd5afd`, `26023c247e8251e3`, `71e4ce1d08e85a3c`, `bf01e524555c556e`, `d92e020065eb5d9e`
  - Ticks per scene: first tick and mid tick (`0`, nearest `74`)
- Artifacts:
  - Batch scripts: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_scripts`
  - Batch manifests: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_manifests`
  - Batch output dirs: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_*`
  - Label summary TXT: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_summary/closed_loop_intervention_summary.txt`
  - Label table CSV: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_summary/closed_loop_intervention_labels.csv`
  - Label summary JSON: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch1_summary/closed_loop_intervention_summary.json`
  - Summary tool: `scripts/anchor/summarize_closed_loop_interventions.py`
- Results:
  - Official rollouts: `10/10` completed.
  - Forced-candidate trace check: `10/10` matched the manifest by candidate metadata.
  - Invalid labels: `0/10`.
  - Labels under the default conservative policy:
    - accept: `9/10`
    - reject: `1/10`
  - Mean weighted-score delta versus `anchor_none`: `-0.7175`.
  - Mean progress delta versus `anchor_none`: `+0.0010`.
  - Only reject:
    - `batch1_04_71e4ce1d08e85a3c_tick000`
    - forced candidate: `anchor_rank=1`, `sample_i=1`
    - weighted score: `83.8047 -> 76.0527` (`-7.7520`)
    - collision metric: `-0.5`
    - progress delta: `-0.0279`
  - The same scene at mid tick was acceptable:
    - `batch1_05_71e4ce1d08e85a3c_tick074`
    - weighted score delta: `+0.0769`
    - progress delta: `+0.0044`
- Conclusion:
  - The collector now produces the kind of label the open-loop selector could not provide: a candidate can be acceptable at one tick and unsafe at another after official closed-loop rollout.
  - This supports switching the selector line from "always take the open-loop best candidate" to "train a closed-loop accept/reject gate".
  - `val20_clean` batch1 is evidence that the data path works, not training data.
- Decision:
  - Keep the runtime snapshot as the official experiment execution environment for this line.
  - Before training, create train-scene intervention batches and summarize them with the same label table format.
  - Do not continue raw selector budget sweeps.

## Experiment: 20260505 closed-loop intervention batch2 val100_clean extra-scenes collection

- Status: running
- Goal:
  - Expand closed-loop accept/reject label collection beyond the `val20_clean` probe scenes.
  - Keep `val20_clean` held out for final method checking; use `val100_clean - val20_clean` as development-label scenes because AutoDL currently has official nuPlan DB cache for `val` / `mini`, but not official `train` DB cache.
- Setup:
  - Runtime repo: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - Formal branch: `anchor`
  - Artifact root: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505`
  - Baseline comparison run: `/root/autodl-tmp/anchor_runs/official_planner_anchor_val100_clean_20260501/anchor_none_val100_clean`
  - Scenario source: successful scenes from `val100_clean`, excluding `val20_clean` and the one `rawbest_smoke` scene.
  - Scene list: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch2_val100_extra20_scenes.txt`
  - Number of scenes: `20`
  - Ticks per scene: `0` and nearest `74`
  - Planned rollouts: `40`
  - Intervention force mode: `raw_best_anchor`
    - The manifest specifies only the scene and tick.
    - At the intervention tick, the runtime computes the candidate pool and forces the highest-logit anchor candidate at that tick.
    - Non-listed ticks still use baseline/unconditioned planning.
- Code update:
  - `flow_planner/dpo/eval_multidim_utils.py`
    - Added manifest type `raw_best_anchor` to avoid requiring a full raw-selector trace before collecting labels.
  - `scripts/anchor/launch_closed_loop_intervention_batch.py`
    - Added `--force raw_best_anchor`, which builds manifests from baseline metric timestamps.
  - `scripts/anchor/summarize_closed_loop_interventions.py`
    - Updated forced-candidate verification for `raw_best_anchor`.
- Artifacts:
  - Batch scripts: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch2_val100_extra20_scripts`
  - Batch manifests: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch2_val100_extra20_manifests`
  - Batch output dirs: `/root/autodl-tmp/anchor_runs/closed_loop_selector_v1_20260505/batch2_val100_extra20_*`
- Running notes:
  - `rawbest_smoke` first confirmed the new manifest type:
    - scene `029bea2e8bd95bbd`
    - trace forced `raw_best_anchor` into final `anchor_rank=0`, `sample_i=1`
  - Batch2 launched in three waves:
    - first `12` rollouts,
    - next `12` rollouts,
    - final `16` rollouts,
    - total `40/40` launched.
  - On launch, the machine had `128` CPU cores, `503GiB` RAM, and one visible `RTX 4090 D`.
  - Peak launch-time load reached about `95`; memory/GPU remained within limits.
- Results:
  - Pending.
- Decision:
  - After batch2 finishes, summarize with `scripts/anchor/summarize_closed_loop_interventions.py`.
  - If valid labels are non-trivial and forced-candidate checks pass, train the first accept/reject closed-loop gate from these development labels.

## Experiment: 20260505 closed-loop gate implementation

- Status: implementation ready, waiting for trajectory-rich closed-loop labels
- Goal:
  - Replace the failed "always trust the open-loop candidate selector" behavior with a closed-loop accept/reject gate.
  - Keep the old candidate selector only as a proposal generator: it picks the anchor candidate it wants, then the new gate decides whether that one tick should take over or fall back to `anchor_none`.
- Setup:
  - Formal branch: `anchor`
  - Runtime target: `/root/autodl-tmp/Flow-Planner-anchor-runtime`
  - New runtime mode: `predicted_anchor_candidate_selector_closed_loop_gate`
  - Gate checkpoint config:
    - `planner.flow_planner.closed_loop_gate_ckpt`
    - `planner.flow_planner.closed_loop_gate_threshold`
  - Training data source:
    - closed-loop intervention label CSV from `scripts/anchor/summarize_closed_loop_interventions.py`
    - matching trace JSONL files written with `planner.flow_planner.candidate_trace_training_payload=true`
- Code:
  - `flow_planner/goal/closed_loop_gate.py`
    - Binary accept/reject model over scene features, selected candidate trajectory, and selected anchor trajectory.
  - `flow_planner/dpo/train_closed_loop_gate.py`
    - Trains `closed_loop_gate_best.pth` from accept/reject intervention labels.
    - Uses scene-grouped train/val split.
  - `flow_planner/dpo/eval_multidim_utils.py`
    - Adds trajectory-rich trace payload fields:
      - `scene_features`
      - `selected_candidate_traj`
      - `selected_anchor_traj`
    - Adds closed-loop gate loading and runtime scoring.
  - `flow_planner/planner.py`
    - Adds the runtime mode `predicted_anchor_candidate_selector_closed_loop_gate`.
    - Behavior:
      - old selector proposes a candidate,
      - gate probability below threshold falls back to unconditioned baseline,
      - gate probability above threshold allows the proposed candidate.
  - `scripts/anchor/launch_closed_loop_intervention_batch.py`
    - Adds `--trace-training-payload` so future intervention batches can be used directly as gate training data.
- Current limitation:
  - Batch2 was launched before trajectory-rich trace payload existed, so it can summarize label balance and bad cases, but it is not sufficient by itself to train the gate.
- Decision:
  - Let current batch2 finish as a label-balance/probe batch.
  - Launch the next development batch with `--trace-training-payload`.
  - Train the first `ClosedLoopGate` checkpoint from that trajectory-rich batch.
  - Evaluate on held-out `official val20_clean` against:
    - `anchor_none`
    - old open-loop candidate selector
    - new candidate selector plus closed-loop gate

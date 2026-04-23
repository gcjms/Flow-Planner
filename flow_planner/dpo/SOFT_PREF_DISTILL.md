# Soft Preference Distillation

这一步放在 `goal-conditioned base` 和 `hard-pair DPO` 之间，用来先学习一个更稳定的 scene-level 软排序信号。

## 为什么加这一阶段

当前 hard DPO 的问题有两个：

1. 每个 scene 最终只吃到很少的 `chosen / rejected` 信息。
2. 如果 `K` 较小或 strict pair mining 太苛刻，很多 scene 可能根本挖不出 pair。

Soft preference distillation 的思路是：

- 一个 scene 先生成 `K` 条候选轨迹
- 每条轨迹都打分
- 把这些分数压成一个 soft target distribution
- 先让 policy 学会“这一组里谁更值得更高概率”

这样每个 scene 的所有候选都能参与训练，不再只依赖最终那一对 hard pair。

## 依赖的数据

这个脚本直接使用：

1. `score_hybrid.py --use_structured_scores --emit_traj_info` 产出的 `scored_dir/*.json`
2. `generate_candidates_goal.py` 产出的 `*_candidates.npz`
3. 原始场景 `scene_dir/*.npz`

其中：

- `scored_dir/*.json` 提供每条 candidate 的 `total_score / ADE / FDE / hard_failures`
- `*_candidates.npz` 提供 candidate trajectory 和 `goal_labels`
- `scene_dir/*.npz` 用于恢复 encoder 条件

如果 `scored_dir` 里的 `source_npz` 是旧机器上的绝对路径，可以额外传 `--candidates_dir` 作为 fallback。

## 目标分布怎么构造

对一个 scene 里的第 `i` 条 candidate，先构两个分量：

1. GT 相似度代理：

```text
gt_sim_i = -(w_ade * ADE_i + w_fde * FDE_i)
```

2. scorer 排序分：

```text
score_i = total_score_i
```

然后分别做 scene 内标准化，再合并成一个 logit：

```text
u_i = lambda_gt * zscore(gt_sim_i) + lambda_score * zscore(score_i)
```

最后用 softmax 得到软目标分布：

```text
q_i = softmax(u_i / T)
```

默认参数是：

- `lambda_gt = 0.5`
- `lambda_score = 1.0`
- `T = 1.0`

这表示先更信 scorer 的整体排序，但保留一部分 GT proximity 作为稳定锚点。

## 训练目标

模型对 scene 内所有 candidate 分别计算近似 log-prob：

```text
log pi_theta(tau_i | scene, goal_i)
```

再把这一组 candidate 的相对概率写成：

```text
p_theta(i) = softmax(log pi_theta(tau_i | scene, goal_i))
```

主损失是 soft distillation：

```text
L_soft = KL(q || p_theta)
```

脚本里还提供两个可选项：

1. `top1_weight`
   - 对 teacher top-1 candidate 再加一个 log-prob anchor
   - 类似一个轻量版的 SFT 项

2. `ref_kl_weight`
   - 如果大于 `0`
   - 会额外建立一个冻结 base policy
   - 用 `KL(p_ref || p_theta)` 抑制分布漂移

## 训练脚本

入口：

- `flow_planner/dpo/train_soft_pref.py`
- 镜像入口：`flow_planner/dpo/dpo/train_soft_pref.py`

推荐先用一个很小的 smoke test：

```bash
python -m flow_planner.dpo.train_soft_pref \
  --config_path /path/to/config_goal.yaml \
  --ckpt_path /path/to/model_goal.pth \
  --scored_dir /path/to/preferences_scored/scored_dir \
  --scene_dir /path/to/dpo_mining \
  --candidates_dir /path/to/dpo_candidates_goal \
  --output_dir /path/to/checkpoints/soft_pref_smoke \
  --epochs 1 \
  --batch_size 2 \
  --max_scenes 64 \
  --num_t_samples 4 \
  --require_goals \
  --save_merged
```

更像样的第一轮建议：

```bash
python -m flow_planner.dpo.train_soft_pref \
  --config_path /path/to/config_goal.yaml \
  --ckpt_path /path/to/model_goal.pth \
  --scored_dir /path/to/preferences_scored/scored_dir \
  --scene_dir /path/to/dpo_mining \
  --candidates_dir /path/to/dpo_candidates_goal \
  --output_dir /path/to/checkpoints/soft_pref_run1 \
  --epochs 3 \
  --batch_size 4 \
  --num_t_samples 8 \
  --gt_weight 0.5 \
  --score_weight 1.0 \
  --top1_weight 0.1 \
  --ref_kl_weight 0.05 \
  --require_goals \
  --save_merged
```

## 推荐流程

完整流程建议变成：

1. `cluster_goals.py`
2. 训练 `goal-conditioned base`
3. `generate_candidates_goal.py`
4. `score_hybrid.py --use_structured_scores --emit_traj_info`
5. `train_soft_pref.py`
6. `build_multi_pairs.py`
7. `train_dpo.py`

也就是：

```text
base -> soft preference distill -> hard DPO
```

## DriveDPO 风格的 hard negative

在这版实现里，`hard DPO` 前的 pair mining 也做了升级。

`build_multi_pairs.py` 现在的 rejected 选择顺序是：

1. `strict_same_group`
   - 优先保留原本“同 cluster / 同 maneuver 里的 subtle bad”
2. `gt_near_unsafe`
   - 从“离 GT 很近但不安全/低分”的候选里补 hard negative
3. `chosen_near_unsafe`
   - 从“离 chosen 很近但不安全/低分”的候选里补 hard negative
4. `same_group_soft`
5. `cross_group_soft`
6. `hard_failure_fallback`

也就是说，现在的逻辑不再是：

```text
strict same-group 找不到 -> 这个 scene 直接空掉
```

而是：

```text
strict subtle bad 优先
-> DriveDPO 风格 unsafe negative
-> 普通 fallback
```

这样做的目标是：

- 保留“同类行为里比优劣”的干净 pair
- 同时补上“看起来接近正确，但其实危险”的 hard negatives

### 新增的 pair mining 配置

`build_multi_pairs.py` 新增了这些参数：

- `--gt_near_unsafe_per_good`
- `--chosen_near_unsafe_per_good`
- `--unsafe_score_threshold`

默认值是：

```text
--gt_near_unsafe_per_good 1
--chosen_near_unsafe_per_good 1
--unsafe_score_threshold 0.55
```

并且 meta 里会新增：

```text
selection_source
```

用来标记这个 pair 是来自：

- `strict_same_group`
- `gt_near_unsafe`
- `chosen_near_unsafe`
- `same_group_soft`
- `cross_group_soft`
- `hard_failure_fallback`

部署时要重点看这个统计，确认新逻辑真的生效了。

## 和 hard DPO 的关系

这一步不是替代 DPO，而是给 DPO 打底。

它解决的是：

- “scene 里整体排序对不对”
- “哪些候选应该更高概率”

后面的 DPO 再解决：

- “hard negative 要不要压下去”
- “chosen 相对 rejected 的偏好边界要不要更 sharp”

## 当前实现的边界

这个版本是最小可落地实现，还没有做下面这些增强：

1. scene 内 candidate 数目不一致时的 padding 训练
2. 更强的 simulator-level safety teacher
3. route-near / GT-near goal 混合采样
4. 更细的 per-dimension soft target 构造

如果后面要继续往 DriveDPO 靠，可以优先做：

1. `K=5 -> 8/10`
2. 更强的 rejected 选择策略
3. 更强的 reference regularization

## 推荐部署命令链

下面是一套适合异机部署的最小完整流程。

### Step 1. 生成 goal-conditioned candidates

建议先把 `K` 提到 `8`，不要继续只用 `5`。

```bash
python -m flow_planner.dpo.generate_candidates_goal \
  --data_dir /path/to/dpo_mining \
  --config_path /path/to/config_goal.yaml \
  --ckpt_path /path/to/model_goal.pth \
  --vocab_path /path/to/goal_vocab.npy \
  --output_dir /path/to/dpo_candidates_goal \
  --num_candidates 8 \
  --use_cfg \
  --cfg_weight 1.8
```

### Step 2. 结构化打分并输出 `scored_dir`

```bash
python -m flow_planner.dpo.score_hybrid \
  --candidates_dir /path/to/dpo_candidates_goal \
  --output_dir /path/to/preferences_scored \
  --use_structured_scores \
  --emit_traj_info \
  --skip_vlm
```

如果要开 VLM，把 `--skip_vlm` 去掉并补 `--api_key`。

### Step 3. soft preference distillation

```bash
python -m flow_planner.dpo.train_soft_pref \
  --config_path /path/to/config_goal.yaml \
  --ckpt_path /path/to/model_goal.pth \
  --scored_dir /path/to/preferences_scored/scored_dir \
  --scene_dir /path/to/dpo_mining \
  --candidates_dir /path/to/dpo_candidates_goal \
  --output_dir /path/to/checkpoints/soft_pref_run1 \
  --epochs 3 \
  --batch_size 4 \
  --num_t_samples 8 \
  --gt_weight 0.5 \
  --score_weight 1.0 \
  --top1_weight 0.1 \
  --ref_kl_weight 0.05 \
  --require_goals \
  --save_merged
```

### Step 4. 构建 hard DPO preference pairs

```bash
python -m flow_planner.dpo.build_multi_pairs \
  --scored_dir /path/to/preferences_scored/scored_dir \
  --candidates_dir /path/to/dpo_candidates_goal \
  --output_path /path/to/dpo_preferences/preferences_multi_drive_like.npz \
  --top_good_per_cluster 1 \
  --subtle_bad_per_good 2 \
  --min_score_gap 0.25 \
  --strict_pair_mining \
  --gt_near_unsafe_per_good 1 \
  --chosen_near_unsafe_per_good 1 \
  --unsafe_score_threshold 0.55
```

### Step 5. hard DPO

```bash
python -m flow_planner.dpo.train_dpo \
  --config_path /path/to/config_goal.yaml \
  --ckpt_path /path/to/checkpoints/soft_pref_run1/model_softpref_merged.pth \
  --preference_path /path/to/dpo_preferences/preferences_multi_drive_like.npz \
  --scene_dir /path/to/dpo_mining \
  --output_dir /path/to/checkpoints/dpo_drive_like \
  --epochs 3 \
  --batch_size 4 \
  --beta 0.1 \
  --sft_weight 0.1 \
  --num_t_samples 8 \
  --save_merged
```

## 部署质量检查

异机部署时，不要只看命令跑没跑完，要看下面这些检查点。

### A. candidate generation 阶段

你应该确认：

- 输出目录下确实有 `*_candidates.npz`
- 每个文件里有 `goal_labels`
- `num_candidates` 是你想要的值，比如 `8`

### B. `score_hybrid.py` 阶段

你应该确认：

- `preferences_scored/scored_dir/*.json` 被正常写出
- 每个 scene json 里有 `candidates`
- 每个 candidate 有 `total_score / metrics.ade / metrics.fde / primary_failure`

### C. `train_soft_pref.py` 阶段

日志里应该看到类似：

```text
Loaded X scored scenes for soft preference distillation (K=8)
Starting soft preference distillation
```

如果 `K` 不是你预期的 `8/10`，说明候选文件不一致或者读错目录了。

### D. `build_multi_pairs.py` 阶段

最重要的是看它最后打印的：

```text
Pair type counts: ...
Selection source counts: ...
Dim label counts: ...
```

这里至少要检查：

1. `Selection source counts` 里不应该只有 `strict_same_group`
2. 最好能看到非零的：
   - `gt_near_unsafe`
   - `chosen_near_unsafe`
3. 如果全部都是 `hard_failure_fallback`，说明 subtle/hard negative 质量不够好
4. 如果 pair 总数非常少，优先检查：
   - `K` 是否太小
   - `strict_pair_mining` 是否太苛刻
   - `unsafe_score_threshold` 是否太低

### E. `train_dpo.py` 阶段

你必须看到：

```text
Loaded pair-specific chosen/rejected goals for goal-conditioned DPO.
```

如果看到的是：

```text
Preference file has no pair-specific goals; DPO train will stay goal-free.
```

说明你喂进去的 preference 文件不对，这轮训练等于没把 goal 接回去。

## 推荐的质量门槛

我建议异机部署时至少做到下面这些再往下跑大实验：

1. `soft_pref` smoke test 能正常收敛，`top1_match` 比随机高
2. `build_multi_pairs.py` 能稳定产出非零 pair，而且 `selection_source` 不只一种
3. `train_dpo.py` 确认读到了 `chosen_goals / rejected_goals`
4. 至少跑一个小规模 eval，看 `goal_mode=none` 和 `route_goal / predicted_goal` 是否方向一致

## 失败时优先排查什么

### 情况 1：soft pref 读不到 scene

先检查：

- `--scored_dir`
- `--scene_dir`
- `scored_dir` 里的 `source_npz` 是否还是旧机器绝对路径

如果是，就补：

```bash
--candidates_dir /path/to/dpo_candidates_goal
```

### 情况 2：pair 数特别少

优先做：

1. `K=5 -> 8/10`
2. 把 `--unsafe_score_threshold` 从 `0.55` 提到 `0.60`
3. 暂时把 `--subtle_bad_per_good` 保持 `2`

### 情况 3：DPO 虽然能跑，但像没吃到 goal

检查：

1. candidate npz 里是否有 `goal_labels`
2. `build_multi_pairs.py` 导出的 preference 文件里是否有 `chosen_goals / rejected_goals`
3. `train_dpo.py` 日志是否打印“loaded pair-specific goals”

## 这版逻辑的总结

现在这条线已经不是简单的：

```text
base -> generate pairs -> DPO
```

而是：

```text
goal-conditioned base
-> structured scoring
-> soft preference distillation
-> DriveDPO-style hard negative mining
-> hard DPO
```

这是当前仓库里一条比较完整、可以拿去异机部署的版本。

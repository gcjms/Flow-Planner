# Goal-DPO 数据流说明

## 这份文档是干什么的
这份文档只解释一件事：

> 在不改 `flow_planner/dpo/train_dpo.py` 主接口的前提下，怎么把上游数据流从“一个场景只产 1 对偏好”升级成“一个场景产多对偏好”。

对应的 3 个脚本是：

1. `flow_planner/dpo/analyze_candidate_modes.py`
2. `flow_planner/dpo/score_hybrid.py`
3. `flow_planner/dpo/build_multi_pairs.py`

一句话主线：

```text
候选轨迹
-> 判断有没有真正的行为分叉
-> 给每条候选做结构化诊断
-> 从每个行为簇里构造 multiple preference pairs
-> 继续喂给 train_dpo.py
```

## 整体数据流

```text
*_candidates.npz
    -> analyze_candidate_modes.py
        -> mode_report.json / mode_report_scenes.jsonl

*_candidates.npz + mode_report
    -> score_hybrid.py
        -> scored_dir/{scenario_id}.json
        -> 旧版 preferences.npz（baseline 兼容）

scored_dir + *_candidates.npz
    -> build_multi_pairs.py
        -> preferences_multi.npz
        -> preference_meta.jsonl

preferences_multi.npz + scene_dir/{scenario_id}.npz
    -> train_dpo.py
```

## 1. `analyze_candidate_modes.py`

### 作用
回答一个问题：

> 候选到底是“真多模态”，还是“只是几何上散了一点”。

### 输入
- `candidates_dir/*.npz`
- 必需字段：
  - `candidates`
- 可选字段：
  - `goal_labels`
  - `ego_agent_past`
  - `ego_agent_future`

### 输出
- `mode_report.json`
  - 全局汇总
- `mode_report_scenes.jsonl`
  - 每个场景一条记录
  - 包含候选级的 `cluster_id` / `maneuver_tag` 等信息

### 主要指标
- `pairwise_ade`
- `pairwise_fde`
- `pairwise_heading_deg`
- `endpoint_spread_mean`
- `cluster_count`
- `cluster_entropy`
- `unique_goal_count`
- `goal_maneuver_consistency`

### 它不负责什么
- 不参与训练
- 不产出 DPO pair
- 只是诊断候选有没有行为簇

### 示例
```bash
python -m flow_planner.dpo.analyze_candidate_modes \
    --candidates_dir outputs/candidates_goal \
    --output_json outputs/mode_report_goal.json
```

## 2. `score_hybrid.py`

### 作用
把原来的“场景级排序器”升级成“候选级结构化诊断器”。

它现在有两种工作模式：

1. 旧模式
   - 继续输出一个 `chosen` 和一个 `rejected`
   - 保持旧版 `preferences.npz` 和 `preference_details.json`

2. 新模式
   - 给每条候选都生成一份 `traj_info`
   - 写入 `scored_dir/{scenario_id}.json`
   - 供 `build_multi_pairs.py` 使用

### 结构化 `traj_info` 里有什么
每条候选会有这些字段：

- `candidate_idx`
- `scenario_id`
- `goal_label`
- `cluster_id`
- `maneuver_tag`
- `hard_ok`
- `hard_failures`
- `scores.margin`
- `scores.progress`
- `scores.comfort`
- `scores.route`
- `scores.legality`
- `scores.semantic`
- `primary_failure`
- `total_score`
- `rank`

### 这些字段在干嘛
- `cluster_id` / `maneuver_tag`
  - 用来区分不同候选属于哪个行为簇
- `hard_ok`
  - 这条轨迹是否通过硬约束
- `hard_failures`
  - 明确写出硬错误，例如碰撞、越界、反向
- `scores.*`
  - 分项分数，而不是只看总分
- `primary_failure`
  - 这条轨迹主要死在哪
- `total_score`
  - 最终排序分

### `hard_ok=False` 的典型情况
- 碰撞余量太小
- 明显偏离可行区域
- 明显负进度 / 反向

### VLM 的定位
- 不是主评分器
- 只在 high-spread 场景下做语义 tie-break
- 主体仍然是规则评分

### 输出
旧输出仍保留：
- `preferences.npz`
- `preference_details.json`

新输出：
- `scored_dir/{scenario_id}.json`

### 示例
```bash
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir outputs/candidates_goal \
    --output_dir outputs/scored_goal \
    --use_structured_scores \
    --emit_traj_info \
    --mode_report_json outputs/mode_report_goal.json \
    --skip_vlm
```

## 3. `build_multi_pairs.py`

### 作用
把 `score_hybrid.py` 产出的结构化 `traj_info` 变成 DPO 能直接吃的多对偏好数据。

### 输入
- `scored_dir/*.json`
- `candidates_dir/*.npz`

### 核心逻辑
1. 先按 `cluster_id` 分组；如果没有，就按 `maneuver_tag` 分组
2. 默认开启严格模式：
   - `good` 必须同时满足：
     - `hard_ok=True`
     - `primary_failure == "none"`
     - `total_score >= min_good_total_score`，默认 `7.0`
3. 严格模式下，`subtle bad` 只在同 group 内找，不再跨簇补样本
4. 一个同 group 的 `subtle bad` 必须同时满足：
   - `hard_ok=True`
   - `score_gap >= max(min_score_gap, strict_same_group_min_score_gap)`，默认至少 `0.75`
   - 并且满足下面二选一：
     - `primary_failure` 属于 soft failure（`progress / comfort / route / legality / semantic`）
     - 至少一个分项分数明显更差，`max(score_drop) >= strict_same_group_min_dim_drop`，默认 `0.15`
5. 如果一个 group 找不到合格的 `good -> bad`，那这个 group 直接跳过，不硬凑 pair
6. 如果想回到旧的宽松策略，可以加：
   - `--no_strict_pair_mining`
   - 这样就会恢复“同 group 优先，不够再跨 group soft failure，最后补 hard failure”的逻辑

### 为什么这样做
因为我们不想只做：

```text
top1 vs worst1
```

而是想做：

```text
每个合理行为簇里都保留 good representative
再给它配 subtle bad
```

这样 DPO 学到的不是“一个最好答案”，而是“多个合理模式都值得保留”。

同时默认把 pair 卡严，是为了尽量避免这三种脏信号：
- 簇内第一名其实也不够好，却被当成 `chosen`
- 同簇内差异太小，却被硬标成明显偏好
- 某个簇本来不够格，却为了凑数量拿跨簇样本来补

### 输出
- `preferences_multi.npz`
  - `chosen`
  - `rejected`
  - `scenario_ids`
  - `dim_labels`
- `preference_meta.jsonl`
  - `pair_id`
  - `scenario_id`
  - `chosen_idx`
  - `rejected_idx`
  - `chosen_cluster_id`
  - `rejected_cluster_id`
  - `pair_type`
  - `score_gap`
  - `failure_type`
  - `score_drops`
  - `goal_labels`

### 示例
```bash
python -m flow_planner.dpo.build_multi_pairs \
    --scored_dir outputs/scored_goal/scored_dir \
    --candidates_dir outputs/candidates_goal \
    --output_path outputs/preferences_multi.npz \
    --top_good_per_cluster 1 \
    --subtle_bad_per_good 2
```

如果你要做对照实验，再额外加：

```bash
--no_strict_pair_mining
```

## 4. 为什么还能直接接 `train_dpo.py`

因为我们故意没有改训练侧主接口。

训练侧现在认的是：

```text
chosen / rejected / scenario_ids / optional dim_labels
```

所以这里的兼容策略是：

- 每个 pair 展平成一行
- 同一个 `scenario_id` 可以重复多次
- 场景条件仍然从：
  - `scene_dir/{scenario_id}.npz`
  读取

也就是说：

- `train_dpo.py` 不需要知道一个场景原来有多少候选
- 它只需要继续读 pair 即可

## 最小跑通顺序

### Step 1：先看候选有没有行为簇
```bash
python -m flow_planner.dpo.analyze_candidate_modes \
    --candidates_dir outputs/candidates_goal \
    --output_json outputs/mode_report_goal.json
```

### Step 2：给每条候选打结构化标签
```bash
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir outputs/candidates_goal \
    --output_dir outputs/scored_goal \
    --use_structured_scores \
    --emit_traj_info \
    --mode_report_json outputs/mode_report_goal.json \
    --skip_vlm
```

### Step 3：把结构化结果压成 multi-pair
```bash
python -m flow_planner.dpo.build_multi_pairs \
    --scored_dir outputs/scored_goal/scored_dir \
    --candidates_dir outputs/candidates_goal \
    --output_path outputs/preferences_multi.npz
```

### Step 4：照常训练 DPO
```bash
python -m flow_planner.dpo.train_dpo \
    --ckpt_path checkpoints/model_goal.pth \
    --config_path checkpoints/config_goal.yaml \
    --preference_path outputs/preferences_multi.npz \
    --scene_dir /path/to/scene_npz_dir \
    --output_dir checkpoints/dpo_goal
```

## 最后一句话
这套新链路的核心不是“把脚本变多”，而是把职责拆清楚：

- `analyze_candidate_modes.py`
  - 负责判断候选有没有真分叉
- `score_hybrid.py`
  - 负责解释每条候选为什么好 / 为什么差
- `build_multi_pairs.py`
  - 负责把这些解释变成 DPO 可训练的数据

最终目标还是一句话：

> 不改 `train_dpo.py` 主接口，但让上游从“单 pair”升级成“multi-pair + 结构化偏好”。 

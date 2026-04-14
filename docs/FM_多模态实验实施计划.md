# FM 多模态实验实施计划

> 这份文档是 `FM_多模态实验路线图.md` 的执行版。  
> 路线图回答“为什么做、先后顺序是什么”；本计划回答“本周先干什么、改哪些文件、跑哪些命令、产出哪些表、通过门槛是多少”。

---

## 一、实验总目标

在不推翻现有 `Flow-Planner` 主体架构的前提下，完成一条可验证的主线：

```text
goal conditioning / SDE 扩展候选
    -> 结构化评分 + 多 pair 构造
        -> DPO 后训练
            -> 验证 N=1 质量和行为级多模态是否同步提升
```

最终要回答 3 个问题：

1. 当前 FM 的“多模态不足”主要是候选层问题，还是训练目标问题？
2. `goal conditioning` 是否真的能稳定造出行为级 mode？
3. `multi-good + subtle-bad + structured judge` 是否比 `GT vs collision` 更能保 mode？

---

## 二、现有资产盘点

当前仓库里已经有的关键脚本：

### 已有，可直接复用

- `flow_planner/goal/cluster_goals.py`
- `flow_planner/dpo/generate_candidates_goal.py`
- `flow_planner/dpo/measure_sde_diversity.py`
- `flow_planner/dpo/score_hybrid.py`
- `flow_planner/dpo/vlm_score_candidates.py`
- `flow_planner/dpo/train_dpo.py`
- `flow_planner/dpo/eval_multidim.py`
- `docs/goal_conditioning_guide.md`
- `docs/FM_多模态实验路线图.md`

### 建议新增

- `flow_planner/dpo/analyze_candidate_modes.py`
  - 用于算 cluster count / entropy / mode coverage
- `flow_planner/dpo/build_multi_pairs.py`
  - 用于把 `traj_info` 展开成一场景多 pair 的 `preferences_multi.npz`
- `flow_planner/dpo/vlm_pair_judge.py`（可选）
  - 边界 case 上做 pairwise VLM tie-break

### 建议修改

- `flow_planner/dpo/score_hybrid.py`
  - 从“输出一个 chosen / rejected”升级到“输出每条候选的结构化评分 + failure type”
- `flow_planner/dpo/train_dpo.py`
  - 确认可直接吃“一个场景多 pair 展平后的 dataset”；如不能，则加 dataset flatten 支持

---

## 三、执行原则

### 原则 1：先诊断，再增强

先确认问题到底卡在哪，再决定是优先强化 `goal` 还是优先强化 `preference`。

### 原则 2：先规则，后 VLM

先把：

- hard safety
- progress
- comfort
- route

这些规则分数打稳，再把 VLM 作为语义层 tie-break 接进来。

### 原则 3：先 DPO，后 GRPO

第一轮只做 DPO，理由：

- 工程闭环更短
- 不需要在线 rollout
- 更适合验证“pair 设计有没有信息量”

GRPO 只作为第二阶段增强对照组。

---

## 四、四周执行排期

下面给一个最现实的 4 周计划。

---

## Week 1：补齐仪表盘，确认 baseline

### 目标

建立一套可信的“多模态 + 质量”评测基线。

### 本周任务

#### T1-1 修好 / 校验 scorer 方向性

重点检查：

- `score_hybrid.py` 里邻居未来外推是否可用
- `collision / TTC` 是否与 closed-loop 方向一致

**产出**

- `reports/baseline_scorer_check.md`
- 50~100 个场景的 open-loop / closed-loop 对齐分析表

#### T1-2 新增 mode 分析脚本

新增：

- `flow_planner/dpo/analyze_candidate_modes.py`

功能：

- 输入：候选 `npz`
- 输出：
  - pairwise ADE
  - endpoint spread
  - cluster count
  - cluster entropy
  - unique goal count
  - goal->maneuver 一致性

**推荐实现**

- 先用终点 + 中点 + 最大横向偏移做聚类
- 后续可再加规则化 `maneuver tag`

#### T1-3 构一份 one-to-many 验证集

目标：单独筛出 100~200 个最可能有多个合理解的场景。

建议来源：

- `hard_scenarios_v2`
- 规则筛选：
  - 前方有慢车 / 静止障碍
  - 左右侧有至少一个可用空间
  - 路口处存在“等待 / 通过”双策略

**产出**

- `splits/one_to_many_val.txt`

### 推荐命令

```bash
# 1) baseline 候选（原始 ODE）
python -m flow_planner.dpo.generate_candidates \
    --data_dir <scene_dir> \
    --config_path <config.yaml> \
    --ckpt_path <model.pth> \
    --output_dir outputs/candidates_ode \
    --num_candidates 8

# 2) 候选分析（新脚本）
python -m flow_planner.dpo.analyze_candidate_modes \
    --candidates_dir outputs/candidates_ode \
    --output_json outputs/mode_report_ode.json
```

### 通过门槛

- scorer 与 closed-loop 至少方向一致
- `analyze_candidate_modes.py` 能稳定产出报告
- one-to-many 集构建完成

如果这周没完成，不进入 Week 2。

---

## Week 2：验证 `goal conditioning` / `SDE` 到底谁在造 mode

### 目标

确认：

- 候选是否只是“几何发散”
- 还是已经出现“行为簇分叉”

### 本周任务

#### T2-1 跑 4 组候选

在同一套 scene 上跑：

1. `ODE baseline`
2. `SDE only`
3. `goal only`
4. `goal + SDE`

#### T2-2 跑 CFG sweep

确认：

- `cfg_weight` 是否正在压制 goal 差异

建议扫：

- `0.5`
- `1.0`
- `1.8`
- `3.0`

#### T2-3 比较行为簇

重点看：

- `cluster count`
- `cluster entropy`
- `unique maneuver count`
- `每个 goal 的平均质量`

### 推荐命令

```bash
# SDE 多样性
python -m flow_planner.dpo.measure_sde_diversity \
    --ckpt_path <model.pth> \
    --config_path <config.yaml> \
    --scene_dir <scene_dir> \
    --num_scenes 100 \
    --num_samples 20 \
    --sigma_base "0.1,0.3,0.5" \
    --sde_steps 20 \
    --cfg_weight 1.8

# goal 候选
python -m flow_planner.dpo.generate_candidates_goal \
    --data_dir <scene_dir> \
    --config_path <goal_config.yaml> \
    --ckpt_path <goal_model.pth> \
    --vocab_path <goal_vocab.npy> \
    --output_dir outputs/candidates_goal \
    --num_candidates 8

# mode 分析
python -m flow_planner.dpo.analyze_candidate_modes \
    --candidates_dir outputs/candidates_goal \
    --output_json outputs/mode_report_goal.json
```

### 通过门槛

至少满足下面一条：

1. `goal only` 的 cluster count 明显高于 `ODE only`
2. `goal + SDE` 的 cluster entropy 明显高于 `goal only`

如果所有设置都只提升几何 spread、不提升行为簇，先暂停 DPO，回头继续修 mode 生成。

---

## Week 3：把评分从“总分”升级成“结构化判断”

### 目标

把现有：

- `top1 vs worst1`

升级成：

- `multi-good vs subtle-bad`

### 本周任务

#### T3-1 改造 `score_hybrid.py`

从当前输出：

- 一个 `chosen_idx`
- 一个 `rejected_idx`

改成输出每条候选的：

```python
traj_info = {
    "traj": ...,
    "goal_label": ...,
    "maneuver_tag": ...,
    "hard_ok": True/False,
    "scores": {
        "margin": ...,
        "progress": ...,
        "comfort": ...,
        "route": ...,
        "legality": ...,
        "semantic": ...,
    },
    "primary_failure": ...,
    "total_score": ...,
}
```

#### T3-2 新增 `build_multi_pairs.py`

输入：

- 一个场景的所有 `traj_info`

输出：

- 展平后的多个 `(chosen, rejected)` pair

逻辑：

1. 先按 maneuver / cluster 分组
2. 每个 good group 取一个 best representative
3. 每个 representative 配 1~2 个 subtle bad
4. 输出 `preferences_multi.npz`

#### T3-3 VLM 只做边界 case

如果接 VLM，建议只在以下条件下调用：

- 两条轨迹都 `hard_ok=True`
- 规则总分接近
- maneuver 不同，或者语义上难分

不要让 VLM 去判断：

- collision
- off-road
- red-light

这些还是规则说了算。

### 推荐命令

```bash
# 结构化评分
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir outputs/candidates_goal \
    --output_dir outputs/scored_goal \
    --emit_traj_info \
    --use_structured_scores \
    --skip_vlm

# 多 pair 构造（新脚本）
python -m flow_planner.dpo.build_multi_pairs \
    --scored_dir outputs/scored_goal \
    --output_path outputs/preferences_multi.npz \
    --top_good_per_cluster 1 \
    --subtle_bad_per_good 2
```

### 通过门槛

- 平均每个场景至少产出 `2~4` 个高质量 pair
- pair 的 `score gap` 明显高于当前 baseline
- `good vs bad` 的 failure 类型可解释

如果 pair 数太少，优先回到 Week 2 增加候选多样性；  
如果 pair 数多但 gap 很小，优先继续打磨 judge。

---

## Week 4：DPO 训练与验证

### 目标

验证：

- `multi-good + subtle-bad + structured judge` 是否真的优于 `GT vs collision`

### 本周任务

#### T4-1 跑 3 组 DPO

至少对比这三组：

1. `Baseline DPO`
   - `GT vs collision`

2. `Goal DPO`
   - `goal candidates + top1 vs worst1`

3. `Structured Multi-Pair DPO`
   - `goal candidates + structured judge + multi-good/subtle-bad`

#### T4-2 开环评测

看：

- collision
- TTC
- progress
- comfort
- route consistency
- cluster entropy
- `N=1` 质量

#### T4-3 闭环评测

重点看：

- `N=1` 是否变好
- 是否减少对 `Best-of-N` 的依赖

### 推荐命令

```bash
# DPO 训练
python -m flow_planner.dpo.train_dpo \
    --config_path <goal_config.yaml> \
    --ckpt_path <goal_model.pth> \
    --preference_path outputs/preferences_multi.npz \
    --scene_dir <scene_dir> \
    --output_dir checkpoints/dpo_multi_pair \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --beta 5.0 \
    --sft_weight 0.1 \
    --num_t_samples 16 \
    --lora_rank 4 \
    --lora_alpha 16 \
    --save_merged

# 开环评估
python -m flow_planner.dpo.eval_multidim \
    --ckpt_path checkpoints/dpo_multi_pair/model_dpo_merged.pth \
    --config_path <goal_config.yaml> \
    --scene_dir <scene_dir> \
    --max_scenes 500
```

### 通过门槛

至少满足下面两个：

1. `DPO accuracy > 0.6`
2. `delta_mean` 持续为正并上升
3. `N=1` 开环碰撞率下降
4. 行为级多模态指标不比 baseline 明显变差

如果安全变好但 entropy 断崖式下降，说明模型在收缩到单一保守模式；  
如果 entropy 变好但安全显著恶化，说明 judge / pair 还不够稳。

---

## 五、每周固定产出

每周必须产出同格式报告，避免后面无法横向比较。

### 表 1：几何多样性

| 方法 | pairwise ADE | endpoint spread | max lateral spread |
|------|--------------|----------------|--------------------|

### 表 2：行为多样性

| 方法 | cluster count | cluster entropy | unique maneuver count |
|------|---------------|----------------|------------------------|

### 表 3：质量

| 方法 | collision | TTC | comfort | progress | route |
|------|-----------|-----|---------|----------|-------|

### 表 4：训练信号

| 方法 | pair count | avg score gap | DPO acc | delta mean |
|------|------------|---------------|---------|------------|

### 表 5：闭环

| 方法 | N=1 NR-CLS | N=5 NR-CLS | Δ |
|------|------------|------------|---|

---

## 六、推荐的优先实现顺序

如果时间有限，按下面顺序实现最划算：

### P0：必须先做

1. `analyze_candidate_modes.py`
2. `score_hybrid.py` 结构化输出
3. `build_multi_pairs.py`

### P1：非常值得做

4. `goal only` vs `goal + SDE`
5. `multi-good vs subtle-bad`
6. `structured judge`

### P2：第二阶段

7. `VLM pairwise judge`
8. `GRPO` 对照组
9. `DPO + Best-of-N` 联合验证

---

## 七、预期风险与对策

### 风险 1：goal 候选确实更散，但只是“假多模态”

**表现**

- geometry spread 提升
- cluster count 不提升

**对策**

- 重新定义 maneuver clustering
- 降低 CFG
- 增强 goal 词典或 goal 选取策略

### 风险 2：pair 数量不够

**表现**

- 每场景只有 0~1 个 pair

**对策**

- 增加 `num_candidates`
- 增强 SDE 探索
- 放宽 subtle bad 筛选阈值

### 风险 3：pair 数量够，但 signal 很 noisy

**表现**

- DPO accuracy 接近 0.5
- delta_mean 很小

**对策**

- 优先加 hard gate
- 改成结构化评分
- 只保留高置信度 pair
- VLM 只做边界 tie-break

### 风险 4：DPO 后安全提升，但 mode 崩了

**表现**

- collision 降
- entropy / cluster count 明显掉

**对策**

- 增加 multi-good pairs
- 降低 β
- 提高 `sft_weight`
- 禁止 good vs good pair

---

## 八、最终成功标准

这个项目成功，不是只看某一个指标，而是三件事同时成立：

1. **候选层**
   - goal / SDE 确实拉出了行为簇

2. **训练层**
   - structured multi-pair DPO 的信号明显强于 baseline DPO

3. **部署层**
   - `N=1` 推理质量提升
   - 对 `Best-of-N` 的依赖下降

一句话：

> **最终目标不是“采样时很花”，而是“单次推理更容易落到多个合理模式中的一个”。**

---

## 九、我对当前最推荐的主线

如果只能押一条线，我建议押：

```text
goal conditioning
    -> structured scoring
        -> multi-good / subtle-bad pairs
            -> DPO
```

而不是：

```text
先全力上 VLM
或
先直接跳到 GRPO
```

原因：

- 这条线和你们现有代码兼容性最高
- 改动面最小
- 最容易在 2~4 周内产出可信的正反结果
- 一旦有效，再往 `VLM judge` 或 `GRPO` 扩展也顺

---

## 十、和现有文档的关系

这份计划和现有文档的关系如下：

- `goal_conditioning_guide.md`
  - 负责已有 goal pipeline 的操作说明

- `FM_多模态不足与_HighLevel_Preference_Shaping.md`
  - 负责解释“为什么只调采样不够”

- `FM_多模态实验路线图.md`
  - 负责实验逻辑与分阶段假设

- **本文件**
  - 负责按周执行、按脚本落地

如果后面再加文档，我建议只再补一个：

- `FM_多模态实验结果模板.md`

专门统一汇总表格和实验记录格式。

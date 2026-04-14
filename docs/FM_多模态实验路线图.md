# FM 多模态实验路线图

> 目标：把“FM 生成轨迹不够多模态”这个问题，拆成一组可验证、可复现、可逐步推进的实验。  
> 这份文档不讲大而全的方法综述，只回答四个问题：
> 1. 先做什么  
> 2. 每一步看什么指标  
> 3. 看到什么现象说明方向对  
> 4. 下一步该往哪条分支走

---

## 一、总目标

我们当前要解决的，不只是“多采几条轨迹”，而是：

> **让 Flow Matching 在同一场景下，既能生成多个行为层面的合理模式，又能在训练后把这些模式保留下来。**

最终希望达到的状态是：

```text
同一场景 x
  -> 采样时能看到多个清晰行为簇
  -> judge 能分辨哪些簇合理、哪些簇不合理
  -> DPO / GRPO 后训练后，单次推理也更容易落到合理簇里
```

---

## 二、核心假设

本路线图基于两个假设。

### 假设 H1：当前 FM 的多模态不足，既是采样问题，也是训练目标问题

具体表现：

- `Best-of-N` 无效，不只是 selector 问题
- 候选之间几何差异小，行为差异更小
- 同一场景下，多数 sample 落在一个主模式附近
- 单 GT 监督让策略分布天然单峰化

### 假设 H2：Goal Conditioning + High-Level Preference Shaping 是互补的

- `goal conditioning` 负责把 mode 显式拆开
- `sampling / SDE / CFG` 负责把潜在 mode 挖出来
- `judge / DPO / GRPO` 负责保留合理 mode、压掉坏 mode

如果 H2 成立，那么最有前景的主线就是：

```text
Goal / SDE 造 mode
    + 
Judge / Preference 保 mode
```

---

## 三、总体分阶段路线

整个实验建议按 5 个阶段推进。

| 阶段 | 目的 | 核心问题 | 输出 |
|------|------|----------|------|
| Phase 0 | 仪表盘与基线校准 | 我们现在到底有没有“真多模态”指标？ | baseline report |
| Phase 1 | 候选层多样性诊断 | 候选不够多样，是 sampler 不行还是 policy 太窄？ | diversity diagnosis |
| Phase 2 | mode 扩展实验 | Goal / SDE / CFG 哪个最能拉开行为簇？ | mode expansion report |
| Phase 3 | preference 设计实验 | 什么样的 pair / judge 真正有助于多模态？ | preference ablation |
| Phase 4 | 后训练验证 | DPO / GRPO 后，多模态与安全是否同时提升？ | final comparison |

---

## 四、Phase 0：先把仪表盘补齐

### 4.1 目标

在开始谈“多模态变好了没有”之前，先确保我们测的不是假指标。

### 4.2 必做事项

#### E0-1 修好 / 校准现有 scorer

根据 `docs/experiment_log.md`，当前 Best-of-N 曾经失败的一个根因是：

- scorer 用了 `neighbor_past` 近似未来
- collision / TTC 两个最重要维度失真

所以本阶段必须确认：

- 未来邻居评估逻辑是否正确
- open-loop scorer 是否和 closed-loop 指标至少方向一致

#### E0-2 增加“行为级多模态”指标

当前只看：

- ADE
- endpoint spread
- max lateral spread

还不够。

必须新增：

- **maneuver cluster count**
- **cluster entropy**
- **unique goal count**
- **route branch coverage**
- **不同 cluster 的 scorer gap**

#### E0-3 建立一套固定评测场景集

建议拆成 3 组：

1. **normal split**
   - 常规验证场景

2. **hard scenarios**
   - 已有 `hard_scenarios_v2`

3. **one-to-many scenes**
   - 人工或规则筛选出最可能存在多个合理解的场景
   - 例如：
     - 前车慢且可借道
     - 静态障碍可绕行
     - 路口有等待 / 通过两种合理策略

### 4.3 推荐指标

#### 几何多样性

- pairwise ADE
- endpoint spread
- mid-point spread
- max lateral spread

#### 行为多样性

- cluster count
- cluster entropy
- 不同 goal / maneuver 的覆盖率
- 不同候选的方向类别数

#### 质量指标

- collision
- TTC
- drivable area
- comfort
- progress
- route consistency

### 4.4 本阶段成功标准

- baseline 指标稳定可复现
- scorer 与 closed-loop 方向基本一致
- 能清楚区分“几何散度”和“行为簇数量”

如果这一步没做完，后面所有多模态实验都容易变成噪声。

---

## 五、Phase 1：先诊断“多模态不足”到底卡在哪

### 5.1 核心问题

当前的多模态不足，到底是：

1. **candidate generation 不够强**
2. **policy 分布太窄**
3. 两者都有

### 5.2 实验组

#### E1-1 ODE baseline

固定模型，只改 seed：

- `ODE + seed sweep`

看：

- 几何散度是否增长
- 行为簇数量是否增长

#### E1-2 SDE baseline

用 `docs/sde_dpo_deploy_guide.md` 里的现有能力：

- 不同 `sigma_base`
- 不同 `sde_steps`

看：

- 几何散度是否增长
- 行为簇数量是否增长
- reward variance 是否增长

#### E1-3 CFG sweep

扫 `cfg_weight`：

- 低 CFG
- 中 CFG
- 高 CFG

验证：

- CFG 是否正在把候选重新拉回一个主模式

### 5.3 要回答的问题

这一阶段不是追求变强，而是要回答：

> **FM 当前有没有潜在多模态能力，只是没被释放出来？**

### 5.4 预期现象

#### 情况 A：几何散度涨，但 cluster count 不涨

说明：

- 只是“假多模态”
- sampler 只是在同一主盆地里搅动

#### 情况 B：cluster count 能涨，但质量迅速崩

说明：

- 有潜在 mode
- 但没有足够的高层约束，采样一放开就发散

#### 情况 C：不同设置能稳定拉出 2-3 个行为簇

说明：

- 你们是有希望做多模态的
- 下一步重点该转向 judge / preference 保 mode

---

## 六、Phase 2：验证 Goal Conditioning 是否真能“造 mode”

### 6.1 核心问题

你们现在的 `goal conditioning` 非常关键，因为它不是只靠噪声，而是显式枚举意图。

这一步要验证的不是“goal 能不能改变终点”，而是：

> **goal conditioning 能不能稳定地产生“行为级别不同”的候选。**

### 6.2 实验矩阵

| 实验 | 设置 | 目的 |
|------|------|------|
| E2-1 | ODE only | baseline |
| E2-2 | SDE only | 看随机扰动能不能出 mode |
| E2-3 | goal only | 看显式 goal 能不能出 mode |
| E2-4 | goal + ODE | 看 goal 是否已经足够 |
| E2-5 | goal + SDE | 看 goal 与随机探索是否互补 |
| E2-6 | goal + CFG sweep | 看 CFG 是否压制 goal 差异 |

### 6.3 关键指标

- `cluster count`
- `cluster entropy`
- 不同 `goal label` 对应的平均轨迹差异
- 不同 `goal` 是否映射到不同 maneuver
- 每个 `goal` 的平均质量分数

### 6.4 最重要的判断标准

不是“5 条轨迹终点是否不同”，而是：

> **不同 goal 是否稳定对应不同的行为簇，而不是只对应几何上略有不同的一类轨迹。**

### 6.5 预期

我更看好下面这个结果：

- `goal only` 比 `SDE only` 更容易提升行为簇数量
- `goal + SDE` 比单独用 goal 更有机会发现稀有簇

如果这个结论成立，后面主线应该优先走：

```text
goal conditioning 作为 mode scaffold
SDE / seed 作为次级探索增强
```

### 6.6 `goal conditioning` 在整个路线里的真实定位

这里需要特别强调，避免后面路线跑偏。

`goal conditioning` 的价值，不是“单独解决最终多模态问题”，而是：

> **先把候选空间撑开，把潜在行为模式显式拆出来，让后续的 judge / DPO 真正有东西可学。**

也就是说，它最直接的作用是：

- 让不同候选不再只是同一条轨迹的微扰版
- 让 pair mining 不再停留在“同类轨迹里比高低分”
- 让 DPO 的 chosen / rejected 更有信息量

但它**不自动保证**：

- 多个合理 mode 最终都会长期保留下来
- 训练后的策略分布会天然变成多峰

原因是：

- 如果后续 preference 仍然是“唯一 GT 正确”
- 那么模型仍然可能只偏向某一个 goal 对应的模式
- 其他本来也合理的 goal / mode 依然可能被训练慢慢压掉

所以在这条路线里，三者分工应该理解成：

```text
goal conditioning
    -> 负责“有不同候选”

high-level preference shaping
    -> 负责“多个合理候选都别被压没”

DPO / GRPO
    -> 负责把这个偏好真正写进参数
```

一句最短总结：

> **`goal conditioning` 不是在替代 `high-level preference shaping`，而是在给后者创造施展空间。**

因此，如果当前目标是“先让 DPO 挖到更有区分度的 pair”，那么 `goal conditioning` 已经非常有价值；  
而如果目标进一步变成“让单次推理本身保留多个合理行为模式”，那就必须继续推进 Phase 3 的 preference 设计实验。

---

## 七、Phase 3：验证“高层偏好”是否真的能保 mode

### 7.1 核心问题

就算 goal / SDE 能造出 mode，如果 preference 设计还是：

- `chosen = GT`
- `rejected = collision`

模型最终仍然可能只保留最接近 GT 的那个模式。

所以这一步的目标是验证：

> **什么样的 pair 构造和 judge 输出，才真正有利于多模态保留。**

### 7.2 实验轴 1：pair 类型

#### E3-1 单一 GT 正样本

- `chosen = GT`
- `rejected = collision`

作用：

- 学安全边界
- 作为 baseline

#### E3-2 subtle negative

- `chosen = GT`
- `rejected = near-collision / low-margin / low-progress`

作用：

- 看“微妙差异”是否比“极端差异”更有利于学边界

#### E3-3 multi-good pairs

- 每个场景不是只保留一个 chosen
- 从多个 good cluster 中各保留一个代表

作用：

- 明确告诉模型：一个场景允许多个 good mode

#### 说明：DPO 仍然是 pairwise，不是 setwise

这里很容易误解，必须说清楚。

即使我们引入：

- 多个 good 候选
- 多个 bad 候选

**标准 DPO 公式本身也不用改。**

因为 DPO 依然只吃：

```text
(chosen, rejected)
```

的二元比较。

真正的变化是：

> **一个场景不再只贡献 1 个 pair，而是可以贡献很多个 pair。**

例如同一个场景里，如果经过 goal conditioning + judge 之后得到：

```text
good set = {g1=跟车, g2=左绕, g3=刹停}
bad set  = {b1=低安全余量跟车, b2=激进左绕, b3=追尾}
```

那么可以构造成：

```text
(g1, b1)
(g1, b3)
(g2, b2)
(g2, b3)
(g3, b3)
...
```

然后对这些 pair 分别计算普通 DPO loss，再求平均。

因此：

- **不是**“把一堆 good 和一堆 bad 一次性塞进一个新公式”
- **而是**“把好集合和坏集合展开成多个 pairwise comparison”

这也是最适合你们现有 `goal-conditioned candidates -> scorer -> preference.npz -> train_dpo.py` 流水线的做法。

#### 关键原则：good 不要互相打架

如果目标是“保留多个合理 mode”，那通常不应该构造：

```text
(g1, g2)
(g2, g3)
```

这种 `good vs good` 的 pair。

原因是这会强迫模型在多个合理模式之间分输赢，反而可能把其中一个压掉。

更合理的组织方式是：

- good 只和 bad 比
- 不同 good mode 各自配自己的 subtle bad

也就是说，DPO 要学的是：

> **多个 good mode 都比 bad mode 好**

而不是：

> **good mode 之间谁才是唯一正确答案**

#### 推荐的 pair 组织方式

对你们当前项目，我建议每个场景按下面的顺序构 pair：

1. 先用 `goal conditioning` 生成 `K` 条候选
2. 用 judge / scorer 给每条轨迹打：
   - 总分
   - 结构化子分
   - maneuver tag
3. 先按 maneuver / cluster 分组
4. 每个 good cluster 只保留一个 best representative
5. 每个 good representative 配 1~2 个最像它但更差的 subtle bad

这样构出来的 pair 最有信息量，因为它们教模型学的是“边界”，不是“极端差异”。

#### 伪代码（当前项目适配版）

```python
def build_scene_pairs(candidates, judge_outputs):
    """
    candidates: K 条 goal-conditioned 候选轨迹
    judge_outputs:
        - total_score
        - safety / comfort / progress / legality / semantic
        - maneuver_tag
    """
    # 1. 先按 maneuver 或 cluster 分组
    groups = group_by_maneuver_or_cluster(candidates, judge_outputs)

    good_reps = []
    bad_pool = []

    for group in groups:
        ranked = sort_by_total_and_constraints(group)

        # 2. 每个 good group 取一个 best representative
        if is_good_group(ranked[0]):
            good_reps.append(ranked[0])

        # 3. 把 near-miss / low-margin / low-progress 的样本放入 bad pool
        bad_pool.extend(select_subtle_bad(ranked))

    pairs = []
    for g in good_reps:
        matched_bads = match_similar_but_worse(g, bad_pool, topk=2)
        for b in matched_bads:
            pairs.append((g, b))

    return pairs
```

这段逻辑最重要的含义是：

- 不是一个场景只产生一个 `(chosen, rejected)`
- 而是一个场景可以产生多个“mode-aware pairs”

如果这一步做得好，DPO 才真正有机会学到：

- 跟车 mode 可以保留
- 左绕 mode 也可以保留
- 刹停 mode 也可以保留
- 但每种 mode 的坏版本要压下去

### 7.3 实验轴 2：judge 类型

#### E3-4 scalar judge

- 只给一个总分

#### E3-5 structured judge

至少输出：

- safety
- comfort
- progress
- legality
- semantic consistency
- maneuver tag

#### 单独小节：多 pair 评分设计与 VLM tie-break

对你们当前项目，我不建议一上来把 VLM 当成“万能总分器”。  
更稳的设计是三层：

##### 第一层：规则 hard gate

这些维度优先用规则 / 几何方法决定：

- collision
- drivable area violation
- red-light / route impossible
- 明显运动学不可行

只要违反硬约束，直接进 `hard_bad_pool`。

##### 第二层：结构化软分

对剩余候选，保留一个结构化 score vector，而不是只算一个总分：

- `s_margin`：最小车距 / TTC / safety margin
- `s_progress`：有效前进进度
- `s_comfort`：加速度 / jerk / yaw-rate 平滑性
- `s_route`：路线一致性
- `s_legality`：边界性违规倾向
- `s_semantic`：场景语义合理性

其中前 5 项建议优先规则化，`s_semantic` 最适合后续交给 VLM。

##### 第三层：VLM 只做语义与社交层判断

VLM 更适合做：

- `semantic_consistency`
- `social_reasonableness`
- `maneuver_tag`
- 边界 case 的 pairwise tie-break

不建议让 VLM 决定：

- 是否碰撞
- 是否出界
- 是否闯灯

这些硬约束，规则系统通常更稳定。

#### 推荐输出 schema

每条候选建议保留：

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

有了这套输出以后，后面的 pair 构造直接复用前面 `E3-3 multi-good pairs` 那一套逻辑即可，不需要再退回：

- `top1 vs worst1`

#### VLM 的最佳用法：pairwise judge，而不是 absolute scorer

如果后面要引入 VLM，我更推荐：

- 先用规则 hard gate + 结构化软分筛掉明显好坏
- 只在“两个都可行但难分”的边界 case 上调用 VLM
- 让 VLM 输出：
  - `winner`
  - `maneuver_a / maneuver_b`
  - `safety_reason`
  - `semantic_reason`
  - `confidence`

一个合适的 JSON 输出可以是：

```json
{
  "winner": "A",
  "maneuver_a": "left_bypass",
  "maneuver_b": "follow_too_close",
  "safety_reason": "A keeps larger margin from the stalled obstacle",
  "semantic_reason": "A is more appropriate because the left lane is free",
  "confidence": 0.82
}
```

低置信度 pair 建议直接丢弃，不进入 DPO。

### 7.4 实验轴 3：positive 数量

#### E3-6 单 positive

- 每场景一个 chosen

#### E3-7 多 positive

- 每个 cluster 一个 chosen
- 每场景 top-k chosen

### 7.5 这一阶段最想验证的结论

> **如果 multi-good + structured judge 显著提升 cluster retention，那么 high-level preference shaping 对多模态是实质有效的。**

### 7.6 推荐观察指标

- DPO accuracy
- delta mean
- 训练后 cluster count
- 训练后 cluster entropy
- 单次推理落入不同 good cluster 的概率
- collision / TTC 是否同步改善

---

## 八、Phase 4：后训练方案对比

### 8.1 核心问题

当 mode 已经能造出来、judge 也更合理以后，再看：

> **DPO 和 GRPO 谁更适合“保 mode + 保安全”？**

### 8.2 推荐比较组

| 组别 | 说明 |
|------|------|
| G1 | goal-only，不做后训练 |
| G2 | goal + DPO（GT vs collision） |
| G3 | goal + DPO（multi-good + subtle-bad） |
| G4 | goal + GRPO（结构化 reward） |
| G5 | goal + DPO + Best-of-N |

### 8.3 推荐判断指标

#### 多模态

- behavior cluster count
- entropy
- distinct good modes

#### 单次推理质量

- N=1 open-loop safety
- N=1 closed-loop NR-CLS

#### 采样效率

- N=1 vs N=5 的收益差
- 是否达到“DPO 替代 Best-of-N”的目标

### 8.4 核心判断

如果 G3 明显优于 G2，说明：

> **真正起作用的不是“有 DPO”，而是“DPO 的 preference 是否允许多 good modes 共存”。**

---

## 九、建议的实验优先级

为了节省时间，不建议一上来全做。

### P0：必须先做

1. `E0-1` scorer / 指标校准
2. `E1-1 ~ E1-3` 基础诊断
3. `E2-3` goal-only

### P1：最值得做

4. `E2-5` goal + SDE
5. `E3-2` subtle negative
6. `E3-3` multi-good pairs
7. `E3-5` structured judge

### P2：第二批

8. `E3-7` 多 positive
9. `G3 vs G4` DPO vs GRPO
10. `G5` DPO + Best-of-N

---

## 十、建议的数据切分

### Split A：快速诊断集

- 100~200 个场景
- 用于 seed / SDE / goal 的快速 sweep

### Split B：hard set

- 用现有 `hard_scenarios_v2`
- 用于观察安全边界和 subtle negative

### Split C：one-to-many set

建议单独构一份小集合，专门包含：

- 可借道
- 可等待
- 可轻微绕行
- 路口可保守 / 可激进通过

因为多模态实验最怕的数据问题是：

> 场景本身就是“一对一”，你怎么采也不会有多个合理解。

---

## 十一、推荐记录的核心表格

每次实验都建议产出这 4 张表。

### 表 1：几何多样性

| 方法 | pairwise ADE | endpoint spread | max lateral spread |
|------|--------------|----------------|--------------------|

### 表 2：行为多样性

| 方法 | cluster count | cluster entropy | unique maneuver count |
|------|---------------|----------------|------------------------|

### 表 3：质量

| 方法 | collision | TTC | comfort | progress | legality |
|------|-----------|-----|---------|----------|----------|

### 表 4：后训练效果

| 方法 | DPO acc | delta mean | N=1 open-loop | N=1 closed-loop | N=5 closed-loop |
|------|---------|------------|----------------|------------------|-----------------|

---

## 十二、判断路线是否成功的标准

这个项目真正成功，不是只看到“散得更开”，而是同时满足：

1. **行为簇数量增加**
2. **不同簇中存在多个 good mode**
3. **训练后这些 good mode 没被收缩回去**
4. **N=1 质量提升**
5. **Best-of-N 的必要性下降**

也就是说最终要达成：

> **从“靠多采几条碰运气选优”  
> 变成“模型单次推理本身就更可能落在多个合理模式中的一个”。**

---

## 十三、建议的论文式故事线

如果后续这条路线成立，论文故事可以这么讲：

1. **问题**
   - FM 在单 GT 监督下存在单峰化倾向
   - Best-of-N 失败说明候选集合缺乏真实行为多样性

2. **观察**
   - 单纯增加采样扰动，只能带来几何散度，不一定带来行为级多模态

3. **方法**
   - 用 goal conditioning 显式建立 mode scaffold
   - 用 high-level preference shaping 保留多个 good modes

4. **结果**
   - 候选行为簇增加
   - 单次推理质量提升
   - 对 Best-of-N 的依赖下降

---

## 十四、最推荐的近期执行顺序

如果只给一个最现实的执行列表，我建议是：

### 第 1 周

- 补齐指标与 baseline
- 跑 `ODE / SDE / CFG sweep`
- 建立 one-to-many 场景小集合

### 第 2 周

- 跑 `goal-only` 与 `goal + SDE`
- 看 cluster count 和 unique maneuver count

### 第 3 周

- 做 `GT vs collision` 与 `multi-good vs subtle-bad` 的 DPO 对比
- judge 从 scalar 升级为 structured

### 第 4 周

- 跑闭环验证
- 看 `N=1` 是否提升
- 看是否能减少对 Best-of-N 的依赖

---

## 十五、最后一句话

这份路线图背后的核心判断只有一句：

> **多模态不是“采样足够散”就自然出现的。真正重要的是：模型是否被训练成愿意把多个合理行为模式都保留为高概率解。**

所以你们接下来最应该做的，不是只问：

> “怎么把 seed 调得更散？”

而是同时问：

> “这个训练目标有没有明确告诉模型：一个场景可以有多个合理驾驶策略？”

如果这两件事一起做，`goal conditioning + judge + DPO/GRPO` 这条线就非常值得继续往下推。

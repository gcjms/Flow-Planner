# FM 多模态不足与 High-Level Preference Shaping

> 这份文档专门回答一个问题：  
> **为什么 Flow Matching 明明是生成式模型，实际采样出来的轨迹还是不够多模态？**  
> 以及：**除了继续调 sampler / seed / SDE 之外，能不能通过更高层的偏好塑形（high-level preference shaping）来改善多模态？**

---

## 一、先说结论

我对当前问题的判断是：

> **你们现在缺的，很可能不只是“采不出更多轨迹”，而是“模型没有学会把多种合理行为都保留为高概率模式”。**

也就是说，当前问题有两层：

1. **显式采样层**
   - 能不能从同一个场景里挖出更多候选
   - 这取决于 `seed / SDE / CFG / noise / sampler`

2. **策略分布层**
   - 模型是否真的认为“同一个场景存在多种合理解”
   - 这取决于 **训练目标、reward、preference、judge 信号**

如果第 2 层不成立，那么第 1 层再怎么调，最后也常常只是在**单一主模式附近抖动**。

---

## 二、为什么“有噪声”不等于“有多模态”

很多人第一次接触 Diffusion / Flow Matching 都会有个直觉：

> 既然输入里有噪声，输出就应该天然多模态。

这个直觉只对了一半。

### 2.1 噪声只是“潜在分叉变量”，不是“模式保证器”

在 FM 里，噪声的确给了模型一个潜在自由度：

```text
场景 x + 噪声 z0 -> 轨迹 y
```

但训练目标如果长期在强化下面这件事：

```text
同一个场景 x -> 唯一 GT y*
```

那么对模型来说，最省 loss 的策略通常是：

> **把不同噪声都压回同一个主模式附近。**

因为：

- GT 只有一条
- 偏离 GT 就要被罚
- 另一条“虽然合理但不是 GT 的轨迹”在 loss 看来也还是错

所以久而久之，噪声会退化成：

- 局部抖动源
- 小尺度扰动源
- 而不是 mode selector

### 2.2 这正是你们现在看到的现象

从 `docs/experiment_log.md` 和 `docs/sde_dpo_deploy_guide.md` 里可以看出，你们已经碰到了这个经典症状：

- `Best-of-N` 没有效果
- 不同候选差异太小
- CFG 引导太强，候选被拉回同一条主路径
- 即使有多个 sample，很多时候也只是“同一类轨迹的微扰版”

这说明：

> **问题不只是 scorer 没选好，更可能是候选集合本身就没有足够多的行为模式。**

---

## 三、什么叫“真正的多模态”

这里要把两个概念分开。

### 3.1 假多模态：几何扰动型

例如同一个场景采出 5 条轨迹：

- 都是跟车
- 终点位置相差 0.3m
- 速度略快略慢
- 横向最大偏移相差 0.2m

这在数值上是“不同的”，但在策略层面并不是不同模式。

它更像：

> **同一种行为模式的多个抖动样本。**

### 3.2 真多模态：行为簇型

例如同一个场景里真的存在不同策略簇：

- 跟车减速
- 左绕超车
- 轻微减速等待再绕
- 直接停车避险

这些轨迹在几何上差异大，在语义上也对应不同的 driving maneuver。

这才是自动驾驶真正想要的“多模态”：

> **不是多条长得略有不同的轨迹，而是多种语义上成立的驾驶决策。**

---

## 四、当前 FM 为什么容易掉进“单模态主盆地”

### 4.1 单 GT 监督天然偏向 mode collapse

如果训练一直是：

```text
给定场景 x
最小化生成轨迹和 GT 的误差
```

那么模型最稳妥的选择通常是：

- 学一个主模式
- 把概率质量集中在最常见、最保守、最不容易出错的轨迹附近

在自动驾驶里这会表现成：

- 更容易学“跟车”
- 不太愿意学“有风险但合理的借道、绕行、进取通过”
- 面对一对多场景时，倾向于收敛到均值化/保守化行为

### 4.2 只调采样，只是在主盆地附近搅动

如果 policy 自身已经把概率分布压得很窄，那么：

- 换更多 seed
- 增大温度
- 用 SDE
- 扫更多 CFG

确实能让候选“散一点”，但很多时候只是：

> **在同一个 basin 里多搅几下。**

这就是为什么很多系统里会出现：

- 候选看起来有差异
- 但 scorer 一排，谁都差不多
- 甚至闭环指标完全不翻转

---

## 五、High-Level Preference Shaping 到底在塑什么

一句话：

> **它塑的不是采样器，而是“模型对好轨迹集合的认知边界”。**

### 5.1 从“点目标”变成“好集合”

传统 imitation 更像在说：

> 这个场景下，只有这条 GT 是标准答案。

而 high-level preference shaping 更像在说：

> 这个场景下，有一组行为是好的；有一组行为是差的。

例如一个“前车慢、左车道可借”的场景：

**可能的 good set：**
- 安全跟车
- 安全左绕
- 稍等片刻后再变道

**可能的 bad set：**
- 追尾
- 激进切出
- 压实线
- 明明可前进却完全停死

一旦训练信号从“单点逼近 GT”变成“落在 good set 即可”，模型就有理由把多个行为簇都保留下来。

### 5.2 它塑的是“概率质量怎么分布”

这是最关键的一点：

- sampler 负责“从分布中取样”
- preference shaping 负责“把概率质量压到哪里”

如果高层偏好告诉模型：

- 跟车是好的
- 左绕也是好的
- 等待也是好的

那么模型会更倾向于学出：

```text
p(y | x) = 多个合理模式的混合
```

而不是：

```text
p(y | x) = 一个 GT 附近的尖峰 + 少量抖动
```

这才是真正能帮助多模态的原因。

---

## 六、为什么“高层偏好”比“几何误差”更容易保留多模态

因为它们衡量的是不同的东西。

### 6.1 几何误差关心“像不像这条 GT”

例如：

- 离 GT 0.5m 远，罚
- 离 GT 1m 远，罚更多

这会把“另一种合理轨迹”也当成错误。

### 6.2 高层偏好关心“这是不是一种合理驾驶”

例如：

- 有没有碰撞
- 是否压线
- 是否过于激进
- 是否完全没有进度
- 是否和场景语义一致

这类目标往往对“小几何差异”不敏感，但对“行为模式差异”敏感。

所以在一个存在多个合理解的场景里：

- imitation 会说：只有 GT 最对
- high-level preference 会说：这些不同轨迹都可以，只是风格不同

于是模型就更有机会保留多解。

---

## 七、把这个思想放到你们当前 DPO 设置里看

### 7.1 当前最自然的 DPO 形式

你们现在已经在探索的路线，最自然的版本通常是：

```text
chosen   = GT 或更安全的轨迹
rejected = 碰撞轨迹 / 很差轨迹
```

这种方法当然有价值，尤其适合：

- 学安全边界
- 压碰撞
- 替代失效的 Best-of-N

但它对“多模态”的帮助是有限的。

原因很简单：

1. 正样本往往只有一个模式
2. 负样本又离正样本太远
3. 中间那些“另一种也合理”的模式没有被正向保留

结果就是：

> **模型会更安全，但不一定会更会“分叉”。**

### 7.2 想让 DPO 帮到多模态，pair 的结构要改

真正更有利于多模态的 pair，不应该只有：

- `GT vs 撞车`

而应该逐步扩展到：

- `good mode A vs subtle bad A`
- `good mode B vs subtle bad B`
- `good mode A` 和 `good mode B` 都被保留为 positive

例如：

**正样本：**
- 跟车但安全
- 左绕但安全
- 稍等再变道也安全

**负样本：**
- 看起来很像左绕，但安全余量不足
- 看起来很像跟车，但 TTC 太小
- 合法但过于激进，舒适度差
- 安全但进度严重不足

这时候 DPO 学到的不是“GT 胜过垃圾”，而是：

> **多个 good behavior manifold 都该被保留，不同 bad manifold 该被压下去。**

---

## 八、一个更具体的例子

### 场景

- 前方慢车
- 左车道可借
- 右侧有障碍
- 交规允许超车

### 只用 GT 监督

如果 GT 恰好是“跟车减速”，那么模型很容易学成：

- 跟车 = 对
- 左绕 = 偏离 GT
- 等待后绕 = 偏离 GT

最后输出分布大概是：

```text
90% 跟车主模式
10% 跟车主模式附近的扰动
```

### 加入高层偏好

如果 judge / preference 认为：

- 安全跟车 = good
- 安全左绕 = good
- 等待后绕 = good
- 追尾 = bad
- 激进切出 = bad

模型就会更倾向于学成：

```text
40% 跟车
35% 左绕
25% 等待后绕
```

这个时候：

- 不同 seed 才真正有机会采出不同模式
- Best-of-N 才会有真实选择空间
- DPO/GRPO 才不只是在局部修边

---

## 九、但必须强调：Preference Shaping 不是采样技巧的替代品

这里最容易误解。

### 它不能凭空创造 mode

如果模型从来没见过左绕这个合理模式，或者 sampler 根本采不出来，那再好的偏好也救不了。

所以：

> **Preference shaping 负责“保 mode”，sampling 负责“找 mode”。**

两者是串联关系，不是二选一。

### 正确链条应该是

```text
更强的采样 / 扰动
    -> 暴露更多潜在行为模式
    -> judge / preference 判断哪些模式值得保留
    -> DPO / GRPO 把这些模式的概率质量抬起来
```

这也是为什么对你们来说，最现实的方向不是“只做 judge”或“只调 SDE”，而是两条一起走。

---

## 十、对你们当前项目最实用的判断

我会把当前问题拆成这两个子问题：

### 问题 A：模型能不能采出不同模式？

如果不能，就先做：

- SDE
- 多 seed
- CFG sweep
- 更强的初始扰动
- 不同 route / condition perturbation

目标是先把潜在 mode 挖出来。

### 问题 B：模型愿不愿意保留这些模式？

如果能采出一些分叉，但训练后仍然收缩回单一模式，就要做：

- high-level reward / preference
- 多 positive 的 pair 构造
- 不只用 GT 当 chosen
- 不只用最差碰撞轨迹当 rejected

这一步才是 preference shaping 真正发挥作用的地方。

---

## 十一、推荐你们下一步做的实验

下面给一个按优先级排序的 ablation 列表。

### Ablation 1：采样增强是否真的增加“行为簇”

**目的：** 区分“只是几何抖动”还是“真的出现不同 maneuver”。

**设置：**
- Baseline ODE
- SDE
- 多 CFG
- 多 seed

**除了几何散度指标外，再加：**
- maneuver cluster 数量
- 每个 cluster 的占比
- 不同 cluster 的 scorer 差异

**建议指标：**
- endpoint spread
- max lateral spread
- average pairwise ADE
- cluster count
- cluster entropy

**预期：**
- 如果几何散了但 cluster count 不涨，说明只是“假多模态”

---

### Ablation 2：单一 GT pair vs 高层 pair

**目的：** 验证 DPO 到底是在学“安全边界”，还是在学“多行为保留”。

**组别：**

1. `GT vs collision`
2. `GT vs near-collision`
3. `multi-good vs bad`
4. `multi-good vs subtle-bad`

**对比看：**
- collision rate
- diversity
- mode coverage
- chosen/rejected margin

**预期：**
- `GT vs collision` 更偏安全收缩
- `multi-good vs subtle-bad` 更可能真正提升多模态

---

### Ablation 3：judge 只给总分 vs judge 给结构化偏好

**目的：** 验证 high-level preference 的分解是不是关键。

**组别：**

1. scalar total score
2. safety-only
3. safety + comfort + progress
4. safety + comfort + progress + legality + semantic consistency

**预期：**
- 总分容易 noisy
- 结构化偏好更容易形成稳定 pair
- 更适合做多 positive 保留

---

### Ablation 4：单 positive vs 多 positive

**目的：** 检验“一个场景是否必须允许多个 chosen”。

**组别：**

1. 每场景只保留 1 个 chosen
2. 每个 maneuver cluster 保留 1 个 chosen
3. 每场景保留 top-k chosen，再配多个 rejected

**关键看：**
- 训练后 mode collapse 是否减弱
- 不同 cluster 的概率质量是否都上升

---

### Ablation 5：DPO vs GRPO

**目的：** 检验偏好优化和 group RL 在多模态塑形上的差别。

**建议只在前 4 个 ablation 有结果后再做。**

因为：
- DPO 更稳，更适合先验证“偏好塑形是否有效”
- GRPO 更像第二阶段增强

---

## 十二、一个更接近落地的训练路线

如果按“最小改动、最大信息量”的原则，我建议你们这样推进：

### Phase 1：确认 mode 是否能被挖出来

- 用 SDE / 多 seed / 多 CFG
- 先在 open-loop 上看 cluster count 和 cluster entropy

**目的：** 证明模型不是完全没有潜在多模态能力。

### Phase 2：把 judge 从“总分器”升级成“结构化 critique”

至少输出：
- 安全
- 进度
- 舒适
- 合法性
- 场景语义一致性
- maneuver tag

**目的：** 为 multi-good preference 做准备。

### Phase 3：从单一 pair 变成多 positive pair

- 不是只拿 GT
- 而是从多个 good cluster 里各取代表轨迹

**目的：** 明确告诉模型“这个场景允许多个合理行为模式”。

### Phase 4：再做 DPO

此时 DPO 优化的就不只是“往 GT 靠”，而是：

> **把多个 good modes 一起托起来，同时把不同类型的 bad modes 压下去。**

---

## 十三、这件事最重要的实验假设

我建议你们把下面这句写成核心 hypothesis：

> **Flow Matching 的多模态不足，不仅来自采样探索不足，也来自训练目标过度单峰化。通过引入 high-level preference shaping，可以把“单点 GT 对齐”改写为“多合理行为集合对齐”，从而提升行为层面的多模态。**

这句非常适合作为后续实验设计的总纲。

---

## 十四、最关键的风险

### 风险 1：judge 太 noisy

如果 VLM / scorer 不能稳定区分：

- 真正合理的不同模式
- 和“看起来像但其实差一点就危险”的模式

那 preference 会很脏，DPO 会学偏。

### 风险 2：模型根本采不出 mode

如果 sampler 还是几乎只产出一类轨迹，那么 preference 再好也没用。

### 风险 3：多样性和安全的 trade-off

多模态不等于越散越好。

你们真正要追求的是：

> **多个“成立的策略簇”**

而不是：

> **一堆乱七八糟互相差很大的轨迹**

所以 diversity 指标一定要和：
- safety
- legality
- comfort
- progress

一起看。

---

## 十五、最后一句话

如果把这个问题说得最简洁一点，就是：

> **采样技巧决定“你能不能看到分叉”，  
> high-level preference shaping 决定“模型会不会把这些分叉当成值得保留的解”。**

对于你们当前的 FM 来说，我更倾向于判断：

> **多模态不足不是一个纯 sampler 问题，而是 sampler 问题 + 训练目标单峰化问题。**

所以后续最值得投入的方向，不是继续只调 seed，而是：

1. 先把潜在 mode 挖出来  
2. 再用更高层的 preference 把 mode 保住

这才是把“生成式 planner 的随机性”真正变成“行为层面的多模态”的关键。

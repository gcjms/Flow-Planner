# AlphaDrive 与 VLM+Diffusion+RL 路线深挖

> 目标：回答一个非常具体的问题  
> 对我们当前这条 `Flow Matching / Diffusion 生成多模态轨迹 -> judge 打分 -> DPO/GRPO 后训练` 的路线来说，`AlphaDrive` 和 `VLM + Diffusion + RL` 这些工作到底值不值得学，应该学什么，不该学什么。

---

## 一、先给结论

如果只从“对我们当前项目有没有直接借鉴意义”这个角度看：

1. **AlphaDrive 值得深看**
   - 不是因为它的 VLM 架构本身一定适合我们
   - 而是因为它代表了一个很重要的范式：**planning reasoning + post-training with GRPO**
   - 它最值得学的是：**后训练 recipe、reward 设计思路、SFT -> RL 的两阶段工程节奏**

2. **ReCogDrive / MindDriver 这类 VLM + Diffusion + RL 更值得深看**
   - 因为它们和我们当前设想的路线更接近：
   - `强语义模型 -> 生成式 planner -> 后训练优化`
   - 它们最值得学的是：**如何让 VLM 不直接替代 planner，而是给 planner 提供认知先验、条件信号、reward 或 preference**

3. **HCRMP 这类 LLM/VLM 只做辅助提示的工作，可以看，但优先级没那么高**
   - 适合以后做 side information / hint injection
   - 不适合当前作为主线

一句话总结：

> **AlphaDrive 值得学“怎么后训练 VLM policy”；ReCogDrive/MindDriver 值得学“怎么把强语义模型接到生成式 planner 上”。**

---

## 二、从我们项目出发，应该怎么给这些工作分类

为了避免被“LLM/VLM/RL/Diffusion”这些名字绕晕，最简单的办法不是看模型名，而是看它在系统里扮演什么角色。

一个自动驾驶后训练系统，通常有 4 个核心角色：

1. **Policy / Planner**
   - 真正输出轨迹或动作的模型
   - 例如：Flow Matching planner、Diffusion planner、VLM policy

2. **Judge / Critic / Reward Source**
   - 负责告诉你“哪个更好”“哪里更危险”
   - 例如：规则打分器、VLM judge、reward model、闭环仿真器

3. **Optimizer**
   - 用什么方法把 judge 的偏好回灌给 policy
   - 例如：PPO、GRPO、DPO、RLHF、RLAIF

4. **Conditioning / Reasoning Source**
   - 给 planner 补充高层语义、认知、解释或场景结构
   - 例如：reference line、language prompt、VLM hidden states、risk tags

从这个角度看，我们当前想做的是：

```
FM / Diffusion planner        = Policy
VLM / scorer                  = Judge
DPO                           = Optimizer
scene features + optional VLM = Conditioning
```

所以你真正该看的，不是“谁用了 LLM”，而是：

- 它的 **policy** 是谁
- 它的 **judge** 是谁
- 它的 **optimizer** 是谁
- 它把 VLM 放在了 **condition / policy / critic** 的哪个位置

---

## 三、AlphaDrive 深挖

论文链接：
- [AlphaDrive arXiv](https://arxiv.org/abs/2503.07608)
- [AlphaDrive OpenReview](https://openreview.net/forum?id=QRm2CEZH41)
- [AlphaDrive GitHub](https://github.com/hustvl/alphadrive)

### 3.1 它到底在做什么

AlphaDrive 的核心命题不是“把 VLM 接进自动驾驶”，而是：

> **如何把大模型时代的 reasoning + RL 后训练方法，迁移到自动驾驶 planning 上。**

它要解决的不是普通 IL 的平均解问题，而是：

- 长尾场景理解不够
- 纯 SFT 的 planning policy 容易学 shortcut
- 只会模仿，不会真正做多解推理和权衡

所以它采取的路线是：

```
VLM policy
  ├─ 先做 SFT，学会基本规划推理
  └─ 再做 GRPO，学会更好的规划偏好与探索
```

### 3.2 它的本质：VLM 自己当 policy

AlphaDrive 和我们当前路线最大的区别是：

- **AlphaDrive 里，VLM 自己就是 planner / policy**
- **而我们更像是 FM planner 当 policy，VLM 当 judge**

这意味着 AlphaDrive 关注的重点是：

- 让 VLM 输出更好的 planning reasoning
- 让 reasoning 最终导向更好的动作或轨迹
- 用 RL 把语言空间里的“说得对”和物理空间里的“开得对”对齐

### 3.3 它最值得学的地方

#### A. 两阶段 recipe：SFT -> RL

这是 AlphaDrive 最值得抄的第一件事。

它不是一上来就 RL，而是：

1. **SFT 阶段**
   - 先把模型训到“会基本做规划推理”
   - 让模型先有一个可用的初始 policy

2. **GRPO 阶段**
   - 再用 RL 去提升长尾、边界、安全、多样性
   - 让模型从“像人”走向“更优”

对我们项目的启发是：

- 不要一开始就拿一个没稳定收敛的 FM/Diffusion 模型直接上 DPO/GRPO
- 应该先有一个性能足够可靠的 base planner
- 后训练只负责**对齐偏好**，不要兼顾“从零学会规划”

#### B. reward 不是一个分数，而是一组规划导向约束

公开摘要明确写到 AlphaDrive 有 **4 个 GRPO-based planning rewards**。  
公开摘要没有完全展开公式，但社区解读通常把它概括为：

- planning accuracy
- action-weighted reward
- diversity reward
- format / structure reward

对我们最有启发的，不是这四个名字本身，而是它背后的思想：

> **自动驾驶 reward 不能只用一个“总分”，而要拆成若干和规划直接相关的子目标。**

这对我们非常重要，因为如果以后要从 DPO 往 GRPO 走，或者把 VLM judge 输出变成更稳定的 preference，最稳的办法不是只给一个 scalar，而是分解成：

- 安全
- 舒适
- 可行
- 语义合理
- 多样性

#### C. 先学 reasoning，再学 multimodality

AlphaDrive 一个很值得注意的现象是：

> **它不是先把多模态做出来，再加推理；而是先让模型学会 reasoning，随后在 RL 后训练中出现 emergent multimodal planning capability。**

这点对我们有个很重要的启发：

- 多模态不一定只能从采样技巧里来
- 也可以来自**更好的 high-level preference shaping**

换句话说：

如果我们的 judge / preference 足够好，`FM + DPO` 的多模态质量可能不只是“采样更多条”的问题，而是“偏好信号够不够会教模型区分多种合理行为”。

### 3.4 它不适合我们直接照搬的地方

#### A. 不建议现在把 FM 改成 VLM policy

原因很简单：

- 你们现在的核心资产是连续空间 planner（FM）
- 不是语言空间 planner
- 直接改成 VLM policy，相当于把整个 policy substrate 换掉了

这个成本太高，且会把当前技术主线打断。

#### B. 不建议一上来就学它的“推理即输出”

对你们现在更现实的做法是：

- **VLM 不直接出轨迹**
- 而是先做：
  - 轨迹打分
  - 风险解释
  - preference 生成
  - semantic tag / critique 输出

也就是：

> **先让 VLM 当 judge，再考虑让 VLM 当 conditioner，最后才考虑 VLM 当 policy。**

### 3.5 AlphaDrive 对我们最可落地的借鉴

如果只保留最关键的 3 条：

1. **训练节奏**
   - 先 base planner 收敛
   - 再做偏好/奖励后训练

2. **reward decomposition**
   - safety / comfort / legality / semantic consistency / diversity 分开建模

3. **structured post-training**
   - 不只要 scalar score
   - 更要可解释的 judge output，帮助构造更干净的 preference pair

---

## 四、VLM + Diffusion + RL 这条路线为什么更值得我们看

这类工作和我们的关系，比 AlphaDrive 更近。

因为它们的共同结构基本都是：

```
VLM / cognition model
      ↓
Diffusion planner / generator
      ↓
RL / preference optimization
```

这和我们的路线只差一步：

```
VLM / judge
      ↓
Flow Matching planner
      ↓
DPO
```

所以从“方法母题”上说，它们是我们最直接的近亲。

---

## 五、ReCogDrive 深挖

论文链接：
- [ReCogDrive arXiv HTML](https://arxiv.org/html/2506.08052v2)

### 5.1 它到底在做什么

ReCogDrive 的核心目标是解决一个很现实的问题：

> 直接让 VLM 用语言空间生成轨迹，常常会出现格式错误、动作不可行、推理慢、语言空间和连续动作空间不匹配。

所以它的答案不是“让 VLM 更强”，而是：

> **让 VLM 提供 cognition，让 Diffusion planner 负责连续轨迹生成。**

这点和我们非常像。

### 5.2 它的三层结构

根据论文摘要和正文概述，ReCogDrive 可以理解成三层：

1. **Cognitive VLM**
   - 通过层级数据流水线学驾驶认知
   - 不是直接输出最终轨迹
   - 而是输出认知 token / hidden states

2. **Diffusion Planner**
   - 接收 noisy trajectory + scene context + VLM cognitive tokens
   - 做连续轨迹去噪生成

3. **DiffGRPO**
   - 用 RL 对 diffusion planner 做后训练
   - 优化 safety / comfort beyond imitation

### 5.3 它最值得我们学的地方

#### A. VLM 不是替代 planner，而是 condition planner

这点极其重要。

ReCogDrive 并不是：

`VLM -> 直接输出轨迹`

而是：

`VLM hidden states -> cross-attention condition -> diffusion planner -> trajectory`

这给我们的直接启发是：

> **如果以后我们要把 VLM 接到 Flow Matching 上，最自然的位置不是输出端，而是 condition 端。**

也就是说，我们可以考虑：

- 当前先做 `VLM judge`
- 下一步做 `VLM semantic token -> encoder / decoder conditioning`
- 最后再看是否需要更深融合

#### B. 先做“认知预训练”，再做 planner 对齐

ReCogDrive 的数据流水线很值得注意：

- generation
- refinement
- quality control

它不是直接拿现成 VLM 就上，而是想办法构造高质量驾驶认知数据，让 VLM 先有**驾驶领域认知**。

对我们来说，这启发不是“我们也要复刻这套 pipeline”，而是：

> **如果 VLM 要当 judge，judge 本身也要做领域适配。**

否则会出现两个问题：

- 说得头头是道，但其实不懂驾驶
- 给出的 preference noisy，最后把 DPO 教偏

#### C. 它证明了“生成式 planner + 强语义条件 + RL”是可行的

对我们来说，ReCogDrive 的最大战略价值是：

> **它说明把高层认知和连续轨迹生成解耦，是一条成立的路。**

这等于从方法学上给我们背书：

- planner 可以继续是 FM / diffusion
- VLM 不必替代 planner
- 只要 VLM 能给 planner 更好的 semantic prior / reward / critique，就有价值

### 5.4 ReCogDrive 对我们的直接借鉴

最值得借鉴的不是它的全部系统，而是这三个点：

1. **VLM hidden states 作为 condition，而不是 action**
2. **judge / conditioner 先做驾驶领域对齐**
3. **生成式 planner 的后训练可以走 RL，不一定必须是纯监督**

---

## 六、MindDriver 深挖

论文链接：
- [MindDriver arXiv](https://arxiv.org/abs/2602.21952)

### 6.1 它解决的问题

MindDriver 认为，传统 CoT 在自动驾驶里有个核心断裂：

- 纯文本 reasoning 离真实物理轨迹太远
- 纯图像 future imagination 又缺少明确 planning objective

所以它提出一个三段式的 progressive reasoning：

1. **Semantic Understanding**
2. **Semantic-to-Physical Space Imagination**
3. **Physical-Space Trajectory Planning**

### 6.2 它为什么值得我们看

MindDriver 值得看的，不是因为它一定和我们架构最像，而是因为它回答了一个我们后面迟早会遇到的问题：

> **如果让 VLM 来评轨迹，VLM 到底是在看“语言合理性”，还是在看“物理可行性”？**

MindDriver 的回答是：

> 不要把这两者混在一起，而要显式分层。

### 6.3 对我们最有价值的启发

#### A. judge 不一定只输出一个分数

MindDriver 提醒我们，强语义模型的输出可以分层：

- 语义层结论
- 物理层想象
- 轨迹层判断

这对我们特别有用，因为如果我们现在直接做：

`VLM(scene, traj) -> scalar score`

会很快遇到两个问题：

1. 分数解释性差
2. preference 噪声大，DPO 容易学偏

更好的做法可能是让 judge 输出：

- 风险类型标签
- 决策解释
- 语义层 preferred / rejected 理由
- 最终 scalar ranking

也就是：

> **先结构化 critique，再压成 preference。**

#### B. progressive reward 的思想值得学

MindDriver 的 progressive reinforcement fine-tuning，核心不是某个具体公式，而是一个很重要的训练思想：

> **不要让一个 reward 同时承担“理解对不对”和“轨迹好不好”两件事。**

这对我们很关键。

如果未来我们做 VLM judge，最容易犯的错误是：

- 一次性给一个“总分”
- 结果 semantic mistake 和 physical mistake 混在一起

更好的办法是分阶段：

1. 先让 judge 学会基本场景语义对齐
2. 再让 judge 对轨迹安全/舒适/意图一致性做排序
3. 最后再把偏好喂给 DPO

### 6.4 它对我们不是最直接，但很有“方法论价值”

MindDriver 和我们不是一模一样，但它有个很大的价值：

> 它教我们不要把“VLM judge”想得过于简单。

如果后面你们想把 judge 从规则分数升级成强语义打分器，MindDriver 这种“语义 -> 物理 -> 轨迹”的分层思路会非常有帮助。

---

## 七、把三条路线放到我们当前项目里对比

| 维度 | AlphaDrive | ReCogDrive | MindDriver | 我们当前路线 |
|------|------------|------------|------------|--------------|
| 主 policy | VLM | Diffusion planner | VLM / multimodal reasoning planner | FM / Diffusion planner |
| VLM 角色 | policy 本体 | cognitive conditioner | reasoning backbone | judge / potential conditioner |
| 优化方式 | GRPO | DiffGRPO | progressive RL fine-tuning | DPO（计划） |
| 轨迹空间 | 多为离散/结构化输出 | 连续轨迹生成 | reasoning + planning | 连续轨迹生成 |
| 最值得借鉴 | 后训练 recipe、reward 分解 | VLM hidden state conditioning | 结构化 critique / progressive reward | judge 打分 + preference alignment |
| 直接可抄程度 | 中 | 高 | 中高 | - |

---

## 八、对我们最现实的技术路线建议

如果目标是尽快把“FM + judge + DPO”做成一个扎实的研究/工程方案，我建议按下面 4 步走。

### Step 1：先把 VLM 定位成 judge，不要让它直接出轨迹

优先做：

`scene + candidate trajectories -> VLM judge -> chosen/rejected / score / critique`

原因：

- 改动小
- 容易接到现有 FM 生成器后面
- 最适合做 DPO 数据构造
- 风险最可控

### Step 2：judge 输出不要只有一个标量

建议至少输出：

- 安全性
- 合法性
- 舒适性
- 意图一致性
- 一段简短解释或标签

然后再把它们压成：

- total rank
- chosen / rejected pair

这一步最受 MindDriver 启发。

### Step 3：先做 DPO，再考虑 GRPO

原因：

- DPO 更稳
- 不需要在线 rollout 的 RL 训练环
- 和你们现有 `docs/flow_matching_dpo_详解.md` 技术栈最一致

GRPO 可以作为后续 ablation / stronger baseline。

这一步最受 AlphaDrive 启发，但不建议直接跳过去。

### Step 4：如果 judge 有价值，再考虑把 VLM hidden states 接进 planner

这一步最受 ReCogDrive 启发。

可能的演化路线：

1. **judge-only**
2. **judge + structured critique**
3. **judge + semantic token conditioning**
4. **joint post-training**

不要倒着来。

---

## 九、哪些东西现在不要急着学

### 9.1 不要急着把 VLM 变成主 policy

原因：

- 改动太大
- 会把你们现有 FM 主线完全打断
- 难以和现有 DPO 文档、实验积累衔接

### 9.2 不要一开始就做最复杂的 online RL

原因：

- 训练不稳定
- 工程成本高
- judge 还没稳定前，reward 噪声会很大

### 9.3 不要只追求“更大的模型”

当前阶段更重要的是：

- preference quality
- chosen/rejected 构造质量
- judge 的结构化能力
- DPO / GRPO 的训练稳定性

不是先把 VLM 尺寸做大。

---

## 十、如果把这份深挖变成我们项目的 action item

### 最短路径版

1. 用现有 FM / diffusion planner 采样多条轨迹
2. 用规则 scorer + VLM judge 混合打分
3. 构造更“微妙”的 chosen / rejected
4. 先做 DPO
5. 再做和 GRPO 的对比实验

### 中期增强版

1. 做结构化 VLM critique
2. 把 critique 拆成多头偏好信号
3. 尝试 semantic token conditioning
4. 做 `judge-only` vs `judge+condition` 对比

### 论文故事版

可以形成这样一条故事线：

1. **问题**
   - IL / FM 存在 narrow policy、对安全边界不敏感

2. **观察**
   - 纯 Best-of-N 或纯 imitation 无法稳定选出语义上真正好的轨迹

3. **方法**
   - 用 VLM 作为 judge 构造 trajectory preference
   - 用 DPO 对齐 FM planner

4. **进阶**
   - 再把 VLM critique 作为 structured signal 注入 planner

这条线和 AlphaDrive / ReCogDrive / MindDriver 有血缘关系，但不会变成“简单复现别人的 VLM policy”。

---

## 十一、最终判断：到底该看什么

如果你接下来只投入有限时间，我建议这样排优先级：

### 必看

1. **AlphaDrive**
   - 学后训练节奏
   - 学 reward decomposition
   - 学为什么 reasoning 会带来多模态能力

2. **ReCogDrive**
   - 学 VLM hidden state 如何 condition diffusion planner
   - 学 cognition 与 continuous planner 的解耦方式

### 强烈建议看

3. **MindDriver**
   - 学 judge / reasoning 的分层设计
   - 学 progressive reward / critique 思路

### 可以后看

4. HCRMP 一类 hint-based 方法
   - 适合以后做 side information injection
   - 不是当前主线

---

## 十二、对我们当前项目的最终建议

最推荐的主线不是：

`把 VLM 直接变成 planner`

而是：

`FM planner + VLM judge + structured preference + DPO`

然后沿着这个主线逐步升级到：

`FM planner + VLM judge/conditioner + DPO/GRPO hybrid`

这条路线的好处是：

- 和现有 Flow Matching 资产兼容
- 和仓库中已有的 DPO 文档兼容
- 工程风险更低
- 论文故事也更完整

一句话收尾：

> **AlphaDrive 教我们怎么后训练“会推理的 policy”，ReCogDrive/MindDriver 教我们怎么把“会理解的模型”接到连续轨迹生成器上。对我们来说，最合理的落点不是 VLM 直接开车，而是 VLM 先当裁判，再逐步变成副驾驶。**

---

## 参考链接

- [AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning](https://arxiv.org/abs/2503.07608)
- [AlphaDrive OpenReview](https://openreview.net/forum?id=QRm2CEZH41)
- [AlphaDrive GitHub](https://github.com/hustvl/alphadrive)
- [ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving](https://arxiv.org/html/2506.08052v2)
- [MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving](https://arxiv.org/abs/2602.21952)

# Flow Matching 方向 3 设计草案

## 背景

我们当前已经在做方向 1：

- 用 `goal_point` 给 FM 提供更强的去歧义条件
- 训练时用 GT endpoint 对应最近的 goal cluster 作为条件
- 推理时可用不同 goal 生成不同候选

这条线本质上是：

`condition disambiguation`

即先把 mode 拆开，再在各个 mode 内做普通的单值 FM 回归。

但当前训练主体仍然是：

- 单个 decoder 输出
- 单个目标轨迹
- 单个 `MSELoss`

所以目前的范式仍然是：

`先分题，再单答`

这还不是方向 3。

## 方向 3 在做什么

方向 3 的核心不是“推理时多采样几条”，而是：

`训练时就让模型学会多个假设`

目标是缓解标准 FM 在多模态区域里的单值回归平均化问题。

一个更准确的说法是：

- 现在：给定 `(x_t, t, goal)`，模型输出一个 velocity / trajectory target
- 方向 3：给定 `(x_t, t, goal)`，模型输出 `K` 个 branch-specific hypotheses
- 训练时不要求所有 branch 都拟合同一个平均答案
- 而是要求“至少有一个 branch 负责当前 GT mode”，同时其余 branch 保持分散

## 为什么值得在方向 1 做完后再做方向 3

建议顺序：

1. 先把方向 1 做干净
2. 再做方向 3

原因：

- 当前还需要先确认 goal-conditioned eval 的真实收益
- 如果方向 1 本身就能显著恢复主指标，那方向 3 不一定是第一优先级
- 如果方向 1 能带来多样性，但规划质量仍然明显不够，方向 3 就非常自然

所以方向 3 更适合作为：

`goal-conditioned FM v2`

而不是立刻替换当前主线。

## 最小可做版本

最推荐先做一个最小版本，不要一上来就做复杂 mixture density 或 latent variable。

### v1 目标

在保留当前 `goal conditioning` 的前提下，把单输出 decoder 改成 `K` 分支输出。

即：

- 输入仍然是 `(x_t, t, encoder context, goal_point)`
- 输出从 `(B, P, T, D)` 变成 `(B, K, P, T, D)`

其中：

- `B`: batch
- `K`: hypothesis branches
- `P`: action token 数
- `T`: action length
- `D`: state dim

推荐先设：

- `K = 3` 或 `K = 4`

不要一开始就太大，不然训练和分析都会变复杂。

## 模型改法

### 方案 A：共享 trunk + 多分支 head

这是最适合当前代码的最小改法。

保留：

- encoder
- DiT trunk
- post_fusion

只改 decoder 最后的输出头。

当前大致是：

- `preproj`
- `DiT`
- `post_fusion`
- `final_layer -> (B, P, output_dim)`

可以改成：

- `preproj`
- `DiT`
- `post_fusion`
- `K` 个并列 `final_layer`
- 输出 `(B, K, P, output_dim)`

优点：

- 改动小
- 能快速验证方向 3 是否有价值
- trunk 仍共享场景理解能力，参数量增长可控

### 方案 B：共享 trunk + branch embedding + 单个共享 head

另一种更省参数的写法是：

- 增加 `branch_embedding: Embedding(K, hidden_dim)`
- 对每个 branch，把 `branch_embedding[k]` 加到 trajectory token 或条件向量里
- 然后用同一个 `final_layer` 逐 branch 输出

优点：

- 参数更省
- 更容易解释为“一个条件化的 set-valued field”

缺点：

- 实现比方案 A 稍复杂

建议：

先做方案 A，跑通后再考虑方案 B。

## 训练目标怎么改

### 当前目标

当前每个样本只有：

- 一条 GT trajectory
- 一个 `MSE(prediction, target_tokens)`

这会逼模型把所有不确定性压成单答案。

### v1 目标：best-of-K token loss

最小版可直接做：

1. 模型输出 `K` 个预测
2. 对每个 branch 分别算 token-level MSE
3. 对每个样本只取最小的那个 branch 参与主重建损失

形式上：

`L_recon = min_k L_mse(pred_k, target)`

或者 batch 维度上：

- 先算 `(B, K)` 的 branch loss
- 每个样本取 `argmin`
- 只回传 best branch

这一步的意义是：

- 不再逼所有 branch 拟合同一个平均轨迹
- 允许不同 branch 向不同 plausible mode 分化

### consistency loss 怎么办

当前有 action overlap consistency loss。

做法建议：

- 对每个 branch 分别计算 consistency
- 只对 best branch 加 consistency loss

即和主重建损失保持一致。

原因：

- 简单
- 不会过早约束所有 branch 都收缩到同一解

### diversity loss

如果只做 best-of-K，最容易出现的问题是：

- 一个 branch 学会所有东西
- 其余 branch 不工作，或者全塌到一起

所以 v1 最好就加一个很轻量的 diversity loss。

推荐从终点分散度开始，而不是整段轨迹两两拉开。

例如对每个样本：

- 取各 branch 最终 endpoint `e_k`
- 计算两两距离
- 鼓励平均 pairwise distance 大于一个 margin

可写成：

`L_div = mean(max(0, m - ||e_i - e_j||))`

推荐：

- 只在不同 branch 的 endpoint 上做
- margin 先取 `2m ~ 4m`

原因：

- 比整条轨迹 pairwise ADE 更稳定
- 更接近“mode 至少在决策层分开”

### 总损失

v1 可以先用：

`L = L_recon_best + lambda_cons * L_cons_best + lambda_div * L_div`

建议初始系数：

- `lambda_cons =` 延续当前配置
- `lambda_div = 0.05` 或 `0.1`

## 代码层面的最小改动点

### 1. `flow_planner/model/flow_planner_model/decoder.py`

增加：

- `num_branches` 配置
- 多个 `final_layer` 或 branch embedding

forward 输出改成：

- 当前：`(B, P, action_len, state_dim)`
- 目标：`(B, K, P, action_len, state_dim)`

### 2. `flow_planner/model/flow_planner_model/flow_planner.py`

训练部分要改：

- `prediction = self.decoder(...)` 后，prediction 不再是单个 branch
- 需要对每个 branch 计算 token loss
- 对每个样本取 best branch
- 记录：
  - `best_branch_idx`
  - `branch_loss_mean`
  - `diversity_loss`

推理部分也要改：

- 若 `num_branches == 1`，行为与当前兼容
- 若 `num_branches > 1`：
  - 单次 ODE 采样后得到 `K` 个 branch trajectory
  - 可直接返回 `(B, K, T, D)`
  - 或结合现有 `TrajectoryScorer` 选 best branch

### 3. config

新增配置项：

```yaml
model:
  num_branches: 3
  branch_diversity_margin: 3.0
  branch_diversity_weight: 0.1
```

decoder 侧也需要加：

```yaml
model_decoder:
  num_branches: 3
```

### 4. recorder / log

建议额外记录：

- `best_branch_histogram`
- `branch_loss_min`
- `branch_loss_mean`
- `branch_endpoint_pairwise_dist`

这些对判断 branch 是否塌缩很重要。

## 推理怎么用

### 当前模式

当前你们是：

- 固定一个 goal
- 输出一条轨迹
- 或换不同 goal 多跑几次

### 方向 3 后的模式

会变成两层多样性：

1. `goal` 层的 mode 拆分
2. `branch` 层的 mode 内多假设

所以一个很自然的使用方式是：

- 先选 `G` 个 goal
- 每个 goal 下输出 `K` 个 branch
- 最后得到 `G * K` 个候选

这个特别适合 DPO / reranking。

最小版本里也可以先不做这么复杂，只做：

- 单个 goal
- `K` 个 branch

先验证方向 3 是否真的缓解了 mode averaging。

## 实验建议

不要只看最终 minADE / minFDE。

方向 3 的价值必须通过“不是普通提分，而是真的缓解平均化”来证明。

### 基础对比

至少要比：

1. Base FlowPlanner
2. Goal-conditioned single-branch FM
3. Goal-conditioned multi-branch FM

### 关键指标

- ADE / FDE
- collision rate
- progress
- pairwise diversity
- unique endpoint ratio

### 最重要的诊断

建议补：

1. branch 使用率
   看是不是总是只有一个 branch 在工作

2. branch endpoint pairwise distance
   看 branch 是否塌缩

3. 高多模态场景分组结果
   如：
   - 路口
   - 变道
   - 避障绕行
   - 让行/会车

4. 同一 goal 下的 branch 多样性
   这点最重要，因为它直接回答：
   “方向 3 解决的是 goal 内部残余歧义，还是只是换了个形式重复方向 1？”

## 可能的失败模式

### 1. branch collapse

表现：

- 多个 branch 输出几乎一样

应对：

- 加 endpoint diversity loss
- 对 branch embedding 加更强区分
- 训练早期临时加大 diversity weight

### 2. 一个 branch 独占全部样本

表现：

- `best_branch_idx` 分布极不均匀

应对：

- 给非 best branch 也加少量辅助损失
- 增加 load balancing 正则

### 3. diversity 上去了，但规划质量掉更多

表现：

- pairwise spread 更大
- ADE / collision / progress 退化

应对：

- 先减小 `K`
- 降低 diversity 权重
- 先只对 endpoint 做分散，不对整段轨迹强行拉开

## 不建议一开始做的东西

先不要马上做：

- latent variable + variational objective
- mixture density head
- 复杂 set transformer decoder
- scene-dependent dynamic branch count

这些都可能有价值，但不适合作为第一版。

第一版最重要的是回答一个简单问题：

`在保留 goal conditioning 的前提下，多分支训练目标能不能比单分支 MSE 更好地缓解 mode averaging？`

## 推荐实施顺序

### Phase 0

先把方向 1 的评估做正确：

- goal-trained checkpoint 必须用带 goal 的 eval

### Phase 1

实现方向 3 的最小版：

- `K=3`
- 共享 trunk
- 多个 final heads
- best-of-K MSE
- endpoint diversity loss

### Phase 2

分析是否有效：

- 主指标
- pairwise diversity
- branch usage
- 高多模态子集表现

### Phase 3

如果 v1 有效果，再考虑：

- branch embedding 替代多头
- branch ranking loss
- `goal x branch` 两层候选生成
- 接入 DPO reranking

## 一句话总结

方向 1 解决的是：

`先给模型更多信息，减少歧义`

方向 3 解决的是：

`即使还有歧义，也不要逼模型只学一个平均答案`

对当前项目来说，最自然的 v2 是：

`Goal-conditioned Multi-Branch Flow Matching`

即：

- 保留当前 goal conditioning
- 在 decoder 输出层引入 `K` 个 branch
- 用 best-of-K + diversity loss 训练
- 用 branch usage 和多模态诊断来证明它确实在缓解 mode averaging

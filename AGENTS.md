# AGENTS.md

这份文件是给后续代理、编码助手、自动化脚本看的远端主仓库说明。

当前主仓库路径：`/root/autodl-tmp/Flow-Planner`

## 1. 当前仓库定位

这个仓库当前主要承担三类工作：

- Flow Planner 基线训练、评测、部署
- `goal_conditioned` 实验线维护
- `anchor_conditioned` 实验线维护，尤其是 AutoDL 上的真实训练和评测

当前如果用户说“跑实验”“看结果”“继续 anchor 线”，默认优先理解为这个远端主仓库，而不是本地工作区。

## 2. 重要路径

- 主仓库：`/root/autodl-tmp/Flow-Planner`
- anchor runtime 快照：`/root/autodl-tmp/Flow-Planner-anchor-runtime`
- anchor 资产：`/root/autodl-tmp/anchor_runs`
- train 数据：`/root/autodl-tmp/train_dataset`
- val 数据：`/root/autodl-tmp/val_dataset`
- 地图：`/root/autodl-tmp/maps_raw/maps`

## 3. 分支语义

后续默认按下面的语义维护：

- `main`
  - 统一入口分支
  - 放公共说明、总览、整理后的入口、跨实验线共享的规则
  - 可以保留实验摘要或统一台账入口
- `goal`
  - `goal_conditioned` 主实验线
  - 保留 goal 线自己的实验演化记录
- `anchor`
  - `anchor_conditioned` 主实验线
  - 保留 anchor 线自己的实验演化记录

很重要的一条：

- 具体实验线的详细实验记录，优先跟着对应实验线分支走
- `main` 只保留统一入口、摘要、总览，尽量不要承担某条实验线的完整演化历史

也就是说，后续推荐的语义是：

- `anchor` 分支保留 anchor 线自己的演化记录
- `goal` 分支保留 goal 线自己的演化记录
- `main` 上保留统一入口或摘要

这样后面继续扩展 `goal` / `anchor` / 其他实验线时，语义会更顺，不容易混。

## 4. 重要约束

- 不要随便新建第二份长期开发仓库。
- 不要把新实验产物乱扔在仓库根目录。
- goal 线和 anchor 线不要混目录。
- 清理目录时，不要误删 checkpoint、数据集、可复用实验结果。

## 5. `Flow-Planner-anchor-runtime` 的使用原则

`/root/autodl-tmp/Flow-Planner-anchor-runtime` 当前是一个 **可运行的 anchor 快照环境**。

如果出现下面情况，允许优先使用这份 runtime 跑实验：

- 远端主仓库还没有完整并回 anchor 相关训练/评测代码
- 主仓库和 runtime 之间存在代码不一致，导致 anchor 实验无法直接启动
- 用户当前目标是“尽快把实验跑起来”，而不是先整理代码结构

但需要注意：

- 正式实验产物仍然优先落到 `/root/autodl-tmp/anchor_runs/`
- 正式实验记录不能只留在 runtime 快照目录里
- 跟实验线强绑定的正式记录，后续优先同步到对应分支

## 6. 实验记录强制规则

从现在开始，凡是对后续论文、报告、答辩可能有用的实验，必须补实验记录。

这不是可选项。

### 6.1 哪些实验必须记录

默认必须记录：

- 新的正式训练
- 新的正式 eval suite
- 重要对比实验
- 会影响是否继续某条实验线的实验
- 高成本 AutoDL 实验
- 任何会进入论文表格、图、结论的实验

默认可以不写成正式记录的：

- 1 到 2 分钟的 smoke test
- 明显失败且立刻终止的临时调试
- 纯粹检查脚本能否 import / 启动的小测试

### 6.2 记录应该落在哪个分支

后续默认按“实验线归属”落记录：

- anchor 线正式记录：优先落在 `anchor` 分支
- goal 线正式记录：优先落在 `goal` 分支
- `main`：保留统一入口、摘要、总览型记录

如果当前实验是在 runtime 快照里跑出来的，也不要把“运行环境在 runtime”误解成“记录只能留在 runtime”。

正确做法是：

- 实验可以在 runtime 跑
- 但实验记录要回收到正式仓库分支
- 并且优先回收到对应实验线分支

### 6.3 记录必须落在哪里

重要实验记录默认写到：

- `docs/experiments/anchor_conditioned.md`
- `docs/experiments/goal_conditioned.md`

推荐语义是：

- `anchor` 分支里的 `docs/experiments/anchor_conditioned.md`
  - 记录 anchor 线的详细演化历史
- `goal` 分支里的 `docs/experiments/goal_conditioned.md`
  - 记录 goal 线的详细演化历史
- `main` 分支里的同名文档
  - 只保留摘要、导航、统一入口，或阶段性总结

### 6.4 为什么不能只靠对话

因为实验主要发生在远端 AutoDL，上下文很容易丢。

如果只在对话里说过：

- 后面换一个代理就会忘
- 过几天回头写论文时会缺过程、参数、结果
- 很容易只剩日志文件，不知道哪次实验对应什么结论

所以要求是：

- 重要实验不能只留在聊天记录里
- 重要实验不能只留在 `train.log` / `launch.log` 里
- 重要实验必须在仓库文档里有一份人能直接看懂的摘要

## 7. 每个实验至少要记录什么

每个重要实验至少记录以下内容：

1. `experiment_id`
   - 稳定名称，例如：`anchor_sched_p0p3_20260426`
2. `goal`
   - 这次实验要验证什么
3. `setup`
   - 核心方法和关键参数
4. `artifacts`
   - 脚本、checkpoint、输出目录、日志路径
5. `data`
   - train/val 路径、样本数、eval scenes 数、manifest
6. `results`
   - 核心指标和主要 baseline 对比
7. `conclusion`
   - 这次实验支持什么，不支持什么
8. `decision`
   - 下一步动作，例如继续跑、停止、改方法

## 8. 记录时机

后续代理做实验时，至少要在两个时机更新文档：

1. 实验启动后
   - 先补目标、配置、路径、数据、当前状态
2. 实验结束后
   - 再补结果、结论、下一步决策

也就是说，哪怕实验还没跑完，也应该先把“这次实验是什么、在哪里跑、产物落哪”记下来，避免中途断掉后完全失联。

## 9. 对后续代理的直接要求

后续代理如果：

- 启动了新的正式训练
- 跑完了新的正式评测
- 对实验结果做出了会影响后续方向的判断

那么在任务结束前，默认要同时交付四样东西：

- 产物路径
- 核心指标
- 简短结论
- 已更新的实验记录文档路径

如果只把命令跑了，没有把实验记录补到对应实验线分支的 `docs/experiments/*.md`，视为任务没有完整收尾。

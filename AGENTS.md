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

## 3. 重要约束

- 不要随便新建第二份长期开发仓库。
- 不要把新实验产物乱扔在仓库根目录。
- goal 线和 anchor 线不要混目录。
- 清理目录时，不要误删 checkpoint、数据集、可复用实验结果。

## 4. `Flow-Planner-anchor-runtime` 的使用原则

`/root/autodl-tmp/Flow-Planner-anchor-runtime` 当前是一个 **可运行的 anchor 快照环境**。

如果出现下面情况，允许优先使用这份 runtime 跑实验：

- 远端主仓库还没有完整并回 anchor 相关训练/评测代码
- 主仓库和 runtime 之间存在代码不一致，导致 anchor 实验无法直接启动
- 用户当前目标是“尽快把实验跑起来”，而不是先整理代码结构

但需要注意：

- 正式实验产物仍然优先落到 `/root/autodl-tmp/anchor_runs/`
- 正式实验记录仍然优先写回主仓库 `docs/experiments/`
- 不能因为用 runtime 跑，就把结果只留在 launch log 或对话里

## 5. 实验记录强制规则

从现在开始，凡是对后续论文、报告、答辩可能有用的实验，必须补实验记录。

这不是可选项。

### 5.1 哪些实验必须记录

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

### 5.2 记录必须落在哪里

重要实验记录默认写到：

- `docs/experiments/anchor_conditioned.md`
- `docs/experiments/goal_conditioned.md`

如果是 anchor 线实验，优先更新：

- `docs/experiments/anchor_conditioned.md`

如果是 goal 线实验，优先更新：

- `docs/experiments/goal_conditioned.md`

### 5.3 为什么不能只靠对话

因为实验主要发生在远端 AutoDL，上下文很容易丢。

如果只在对话里说过：

- 后面换一个代理就会忘
- 过几天回头写论文时会缺过程、参数、结果
- 很容易只剩日志文件，不知道哪次实验对应什么结论

所以要求是：

- 重要实验不能只留在聊天记录里
- 重要实验不能只留在 `train.log` / `launch.log` 里
- 重要实验必须在仓库文档里有一份人能直接看懂的摘要

## 6. 每个实验至少要记录什么

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

## 7. 记录时机

后续代理做实验时，至少要在两个时机更新文档：

1. 实验启动后
   - 先补目标、配置、路径、数据、当前状态
2. 实验结束后
   - 再补结果、结论、下一步决策

也就是说，哪怕实验还没跑完，也应该先把“这次实验是什么、在哪里跑、产物落哪”记下来，避免中途断掉后完全失联。

## 8. 对后续代理的直接要求

后续代理如果：

- 启动了新的正式训练
- 跑完了新的正式评测
- 对实验结果做出了会影响后续方向的判断

那么在任务结束前，默认要同时交付四样东西：

- 产物路径
- 核心指标
- 简短结论
- 已更新的实验记录文档路径

如果只把命令跑了，没有把实验记录补到 `docs/experiments/*.md`，视为任务没有完整收尾。

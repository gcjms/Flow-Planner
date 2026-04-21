# Goal-Conditioned DPO 评估部署手册（含 Oracle 模式）

## 现状

之前做了：Goal_Base 模型跑过 4 种 goal 模式，其中 oracle_goal 碰撞率 0%。但**另一个 goal-aware DPO（cross-goal pair 版本）在 `none` / `route_goal` 下碰撞率 21% / 27%，已判死**。本轮目标换成 `dpo_goal_tune_b3.0_s0.3_e1`，这个模型上轮卡在 CFG shape bug 没跑通。

本轮要补齐的是 `dpo_goal_tune_b3.0_s0.3_e1` 在 4 种 goal 模式下的完整评估，**重点是 `predicted_goal`（代码早写好，没跑）和 `oracle_goal`（本轮刚加）**。跑完就能判断 DPO 是不是把 decoder 训坏了。

### 已有实验结果

**Goal_Base（老 checkpoint，100 scenes，来自 `docs/dpo_自动驾驶综述.md` Ch 4.3）**

| Model | none | route_goal | oracle (gt_nearest) |
|---|---:|---:|---:|
| Base Flow Planner | 7% | 5% | 5% |
| Goal_Base e50 | 20% | 23% | **0%** |
| Goal_Base e60 | 20% | 24% | **0%** |
| Goal_Base e70 | 18% | 25% | **0%** |

结论：decoder 本身会用 goal（作弊下碰撞降到 0），但没作弊时选不对 goal，反而更差。

**goal-aware DPO（cross-goal pair 版，已判死，1000 scenes，来自 `docs/goalaware_dpo_eval_summary.md`）**

| Model | none | route_goal |
|---|---:|---:|
| Base Flow Planner | 4.9% | — |
| old strict_dimfix adaptive | 19.2% | — |
| goal-aware adaptive | 21.3% | 26.6% |

训练 pair accuracy 只有 48.69%（比随机还差），已确认 pair 条件不一致导致训练信号崩了。不再追这条路。

**本轮目标 `dpo_goal_tune_b3.0_s0.3_e1`（200 scenes，待跑）**

| Mode | 状态 |
|---|---|
| none | ⏳ 待跑 |
| route_goal | ⏳ 待跑 |
| predicted_goal | ⏳ 待跑（代码已就绪） |
| oracle_goal | ⏳ 待跑（本轮新增代码支持） |

## 本轮代码改动

| 文件 | 改了什么 |
|---|---|
| `flow_planner/dpo/eval_multidim_utils.py` | `choose_goal_point` 加 `oracle_goal` 分支（GT 终点 snap 到 `goal_vocab` 最近 cluster） |
| `flow_planner/dpo/eval_multidim_goal_ablation.py` | `--goal_mode` 多一个 `oracle_goal` 选项 |
| `flow_planner/run_script/run_goal_dpo_eval.sh` | 末尾加一步 DPO × oracle_goal |

## 目标机器准备

```bash
BASE_CKPT=/path/to/base/model.pth
BASE_CONFIG=/path/to/base/config.yaml
DPO_CKPT=/path/to/dpo_goal_tune_b3.0_s0.3_e1/model_dpo_merged.pth
DPO_CONFIG=/path/to/dpo_goal_tune_b3.0_s0.3_e1/config.yaml
GOAL_VOCAB=/path/to/goal_vocab.npy
GOAL_PREDICTOR_CKPT=/path/to/goal_predictor.ckpt
SCENE_DIR=/path/to/hard_scenarios_v2
REPO_ROOT=$(pwd)
```

## 执行

先 5 场景冒烟：

```bash
SMOKE_SCENES=5 MAX_SCENES=5 bash flow_planner/run_script/run_goal_dpo_eval.sh
```

通过后正式跑 200 场景：

```bash
MAX_SCENES=200 bash flow_planner/run_script/run_goal_dpo_eval.sh
```

脚本会顺序跑 7 步，结果落在 `outputs/goal_dpo_eval/*.json`。总耗时 50-90 分钟。

## 看结果

关注每个 JSON 的 `summary.collision_rate` / `avg_progress` / `avg_route`。填这张表：

| | collision_rate | progress | route |
|---|---:|---:|---:|
| BASE × none | | | |
| DPO × none | | | |
| DPO × route_goal | | | |
| **DPO × predicted_goal**（新） | | | |
| **DPO × oracle_goal**（新） | | | |

判断规则：

- `DPO × oracle` 明显好于 `DPO × none` → decoder 没被 DPO 搞坏
- `DPO × oracle` 和 `DPO × none` 差不多 → DPO 把 decoder 对 goal 的响应洗掉了
- `DPO × predicted` ≈ `DPO × oracle` → goal_predictor 够用，可以上车
- `DPO × predicted` 远差于 `DPO × oracle` → 瓶颈在 goal_predictor，下一步改它

**oracle 先看**。oracle 过不了，predicted 结果不用看了。

## 常见报错

- `oracle_goal requires scene NPZ to contain 'ego_agent_future'` → 场景不是带 GT 的 benchmark，换 `hard_scenarios_v2`
- `oracle_goal requires a goal vocabulary` → 没传 `--goal_vocab_path`
- decoder shape mismatch → checkpoint 和 config 对不上，检查 `model.goal_dim`
- goal predictor load 失败 → `GOAL_PREDICTOR_HIDDEN_DIM` 要和训练时一致

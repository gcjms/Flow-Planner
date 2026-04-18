# Goal DPO Eval Deploy

这份文档给 GPU 机器上的部署和执行使用，目标是把 `multidim open-loop evaluation` 跑通，并完成 handoff note 里要求的三类检查。

## 1. 已确认的问题

旧版 `dpo/eval_multidim.py` 的问题不是“怀疑”，而是和当前 planner 推理链路不一致的真实问题。

### 1.1 CFG shape bug 是什么

旧脚本在 `use_cfg=True` 时，手动构造的是：

- `cfg_flags = torch.ones(1, ...)`
- 单 batch 的 `inputs`
- 直接调用 `model.extract_encoder_inputs(...)`
- 直接调用 `model.flow_ode.generate(... use_cfg=True, cfg_weight=1.8, ...)`

但当前正式推理链路 `FlowPlanner.forward_inference()` 的逻辑是：

1. 先构造 `NuPlanDataSample`
2. 如果 `use_cfg=True`，就把 batch 扩成 `2B`
3. `cfg_flags` 变成 `[1, 0]`，即 conditional + unconditional
4. `prepare_model_input()` 负责把 `data.repeat(2)`、CFG masking、normalization 都做对
5. 如果传了 `goal_point`，还要把它变成：
   - conditional half: 真 goal
   - unconditional half: 零向量

所以旧脚本的问题在于：

- `x_init` 是按 `B=1` 生成的
- `flow_ode.generate(... use_cfg=True)` 内部会把 latent 复制成 `2B`
- 但 `decoder_inputs` 还是按单 batch 构出来的
- `cfg_flags` 也还是单元素，不是 `[1, 0]`

这会让 decoder 侧的 batch 维和 CFG 预期对不上，导致 shape bug，进一步引发评测失败。handoff note 里提到的“昨晚 report ended with Scenes evaluated: 0”与这个问题一致。

### 1.2 旧脚本还绕过了标准推理入口

除了 CFG 之外，旧脚本还绕过了当前标准链路：

- 没有走 `NuPlanDataSample`
- 没有走 `prepare_model_input()`
- 没有走 `model(data, mode="inference", ...)`
- 没有办法正确复用当前的 `goal_point` 注入逻辑

这意味着即便某些场景不直接报错，评测逻辑也可能和实际 planner inference 不一致。

## 2. 我已经做的代码修改

### 2.1 重写 `dpo/eval_multidim.py`

现在它会：

- 把每个 NPZ 场景构造成 `NuPlanDataSample`
- 调用高层推理入口：

```python
model(
    data,
    mode="inference",
    use_cfg=...,
    cfg_weight=...,
    num_candidates=1,
    return_all_candidates=False,
    bon_seed=...,
    goal_point=None,
)
```

- 这样 CFG、normalization、neighbor masking、goal 注入都和当前 planner 实际推理保持一致
- 增加了 `--output_json`，便于把结果保存成结构化文件

### 2.2 新增 `dpo/eval_multidim_goal_ablation.py`

支持：

- `--goal_mode none`
- `--goal_mode route_goal`
- `--goal_mode predicted_goal`

其中：

- `none`: 不传 `goal_point`
- `route_goal`: 用 `goal_utils.select_goal_from_route()` 从 route geometry 检索一个非 oracle goal
- `predicted_goal`: 用 `GoalPredictor.predict_topk(..., top_k=1)` 拿 top-1 goal point

### 2.3 新增共享工具 `dpo/eval_multidim_utils.py`

统一了下面这些逻辑：

- planner model 加载
- goal predictor 加载
- `NPZ -> NuPlanDataSample`
- goal 选择
- 单场景 inference
- multidim metrics 汇总
- JSON summary 保存

## 3. 需要同步到 GPU 机器的文件

至少同步这 4 个文件：

- `flow_planner/dpo/eval_multidim.py`
- `flow_planner/dpo/eval_multidim_goal_ablation.py`
- `flow_planner/dpo/eval_multidim_utils.py`
- `flow_planner/run_script/run_goal_dpo_eval.sh`

如果你是直接同步整个仓库工作树，当然更简单。

如果你不想逐条复制命令，优先直接使用：

- `run_script/run_goal_dpo_eval.sh`

这个脚本会先跑 `5` 个场景的 smoke test，再顺序跑 base / DPO / goal ablation，并把日志和 JSON summary 全部写到 `OUTPUT_DIR`。

## 4. GPU 机器准备项

在 GPU 机器上，保证下面这些路径变量是清楚的：

```bash
REPO_ROOT=/root/Flow-Planner
SCENE_DIR=/path/to/hard_scenarios_v2

BASE_CKPT=/path/to/base/model.pth
BASE_CONFIG=/path/to/base/config.yaml

DPO_CKPT=/path/to/dpo_goal_tune_b3.0_s0.3_e1/model_dpo_merged.pth
DPO_CONFIG=/path/to/dpo_goal_tune_b3.0_s0.3_e1/config.yaml

GOAL_VOCAB=/path/to/goal_vocab.npy
GOAL_PREDICTOR_CKPT=/path/to/goal_predictor.ckpt
GOAL_PREDICTOR_HIDDEN_DIM=256
GOAL_PREDICTOR_DROPOUT=0.1
```

注意：

- `BASE_CONFIG` / `DPO_CONFIG` 请填和对应 checkpoint 匹配的 config
- `GOAL_VOCAB` 在 `route_goal` / `predicted_goal` 时需要
- `GOAL_PREDICTOR_CKPT` 只在 `predicted_goal` 时需要
- 如果 goal predictor 训练时 `hidden_dim` 不是默认的 `256`，运行时要改 `GOAL_PREDICTOR_HIDDEN_DIM` 或显式加 `--goal_predictor_hidden_dim`

## 5. 推荐执行顺序

按照 handoff note，推荐跑下面 5 个命令。

### 5.1 检查 1: Original Flow Planner，无显式 goal

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$BASE_CKPT" \
  --config_path "$BASE_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --max_scenes 200 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/eval_multidim_base.json
```

### 5.2 检查 2: DPO_b3.0_s0.3，无显式 goal

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --max_scenes 200 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/eval_multidim_dpo_none.json
```

### 5.3 检查 3a: DPO_b3.0_s0.3，`goal_mode=none`

这个命令和上一个的指标应该基本一致，它的意义是让 goal ablation 脚本本身也有一个 `none` 基线。

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim_goal_ablation \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --goal_mode none \
  --max_scenes 200 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/eval_multidim_goal_none.json
```

### 5.4 检查 3b: DPO_b3.0_s0.3，`goal_mode=route_goal`

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim_goal_ablation \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --goal_mode route_goal \
  --goal_vocab_path "$GOAL_VOCAB" \
  --max_scenes 200 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/eval_multidim_goal_route.json
```

### 5.5 检查 3c: DPO_b3.0_s0.3，`goal_mode=predicted_goal`

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim_goal_ablation \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --goal_mode predicted_goal \
  --goal_vocab_path "$GOAL_VOCAB" \
  --goal_predictor_ckpt "$GOAL_PREDICTOR_CKPT" \
  --goal_predictor_hidden_dim 256 \
  --max_scenes 200 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/eval_multidim_goal_predicted.json
```

## 6. 跑完以后看什么

每次运行至少看两处：

### 6.1 控制台 summary

必须重点看：

- `Scenes requested`
- `Scenes evaluated`
- `Scenes failed`
- `collision_rate`
- `avg_collision_score`
- `avg_ttc`
- `avg_comfort`
- `avg_progress`
- `avg_route`

### 6.2 `output_json`

每个命令都会输出一个 JSON 文件，里面有：

- `summary`
- `failures`
- `extra`

如果发现：

- `Scenes evaluated = 0`
- 或者 `failures` 里几乎全是同一种错误

就不要继续汇总结论，先看错误信息。

## 7. 最小验收标准

在 GPU 机器上，这次修复至少要满足下面几点：

1. `eval_multidim.py` 不再出现旧版 CFG shape bug
2. `Scenes evaluated` 必须大于 0
3. `eval_multidim_goal_ablation.py` 能在 `none` / `route_goal` / `predicted_goal` 三种模式下分别运行
4. `route_goal` 和 `predicted_goal` 在非 goal-conditioned checkpoint 上应直接报清晰错误，而不是静默跑错
5. 每次运行都能落一个 JSON summary，便于后续整理表格

## 8. 已知限制

这次我在本机没有完整 Python / CUDA 运行环境，所以：

- 我做了代码级核对
- 做了静态 lint 检查
- 但没有在本机实际跑通 eval

因此，GPU 机器上的第一次运行建议先用：

- `--max_scenes 5`

做 smoke test，确认：

- 模型能加载
- scene NPZ key 是齐的
- `Scenes evaluated > 0`
- JSON 能正确落盘

然后再把 `--max_scenes` 提到 `200`。

## 9. 建议的首次 smoke test

先跑这个：

```bash
cd "$REPO_ROOT"

python -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$BASE_CKPT" \
  --config_path "$BASE_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --max_scenes 5 \
  --use_cfg \
  --cfg_weight 1.8 \
  --output_json outputs/smoke_eval_base.json
```

只要这个能正常得到：

- `Scenes evaluated > 0`
- 没有全量失败

再继续跑后面的 200-scene 正式命令。

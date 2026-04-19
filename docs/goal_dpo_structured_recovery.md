# Goal DPO Structured Recovery

## 目标

这次不是继续在旧的 `top1 vs worst1` preference 上盲调 beta。

要先把 DPO 的训练目标改回更接近规划质量的版本：

1. 候选先做结构化诊断
2. 再按 cluster / maneuver 挖 multi-pair
3. 把 `dim_labels` 真正写进 `preferences.npz`
4. 再用保守超参做 1 个 recovery run

## 为什么要这样改

当前坏信号已经比较明确：

- 旧 preference 几乎全是 rule-based `FDE/ADE/obs` 驱动
- 训练集没有真实 `dim_labels`
- 训练日志里一直显示 `[collision]`，但那其实是缺省值，不是数据事实
- open-loop 上 DPO 比 base 明显更差，说明现在优化目标和最终指标是错位的

所以接下来最小可行修复，不是继续扫更多 `beta/sft_weight`，而是先换 preference construction。

## 需要改的代码

- [flow_planner/dpo/score_hybrid.py](/home/gcjms/Flow-Planner/flow_planner/dpo/score_hybrid.py)
  - structured scoring 结果会额外写出 `dim_labels` 和 `score_gaps`
- [flow_planner/dpo/build_multi_pairs.py](/home/gcjms/Flow-Planner/flow_planner/dpo/build_multi_pairs.py)
  - 从 `scored_dir/*.json` 构造 strict multi-pair
- [flow_planner/dpo/train_dpo.py](/home/gcjms/Flow-Planner/flow_planner/dpo/train_dpo.py)
  - 没有 `dim_labels` 时不再伪装成 `collision`，而是标成 `legacy_unlabeled`
- [auto_goal_dpo_pipeline.sh](/home/gcjms/Flow-Planner/auto_goal_dpo_pipeline.sh)
  - Step 4 改成 structured scoring + multi-pair build

## 一次完整 recovery run

### Step 1. 结构化打分

```bash
python -u -m flow_planner.dpo.score_hybrid \
    --candidates_dir /root/autodl-tmp/dpo_candidates_goal \
    --output_dir /root/autodl-tmp/dpo_preferences_goal_structured \
    --scored_dir /root/autodl-tmp/dpo_preferences_goal_structured/scored_dir \
    --use_structured_scores \
    --emit_traj_info \
    --skip_vlm
```

### Step 2. 构造 multi-pair preference

```bash
python -u -m flow_planner.dpo.build_multi_pairs \
    --scored_dir /root/autodl-tmp/dpo_preferences_goal_structured/scored_dir \
    --candidates_dir /root/autodl-tmp/dpo_candidates_goal \
    --output_path /root/autodl-tmp/dpo_preferences_goal_structured/preferences_multi.npz \
    --meta_path /root/autodl-tmp/dpo_preferences_goal_structured/preferences_multi_meta.jsonl \
    --top_good_per_cluster 1 \
    --subtle_bad_per_good 2
```

### Step 3. 快速检查 preference

```bash
python - <<'PY'
import numpy as np
d = np.load('/root/autodl-tmp/dpo_preferences_goal_structured/preferences_multi.npz', allow_pickle=True)
print('pairs =', len(d['chosen']))
u, c = np.unique(d['dim_labels'], return_counts=True)
print('dim_labels =', dict(zip([str(x) for x in u], [int(x) for x in c])))
PY
```

期望至少确认两件事：

- `pairs` 不再只是每场景 1 对
- `dim_labels` 不再只有一个假 `collision`

### Step 4. 保守版 DPO recovery run

```bash
python -u -m flow_planner.dpo.train_dpo \
    --config_path checkpoints/config_goal.yaml \
    --ckpt_path checkpoints/model_goal.pth \
    --preference_path /root/autodl-tmp/dpo_preferences_goal_structured/preferences_multi.npz \
    --scene_dir /root/autodl-tmp/dpo_mining \
    --output_dir checkpoints/dpo_goal_structured_b2.0_s0.3_e1 \
    --beta 2.0 \
    --epochs 1 \
    --lr 5e-5 \
    --lora_rank 4 \
    --lora_alpha 16 \
    --batch_size 8 \
    --sft_weight 0.3 \
    --num_t_samples 16 \
    --save_merged
```

## 第一优先级验证

新 run 跑完以后，只看 3 件事：

1. `without goal` 的 200-scene open-loop 是否至少回到接近 base，而不是继续恶化
2. collision rate 是否从目前 DPO 的 `15.5%` 明显往回拉
3. progress / route 是否不再一起塌

## 这轮不要再做的事

- 不要把 `predicted_goal` 或 `route_goal` 的推理 ablation 当主结论
- 不要再拿旧的 `preferences.npz` 继续扫很多组超参
- 不要用 best-of-n scorer 作为主要补救方案

## reviewer 最该看什么

如果 reviewer 只想看最关键的，先看这三个文件：

- [flow_planner/dpo/score_hybrid.py](/home/gcjms/Flow-Planner/flow_planner/dpo/score_hybrid.py)
- [flow_planner/dpo/build_multi_pairs.py](/home/gcjms/Flow-Planner/flow_planner/dpo/build_multi_pairs.py)
- [auto_goal_dpo_pipeline.sh](/home/gcjms/Flow-Planner/auto_goal_dpo_pipeline.sh)

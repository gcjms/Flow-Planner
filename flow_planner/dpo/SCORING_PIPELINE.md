# DPO 偏好打分管线

## 概述

本模块实现了**混合规则 + VLM**的偏好打分管线，用于 DPO 训练。
从 Flow-Planner（`use_cfg=False`）生成的 10,000 个场景（每个 5 条候选轨迹）中，
产出 `(chosen, rejected)` 偏好对。

## 架构

```
10,000 个候选 NPZ 文件
    ├── 第一层：纯规则打分（横向分散 < 5m，约 99.5% 场景）
    │   └── Score = -FDE - 0.5*ADE - 碰撞惩罚
    │       chosen = 最高分, rejected = 最低分
    │
    └── 第二层：VLM 融合打分（横向分散 ≥ 5m，约 49 个场景）
        └── Gemini 3.1 Pro：BEV图(含GT白色虚线) + 物理距离数据 + 语义提问
            检查：是否驶出道路？是否逆行？是否与GT方向一致？碰撞风险？
```

## 关键设计决策

### 为什么用混合方案而不是纯 VLM？
1. 大多数场景横向分散 <0.5m —— 5 条轨迹几乎完全重叠，VLM 从图上根本看不出区别
2. 把预计算的距离作为文本喂给 VLM 时，VLM 只是按数字排序 —— 等同于规则但慢 1000 倍
3. VLM 的独特价值是**语义理解**（识别驶出道路、逆行等），仅在约 1% 的场景有意义

### 为什么 `use_cfg=False` 下多样性有限？
1. `use_cfg=False` 只是不做 CFG 两次推理，模型仍然看到完整场景信息（不是无条件生成）
2. `routes` 输入告诉模型走哪些 lane → 锁死了方向（左转/右转/直行）
3. nuPlan 每个场景只有 1 条 GT → 模型学到的是以 route 为条件的**单模态分布**
4. 多样性仅来自不同随机种子 → 主要是速度/进度差异，几乎没有方向分歧

### Prompt 演进过程
| 版本 | 问题 |
|------|------|
| 纯图片 | VLM 在重叠轨迹上产生幻觉，声称看到不存在的差异 |
| 图片 + 预计算距离 | VLM 变成昂贵的排序器，丧失语义价值 |
| 图片 + 距离 + GT参考 | VLM 正确识别方向 + 语义违规，排名与规则完全一致 |

## 使用方式

```bash
# 完整 10K 混合打分
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir dpo_data/candidates \
    --output_dir dpo_data/preferences_final \
    --api_key YOUR_KEY

# 纯规则（更快，不需要 API）
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir dpo_data/candidates \
    --output_dir dpo_data/preferences_final \
    --skip_vlm

# 自定义分散阈值
python -m flow_planner.dpo.score_hybrid \
    --candidates_dir dpo_data/candidates \
    --output_dir dpo_data/preferences_final \
    --api_key YOUR_KEY \
    --spread_threshold 10.0
```

## 输出格式

### `preferences.npz`
- `chosen`: `(N, T, D)` — 被选中的优质轨迹
- `rejected`: `(N, T, D)` — 被拒绝的劣质轨迹
- `scenario_ids`: 场景名列表
- `rankings`: 完整 5 路排名列表
- `reasons`: 打分理由列表

### `preference_details.json`
```json
[
  {
    "scenario_id": "us-ma-boston_xxxx",
    "chosen_idx": 3,
    "rejected_idx": 0,
    "ranking": [4, 2, 5, 1, 3],
    "reason": "规则打分: chosen=#4(FDE=1.8m) vs rejected=#3(FDE=20.0m)",
    "method": "rule",
    "lateral_spread": 0.42
  }
]
```

### `bev_images/`（仅 VLM 场景）
带 GT 白色虚线的 BEV 渲染图，包含候选轨迹（彩色）和周围车辆（红色方块）。

## 横向分散统计（10K 候选，use_cfg=False）

| 阈值 | 场景数 | 占比 |
|------|--------|------|
| ≥ 0.5m | 3,607 | 36.1% |
| ≥ 2.0m | 989 | 9.9% |
| ≥ 5.0m | 49 | 0.5% |
| ≥ 10.0m | 19 | 0.2% |

## DPO 在当前数据下能学到什么

- ✅ 方向对齐：跟随 GT 方向比偏离好
- ✅ 碰撞回避：离车远比离车近好
- ✅ 进度合理：完成转弯比半途停车好
- ❌ 决策偏好：无法学到"左绕 vs 右绕"（训练数据缺乏多模态 GT）

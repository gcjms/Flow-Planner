# Boston Training & Val14 Closed-Loop Evaluation Results

## Training

| Item | Value |
|------|-------|
| **Data** | Boston nuPlan, 8000 train / 2000 val |
| **Epochs** | 31→100 (resumed from epoch 30) |
| **Duration** | ~9.5 hours |
| **Final Loss** | 0.0333 |

## Open-Loop Evaluation (全部 Checkpoint)

| Epoch | ADE (m) | FDE (m) | ADE@1s | ADE@3s | Heading (°) |
|-------|---------|---------|--------|--------|------------|
| 40 | 4.250 | 9.648 | 0.566 | 1.457 | 3.81 |
| 50 | 4.130 | 9.463 | 0.530 | 1.378 | 3.68 |
| 60 | 4.172 | 9.625 | 0.512 | 1.362 | 3.83 |
| 70 | 4.003 | 9.555 | 0.546 | 1.303 | 3.41 |
| 80 | 4.004 | 9.289 | 0.521 | 1.346 | 3.36 |
| 90 | 3.889 | 8.982 | 0.514 | 1.312 | 3.52 |
| **100** | **3.817** | **8.978** | **0.521** | **1.268** | **3.23** |
| latest | 3.724 | 8.774 | — | — | — |

> [!NOTE]
> **Epoch 100 / latest** 是最优 checkpoint，ADE 持续下降趋势明显。

## Closed-Loop NR Simulation (Val14)

**Checkpoint**: `latest.pth` (epoch 100)

| Metric | Score |
|--------|-------|
| **Final NR Score** | **71.20%** |
| No At-Fault Collision | 75.00% |
| Drivable Area Compliance | 100.00% |
| Driving Direction Compliance | 91.67% |
| Making Progress | 100.00% |
| Time to Collision | 75.00% |
| Comfortable | 75.00% |
| Speed Limit Compliance | 97.06% |

### Per-Scenario Type Breakdown

| Scenario Type | n | Score |
|--------------|---|-------|
| starting_right_turn | 1 | 0.810 |
| stopping_with_lead | 1 | 1.000 |
| straight_traffic_light_intersection | 1 | 0.946 |
| stationary_in_traffic | 3 | 1.000 |
| starting_left_turn | 2 | 0.480 |
| low_magnitude_speed | 3 | 0.292 |
| high_magnitude_speed | 1 | 0.953 |

### vs 论文报告值

| | 论文 Val14 NR | 我们的结果 | 差距 |
|---|---|---|---|
| **NR Score** | 90.43% | 71.20% | -19.23% |

> [!IMPORTANT]
> 差距主要来自：
> - **low_magnitude_speed** 类型得分极低 (0.292)，3个场景中有碰撞/不舒适事件
> - **starting_left_turn** 得分也偏低 (0.480)
> - 论文使用完整训练集（全部城市），我们仅用 Boston 子集 (8000 samples)
> - 论文可能训练了更多 epoch 或使用了不同的超参数

## Cleanup

删除了 ~1.1GB 不需要的文件：
- 旧 mini 数据训练 checkpoints (874MB)
- 失败的闭环测试结果
- 大训练日志 (~30MB)

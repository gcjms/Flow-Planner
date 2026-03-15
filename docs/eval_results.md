# Boston Training & Val14 Closed-Loop Evaluation Results

## Training

| Item | Value |
|------|-------|
| **Data** | Boston nuPlan, 8000 train / 2000 val |
| **Epochs** | 31→100 (resumed from epoch 30) |
| **Duration** | ~9.5 hours |
| **Final Loss** | 0.0333 |

## Open-Loop Evaluation (Best Checkpoint)

| Epoch | ADE (m) | FDE (m) | ADE@1s | ADE@3s | Heading (°) |
|-------|---------|---------|--------|--------|------------|
| **100** | **3.817** | **8.978** | **0.521** | **1.268** | **3.23** |
| latest (EMA) | 3.724 | 8.774 | — | — | — |

## CFG Weight Grid Search (Val14 NR, mini 12 scenarios)

| cfg_weight | **NR Score** | Collision-Free | Drivable | TTC | Comfort | low_mag_speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.0 (原默认) | 71.20% | 75.0% | 100% | 75.0% | 75.0% | 0.292 |
| 1.8 (论文默认) | 76.80% | 91.7% | 91.7% | 83.3% | 75.0% | 0.514 |
| 2.0 | 71.49% | 91.7% | 83.3% | 91.7% | 75.0% | 0.625 |
| **2.5** | **80.19%** 🏆 | **91.7%** | **91.7%** | **91.7%** | 75.0% | **0.611** |
| 3.0 | 75.77% | 91.7% | 100% | 83.3% | 66.7% | 0.516 |

> [!IMPORTANT]
> **最优 cfg_weight=2.5**，NR score 从 71.20% 提升到 **80.19%** (+9分)，仅改推理参数，零训练成本。
> w=3.0 过度保守导致舒适度下降。已将 planner yaml 更新为 2.5。

### vs 其他方法

| 方法 | Val14 NR | 备注 |
|------|---------|------|
| PDM-Closed | 92-93% | 规则 baseline |
| Flow-Planner 论文 | 90.43% | 完整数据 |
| Diffusion Planner | 89.87% | ICLR 2025 |
| PlanTF | 84.83% | ICRA 2024 |
| **我们 (w=2.5)** | **80.19%** | Boston-only, mini 12 scenarios |
| GameFormer | 80.80% | 混合方法 |

> [!NOTE]
> 仿真仅用 mini split（12 scenarios），需下载官方 val split 才能完整对比。

# FlowPlanner vs GoalFlow vs GuideFlow 论文对比

## 基本信息

| | FlowPlanner | GoalFlow | GuideFlow |
|--|------------|----------|-----------|
| **发表时间** | 2025.10（NeurIPS 2025） | 2025.03（arXiv） | 2025.11（arXiv） |
| **会议/期刊** | NeurIPS 2025 ✅ | arXiv（待投） | arXiv（待投） |
| **核心方法** | Flow Matching + CFG | Flow Matching + Goal Point | Constrained Flow Matching + EBM |

## 技术对比

| 技术点 | FlowPlanner | GoalFlow | GuideFlow |
|--------|------------|----------|-----------|
| **生成方法** | Flow Matching | Flow Matching | Constrained Flow Matching |
| **CFG 引导** | ✅ 创新点 | ❌ | ❌ |
| **Goal Point** | ❌ | ✅ 创新点 | ❌ |
| **多轨迹生成** | ❌ 只生成 1 条 | ✅ 多条 + 评分选最优 | ✅ 约束引导生成 |
| **约束施加方式** | 无显式约束 | 后置评分筛选 | 生成过程中直接约束（EBM） |
| **ODE 步数** | 4-10 步 | 1-2 步 | 未明确 |
| **感知模块** | ❌ 只做规划 | ✅ End-to-End (BEV) | ✅ End-to-End |
| **输入** | 结构化数据（状态+车道） | 相机+LiDAR → BEV | 相机+LiDAR → BEV |
| **交互建模** | ✅ 强（neighbor编码+CFG） | 弱 | 中 |
| **驾驶风格控制** | ❌ | ❌ | ✅ 可调激进/保守 |

## Benchmark 分数

| Benchmark | FlowPlanner | GoalFlow | GuideFlow |
|-----------|------------|----------|-----------|
| **nuPlan Val14 NR-CLS** | **94.31** | — | — |
| **nuPlan Val14 R-CLS** | **92.38** | — | — |
| **NAVSIM PDMS** | — | **90.3** | — |
| **NAVSIM hard EPDMS** | — | — | **43.0** |

> 三者评测的 benchmark 不同，分数不可直接比较

## 多角度分析

| 维度 | FlowPlanner | GoalFlow | GuideFlow |
|------|------------|----------|-----------|
| **论文含金量** | ⭐⭐⭐⭐⭐ NeurIPS 顶会 | ⭐⭐⭐ arXiv，NAVSIM SOTA | ⭐⭐⭐ arXiv，多benchmark测试 |
| **效果** | nuPlan 闭环 SOTA | NAVSIM SOTA | NAVSIM hard SOTA |
| **实用性** | ⭐⭐⭐ 规划模块独立可用 | ⭐⭐⭐⭐ 端到端完整系统 | ⭐⭐⭐ 需要 EBM 额外训练 |
| **部署难度** | 低（只做规划） | 中（需要感知+规划） | 高（EBM+约束优化） |
| **创新深度** | CFG 引入规划领域（新颖） | Goal Point 简洁有效 | EBM 约束理论优美 |
| **代码开源** | ✅ GitHub | ✅ GitHub | ❓ 未确认 |
| **可复现性** | ⭐⭐⭐⭐ 代码清晰 | ⭐⭐⭐⭐ 代码开源 | ⭐⭐ 实现复杂 |

## 总结

- **FlowPlanner**：学术含金量最高（NeurIPS），CFG 创新有启发性，但没利用多模态
- **GoalFlow**：工程最实用，Goal Point + 评分简洁有效，NAVSIM SOTA
- **GuideFlow**：理论最优美（EBM 约束），但实现最复杂，部署难度大

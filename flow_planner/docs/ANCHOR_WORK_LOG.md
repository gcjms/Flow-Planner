# Anchor 分支工作日志

**主分支**：`feature/anchor`  
**当前重点**：Flow Planner 的 trajectory anchor + learned selector 路线（从 goal_conditioned 切换而来）。同时维护创新项目申报材料。

## 关键决策记录
- **2026-04-28**：创新项目 PPT 从技术重写转向**在原模板 `Innovation Projects 1.pptx` 上手动修改**。避免破坏母版和布局。
- **2026-04-28**：PPT 风格迭代 - 减少英文专业术语（DPO、oracle、selector、CFG、LoRA 等），全部改为通俗中文描述，面向汽车领域非端到端专家的评审委员。强调项目可行性、已有基础、2k内部测试正面结论。
- **2026-04-28**：PPT 叙事原则修正 - 创新项目申报材料应以“拟开展的创新内容、价值、可行性和资源需求”为主，不再写成技术路线迁移/失败复盘记录。历史尝试只作为已有基础和立项依据的轻量背景。
- 主线从 "goal conditioning + soft preference" 切换到 "参考轨迹指引（anchor）+ 智能选择器（selector）+ 候选池优化"。planner-level 偏好学习暂缓，重点攻 candidate-level selector。
- **2026-04-28**：为最终版 `Innovation Projects.pptx` 制作**20分钟演讲稿**（speaker_notes_for_innovation_project.md）。**仅为主内容1-18页**编写自然口语化演讲稿，附录19-32页明确不讲（留作Q&A参考）。演讲稿强调anchor指引+智能选择器、2k内部最新指标（oracle anchor~2.0%、predicted 3.2-4.2%、baseline~2.4%）、公司价值与可行性。语言采用聊天式口语，避免书面腔。
- **2026-04-28 23:13**：已将 `speaker_notes_for_innovation_project.md` 中第1-18页演讲稿直接写入最终版 `Innovation Projects.pptx` 的备注区；附录页未写入正式演讲稿。写入前生成备份 `Innovation Projects.before_speaker_notes_20260428_231323.pptx`。
- **2026-04-28 23:43**：修复最终版 PPT 备注错位问题。当前 `Innovation Projects.pptx` 实际为36页，主讲范围修正为**第1-22页**，第23页起为附录。新增第7页“Why delta = candidate - anchor matters”图解讲稿，用该图解释 anchor、candidate 和 delta，并自然引出“Trajectory Anchor + Anchor Selector + Candidate Selector”的完整方案。重新写入第1-22页备注并清空附录页备注，备份为 `Innovation Projects.before_realigned_notes_20260428_234338.pptx`。
- **2026-04-29 16:24**：在 `Innovation Projects.pptx` 附录新增第26页“为什么 Anchor 能让候选真正离散？”。核心观点：生成式采样不等于有效离散；anchor 作为显式 mode scaffold，用 top-k 意图拉开候选池语义差异，再由 planner 做场景化生成、Candidate Selector 判断偏离是否安全。当前 PPT 为37页，主讲范围仍为第1-22页。

## 实验记录 (重点使用2k内部surrogate eval口径)
- no anchor / raw FP：碰撞风险约 5.45%
- oracle anchor (完美参考指引)：**2.20% ~ 2.0%** （证明指引架构有显著价值）
- predicted anchor (模型预测指引)：约 6.2%
- collision-only selector-DPO (智能选择器) top1：**3.15%** （追平 hand rerank）
- candidate oracle (候选池完美选择)：**1.20%** （显示巨大潜力，下一步主攻方向）
- **2026-04-28 用户补充 PPT 申报口径**：用于“项目可行性/瓶颈诊断”页的最新表述为：`oracle anchor` 碰撞率约 **2.0%**，`predicted anchor` 最新已改善到约 **3.2%-4.2%**，原始 `Flow Planner` 基准约 **2.4%**。该页用于说明方向有效、差距主要在预测 Anchor 与 planner robustness / exposure bias。

## 已修复问题清单
- PPT 模板破坏问题：已回退到 `Innovation Projects 1.pptx` + before_feasibility_update 备份。
- 数据口径混淆：统一使用2k内部评估结论，避免与500场景raw FP混用。
- 内容过于技术化：本次迭代全面中文化，减少英文缩写。
- PPT备注错位问题：旧稿按1-18页主内容写入，但最终版实际主内容为1-22页且第7页为图示页，导致第7页后备注串页。已按当前页序重排并重写PPT备注。

## 关键代码文件索引
- `Innovation Projects.pptx`：**最终版**创新项目申报/答辩材料（37页），已内嵌第1-22页20分钟演讲稿备注；第23页起附录不主动讲；第26页新增 anchor/mode scaffold 解释页
- `Innovation Projects.before_speaker_notes_20260428_231323.pptx`：写入演讲稿备注前的PPT备份
- `Innovation Projects.before_realigned_notes_20260428_234338.pptx`：修复备注错位前的PPT备份
- `speaker_notes_for_innovation_project.md`：**当前页序版20分钟演讲稿**（仅主内容1-22页，自然口语化；第7页图解delta与方案，已对齐申报口径和最新2k指标）
- `Innovation Projects 1.pptx`：早期模板版本（供参考）
- `Innovation_Project_Feasibility_2slides.pptx`：单独生成的 2 页项目可行性/实验证据 PPT，可复制到原申报 PPT 中
- `Innovation_Project_A1A2B1B2_updated.pptx`：单独生成的 4 页创新点更新 PPT，按原模板风格重写 A1/A2/B1/B2，可复制到原申报 PPT 中
- `ppt1_text_map.txt`：PPT文本提取映射，用于精准替换
- `docs/experiments/anchor_conditioned.md`：详细实验记录（暂未创建，本地以本文件为主）

## 下一步 TODO
- [x] 重写 PPT Slide 6-13：从旧的 goal/DPO 技术路线叙事，改为创新项目申报口径（Anchor 引导、Anchor Selector、Candidate Selector、安全评分、迁移验证、2k 内部验证总结）
- [x] 新增/替换 PPT Slide 8-9：项目可行性与实验证据/瓶颈诊断，采用三栏布局和简洁柱状图
- [x] 为最终版 `Innovation Projects.pptx` 制作20分钟自然口语演讲稿（当前主内容1-22页，附录不讲）
- [x] 修复PPT备注错位，并将当前页序版演讲稿直接写入最终版 `Innovation Projects.pptx` 的备注窗格（第1-22页）
- [ ] 用户实际演练验证时长与流畅度
- [ ] 若整体通过，更新实验记录文档（docs/experiments/anchor_conditioned.md）并 commit
- [ ] 继续 anchor selector 实验验证（candidate-level selector，重点candidate oracle方向）

**更新时间**：2026-04-29 16:24（已新增第26页附录 C-2，解释生成式采样不等于有效离散与 anchor mode scaffold；PPT 当前37页）

---
**提醒**：日志更新完成后建议 commit，避免跨机器丢失。**请检查 `Innovation Projects.pptx` 备注区与 `speaker_notes_for_innovation_project.md`，确认无误后一起commit。**

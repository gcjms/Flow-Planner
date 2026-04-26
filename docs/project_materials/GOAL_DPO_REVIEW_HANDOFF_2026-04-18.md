Current usable conclusions:
- Goal scaffolding is useful for opening multimodal candidates; diversity supports Goal_Base < DPO_1ep < DPO_3ep in spread.
- Best-candidate quality drops as spread increases: Goal_Base 9.20/12.00/99.0%%, DPO_1ep 8.52/15.93/99.0%%, DPO_3ep 7.72/21.91/96.5%%.
- Last nights multidim open-loop report is invalid because it ended with Scenes evaluated: 0 and eval_multidim.py hit a CFG shape bug.

Required code changes:
- Fix flow_planner/dpo/eval_multidim.py so it builds NuPlanDataSample and calls model(data, mode="inference", use_cfg=True, cfg_weight=1.8) instead of manually building decoder CFG inputs.
- Add flow_planner/dpo/eval_multidim_goal_ablation.py with --goal_mode {none,route_goal,predicted_goal}.

Core commands:
- Run eval_multidim on checkpoints/model.pth for 200 scenes.
- Run eval_multidim on checkpoints/dpo_goal_tune_b3.0_s0.3_e1/model_dpo_merged.pth for 200 scenes.
- Run eval_multidim_goal_ablation on the same DPO checkpoint with goal_mode none / route_goal / predicted_goal.

Estimated time: about 50-90 min total.
What Goal_Base means:
- Goal_Base is a scaffold, not the final target planner.
- It is only used to verify goal conditioning opens modes and to provide the direct predecessor baseline for DPO.

Three necessary checks:
1. Original Flow Planner vs DPO_b3.0_s0.3 under normal single-shot inference with no explicit goal.
2. Goal_Base vs DPO_b3.0_s0.3 to isolate DPO incremental value.
3. DPO_b3.0_s0.3 in three settings: none, route_goal, predicted_goal.


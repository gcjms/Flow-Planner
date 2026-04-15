# Goal Predictor Run Plan

## Goal

Run a first end-to-end `goal_predictor` experiment on AutoDL and answer one practical question:

`Can scene features predict the correct goal cluster well enough to support top-K goal-conditioned inference?`

## Constraints Observed On AutoDL

- Available backbone assets:
  - `checkpoints/model_goal.pth`
  - `checkpoints/config_goal.yaml`
- Available dataset:
  - `/root/autodl-tmp/hard_scenarios_v2`
- Missing after cleanup:
  - repo-root `goal_vocab.npy`
- Predictor scripts need Hydra config composition instead of raw `OmegaConf.load`.

## Execution Order

1. Regenerate `goal_vocab.npy`
   - Use `flow_planner.goal.cluster_goals`
   - Source dataset: `hard_scenarios_v2`
   - Output: `/root/autodl-tmp/Flow-Planner/goal_vocab.npy`

2. Smoke test training
   - Train on a small subset from `hard_scenarios_v2`
   - Validate on a small subset from the same split
   - Confirm:
     - planner loads
     - labels build correctly
     - feature extraction runs
     - checkpoints save
     - top-k metrics are emitted

3. Fix any runtime issues
   - Config loading
   - goal vocab path
   - batch/device/shape issues

4. Fuller first-pass run
   - Keep `freeze_backbone=True`
   - Train a lightweight MLP head
   - Use `hard_scenarios_v2` as the immediately available dataset

5. Evaluate and decide
   - Report top-1 / top-3 / top-5
   - If top-3 or top-5 is usable, next step is integrating:
     - `scene -> top-K goals -> K trajectories -> rerank`

## Expected Deliverables

- `outputs/goal_predictor_smoke/`
- `outputs/goal_predictor_run1/`
- predictor checkpoints
- `history.json`
- evaluation JSON
- short conclusion on whether the predictor is ready for inference integration

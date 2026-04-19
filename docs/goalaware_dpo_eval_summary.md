# Goal-Aware DPO Current Status

## Scope

This note summarizes the latest `goal-aware DPO` experiments after we changed training to:

- save `chosen_goals` / `rejected_goals` into the rebuilt preference file
- feed `scene + chosen_goal` when scoring the chosen trajectory
- feed `scene + rejected_goal` when scoring the rejected trajectory

The intent was to make DPO training consistent with goal-conditioned candidate generation.

## Main Runs

### Training setup

| Run | Preference file | Key change | Beta | SFT weight | Adaptive ratio | Epochs |
|------|------|------|------:|------:|------:|------:|
| old strict_dimfix adaptive | `preferences_multi_strict_dimfix.npz` | no goal passed back during DPO train | 2.0 | 1.0 | 0.2 | 1 |
| new goal-aware adaptive | `preferences_multi_strict_dimfix_goalaware.npz` | chosen/rejected each use their own goal during DPO train | 2.0 | 1.0 | 0.2 | 1 |

### Preference rebuild

Full rebuilt file:

- `1296` pairs
- all pairs are `same_cluster_subtle_bad`
- dim label counts:
  - `collision: 641`
  - `progress: 174`
  - `route: 253`
  - `semantic: 228`

Stored arrays:

- `chosen`
- `rejected`
- `chosen_goals`
- `rejected_goals`
- `scenario_ids`
- `dim_labels`
- `score_gaps`

## Validation Table

All metrics below use the same 1000-scene open-loop protocol on `hard_scenarios_v2`.

| Model | Goal mode | collision_rate | avg_progress | avg_route |
|------|------|------:|------:|------:|
| Base Flow Planner | `none` | `4.9%` | `0.6084` | `0.8591` |
| old strict_dimfix adaptive | `none` | `19.2%` | `0.4153` | `0.8347` |
| new goal-aware adaptive | `none` | `21.3%` | `0.4206` | `0.8125` |
| new goal-aware adaptive | `route_goal` | `26.6%` | `0.4548` | `0.8682` |

## Training-side Signals

### Smoke test

Small smoke run:

- `32` pairs
- `1` epoch
- completed successfully
- confirmed that the new `chosen_goal / rejected_goal` path is wired correctly

### Full goal-aware run

Full run:

- output dir: `goalaware_strict_dimfix_adaptive_b2.0_s1.0_r0.2_e1`
- final pair accuracy: `48.69%`
- final delta mean: approximately `0`

Representative grad probes:

| Step | lambda_dpo | grad_dpo | grad_sft | ratio dpo/sft |
|------:|------:|------:|------:|------:|
| 20 | 0.1667 | 0.6339 | 0.6086 | 1.042 |
| 40 | 0.1672 | 0.6328 | 0.4187 | 1.511 |
| 100 | 0.1684 | 0.8786 | 0.5819 | 1.510 |
| 140 | 0.1792 | 1.0492 | 1.1491 | 0.913 |
| 160 | 0.1692 | 1.2517 | 0.4991 | 2.508 |

This run never showed a clear, stable preference-learning signal.

## Why It Still Failed

The key issue is not just "whether goal is present during training".

The deeper issue is that the rebuilt pairs are still not "same-condition comparisons" in the DPO sense.

### Important observation

Although all rebuilt pairs are `same_cluster_subtle_bad`, the chosen and rejected goals are often far apart.

Goal distance statistics over the full rebuilt file:

- mean: `24.3 m`
- median: `17.1 m`
- p75: `29.6 m`
- p90: `51.1 m`
- p95: `51.1 m`
- max: `51.1 m`

So in practice many pairs look like:

- `scene + goal_A -> chosen`
- `scene + goal_B -> rejected`

with `goal_A` and `goal_B` still very different.

### Consequence

That means the new goal-aware DPO is often comparing:

- `log p(chosen | scene, chosen_goal)`
- against `log p(rejected | scene, rejected_goal)`

This is no longer a clean "same condition, better vs worse" comparison.

As a result:

- the training signal becomes weak or inconsistent
- pair accuracy collapses toward or below random
- open-loop validation remains poor

## Current Conclusion

The current `cross-goal pair + goal-aware DPO` route is not working.

More specifically:

1. Adding goal back into DPO training did not rescue the run.
2. Under the standard `none` eval protocol, the new model is slightly worse than the old strict baseline.
3. Even under `route_goal` inference, collision gets even worse.

## Recommended Next Step

Do not continue sweeping `beta` or `sft_weight` on this exact setup.

More promising next directions:

1. tighten pair mining from `same cluster` to `same cluster + near goal`
2. move to a reranker pipeline instead of forcing cross-goal DPO
3. if DPO is kept, prefer pairs that compare trajectories under the same or very near goal condition

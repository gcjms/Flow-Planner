# Related Work for Goal-Conditioned Preference / RL Planning

## Goal

This note lists the papers that are most relevant to our current situation:

- many-goal candidate generation works
- preference alignment / DPO works
- RLHF / RL post-training works for autonomous driving

The focus is not "all papers in the area", but "which ones are most helpful for deciding what to do next".

## Short List

| Work | Why it matters to us | Closest part of our pipeline |
|------|------|------|
| GoalFlow | shows why many goals are useful for producing diverse trajectory candidates | candidate generation |
| DriveDPO | closest preference-optimization paper to our current DPO direction | DPO / preference optimization |
| TrajHF | shows that generative trajectory models can be aligned with human feedback | RLHF-style alignment |
| Plan-R1 | strong nuPlan post-training baseline using GRPO rather than DPO | RL post-training baseline |
| DIVER | reinforced diffusion for end-to-end driving, useful contrast to our FM+DPO line | RL on generative driving model |
| READ | RL fine-tuning on top of a pretrained diffusion driving model | post-training comparison |

## 1. GoalFlow

- venue: CVPR 2025
- link: <https://cvpr.thecvf.com/virtual/2025/poster/35027>

### Why it matters

This is the cleanest reference for the "many goals first" part of our pipeline.

Its main lesson for us is:

- many goals are useful for candidate generation
- goal selection should be scene-aware
- goal-conditioning is good at opening up trajectory diversity

### What it does not imply

It does not automatically justify using trajectories from very different goals as direct left-vs-right DPO pairs.

So GoalFlow supports our candidate-generation intuition, but not our current cross-goal DPO construction.

## 2. DriveDPO

- link: <https://openreview.net/forum?id=eIf9GNcA5n>

### Why it matters

This is the closest preference-learning paper to what we hoped to do.

The key takeaway is:

- DPO works best when the comparison is made under a stable conditioning context
- preference optimization should not quietly mix too many hidden variables

### Relevance to our failure

Our latest failure suggests that even after adding goals back into training, the pair itself may still be too cross-condition:

- `scene + goal_A -> chosen`
- `scene + goal_B -> rejected`

That is much weaker than a clean same-condition DPO setup.

## 3. TrajHF

- link: <https://openreview.net/forum?id=5VXHGJiQuW>

### Why it matters

This is the most directly relevant "trajectory generation + human feedback alignment" paper family.

Useful lessons:

- preference alignment on top of a generative planner is feasible
- the alignment signal matters as much as the generator backbone
- human-feedback style objectives can target style or safety, not just imitation quality

### What we should take from it

If DPO keeps failing, the broader RLHF framing is still valid.
The issue may be our pair construction, not the idea of post-training with feedback itself.

## 4. Plan-R1

- arXiv: `2505.17659`

### Why it matters

This is a strong nuPlan post-training reference using GRPO rather than DPO.

Useful lessons:

- planning post-training can work on nuPlan
- rule-based rewards are already competitive in this space
- if our DPO route stays unstable, GRPO-style approaches remain a serious fallback

## 5. DIVER

- reinforced diffusion for autonomous driving

### Why it matters

It is useful as a "same family, different optimizer" comparison:

- they use a generative driving model
- they use RL instead of DPO
- diversity is treated as a first-class issue

This matters because our current bottleneck is exactly the tension between:

- preserving multi-modality
- keeping the preference comparison condition clean

## 6. READ

- reinforcement-based adaptive driving fine-tuning

### Why it matters

This is another concrete reference that post-training a pretrained driving generator can work.

The contrast with us is:

- they accept the complexity of RL
- we wanted a simpler DPO route

If our DPO supervision keeps breaking due to pair construction, READ-like methods become more relevant.

## Current Takeaway

The literature suggests a fairly clean split:

### What is supported

- many goals are good for candidate generation
- post-training on top of a pretrained generative planner is viable
- preference or RL alignment is a reasonable direction in driving

### What is not strongly supported by the literature

- directly comparing trajectories from far-apart goals inside a DPO-style pair

That means our current negative result is not surprising:

- GoalFlow supports "many goals to create diversity"
- DriveDPO supports "clean conditional preference optimization"
- our current setup sits awkwardly between them

## Side-by-Side Comparison

The table below focuses on the three most relevant works for our current design question:

- how candidates are constructed
- how pair / reward supervision is formed
- why they do not fall into our current cross-goal DPO pitfall

| Work | How candidates are built | How pair / reward is built | Why they do not hit our current pitfall |
|------|------|------|------|
| GoalFlow | multiple goal points are used to generate diverse trajectory candidates with a goal-conditioned flow model | not centered on DPO pair mining for planner training; the main loop is candidate generation plus scene-aware selection | many goals are used for candidate generation and selection, not for directly forming cross-goal DPO left-vs-right pairs |
| DriveDPO | trajectory candidates come from a unified driving policy distribution | trajectory-level DPO uses imitation / safety style supervision under a more stable policy context | the comparison is much closer to "same context, better vs worse", instead of comparing trajectories produced under very different goals |
| TrajHF | a generative trajectory model produces multi-modal trajectory candidates | human feedback is used through an RLHF pipeline rather than our current cross-goal DPO setup | it treats preference alignment as policy improvement under feedback, not as a direct comparison between trajectories from far-apart hidden conditions |

### What this means for us

- GoalFlow validates our use of many goals for candidate generation.
- DriveDPO validates preference optimization when the conditioning context is clean.
- TrajHF validates the broader idea of post-training a generative planner with feedback.

The problem is that our current setup mixes these ideas in an unstable way:

- candidate generation behaves more like GoalFlow
- optimization is intended to behave more like DriveDPO
- but pair construction still compares trajectories whose hidden goal conditions can be far apart

So the negative result is not just "DPO failed".
It is more specifically:

- many-goal candidate generation may still be correct
- but cross-goal pair construction is likely the wrong bridge from candidate generation to DPO

## Practical Reading Order

If time is limited, read in this order:

1. GoalFlow
2. DriveDPO
3. TrajHF
4. Plan-R1

## Bottom Line

The literature does not tell us to give up on preference/RL post-training.
It does suggest that we should stop treating "many-goal candidate generation" and "DPO pair construction" as if they were the same problem.

More concretely:

- many goals still make sense for generating diverse candidates
- but DPO pairs likely need tighter conditioning, such as same-goal or near-goal comparisons
- otherwise a reranker may be the more natural next step

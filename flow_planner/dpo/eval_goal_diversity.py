"""
Goal-Conditioned Candidate Diversity Evaluation
==============================================
Evaluate whether a goal-conditioned model is suitable as a DPO candidate generator.

Metrics focus on diversity/separability instead of single-trajectory planning accuracy:
  - endpoint spread
  - pairwise ADE / FDE
  - pairwise final-heading difference
  - scorer margin between best / worst candidates
  - unique endpoint ratio after simple clustering

Usage:
  python -m flow_planner.dpo.eval_goal_diversity \
      --data_dir /root/autodl-tmp/hard_scenarios_v2 \
      --config_path checkpoints/model_config.yaml \
      --ckpt_path checkpoints/model_goal.pth \
      --vocab_path goal_vocab.npy \
      --num_candidates 5 \
      --max_scenarios 200
"""

import argparse
import glob
import logging
import os
from typing import Dict, List

import numpy as np
import torch

from flow_planner.dpo.generate_candidates_goal import (
    generate_candidates_with_goals,
    load_model,
)
from flow_planner.goal.goal_utils import load_goal_vocab
from flow_planner.risk.trajectory_scorer import TrajectoryScorer

logger = logging.getLogger(__name__)


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Return upper-triangular pairwise L2 distances for x: (K, D)."""
    if x.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    d = x[:, None, :] - x[None, :, :]
    m = np.sqrt((d ** 2).sum(axis=-1))
    iu = np.triu_indices(x.shape[0], k=1)
    return m[iu]


def _pairwise_traj_metrics(candidates: np.ndarray) -> Dict[str, float]:
    """
    candidates: (K, T, D)
    Returns mean pairwise ADE / FDE / heading delta across candidate pairs.
    """
    K = candidates.shape[0]
    if K < 2:
        return {
            "pairwise_ade": 0.0,
            "pairwise_fde": 0.0,
            "pairwise_heading_deg": 0.0,
        }

    ades: List[float] = []
    fdes: List[float] = []
    hds: List[float] = []
    for i in range(K):
        for j in range(i + 1, K):
            traj_i = candidates[i]
            traj_j = candidates[j]
            pos_err = np.linalg.norm(traj_i[:, :2] - traj_j[:, :2], axis=-1)
            ades.append(float(pos_err.mean()))
            fdes.append(float(pos_err[-1]))

            hi = np.arctan2(traj_i[-1, 3], traj_i[-1, 2])
            hj = np.arctan2(traj_j[-1, 3], traj_j[-1, 2])
            dh = abs(hi - hj)
            dh = min(dh, 2 * np.pi - dh)
            hds.append(float(np.degrees(dh)))

    return {
        "pairwise_ade": float(np.mean(ades)),
        "pairwise_fde": float(np.mean(fdes)),
        "pairwise_heading_deg": float(np.mean(hds)),
    }


def _unique_endpoint_ratio(endpoints: np.ndarray, threshold: float = 2.0) -> float:
    """
    Greedy clustering on endpoints. Ratio close to 1.0 means candidates end distinctly.
    """
    if len(endpoints) == 0:
        return 0.0
    clusters: List[np.ndarray] = []
    for p in endpoints:
        matched = False
        for c in clusters:
            if np.linalg.norm(p - c) < threshold:
                matched = True
                break
        if not matched:
            clusters.append(p)
    return len(clusters) / len(endpoints)


def evaluate_scene(
    model,
    npz_path: str,
    vocab: np.ndarray,
    num_candidates: int,
    device: str,
    use_cfg: bool,
    cfg_weight: float,
) -> Dict[str, float]:
    candidates, goals = generate_candidates_with_goals(
        model,
        npz_path,
        vocab,
        num_candidates=num_candidates,
        device=device,
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
    )

    endpoints = candidates[:, -1, :2]
    endpoint_pairwise = _pairwise_distances(endpoints)
    goal_pairwise = _pairwise_distances(goals)
    traj_metrics = _pairwise_traj_metrics(candidates)

    scorer = TrajectoryScorer(verbose=False)
    scores = scorer.score_trajectories(torch.from_numpy(candidates).float())
    scores_np = scores.detach().cpu().numpy()

    return {
        "endpoint_spread_mean": float(endpoint_pairwise.mean()) if len(endpoint_pairwise) else 0.0,
        "endpoint_spread_min": float(endpoint_pairwise.min()) if len(endpoint_pairwise) else 0.0,
        "goal_spread_mean": float(goal_pairwise.mean()) if len(goal_pairwise) else 0.0,
        "score_margin": float(scores_np.max() - scores_np.min()),
        "score_std": float(scores_np.std()),
        "unique_endpoint_ratio": float(_unique_endpoint_ratio(endpoints)),
        **traj_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate goal-conditioned candidate diversity")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--max_scenarios", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg_weight", type=float, default=1.8)
    parser.add_argument("--use_cfg", action="store_true", default=True)
    parser.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    vocab = load_goal_vocab(args.vocab_path)
    model = load_model(args.config_path, args.ckpt_path, device=args.device)

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_scenarios:
        npz_files = npz_files[: args.max_scenarios]

    logger.info("Evaluating %d scenes with %d candidates each", len(npz_files), args.num_candidates)

    keys = [
        "endpoint_spread_mean",
        "endpoint_spread_min",
        "goal_spread_mean",
        "pairwise_ade",
        "pairwise_fde",
        "pairwise_heading_deg",
        "score_margin",
        "score_std",
        "unique_endpoint_ratio",
    ]
    agg = {k: [] for k in keys}
    failed = 0

    for idx, npz_path in enumerate(npz_files, 1):
        try:
            metrics = evaluate_scene(
                model,
                npz_path,
                vocab,
                num_candidates=args.num_candidates,
                device=args.device,
                use_cfg=args.use_cfg,
                cfg_weight=args.cfg_weight,
            )
            for k in keys:
                agg[k].append(metrics[k])
        except Exception as exc:
            failed += 1
            logger.warning("Failed on %s: %s", os.path.basename(npz_path), exc)

        if idx % 50 == 0:
            logger.info(
                "[%d/%d] endpoint_spread=%.2f | pairwise_fde=%.2f | score_margin=%.3f",
                idx,
                len(npz_files),
                np.mean(agg["endpoint_spread_mean"]) if agg["endpoint_spread_mean"] else 0.0,
                np.mean(agg["pairwise_fde"]) if agg["pairwise_fde"] else 0.0,
                np.mean(agg["score_margin"]) if agg["score_margin"] else 0.0,
            )

    logger.info("=" * 60)
    logger.info("Goal Candidate Diversity Summary")
    logger.info("  scenes_ok: %d", len(npz_files) - failed)
    logger.info("  scenes_failed: %d", failed)
    for k in keys:
        value = float(np.mean(agg[k])) if agg[k] else float("nan")
        logger.info("  %s: %.4f", k, value)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

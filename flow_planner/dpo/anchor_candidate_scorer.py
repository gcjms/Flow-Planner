"""Shared scoring helpers for anchor-conditioned candidate sets.

The anchor preference pipeline has several consumers of the same candidate
scores: soft selector targets, selector-DPO pair mining, diagnostics, and the
future candidate-level selector.  This module keeps the score components and
scene/anchor summaries in one place while preserving the existing ``total_score``
semantics used by previous experiments.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AnchorCandidateScoreWeights:
    """Weights for the current moderate safety-first teacher score."""

    safety_weight: float = 5.0
    route_weight: float = 1.0
    progress_weight: float = 1.0
    ttc_weight: float = 0.1
    comfort_weight: float = 0.05
    collision_weight: float = 0.05

    @classmethod
    def from_args(cls, args) -> "AnchorCandidateScoreWeights":
        return cls(
            safety_weight=float(args.safety_weight),
            route_weight=float(args.route_weight),
            progress_weight=float(args.progress_weight),
            ttc_weight=float(args.ttc_weight),
            comfort_weight=float(args.comfort_weight),
            collision_weight=float(args.collision_weight),
        )

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _metric(metrics: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


def score_components(
    metrics: Mapping[str, float],
    weights: AnchorCandidateScoreWeights,
) -> Dict[str, float]:
    """Return explicit candidate score components plus ``final_score``.

    ``collision_score`` is a safety-margin score: larger means farther from
    nearby agents and therefore safer.
    """

    collided = _metric(metrics, "collided")
    safe_score = 1.0 - collided
    collision_score = _metric(metrics, "collision_score")
    ttc_score = _metric(metrics, "ttc_score")
    route_score = _metric(metrics, "route_score")
    progress_score = _metric(metrics, "progress_score")
    comfort_score = _metric(metrics, "comfort_score")
    final_score = (
        weights.safety_weight * safe_score
        + weights.route_weight * route_score
        + weights.progress_weight * progress_score
        + weights.ttc_weight * ttc_score
        + weights.comfort_weight * comfort_score
        + weights.collision_weight * collision_score
    )
    return {
        "collided": collided,
        "safe_score": float(safe_score),
        "collision_score": collision_score,
        "ttc_score": ttc_score,
        "route_score": route_score,
        "progress_score": progress_score,
        "comfort_score": comfort_score,
        "final_score": float(final_score),
    }


def build_candidate_record(
    *,
    candidate_idx: int,
    anchor_index: int,
    anchor_rank: int,
    sample_i: int,
    seed: int,
    metrics: Mapping[str, float],
    weights: AnchorCandidateScoreWeights,
) -> Dict[str, object]:
    """Build the JSON record stored for one generated candidate."""

    components = score_components(metrics, weights)
    float_metrics = {key: float(value) for key, value in metrics.items()}
    return {
        "candidate_idx": int(candidate_idx),
        "anchor_index": int(anchor_index),
        "anchor_rank": int(anchor_rank),
        "sample_i": int(sample_i),
        "seed": int(seed),
        # Backward-compatible field consumed by existing selector scripts.
        "total_score": float(components["final_score"]),
        "metrics": float_metrics,
        "score_components": components,
    }


def get_candidate_score(candidate: Mapping[str, object]) -> float:
    if "total_score" in candidate:
        return float(candidate["total_score"])
    components = candidate.get("score_components", {})
    if isinstance(components, Mapping) and "final_score" in components:
        return float(components["final_score"])
    raise KeyError("candidate record missing total_score/final_score")


def get_candidate_collided(candidate: Mapping[str, object]) -> float:
    metrics = candidate.get("metrics", {})
    if isinstance(metrics, Mapping) and "collided" in metrics:
        return float(metrics["collided"])
    components = candidate.get("score_components", {})
    if isinstance(components, Mapping) and "collided" in components:
        return float(components["collided"])
    return 0.0


def logmeanexp(values: Sequence[float], temperature: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float64) / temperature
    m = float(arr.max())
    return float(temperature * (m + math.log(np.exp(arr - m).mean())))


def aggregate_anchor_scores(
    candidates: Sequence[Mapping[str, object]],
    method: str,
) -> Tuple[List[int], List[float]]:
    """Aggregate per-candidate scores into one score per anchor."""

    grouped: Dict[int, List[float]] = {}
    for candidate in candidates:
        anchor_idx = int(candidate["anchor_index"])
        grouped.setdefault(anchor_idx, []).append(get_candidate_score(candidate))

    anchor_indices = sorted(grouped)
    anchor_scores: List[float] = []
    for anchor_idx in anchor_indices:
        scores = grouped[anchor_idx]
        if method == "mean":
            score = float(np.mean(scores))
        elif method == "max":
            score = float(np.max(scores))
        elif method == "logmeanexp":
            score = logmeanexp(scores)
        else:
            raise ValueError(f"unknown anchor score aggregation: {method}")
        anchor_scores.append(score)
    return anchor_indices, anchor_scores


def summarize_anchor_groups(
    candidates: Sequence[Mapping[str, object]],
    method: str = "mean",
) -> Dict[int, Dict[str, float]]:
    """Return per-anchor group diagnostics for scored candidate JSON."""

    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for candidate in candidates:
        grouped.setdefault(int(candidate["anchor_index"]), []).append(candidate)

    summaries: Dict[int, Dict[str, float]] = {}
    for anchor_idx, rows in grouped.items():
        scores = np.asarray([get_candidate_score(row) for row in rows], dtype=np.float64)
        collided = np.asarray([get_candidate_collided(row) for row in rows], dtype=np.float64)
        if method == "mean":
            agg_score = float(scores.mean())
        elif method == "max":
            agg_score = float(scores.max())
        elif method == "logmeanexp":
            agg_score = logmeanexp(scores.tolist())
        else:
            raise ValueError(f"unknown anchor score aggregation: {method}")

        best_local_idx = int(scores.argmax())
        summaries[anchor_idx] = {
            "anchor_index": int(anchor_idx),
            "anchor_rank": float(rows[0].get("anchor_rank", -1)),
            "num_samples": int(len(rows)),
            "num_safe": int((collided <= 0.5).sum()),
            "num_collided": int((collided > 0.5).sum()),
            "collision_rate": float(collided.mean()) if len(collided) else 0.0,
            "has_collision": float(bool((collided > 0.5).any())),
            "all_safe": float(bool(len(collided) and (collided <= 0.5).all())),
            "all_collide": float(bool(len(collided) and (collided > 0.5).all())),
            "score": agg_score,
            "mean_score": float(scores.mean()),
            "max_score": float(scores.max()),
            "min_score": float(scores.min()),
            "best_candidate_idx": int(rows[best_local_idx]["candidate_idx"]),
        }
    return summaries


def summarize_scene(candidates: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    """Summarize whether the scene candidate pool has useful safety coverage."""

    if not candidates:
        return {
            "num_candidates": 0,
            "num_safe": 0,
            "num_collided": 0,
            "collision_rate": 0.0,
            "has_mixed_collision": 0.0,
            "all_safe": 0.0,
            "all_collide": 0.0,
            "best_candidate_idx": -1,
            "best_candidate_score": 0.0,
            "best_candidate_collided": 0.0,
            "score_gap": 0.0,
        }

    scores = np.asarray([get_candidate_score(candidate) for candidate in candidates], dtype=np.float64)
    collided = np.asarray([get_candidate_collided(candidate) for candidate in candidates], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    num_collided = int((collided > 0.5).sum())
    num_safe = int(len(collided) - num_collided)
    return {
        "num_candidates": int(len(candidates)),
        "num_safe": num_safe,
        "num_collided": num_collided,
        "collision_rate": float(num_collided / max(len(candidates), 1)),
        "has_mixed_collision": float(num_safe > 0 and num_collided > 0),
        "all_safe": float(num_collided == 0),
        "all_collide": float(num_safe == 0),
        "best_candidate_idx": int(candidates[int(order[0])]["candidate_idx"]),
        "best_candidate_score": float(scores[int(order[0])]),
        "best_candidate_collided": float(collided[int(order[0])]),
        "score_gap": float(scores[int(order[0])] - scores[int(order[1])]) if len(order) > 1 else 0.0,
    }


def pair_label(chosen: Mapping[str, float], rejected: Mapping[str, float]) -> str:
    if float(chosen["has_collision"]) < float(rejected["has_collision"]):
        return "anchor_collision"
    if float(chosen["collision_rate"]) < float(rejected["collision_rate"]):
        return "anchor_collision_rate"
    return "anchor_quality"


def json_ready_anchor_summaries(
    summaries: Mapping[int, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Convert integer-keyed anchor summaries to JSON object keys."""

    return {str(anchor_idx): dict(summary) for anchor_idx, summary in summaries.items()}

#!/usr/bin/env python3
"""
Analyze whether candidate trajectories form distinct behavior modes.

This script reads *_candidates.npz files and produces:
1. A global JSON summary.
2. A per-scene JSONL report with candidate-level annotations.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CandidateFeatures:
    endpoint_x: float
    endpoint_y: float
    midpoint_x: float
    midpoint_y: float
    max_lateral: float
    progress_x: float
    final_heading_deg: float
    endpoint_angle_deg: float

    def as_cluster_vector(self) -> np.ndarray:
        return np.array(
            [
                self.endpoint_x,
                self.endpoint_y,
                self.midpoint_y,
                self.max_lateral,
                self.final_heading_deg,
                self.progress_x,
            ],
            dtype=np.float32,
        )


def ensure_candidates_shape(candidates: np.ndarray) -> np.ndarray:
    """Normalize candidates to (K, T, D)."""
    cands = np.asarray(candidates)
    if cands.ndim == 4:
        cands = np.squeeze(cands, axis=1)
    if cands.ndim != 3:
        raise ValueError(f"Expected candidates with 3 dims, got shape {cands.shape}")
    return cands


def pairwise_upper_triangle_l2(x: np.ndarray) -> np.ndarray:
    """Return upper-triangular pairwise distances for x: (K, D)."""
    if x.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    matrix = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(x.shape[0], k=1)
    return matrix[iu]


def compute_pairwise_traj_metrics(candidates: np.ndarray) -> Dict[str, float]:
    """Compute average pairwise ADE/FDE/final-heading deltas across candidates."""
    cands = ensure_candidates_shape(candidates)
    if cands.shape[0] < 2:
        return {
            "pairwise_ade": 0.0,
            "pairwise_fde": 0.0,
            "pairwise_heading_deg": 0.0,
        }

    ades: List[float] = []
    fdes: List[float] = []
    heading_deltas: List[float] = []
    for idx in range(cands.shape[0]):
        for jdx in range(idx + 1, cands.shape[0]):
            traj_i = cands[idx]
            traj_j = cands[jdx]
            pos_err = np.linalg.norm(traj_i[:, :2] - traj_j[:, :2], axis=-1)
            ades.append(float(pos_err.mean()))
            fdes.append(float(pos_err[-1]))

            heading_i = estimate_heading_deg(traj_i)
            heading_j = estimate_heading_deg(traj_j)
            delta = abs(heading_i - heading_j)
            delta = min(delta, 360.0 - delta)
            heading_deltas.append(float(delta))

    return {
        "pairwise_ade": float(np.mean(ades)),
        "pairwise_fde": float(np.mean(fdes)),
        "pairwise_heading_deg": float(np.mean(heading_deltas)),
    }


def estimate_heading_deg(traj: np.ndarray) -> float:
    """Estimate final heading in degrees from cos/sin if available, else from displacement."""
    if traj.shape[1] >= 4:
        cos_h = float(traj[-1, 2])
        sin_h = float(traj[-1, 3])
        if abs(cos_h) + abs(sin_h) > 1e-6:
            return float(np.degrees(np.arctan2(sin_h, cos_h)))

    if traj.shape[0] >= 2:
        delta = traj[-1, :2] - traj[-2, :2]
        if np.linalg.norm(delta) > 1e-6:
            return float(np.degrees(np.arctan2(delta[1], delta[0])))
    return 0.0


def compute_candidate_features(traj: np.ndarray) -> CandidateFeatures:
    """Compute lightweight geometry features for one trajectory."""
    endpoint = traj[-1, :2]
    midpoint = traj[len(traj) // 2, :2]
    final_heading_deg = estimate_heading_deg(traj)
    endpoint_angle_deg = float(np.degrees(np.arctan2(endpoint[1], endpoint[0] + 1e-6)))
    max_lateral = float(np.max(np.abs(traj[:, 1])))
    progress_x = float(endpoint[0])

    return CandidateFeatures(
        endpoint_x=float(endpoint[0]),
        endpoint_y=float(endpoint[1]),
        midpoint_x=float(midpoint[0]),
        midpoint_y=float(midpoint[1]),
        max_lateral=max_lateral,
        progress_x=progress_x,
        final_heading_deg=final_heading_deg,
        endpoint_angle_deg=endpoint_angle_deg,
    )


def classify_maneuver(features: CandidateFeatures) -> str:
    """Assign a coarse maneuver tag from ego-frame trajectory geometry."""
    endpoint_y = features.endpoint_y
    progress_x = features.progress_x
    max_lateral = features.max_lateral
    endpoint_angle = features.endpoint_angle_deg
    heading_deg = features.final_heading_deg

    if progress_x < 3.0 and abs(endpoint_y) < 2.0 and max_lateral < 2.5:
        return "wait_or_stop"
    if endpoint_angle > 45.0 or heading_deg > 35.0:
        return "left_turn"
    if endpoint_angle < -45.0 or heading_deg < -35.0:
        return "right_turn"
    if max_lateral >= 3.0 and endpoint_y > 1.5:
        return "left_bypass"
    if max_lateral >= 3.0 and endpoint_y < -1.5:
        return "right_bypass"
    return "follow_or_straight"


def greedy_cluster(
    vectors: np.ndarray,
    threshold: float = 1.75,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Greedy online clustering over normalized vectors."""
    if len(vectors) == 0:
        return np.zeros((0,), dtype=np.int32), []

    means = vectors.mean(axis=0, keepdims=True)
    scales = vectors.std(axis=0, keepdims=True)
    scales[scales < 1e-6] = 1.0
    normalized = (vectors - means) / scales

    cluster_ids = np.full((len(normalized),), -1, dtype=np.int32)
    centroids: List[np.ndarray] = []
    counts: List[int] = []

    for idx, vec in enumerate(normalized):
        if not centroids:
            centroids.append(vec.copy())
            counts.append(1)
            cluster_ids[idx] = 0
            continue

        dists = np.array([np.linalg.norm(vec - c) for c in centroids], dtype=np.float32)
        nearest = int(np.argmin(dists))
        if float(dists[nearest]) <= threshold:
            cluster_ids[idx] = nearest
            old_count = counts[nearest]
            counts[nearest] = old_count + 1
            centroids[nearest] = centroids[nearest] + (vec - centroids[nearest]) / counts[nearest]
        else:
            cluster_ids[idx] = len(centroids)
            centroids.append(vec.copy())
            counts.append(1)

    return cluster_ids, centroids


def normalized_entropy(ids: Sequence[int]) -> float:
    """Entropy normalized to [0, 1] when more than one cluster exists."""
    if not ids:
        return 0.0
    counts = np.array(list(Counter(ids).values()), dtype=np.float32)
    probs = counts / counts.sum()
    raw = float(-(probs * np.log(probs + 1e-12)).sum())
    if len(counts) <= 1:
        return 0.0
    return raw / math.log(len(counts))


def unique_goal_count(goal_labels: Optional[np.ndarray], threshold: float = 1e-3) -> Optional[int]:
    """Count distinct goals with a small tolerance."""
    if goal_labels is None:
        return None
    goals = np.asarray(goal_labels, dtype=np.float32)
    if len(goals) == 0:
        return 0

    centers: List[np.ndarray] = []
    for goal in goals:
        matched = False
        for center in centers:
            if np.linalg.norm(goal - center) <= threshold:
                matched = True
                break
        if not matched:
            centers.append(goal)
    return len(centers)


def goal_maneuver_consistency(
    goal_labels: Optional[np.ndarray],
    maneuver_tags: Sequence[str],
    threshold: float = 1e-3,
) -> Optional[float]:
    """Measure how consistently the same goal maps to one maneuver tag."""
    if goal_labels is None or len(goal_labels) == 0:
        return None

    centers: List[np.ndarray] = []
    goal_cluster_ids: List[int] = []
    for goal in np.asarray(goal_labels, dtype=np.float32):
        matched = False
        for idx, center in enumerate(centers):
            if np.linalg.norm(goal - center) <= threshold:
                goal_cluster_ids.append(idx)
                matched = True
                break
        if not matched:
            goal_cluster_ids.append(len(centers))
            centers.append(goal)

    cluster_to_tags: Dict[int, List[str]] = {}
    for cluster_id, tag in zip(goal_cluster_ids, maneuver_tags):
        cluster_to_tags.setdefault(cluster_id, []).append(tag)

    weighted_scores: List[float] = []
    weights: List[int] = []
    for tags in cluster_to_tags.values():
        tag_counts = Counter(tags)
        weighted_scores.append(max(tag_counts.values()) / len(tags))
        weights.append(len(tags))
    return float(np.average(weighted_scores, weights=weights)) if weights else None


def build_scene_mode_report(
    candidates: np.ndarray,
    goal_labels: Optional[np.ndarray] = None,
    scene_id: Optional[str] = None,
    cluster_threshold: float = 1.75,
) -> Dict[str, object]:
    """Build a per-scene mode report from candidate trajectories."""
    cands = ensure_candidates_shape(candidates)
    features = [compute_candidate_features(traj) for traj in cands]
    feature_matrix = np.stack([f.as_cluster_vector() for f in features], axis=0)
    cluster_ids, _ = greedy_cluster(feature_matrix, threshold=cluster_threshold)
    maneuver_tags = [classify_maneuver(feature) for feature in features]

    endpoints = cands[:, -1, :2]
    endpoint_pairwise = pairwise_upper_triangle_l2(endpoints)
    pairwise_metrics = compute_pairwise_traj_metrics(cands)

    cluster_sizes = Counter(cluster_ids.tolist())
    maneuver_counts = Counter(maneuver_tags)

    candidate_reports: List[Dict[str, object]] = []
    for idx, (feature, cluster_id, maneuver_tag) in enumerate(
        zip(features, cluster_ids.tolist(), maneuver_tags)
    ):
        goal_label = None
        if goal_labels is not None and idx < len(goal_labels):
            goal_label = [float(goal_labels[idx][0]), float(goal_labels[idx][1])]

        candidate_reports.append(
            {
                "candidate_idx": idx,
                "cluster_id": int(cluster_id),
                "maneuver_tag": maneuver_tag,
                "goal_label": goal_label,
                "endpoint": [feature.endpoint_x, feature.endpoint_y],
                "midpoint": [feature.midpoint_x, feature.midpoint_y],
                "max_lateral": feature.max_lateral,
                "progress_x": feature.progress_x,
                "final_heading_deg": feature.final_heading_deg,
                "endpoint_angle_deg": feature.endpoint_angle_deg,
            }
        )

    report: Dict[str, object] = {
        "scene_id": scene_id,
        "num_candidates": int(cands.shape[0]),
        "cluster_count": int(len(cluster_sizes)),
        "cluster_entropy": float(normalized_entropy(cluster_ids.tolist())),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(cluster_sizes.items())},
        "maneuver_counts": dict(sorted(maneuver_counts.items())),
        "unique_goal_count": unique_goal_count(goal_labels),
        "goal_maneuver_consistency": goal_maneuver_consistency(goal_labels, maneuver_tags),
        "endpoint_spread_mean": float(endpoint_pairwise.mean()) if len(endpoint_pairwise) else 0.0,
        "endpoint_spread_min": float(endpoint_pairwise.min()) if len(endpoint_pairwise) else 0.0,
        "endpoint_spread_max": float(endpoint_pairwise.max()) if len(endpoint_pairwise) else 0.0,
        **pairwise_metrics,
        "candidates": candidate_reports,
    }
    return report


def scene_report_to_line(report: Dict[str, object]) -> str:
    """Serialize one scene report for JSONL output."""
    return json.dumps(report, ensure_ascii=False)


def summarize_reports(scene_reports: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate numeric scene metrics into one JSON summary."""
    reports = list(scene_reports)
    metric_keys = [
        "cluster_count",
        "cluster_entropy",
        "unique_goal_count",
        "goal_maneuver_consistency",
        "endpoint_spread_mean",
        "endpoint_spread_min",
        "endpoint_spread_max",
        "pairwise_ade",
        "pairwise_fde",
        "pairwise_heading_deg",
    ]

    aggregates: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        if not values:
            continue
        aggregates[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    maneuver_counts: Counter[str] = Counter()
    for report in reports:
        maneuver_counts.update(report.get("maneuver_counts", {}))

    return {
        "num_scenes": len(reports),
        "metrics": aggregates,
        "maneuver_counts": dict(sorted(maneuver_counts.items())),
    }


def _default_scene_report_path(output_json: str) -> str:
    base = os.path.splitext(output_json)[0]
    return base + "_scenes.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate behavior modes")
    parser.add_argument("--candidates_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--scenes_jsonl", type=str, default=None)
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--cluster_threshold", type=float, default=1.75)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    npz_files = sorted(glob.glob(os.path.join(args.candidates_dir, "*_candidates.npz")))
    if args.max_scenarios:
        npz_files = npz_files[: args.max_scenarios]

    if not npz_files:
        raise FileNotFoundError(f"No *_candidates.npz files found in {args.candidates_dir}")

    scenes_jsonl = args.scenes_jsonl or _default_scene_report_path(args.output_json)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(scenes_jsonl)), exist_ok=True)

    scene_reports: List[Dict[str, object]] = []
    with open(scenes_jsonl, "w", encoding="utf-8") as scene_fp:
        for idx, npz_path in enumerate(npz_files, start=1):
            raw = np.load(npz_path, allow_pickle=True)
            candidates = ensure_candidates_shape(raw["candidates"])
            goal_labels = raw["goal_labels"] if "goal_labels" in raw.files else None
            scene_id = Path(npz_path).stem.replace("_candidates", "")

            report = build_scene_mode_report(
                candidates=candidates,
                goal_labels=goal_labels,
                scene_id=scene_id,
                cluster_threshold=args.cluster_threshold,
            )
            scene_reports.append(report)
            scene_fp.write(scene_report_to_line(report) + "\n")

            if idx % 100 == 0:
                logger.info(
                    "[%d/%d] clusters=%.2f entropy=%.3f pairwise_fde=%.2f",
                    idx,
                    len(npz_files),
                    np.mean([float(r["cluster_count"]) for r in scene_reports]),
                    np.mean([float(r["cluster_entropy"]) for r in scene_reports]),
                    np.mean([float(r["pairwise_fde"]) for r in scene_reports]),
                )

    summary = summarize_reports(scene_reports)
    summary.update(
        {
            "candidates_dir": os.path.abspath(args.candidates_dir),
            "scene_report_path": os.path.abspath(scenes_jsonl),
            "cluster_threshold": args.cluster_threshold,
        }
    )

    with open(args.output_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Mode report saved to %s", os.path.abspath(args.output_json))
    logger.info("Scene reports saved to %s", os.path.abspath(scenes_jsonl))
    logger.info("Scenes analyzed: %d", len(scene_reports))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build multi-pair DPO data from structured scene scoring outputs.

Input:
  - scored_dir/{scenario_id}.json from score_hybrid.py --emit_traj_info
  - candidates_dir/*_candidates.npz

Output:
  - preferences_multi.npz compatible with train_dpo.py
  - preference_meta.jsonl with pair provenance
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from flow_planner.dpo.analyze_candidate_modes import ensure_candidates_shape


SOFT_FAILURES = {"progress", "comfort", "route", "semantic", "legality"}
SCORE_DIMS = ("margin", "progress", "comfort", "route", "legality", "semantic")
GOOD_SOFT_FAILURES = {"comfort"}
UNSAFE_PRIMARY_FAILURES = {"collision", "off_lane", "reverse", "route", "legality"}


@dataclass(frozen=True)
class PairMiningConfig:
    strict_pair_mining: bool = True
    min_score_gap: float = 0.25
    min_good_total_score: float = 7.0
    strict_same_group_min_score_gap: float = 0.75
    strict_same_group_min_dim_drop: float = 0.15
    gt_near_unsafe_per_good: int = 1
    chosen_near_unsafe_per_good: int = 1
    unsafe_score_threshold: float = 0.55


def _load_scene_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_candidates_from_scene(
    scene_payload: Dict[str, object],
    candidates_dir: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    source_npz = scene_payload.get("source_npz")
    scenario_id = str(scene_payload["scenario_id"])
    if source_npz and os.path.exists(source_npz):
        npz_path = source_npz
    else:
        npz_path = os.path.join(candidates_dir, f"{scenario_id}_candidates.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Candidate npz not found for scene {scenario_id}: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    candidates = ensure_candidates_shape(raw["candidates"])
    goal_labels = raw["goal_labels"] if "goal_labels" in raw.files else None
    return candidates, goal_labels, npz_path


def _group_key(candidate: Dict[str, object]) -> str:
    cluster_id = int(candidate.get("cluster_id", -1))
    if cluster_id >= 0:
        return f"cluster:{cluster_id}"
    maneuver_tag = str(candidate.get("maneuver_tag", "ungrouped"))
    return f"maneuver:{maneuver_tag}"


def _candidate_sort_key(candidate: Dict[str, object]) -> Tuple[float, float, float]:
    scores = candidate.get("scores", {})
    return (
        float(candidate.get("total_score", 0.0)),
        float(scores.get("margin", 0.0)),
        float(scores.get("progress", 0.0)),
    )


def _score_value(candidate: Dict[str, object], dim: str) -> float:
    return float(candidate.get("scores", {}).get(dim, 0.0))


def _primary_failure(candidate: Dict[str, object]) -> str:
    return str(candidate.get("primary_failure", "none")).lower()


def _score_drops(chosen: Dict[str, object], rejected: Dict[str, object]) -> Dict[str, float]:
    return {
        dim: float(_score_value(chosen, dim) - _score_value(rejected, dim))
        for dim in SCORE_DIMS
    }


def _dim_label_from_failure(failure_type: str) -> str:
    failure = (failure_type or "collision").lower()
    mapping = {
        "collision": "collision",
        "off_lane": "route",
        "reverse": "route",
        "progress": "progress",
        "comfort": "comfort",
        "route": "route",
        "legality": "route",
        "semantic": "semantic",
        "none": "collision",
    }
    return mapping.get(failure, "collision")


def _dim_label_from_score_drops(score_drops: Dict[str, float]) -> str:
    dim_candidates = {
        "collision": float(score_drops.get("margin", 0.0)),
        "progress": float(score_drops.get("progress", 0.0)),
        "comfort": float(score_drops.get("comfort", 0.0)),
        "route": max(
            float(score_drops.get("route", 0.0)),
            float(score_drops.get("legality", 0.0)),
        ),
        "semantic": float(score_drops.get("semantic", 0.0)),
    }
    best_label, best_value = max(
        dim_candidates.items(),
        key=lambda item: (item[1], item[0]),
    )
    if best_value > 0.0:
        return best_label
    return "comfort"


def _pair_dim_label(
    chosen: Dict[str, object],
    rejected: Dict[str, object],
    failure_type: str,
    score_drops: Dict[str, float],
) -> str:
    # For hard failures, keep the explicit failure semantics.
    if not rejected.get("hard_ok", False):
        return _dim_label_from_failure(failure_type)

    # For soft failures, use the dominant relative score drop instead of the
    # rejected candidate's coarse primary_failure, which is often just "comfort".
    return _dim_label_from_score_drops(score_drops)


def _pair_type(chosen: Dict[str, object], rejected: Dict[str, object]) -> str:
    if int(chosen.get("cluster_id", -1)) == int(rejected.get("cluster_id", -2)):
        if rejected.get("hard_ok", False):
            return "same_cluster_subtle_bad"
        return "same_cluster_hard_failure"
    if rejected.get("hard_ok", False):
        return "cross_cluster_subtle_bad"
    return "cross_cluster_hard_failure"


def _candidate_score_gap(chosen: Dict[str, object], rejected: Dict[str, object]) -> float:
    return float(chosen.get("total_score", 0.0)) - float(rejected.get("total_score", 0.0))


def _candidate_idx(candidate: Dict[str, object]) -> int:
    return int(candidate["candidate_idx"])


def _candidate_metric(candidate: Dict[str, object], key: str, default: float = 0.0) -> float:
    return float(candidate.get("metrics", {}).get(key, default))


def _traj_pairwise_distance(
    chosen_idx: int,
    rejected_idx: int,
    candidate_trajs: np.ndarray,
) -> Tuple[float, float]:
    chosen_traj = np.asarray(candidate_trajs[chosen_idx], dtype=np.float32)
    rejected_traj = np.asarray(candidate_trajs[rejected_idx], dtype=np.float32)
    if chosen_traj.ndim != 2 or rejected_traj.ndim != 2:
        return float("inf"), float("inf")
    min_len = min(chosen_traj.shape[0], rejected_traj.shape[0])
    if min_len <= 0:
        return float("inf"), float("inf")
    pos_err = np.linalg.norm(
        chosen_traj[:min_len, :2] - rejected_traj[:min_len, :2],
        axis=-1,
    )
    return float(pos_err.mean()), float(pos_err[-1])


def _is_unsafe_candidate(
    chosen: Dict[str, object],
    candidate: Dict[str, object],
    config: PairMiningConfig,
) -> bool:
    if _candidate_idx(candidate) == _candidate_idx(chosen):
        return False
    if _candidate_score_gap(chosen, candidate) < config.min_score_gap:
        return False
    if not candidate.get("hard_ok", True):
        return True

    primary_failure = _primary_failure(candidate)
    if primary_failure in UNSAFE_PRIMARY_FAILURES:
        return True

    unsafe_threshold = float(config.unsafe_score_threshold)
    return any(
        _score_value(candidate, dim) < unsafe_threshold
        for dim in ("margin", "route", "legality")
    )


def _is_good_representative(candidate: Dict[str, object], config: PairMiningConfig) -> bool:
    if not candidate.get("hard_ok", False):
        return False
    if not config.strict_pair_mining:
        return True
    primary_failure = _primary_failure(candidate)
    if primary_failure not in {"none"} | GOOD_SOFT_FAILURES:
        return False
    return float(candidate.get("total_score", 0.0)) >= config.min_good_total_score


def _is_strict_same_group_subtle_bad(
    chosen: Dict[str, object],
    rejected: Dict[str, object],
    config: PairMiningConfig,
) -> bool:
    if not rejected.get("hard_ok", False):
        return False

    score_gap = _candidate_score_gap(chosen, rejected)
    required_gap = max(config.min_score_gap, config.strict_same_group_min_score_gap)
    if score_gap < required_gap:
        return False

    if _primary_failure(rejected) in SOFT_FAILURES:
        return True

    score_drops = _score_drops(chosen, rejected)
    return max(score_drops.values(), default=0.0) >= config.strict_same_group_min_dim_drop


def _select_rejected_candidates(
    chosen: Dict[str, object],
    group_members: Sequence[Dict[str, object]],
    all_candidates: Sequence[Dict[str, object]],
    candidate_trajs: np.ndarray,
    subtle_bad_per_good: int,
    config: PairMiningConfig,
) -> List[Tuple[Dict[str, object], str]]:
    chosen_idx = int(chosen["candidate_idx"])
    selected: List[Tuple[Dict[str, object], str]] = []
    selected_ids = {chosen_idx}

    def sort_pool(pool: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        return sorted(
            pool,
            key=lambda candidate: (
                _candidate_score_gap(chosen, candidate),
                -float(candidate.get("scores", {}).get("margin", 0.0)),
            ),
        )

    def append_candidates(
        pool: Iterable[Dict[str, object]],
        source: str,
        *,
        max_take: Optional[int] = None,
        ranking_key=None,
    ) -> None:
        if max_take is not None and max_take <= 0:
            return
        ranked = list(pool)
        if ranking_key is None:
            ranked = sort_pool(ranked)
        else:
            ranked = sorted(ranked, key=ranking_key)

        added = 0
        for candidate in ranked:
            candidate_id = _candidate_idx(candidate)
            if candidate_id in selected_ids:
                continue
            selected.append((candidate, source))
            selected_ids.add(candidate_id)
            added += 1
            if len(selected) >= subtle_bad_per_good:
                return
            if max_take is not None and added >= max_take:
                return

    if config.strict_pair_mining:
        strict_same_group = [
            candidate
            for candidate in group_members
            if _candidate_idx(candidate) != chosen_idx
            and _is_strict_same_group_subtle_bad(chosen, candidate, config)
        ]
        append_candidates(strict_same_group, "strict_same_group")
        if len(selected) >= subtle_bad_per_good:
            return selected

    unsafe_pool = [
        candidate
        for candidate in all_candidates
        if _is_unsafe_candidate(chosen, candidate, config)
    ]
    if config.gt_near_unsafe_per_good > 0:
        append_candidates(
            unsafe_pool,
            "gt_near_unsafe",
            max_take=config.gt_near_unsafe_per_good,
            ranking_key=lambda candidate: (
                _candidate_metric(candidate, "fde") + 0.5 * _candidate_metric(candidate, "ade"),
                _candidate_metric(candidate, "fde"),
                _candidate_metric(candidate, "ade"),
                _candidate_score_gap(chosen, candidate),
            ),
        )
        if len(selected) >= subtle_bad_per_good:
            return selected

    if config.chosen_near_unsafe_per_good > 0:
        append_candidates(
            unsafe_pool,
            "chosen_near_unsafe",
            max_take=config.chosen_near_unsafe_per_good,
            ranking_key=lambda candidate: (
                *_traj_pairwise_distance(chosen_idx, _candidate_idx(candidate), candidate_trajs),
                _candidate_score_gap(chosen, candidate),
            ),
        )
        if len(selected) >= subtle_bad_per_good:
            return selected

    same_group_soft = [
        candidate
        for candidate in group_members
        if _candidate_idx(candidate) != chosen_idx
        and candidate.get("hard_ok", False)
        and _candidate_score_gap(chosen, candidate) >= config.min_score_gap
    ]
    append_candidates(
        same_group_soft,
        "same_group_soft",
        max_take=max(subtle_bad_per_good - len(selected), 0),
    )
    if len(selected) >= subtle_bad_per_good:
        return selected

    cross_group_soft = [
        candidate
        for candidate in all_candidates
        if _candidate_idx(candidate) not in selected_ids
        and candidate.get("hard_ok", False)
        and candidate.get("primary_failure") in SOFT_FAILURES
        and _candidate_score_gap(chosen, candidate) >= config.min_score_gap
    ]
    append_candidates(cross_group_soft, "cross_group_soft")
    if len(selected) >= subtle_bad_per_good:
        return selected

    hard_failures = [
        candidate
        for candidate in all_candidates
        if _candidate_idx(candidate) not in selected_ids
        and not candidate.get("hard_ok", True)
        and _candidate_score_gap(chosen, candidate) >= config.min_score_gap
    ]
    append_candidates(hard_failures, "hard_failure_fallback")

    return selected


def _scene_pairs(
    scene_payload: Dict[str, object],
    candidate_trajs: np.ndarray,
    top_good_per_cluster: int,
    subtle_bad_per_good: int,
    config: PairMiningConfig,
) -> List[Tuple[Dict[str, object], Dict[str, object], str]]:
    candidates = list(scene_payload.get("candidates", []))
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        groups[_group_key(candidate)].append(candidate)

    pairs: List[Tuple[Dict[str, object], Dict[str, object], str]] = []
    seen: set[Tuple[int, int]] = set()

    for group_members in groups.values():
        ranked_group = sorted(group_members, key=_candidate_sort_key, reverse=True)
        good_reps = [
            candidate for candidate in ranked_group if _is_good_representative(candidate, config)
        ]
        if not good_reps:
            continue
        for chosen in good_reps[:top_good_per_cluster]:
            for rejected, selection_source in _select_rejected_candidates(
                chosen=chosen,
                group_members=group_members,
                all_candidates=candidates,
                candidate_trajs=candidate_trajs,
                subtle_bad_per_good=subtle_bad_per_good,
                config=config,
            ):
                key = (int(chosen["candidate_idx"]), int(rejected["candidate_idx"]))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((chosen, rejected, selection_source))
    return pairs


def _meta_record(
    pair_id: int,
    scene_payload: Dict[str, object],
    chosen: Dict[str, object],
    rejected: Dict[str, object],
    selection_source: str,
    goal_labels: Optional[np.ndarray],
) -> Dict[str, object]:
    chosen_idx = int(chosen["candidate_idx"])
    rejected_idx = int(rejected["candidate_idx"])
    score_gap = _candidate_score_gap(chosen, rejected)
    failure_type = str(rejected.get("primary_failure", "collision"))
    score_drops = _score_drops(chosen, rejected)
    dim_label = _pair_dim_label(
        chosen=chosen,
        rejected=rejected,
        failure_type=failure_type,
        score_drops=score_drops,
    )

    chosen_goal = None
    rejected_goal = None
    if goal_labels is not None:
        if chosen_idx < len(goal_labels):
            chosen_goal = [float(goal_labels[chosen_idx][0]), float(goal_labels[chosen_idx][1])]
        if rejected_idx < len(goal_labels):
            rejected_goal = [float(goal_labels[rejected_idx][0]), float(goal_labels[rejected_idx][1])]

    return {
        "pair_id": pair_id,
        "scenario_id": str(scene_payload["scenario_id"]),
        "chosen_idx": chosen_idx,
        "rejected_idx": rejected_idx,
        "chosen_cluster_id": int(chosen.get("cluster_id", -1)),
        "rejected_cluster_id": int(rejected.get("cluster_id", -1)),
        "pair_type": _pair_type(chosen, rejected),
        "selection_source": selection_source,
        "score_gap": score_gap,
        "failure_type": failure_type,
        "dim_label": dim_label,
        "score_drops": score_drops,
        "goal_labels": {
            "chosen": chosen_goal,
            "rejected": rejected_goal,
        },
    }


def _default_meta_path(output_path: str) -> str:
    base = os.path.splitext(output_path)[0]
    return base + "_meta.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-pair DPO preferences")
    parser.add_argument("--scored_dir", type=str, required=True)
    parser.add_argument("--candidates_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--top_good_per_cluster", type=int, default=1)
    parser.add_argument("--subtle_bad_per_good", type=int, default=2)
    parser.add_argument("--min_score_gap", type=float, default=0.25)
    parser.add_argument("--strict_pair_mining", dest="strict_pair_mining", action="store_true")
    parser.add_argument("--no_strict_pair_mining", dest="strict_pair_mining", action="store_false")
    parser.set_defaults(strict_pair_mining=True)
    parser.add_argument("--min_good_total_score", type=float, default=7.0)
    parser.add_argument("--strict_same_group_min_score_gap", type=float, default=0.75)
    parser.add_argument("--strict_same_group_min_dim_drop", type=float, default=0.15)
    parser.add_argument("--gt_near_unsafe_per_good", type=int, default=1)
    parser.add_argument("--chosen_near_unsafe_per_good", type=int, default=1)
    parser.add_argument("--unsafe_score_threshold", type=float, default=0.55)
    parser.add_argument("--max_scenarios", type=int, default=None)
    args = parser.parse_args()

    mining_config = PairMiningConfig(
        strict_pair_mining=args.strict_pair_mining,
        min_score_gap=args.min_score_gap,
        min_good_total_score=args.min_good_total_score,
        strict_same_group_min_score_gap=args.strict_same_group_min_score_gap,
        strict_same_group_min_dim_drop=args.strict_same_group_min_dim_drop,
        gt_near_unsafe_per_good=args.gt_near_unsafe_per_good,
        chosen_near_unsafe_per_good=args.chosen_near_unsafe_per_good,
        unsafe_score_threshold=args.unsafe_score_threshold,
    )

    scored_files = sorted(Path(args.scored_dir).glob("*.json"))
    if args.max_scenarios:
        scored_files = scored_files[: args.max_scenarios]
    if not scored_files:
        raise FileNotFoundError(f"No scored scene json files found in {args.scored_dir}")

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    meta_path = args.meta_path or _default_meta_path(args.output_path)
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)

    chosen_rows: List[np.ndarray] = []
    rejected_rows: List[np.ndarray] = []
    scenario_ids: List[str] = []
    dim_labels: List[str] = []
    score_gaps: List[float] = []
    chosen_goals: List[np.ndarray] = []
    rejected_goals: List[np.ndarray] = []
    has_pair_goals = True
    meta_records: List[Dict[str, object]] = []
    pair_type_counts: Counter[str] = Counter()
    selection_source_counts: Counter[str] = Counter()
    dim_label_counts: Counter[str] = Counter()

    pair_id = 0
    for scored_file in scored_files:
        scene_payload = _load_scene_json(str(scored_file))
        candidates, goal_labels, _ = _load_candidates_from_scene(scene_payload, args.candidates_dir)
        pairs = _scene_pairs(
            scene_payload=scene_payload,
            candidate_trajs=candidates,
            top_good_per_cluster=args.top_good_per_cluster,
            subtle_bad_per_good=args.subtle_bad_per_good,
            config=mining_config,
        )
        for chosen, rejected, selection_source in pairs:
            chosen_idx = int(chosen["candidate_idx"])
            rejected_idx = int(rejected["candidate_idx"])
            chosen_rows.append(candidates[chosen_idx])
            rejected_rows.append(candidates[rejected_idx])
            scenario_ids.append(str(scene_payload["scenario_id"]))

            record = _meta_record(
                pair_id=pair_id,
                scene_payload=scene_payload,
                chosen=chosen,
                rejected=rejected,
                selection_source=selection_source,
                goal_labels=goal_labels,
            )
            meta_records.append(record)
            dim_labels.append(str(record["dim_label"]))
            score_gaps.append(float(record["score_gap"]))
            chosen_goal = record["goal_labels"]["chosen"]
            rejected_goal = record["goal_labels"]["rejected"]
            if chosen_goal is None or rejected_goal is None:
                has_pair_goals = False
            chosen_goals.append(
                np.asarray(chosen_goal if chosen_goal is not None else [np.nan, np.nan], dtype=np.float32)
            )
            rejected_goals.append(
                np.asarray(rejected_goal if rejected_goal is not None else [np.nan, np.nan], dtype=np.float32)
            )
            pair_type_counts.update([str(record["pair_type"])])
            selection_source_counts.update([str(record["selection_source"])])
            dim_label_counts.update([str(record["dim_label"])])
            pair_id += 1

    if not chosen_rows:
        raise RuntimeError("No multi-pairs were generated. Check score gaps and scene quality.")

    payload = {
        "chosen": np.array(chosen_rows),
        "rejected": np.array(rejected_rows),
        "scenario_ids": np.array(scenario_ids),
        "dim_labels": np.array(dim_labels),
        "score_gaps": np.array(score_gaps, dtype=np.float32),
    }
    if has_pair_goals:
        payload["chosen_goals"] = np.stack(chosen_goals).astype(np.float32)
        payload["rejected_goals"] = np.stack(rejected_goals).astype(np.float32)

    np.savez_compressed(args.output_path, **payload)

    with open(meta_path, "w", encoding="utf-8") as fp:
        for record in meta_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 60)
    print(f"Saved {len(chosen_rows)} preference pairs to {args.output_path}")
    print(f"Saved pair metadata to {meta_path}")
    print(f"Strict pair mining: {mining_config.strict_pair_mining}")
    print(f"Pair type counts: {dict(sorted(pair_type_counts.items()))}")
    print(f"Selection source counts: {dict(sorted(selection_source_counts.items()))}")
    print(f"Dim label counts: {dict(sorted(dim_label_counts.items()))}")
    print("=" * 60)


if __name__ == "__main__":
    main()

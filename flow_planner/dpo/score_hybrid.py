#!/usr/bin/env python3
"""
Hybrid Rule + VLM preference scoring pipeline.

Default mode preserves the legacy top1-vs-worst1 behavior.
Structured mode emits candidate-level diagnostics for multi-pair DPO.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flow_planner.dpo.analyze_candidate_modes import (
    build_scene_mode_report,
    classify_maneuver,
    compute_candidate_features,
    ensure_candidates_shape,
)

logger = logging.getLogger(__name__)

STRUCTURED_SCORE_WEIGHTS = {
    "margin": 4.0,
    "progress": 2.5,
    "comfort": 1.5,
    "route": 2.0,
    "legality": 2.0,
    "semantic": 1.0,
}


def _extrapolate_neighbor_future(
    neighbor_past: np.ndarray,
    future_steps: int,
    dt: float = 0.1,
) -> np.ndarray:
    """Extrapolate neighbor futures with a constant-velocity model."""
    future = np.zeros((neighbor_past.shape[0], future_steps, 2), dtype=np.float32)
    for idx, last in enumerate(neighbor_past[:, -1]):
        x, y = last[0], last[1]
        vx, vy = last[4], last[5]
        for step in range(future_steps):
            future[idx, step, 0] = x + vx * dt * (step + 1)
            future[idx, step, 1] = y + vy * dt * (step + 1)
    return future


def _load_candidate_bundle(candidate_npz_path: str) -> Dict[str, object]:
    """Load all arrays needed to score one candidate file."""
    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = ensure_candidates_shape(data["candidates"])
    return {
        "scenario_id": Path(candidate_npz_path).stem.replace("_candidates", ""),
        "path": candidate_npz_path,
        "candidates": cands,
        "gt_future": data["ego_agent_future"],
        "neighbors": data["neighbor_agents_past"],
        "lanes": data["lanes"] if "lanes" in data.files else None,
        "goal_labels": data["goal_labels"] if "goal_labels" in data.files else None,
        "ego_agent_past": data["ego_agent_past"],
        "ego_current_state": data["ego_current_state"],
    }


def _extract_valid_lane_points(lanes: Optional[np.ndarray]) -> np.ndarray:
    """Flatten lane polylines into valid XY points."""
    if lanes is None:
        return np.zeros((0, 2), dtype=np.float32)

    lane_array = np.asarray(lanes)
    if lane_array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points = lane_array[..., :2].reshape(-1, 2)
    finite_mask = np.isfinite(points).all(axis=1)
    nonzero_mask = np.linalg.norm(points, axis=-1) > 1e-3
    return points[finite_mask & nonzero_mask]


def _nearest_reference_distances(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return nearest distance from each point to a reference point set."""
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(reference) == 0:
        return np.full((len(points),), 999.0, dtype=np.float32)
    diff = points[:, None, :] - reference[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return dists.min(axis=1)


def _compute_comfort_metrics(traj_xy: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """Compute simple comfort features from xy positions."""
    if len(traj_xy) < 3:
        return {"max_acc": 0.0, "max_jerk": 0.0, "comfort_score": 1.0}

    velocity = np.diff(traj_xy, axis=0) / dt
    if len(velocity) < 2:
        return {"max_acc": 0.0, "max_jerk": 0.0, "comfort_score": 1.0}

    accel = np.diff(velocity, axis=0) / dt
    jerk = np.diff(accel, axis=0) / dt if len(accel) >= 2 else np.zeros((0, 2), dtype=np.float32)
    max_acc = float(np.linalg.norm(accel, axis=-1).max()) if len(accel) else 0.0
    max_jerk = float(np.linalg.norm(jerk, axis=-1).max()) if len(jerk) else 0.0
    comfort_score = 1.0 / (1.0 + max_acc / 4.0 + max_jerk / 8.0)
    return {
        "max_acc": max_acc,
        "max_jerk": max_jerk,
        "comfort_score": float(np.clip(comfort_score, 0.0, 1.0)),
    }


def _maneuver_similarity(candidate_tag: str, gt_tag: str) -> float:
    """Soft similarity between coarse maneuver tags."""
    if candidate_tag == gt_tag:
        return 1.0

    left_family = {"left_turn", "left_bypass"}
    right_family = {"right_turn", "right_bypass"}
    forward_family = {"follow_or_straight", "wait_or_stop"}
    if candidate_tag in left_family and gt_tag in left_family:
        return 0.75
    if candidate_tag in right_family and gt_tag in right_family:
        return 0.75
    if candidate_tag in forward_family and gt_tag in forward_family:
        return 0.75
    return 0.35


def _direction_label(gt_end: np.ndarray) -> str:
    """Map GT endpoint to a coarse direction string for prompting."""
    angle = float(np.degrees(np.arctan2(gt_end[1], gt_end[0] + 1e-6)))
    if abs(angle) < 30.0:
        return "直行"
    if angle < -30.0:
        return "右转"
    return "左转"


def _score_primary_failure(scores: Dict[str, float], hard_failures: Sequence[str]) -> str:
    """Pick one main failure label for later pair construction."""
    if hard_failures:
        return hard_failures[0]

    ordered_dims = ["margin", "progress", "comfort", "route", "legality", "semantic"]
    lowest_dim = min(ordered_dims, key=lambda dim: scores[dim])
    if scores[lowest_dim] >= 0.55:
        return "none"
    return {
        "margin": "collision",
        "progress": "progress",
        "comfort": "comfort",
        "route": "route",
        "legality": "legality",
        "semantic": "semantic",
    }[lowest_dim]


def _mode_report_index_from_jsonl(scene_path: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    with open(scene_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scene_id = record.get("scene_id")
            if scene_id:
                index[scene_id] = record
    return index


def load_mode_report_index(mode_report_path: Optional[str]) -> Dict[str, Dict[str, object]]:
    """Load precomputed mode annotations from JSON or JSONL."""
    if not mode_report_path:
        return {}

    path = Path(mode_report_path)
    if not path.exists():
        raise FileNotFoundError(f"Mode report path does not exist: {mode_report_path}")

    if path.suffix == ".jsonl":
        return _mode_report_index_from_jsonl(path)

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    scene_report_path = payload.get("scene_report_path")
    if scene_report_path:
        scene_path = Path(scene_report_path)
        if not scene_path.is_absolute():
            scene_path = (path.parent / scene_report_path).resolve()
        return _mode_report_index_from_jsonl(scene_path)

    scenes = payload.get("scenes")
    if isinstance(scenes, dict):
        return scenes
    if isinstance(scenes, list):
        return {scene["scene_id"]: scene for scene in scenes if "scene_id" in scene}
    return {}


def rule_score(candidate_npz_path: str) -> Dict[str, object]:
    """Legacy rule score used by the existing top1-vs-worst1 pipeline."""
    bundle = _load_candidate_bundle(candidate_npz_path)
    cands = bundle["candidates"]
    gt_future = np.asarray(bundle["gt_future"])
    gt_end = gt_future[-1, :2]
    neighbors = np.asarray(bundle["neighbors"])

    valid_mask = np.abs(neighbors[:, -1, :2]).sum(axis=1) > 1e-6
    valid_neighbors = neighbors[valid_mask]
    neighbor_future = None
    if len(valid_neighbors) > 0:
        neighbor_future = _extrapolate_neighbor_future(valid_neighbors, cands.shape[1])

    scores: List[float] = []
    details: List[Dict[str, float]] = []
    for traj in cands[:, :, :2]:
        fde = float(np.linalg.norm(traj[-1] - gt_end))
        min_len = min(len(traj), len(gt_future))
        ade = float(np.mean(np.linalg.norm(traj[:min_len] - gt_future[:min_len, :2], axis=-1)))

        obs_dist = 99.0
        if neighbor_future is not None:
            min_t = min(cands.shape[1], neighbor_future.shape[1])
            dists = np.linalg.norm(
                traj[:min_t, None, :] - neighbor_future[:, :min_t, :].transpose(1, 0, 2),
                axis=-1,
            )
            obs_dist = float(dists.min())

        collision_penalty = max(0.0, 3.0 - obs_dist) * 10.0
        score = -fde - 0.5 * ade - collision_penalty
        scores.append(score)
        details.append({"fde": fde, "ade": ade, "obs_dist": obs_dist})

    ranking_idx = np.argsort(scores)[::-1]
    ranking = [int(idx + 1) for idx in ranking_idx]
    chosen_idx = int(ranking_idx[0])
    rejected_idx = int(ranking_idx[-1])
    d_chosen = details[chosen_idx]
    d_rejected = details[rejected_idx]
    reason = (
        f"规则打分: chosen=#{chosen_idx+1}(FDE={d_chosen['fde']:.1f}m, "
        f"ADE={d_chosen['ade']:.1f}m, obs={d_chosen['obs_dist']:.1f}m) vs "
        f"rejected=#{rejected_idx+1}(FDE={d_rejected['fde']:.1f}m, "
        f"ADE={d_rejected['ade']:.1f}m, obs={d_rejected['obs_dist']:.1f}m)"
    )
    return {
        "chosen_idx": chosen_idx,
        "rejected_idx": rejected_idx,
        "ranking": ranking,
        "reason": reason,
        "method": "rule",
        "scores": scores,
    }


def compute_lateral_spread(candidate_npz_path: str) -> float:
    """Compute lateral spread in ego frame across several horizon points."""
    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = ensure_candidates_shape(data["candidates"])

    ego = data["ego_agent_past"]
    cos_h, sin_h = ego[-1, 2], ego[-1, 3]
    heading = np.arctan2(sin_h, cos_h)
    rot = np.array(
        [[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]],
        dtype=np.float32,
    )
    xy_rot = np.einsum("ij,ktj->kti", rot, cands[:, :, :2])

    max_lateral = 0.0
    for step in [19, 39, 59, 79]:
        if step < xy_rot.shape[1]:
            y_vals = xy_rot[:, step, 1]
            max_lateral = max(max_lateral, float(y_vals.max() - y_vals.min()))
    return max_lateral


def vlm_score(candidate_npz_path: str, bev_dir: str, client, model_name: str) -> Optional[Dict[str, object]]:
    """Render a BEV image and ask the VLM to rank trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    from flow_planner.dpo.bev_renderer import TRAJECTORY_COLORS

    bundle = _load_candidate_bundle(candidate_npz_path)
    cands = bundle["candidates"]
    ego_future = np.asarray(bundle["gt_future"])
    neighbors = np.asarray(bundle["neighbors"])
    lanes = bundle["lanes"]
    gt_end = ego_future[-1, :2]

    bev_path = os.path.join(bev_dir, Path(candidate_npz_path).stem + ".png")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.3)

    if lanes is not None:
        for lane in np.asarray(lanes):
            valid = np.abs(lane).sum(axis=1) > 1e-6
            if valid.sum() > 1:
                ax.plot(lane[valid, 0], lane[valid, 1], "-", color="#555555", linewidth=0.8, alpha=0.6)

    for neighbor in neighbors:
        if np.abs(neighbor).sum() < 1e-6:
            continue
        current = neighbor[-1]
        nx, ny = current[0], current[1]
        if abs(nx) > 60 or abs(ny) > 60:
            continue
        width = max(float(current[6]), 1.5) if neighbor.shape[1] >= 8 else 2.0
        length = max(float(current[7]), 3.5) if neighbor.shape[1] >= 8 else 4.5
        angle = np.degrees(np.arctan2(current[3], current[2])) if neighbor.shape[1] >= 4 else 0.0
        rect = patches.Rectangle(
            (nx - length / 2, ny - width / 2),
            length,
            width,
            angle=angle,
            rotation_point="center",
            linewidth=1,
            edgecolor="#FF6B6B",
            facecolor="#FF6B6B",
            alpha=0.5,
            zorder=8,
        )
        ax.add_patch(rect)

    ax.plot(ego_future[:, 0], ego_future[:, 1], "--", color="white", linewidth=2.5, alpha=0.8, zorder=9)
    ax.text(
        ego_future[-1, 0] + 1,
        ego_future[-1, 1] + 1,
        "GT",
        color="white",
        fontsize=9,
        fontweight="bold",
        zorder=15,
        bbox=dict(facecolor="black", alpha=0.7, pad=2),
    )

    label_offsets = np.linspace(-3, 3, cands.shape[0])
    for idx, traj in enumerate(cands[:, :, :2]):
        color = TRAJECTORY_COLORS[idx % len(TRAJECTORY_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.9, zorder=10)
        ax.text(
            traj[-1, 0] + 1,
            traj[-1, 1] + label_offsets[idx],
            f"#{idx+1}",
            color=color,
            fontsize=11,
            fontweight="bold",
            zorder=15,
            bbox=dict(facecolor="#1a1a2e", edgecolor="none", alpha=0.7, pad=0.5),
        )

    from matplotlib.patches import Polygon as MplPolygon

    ax.add_patch(
        MplPolygon(
            [(2.7, 0), (-1.8, 1), (-1.8, -1)],
            closed=True,
            facecolor="#448AFF",
            edgecolor="white",
            linewidth=1.5,
            zorder=20,
        )
    )

    plt.tight_layout()
    fig.savefig(bev_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    valid_mask = np.abs(neighbors[:, -1, :2]).sum(axis=1) > 1e-6
    valid_neighbors = neighbors[valid_mask]
    neighbor_future = None
    if len(valid_neighbors) > 0:
        neighbor_future = _extrapolate_neighbor_future(valid_neighbors, cands.shape[1])

    lines: List[str] = []
    for idx, traj in enumerate(cands[:, :, :2]):
        obs_dist = 99.0
        if neighbor_future is not None:
            min_t = min(cands.shape[1], neighbor_future.shape[1])
            dists = np.linalg.norm(
                traj[:min_t, None, :] - neighbor_future[:, :min_t, :].transpose(1, 0, 2),
                axis=-1,
            )
            obs_dist = float(dists.min())
        fde = float(np.linalg.norm(traj[-1] - gt_end))
        lines.append(f"- 轨迹#{idx+1}: 距最近车辆 {obs_dist:.1f}m, 终点距GT终点 {fde:.1f}m")

    prompt = f"""你是自动驾驶安全评审专家。

图中用不同颜色标注了候选轨迹（编号1-{cands.shape[0]}），白色虚线是专家参考轨迹（GT），蓝色三角形是自车，红色方块是周围车辆。

【物理数据】
{chr(10).join(lines)}
参考轨迹终点: ({gt_end[0]:.1f}, {gt_end[1]:.1f}), 大致方向为{_direction_label(gt_end)}。

【请你结合图像和数据综合判断】
1. 哪些轨迹驶出了结构化道路或驶入了不合理的区域？
2. 哪些轨迹与参考轨迹方向一致？
3. 哪些轨迹存在碰撞风险？

请按综合评分从最优到最差排序，严格输出JSON：
{{"ranking": [最优, ..., 最差], "reason": "综合分析..."}}
"""

    img = Image.open(bev_path)
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=model_name, contents=[prompt, img])
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            if "ranking" in result and len(result["ranking"]) >= 2:
                chosen_idx = result["ranking"][0] - 1
                rejected_idx = result["ranking"][-1] - 1
                return {
                    "chosen_idx": chosen_idx,
                    "rejected_idx": rejected_idx,
                    "ranking": result["ranking"],
                    "reason": result.get("reason", ""),
                    "method": "vlm",
                }
        except Exception as exc:
            logger.warning("VLM attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)

    return None


def _structured_candidate_infos(
    bundle: Dict[str, object],
    mode_report: Dict[str, object],
) -> List[Dict[str, object]]:
    """Compute structured per-candidate diagnostics."""
    candidates = np.asarray(bundle["candidates"])
    gt_future = np.asarray(bundle["gt_future"])
    gt_xy = gt_future[:, :2]
    gt_end = gt_xy[-1]
    gt_feature = compute_candidate_features(gt_future)
    gt_maneuver = classify_maneuver(gt_feature)
    lane_points = _extract_valid_lane_points(bundle["lanes"])
    goal_labels = bundle["goal_labels"]

    neighbors = np.asarray(bundle["neighbors"])
    valid_neighbors = neighbors[np.abs(neighbors[:, -1, :2]).sum(axis=1) > 1e-6]
    neighbor_future = None
    if len(valid_neighbors) > 0:
        neighbor_future = _extrapolate_neighbor_future(valid_neighbors, candidates.shape[1])

    gt_progress_ref = max(float(np.linalg.norm(gt_end)), 10.0)
    gt_unit = gt_end / max(np.linalg.norm(gt_end), 1.0)
    mode_candidates = {
        int(item["candidate_idx"]): item for item in mode_report.get("candidates", [])
    }

    infos: List[Dict[str, object]] = []
    for idx, traj in enumerate(candidates):
        traj_xy = traj[:, :2]
        mode_item = mode_candidates.get(idx, {})
        candidate_maneuver = mode_item.get(
            "maneuver_tag",
            classify_maneuver(compute_candidate_features(traj)),
        )

        min_len = min(len(traj_xy), len(gt_xy))
        ade = float(np.mean(np.linalg.norm(traj_xy[:min_len] - gt_xy[:min_len], axis=-1)))
        fde = float(np.linalg.norm(traj_xy[-1] - gt_end))

        obs_dist = 99.0
        if neighbor_future is not None:
            min_t = min(candidates.shape[1], neighbor_future.shape[1])
            dists = np.linalg.norm(
                traj_xy[:min_t, None, :] - neighbor_future[:, :min_t, :].transpose(1, 0, 2),
                axis=-1,
            )
            obs_dist = float(dists.min())

        route_dists = _nearest_reference_distances(traj_xy, gt_xy)
        route_mean_dist = float(route_dists.mean()) if len(route_dists) else 0.0
        route_score = float(np.exp(-route_mean_dist / 4.0))

        lane_dists = _nearest_reference_distances(traj_xy, lane_points)
        lane_mean_dist = float(lane_dists.mean()) if len(lane_dists) else route_mean_dist
        lane_max_dist = float(lane_dists.max()) if len(lane_dists) else route_mean_dist
        legality_score = float(np.exp(-lane_mean_dist / 4.5))

        progress_proj = float(np.dot(traj_xy[-1], gt_unit))
        progress_score = float(np.clip(progress_proj / gt_progress_ref, 0.0, 1.0))
        margin_score = float(np.clip((obs_dist - 1.0) / 4.0, 0.0, 1.0))
        comfort_metrics = _compute_comfort_metrics(traj_xy)
        semantic_score = _maneuver_similarity(candidate_maneuver, gt_maneuver)

        hard_failures: List[str] = []
        if obs_dist < 0.75:
            hard_failures.append("collision")
        if lane_max_dist > 8.0:
            hard_failures.append("off_lane")
        if progress_proj < -1.0:
            hard_failures.append("reverse")

        scores = {
            "margin": margin_score,
            "progress": progress_score,
            "comfort": float(comfort_metrics["comfort_score"]),
            "route": route_score,
            "legality": legality_score,
            "semantic": semantic_score,
        }
        total_score = float(sum(STRUCTURED_SCORE_WEIGHTS[key] * value for key, value in scores.items()))
        if hard_failures:
            total_score -= 8.0 * len(hard_failures)

        primary_failure = _score_primary_failure(scores, hard_failures)
        goal_label = None
        if goal_labels is not None and idx < len(goal_labels):
            goal_label = [float(goal_labels[idx][0]), float(goal_labels[idx][1])]

        infos.append(
            {
                "candidate_idx": idx,
                "scenario_id": bundle["scenario_id"],
                "goal_label": goal_label or mode_item.get("goal_label"),
                "cluster_id": int(mode_item.get("cluster_id", -1)),
                "maneuver_tag": candidate_maneuver,
                "hard_ok": len(hard_failures) == 0,
                "hard_failures": hard_failures,
                "scores": scores,
                "metrics": {
                    "fde": fde,
                    "ade": ade,
                    "obs_dist": obs_dist,
                    "route_mean_dist": route_mean_dist,
                    "lane_mean_dist": lane_mean_dist,
                    "lane_max_dist": lane_max_dist,
                    "progress_proj": progress_proj,
                    "max_acc": float(comfort_metrics["max_acc"]),
                    "max_jerk": float(comfort_metrics["max_jerk"]),
                },
                "primary_failure": primary_failure,
                "total_score": total_score,
            }
        )
    return infos


def _apply_vlm_bonus(traj_infos: List[Dict[str, object]], vlm_result: Optional[Dict[str, object]]) -> None:
    """Use VLM ranking as a small semantic tie-break bonus."""
    if not vlm_result or "ranking" not in vlm_result:
        return

    ordered = list(vlm_result["ranking"])
    if len(ordered) <= 1:
        return

    for position, label in enumerate(ordered):
        idx = int(label) - 1
        if idx < 0 or idx >= len(traj_infos):
            continue
        normalized_rank = (len(ordered) - 1 - position) / max(len(ordered) - 1, 1)
        semantic_bonus = 0.30 * normalized_rank
        traj_infos[idx]["scores"]["semantic"] = float(
            traj_infos[idx]["scores"]["semantic"] + semantic_bonus
        )
        traj_infos[idx]["vlm_rank"] = position + 1
        traj_infos[idx]["vlm_reason"] = vlm_result.get("reason", "")

        total_score = float(
            sum(
                STRUCTURED_SCORE_WEIGHTS[key] * value
                for key, value in traj_infos[idx]["scores"].items()
            )
        )
        if not traj_infos[idx]["hard_ok"]:
            total_score -= 8.0 * len(traj_infos[idx]["hard_failures"])
        traj_infos[idx]["total_score"] = total_score


def _rank_structured_candidates(traj_infos: List[Dict[str, object]]) -> List[int]:
    """Rank candidates by total score, then by margin and progress."""
    return sorted(
        range(len(traj_infos)),
        key=lambda idx: (
            traj_infos[idx]["total_score"],
            traj_infos[idx]["scores"]["margin"],
            traj_infos[idx]["scores"]["progress"],
        ),
        reverse=True,
    )


def _structured_reason(
    chosen: Dict[str, object],
    rejected: Dict[str, object],
    method: str,
) -> str:
    chosen_scores = chosen["scores"]
    rejected_scores = rejected["scores"]
    return (
        f"结构化评分({method}): chosen=#{chosen['candidate_idx']+1}"
        f"(total={chosen['total_score']:.2f}, fail={chosen['primary_failure']}, "
        f"margin={chosen_scores['margin']:.2f}, progress={chosen_scores['progress']:.2f}, "
        f"comfort={chosen_scores['comfort']:.2f}) vs "
        f"rejected=#{rejected['candidate_idx']+1}"
        f"(total={rejected['total_score']:.2f}, fail={rejected['primary_failure']}, "
        f"margin={rejected_scores['margin']:.2f}, progress={rejected_scores['progress']:.2f}, "
        f"comfort={rejected_scores['comfort']:.2f})"
    )


def structured_score(
    candidate_npz_path: str,
    mode_index: Dict[str, Dict[str, object]],
    bev_dir: str,
    client,
    model_name: str,
    use_vlm: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Produce candidate-level structured diagnostics and one chosen/rejected pair."""
    bundle = _load_candidate_bundle(candidate_npz_path)
    scenario_id = str(bundle["scenario_id"])
    mode_report = mode_index.get(scenario_id)
    mode_report_source = "precomputed"
    if mode_report is None:
        mode_report_source = "computed"
        mode_report = build_scene_mode_report(
            candidates=np.asarray(bundle["candidates"]),
            goal_labels=bundle["goal_labels"],
            scene_id=scenario_id,
        )

    traj_infos = _structured_candidate_infos(bundle, mode_report)
    vlm_result = None
    method = "structured_rule"
    if use_vlm:
        vlm_result = vlm_score(candidate_npz_path, bev_dir, client, model_name)
        if vlm_result is None:
            method = "structured_rule_fallback"
        else:
            _apply_vlm_bonus(traj_infos, vlm_result)
            method = "structured_vlm"

    ranking_idx = _rank_structured_candidates(traj_infos)
    for rank, idx in enumerate(ranking_idx, start=1):
        traj_infos[idx]["rank"] = rank

    chosen_idx = int(ranking_idx[0])
    rejected_idx = int(ranking_idx[-1])
    chosen = traj_infos[chosen_idx]
    rejected = traj_infos[rejected_idx]
    ranking_1based = [int(idx + 1) for idx in ranking_idx]

    result = {
        "chosen_idx": chosen_idx,
        "rejected_idx": rejected_idx,
        "ranking": ranking_1based,
        "reason": _structured_reason(chosen, rejected, method),
        "method": method,
        "scores": [float(info["total_score"]) for info in traj_infos],
        "traj_infos": traj_infos,
    }
    if vlm_result is not None:
        result["vlm_reason"] = vlm_result.get("reason", "")

    scene_payload = {
        "scenario_id": scenario_id,
        "source_npz": os.path.abspath(candidate_npz_path),
        "mode_report_source": mode_report_source,
        "ordered_candidate_indices": [int(idx) for idx in ranking_idx],
        "ranking_1based": ranking_1based,
        "chosen_idx": chosen_idx,
        "rejected_idx": rejected_idx,
        "method": method,
        "reason": result["reason"],
        "mode_summary": {
            "cluster_count": mode_report.get("cluster_count"),
            "cluster_entropy": mode_report.get("cluster_entropy"),
            "unique_goal_count": mode_report.get("unique_goal_count"),
            "goal_maneuver_consistency": mode_report.get("goal_maneuver_consistency"),
        },
        "candidates": traj_infos,
    }
    if vlm_result is not None:
        scene_payload["vlm"] = {
            "ranking": vlm_result.get("ranking"),
            "reason": vlm_result.get("reason", ""),
        }
    return result, scene_payload


def _save_scene_payload(scored_dir: str, scenario_id: str, payload: Dict[str, object]) -> None:
    os.makedirs(scored_dir, exist_ok=True)
    out_path = os.path.join(scored_dir, f"{scenario_id}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Rule+VLM preference scoring")
    parser.add_argument("--candidates_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dpo_data/preferences_final")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (required for VLM tier)")
    parser.add_argument("--model_name", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument(
        "--spread_threshold",
        type=float,
        default=5.0,
        help="Lateral spread threshold for VLM scoring (meters)",
    )
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM scoring entirely, use only rules")
    parser.add_argument(
        "--use_structured_scores",
        action="store_true",
        help="Use candidate-level structured scores instead of the legacy top1-vs-worst1 rule ranker",
    )
    parser.add_argument(
        "--emit_traj_info",
        action="store_true",
        help="Write candidate-level scene JSON files for later build_multi_pairs.py",
    )
    parser.add_argument(
        "--mode_report_json",
        type=str,
        default=None,
        help="Optional mode report JSON/JSONL generated by analyze_candidate_modes.py",
    )
    parser.add_argument(
        "--scored_dir",
        type=str,
        default=None,
        help="Optional output directory for per-scene structured JSON files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, "bev_images")
    os.makedirs(bev_dir, exist_ok=True)
    scored_dir = args.scored_dir or os.path.join(args.output_dir, "scored_dir")

    npz_files = sorted(Path(args.candidates_dir).glob("*_candidates.npz"))
    if args.max_scenarios:
        npz_files = npz_files[: args.max_scenarios]
    logger.info("Found %d candidate files", len(npz_files))

    client = None
    if args.api_key and not args.skip_vlm:
        from google import genai

        client = genai.Client(api_key=args.api_key)

    mode_index = load_mode_report_index(args.mode_report_json)
    if mode_index:
        logger.info("Loaded mode annotations for %d scenes", len(mode_index))

    logger.info("Phase 1: Scanning lateral spread...")
    spreads: Dict[str, float] = {}
    for npz_path in npz_files:
        try:
            spreads[str(npz_path)] = compute_lateral_spread(str(npz_path))
        except Exception as exc:
            logger.warning("Spread computation failed for %s: %s", npz_path, exc)
            spreads[str(npz_path)] = 0.0

    high_spread = {path for path, spread in spreads.items() if spread >= args.spread_threshold}
    logger.info("High-spread scenes (>= %.2fm): %d", args.spread_threshold, len(high_spread))

    logger.info("Phase 2: Scoring...")
    preferences: List[Dict[str, object]] = []
    stats = {"rule": 0, "vlm": 0, "vlm_fail": 0, "structured": 0}
    start = time.time()

    for idx, npz_path in enumerate(npz_files, start=1):
        npz_str = str(npz_path)
        scenario_id = npz_path.stem.replace("_candidates", "")
        use_vlm = npz_str in high_spread and client is not None

        if args.use_structured_scores or args.emit_traj_info:
            result, scene_payload = structured_score(
                candidate_npz_path=npz_str,
                mode_index=mode_index,
                bev_dir=bev_dir,
                client=client,
                model_name=args.model_name,
                use_vlm=use_vlm,
            )
            if args.emit_traj_info:
                _save_scene_payload(scored_dir, scenario_id, scene_payload)
            if result["method"] == "structured_vlm":
                stats["vlm"] += 1
                stats["structured"] += 1
            elif result["method"] == "structured_rule_fallback":
                stats["vlm_fail"] += 1
                stats["structured"] += 1
            else:
                stats["structured"] += 1
        else:
            if use_vlm:
                result = vlm_score(npz_str, bev_dir, client, args.model_name)
                if result is None:
                    result = rule_score(npz_str)
                    result["method"] = "rule_fallback"
                    stats["vlm_fail"] += 1
                else:
                    stats["vlm"] += 1
            else:
                result = rule_score(npz_str)
                stats["rule"] += 1

        data = np.load(npz_str, allow_pickle=True)
        candidates = ensure_candidates_shape(data["candidates"])
        preferences.append(
            {
                "scenario_id": scenario_id,
                "chosen": candidates[result["chosen_idx"]],
                "rejected": candidates[result["rejected_idx"]],
                "chosen_idx": result["chosen_idx"],
                "rejected_idx": result["rejected_idx"],
                "ranking": result["ranking"],
                "reason": result["reason"],
                "method": result["method"],
                "lateral_spread": spreads.get(npz_str, 0.0),
            }
        )

        if idx % 500 == 0:
            elapsed = time.time() - start
            rate = idx / elapsed
            logger.info(
                "[%d/%d] rule=%d structured=%d vlm=%d vlm_fail=%d rate=%.1f/s ETA=%.1fmin",
                idx,
                len(npz_files),
                stats["rule"],
                stats["structured"],
                stats["vlm"],
                stats["vlm_fail"],
                rate,
                ((len(npz_files) - idx) / max(rate, 1e-6)) / 60.0,
            )

    elapsed = time.time() - start
    logger.info("Done: %d pairs in %.1fmin", len(preferences), elapsed / 60.0)
    logger.info(
        "Stats: rule=%d structured=%d vlm=%d vlm_fail=%d",
        stats["rule"],
        stats["structured"],
        stats["vlm"],
        stats["vlm_fail"],
    )

    output_path = os.path.join(args.output_dir, "preferences.npz")
    np.savez_compressed(
        output_path,
        chosen=np.array([pref["chosen"] for pref in preferences]),
        rejected=np.array([pref["rejected"] for pref in preferences]),
        scenario_ids=[pref["scenario_id"] for pref in preferences],
        rankings=[pref["ranking"] for pref in preferences],
        reasons=[pref["reason"] for pref in preferences],
    )
    logger.info("Saved %d pairs to %s", len(preferences), output_path)

    json_path = os.path.join(args.output_dir, "preference_details.json")
    details = [
        {
            "scenario_id": pref["scenario_id"],
            "chosen_idx": int(pref["chosen_idx"]),
            "rejected_idx": int(pref["rejected_idx"]),
            "ranking": pref["ranking"],
            "reason": pref["reason"],
            "method": pref["method"],
            "lateral_spread": round(float(pref["lateral_spread"]), 2),
        }
        for pref in preferences
    ]
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(details, fp, indent=2, ensure_ascii=False)
    logger.info("Saved details to %s", json_path)


if __name__ == "__main__":
    main()

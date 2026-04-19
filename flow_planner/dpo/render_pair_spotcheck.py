#!/usr/bin/env python3
"""
Render chosen/rejected BEV spot-checks for structured DPO preference pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from flow_planner.dpo.bev_renderer import BEVRenderer


def _extrapolate_neighbor_future(
    neighbor_past: np.ndarray,
    future_steps: int,
    dt: float = 0.1,
) -> np.ndarray:
    """Match the scoring-time constant-velocity future rollout for neighbors."""
    if neighbor_past.size == 0:
        return np.zeros((0, future_steps, 2), dtype=np.float32)

    future = np.zeros((neighbor_past.shape[0], future_steps, 2), dtype=np.float32)
    for idx, last in enumerate(neighbor_past[:, -1]):
        x, y = float(last[0]), float(last[1])
        vx = float(last[4]) if last.shape[0] > 5 else 0.0
        vy = float(last[5]) if last.shape[0] > 5 else 0.0
        for step in range(future_steps):
            future[idx, step, 0] = x + vx * dt * (step + 1)
            future[idx, step, 1] = y + vy * dt * (step + 1)
    return future


def _draw_neighbor_future(ax, neighbors: np.ndarray, future_xy: np.ndarray, view_range: float) -> None:
    """Overlay the scoring-time neighbor future so ego futures are not compared only to current boxes."""
    import matplotlib.patches as patches

    if neighbors.size == 0 or future_xy.size == 0:
        return

    future_label_used = False
    for idx in range(neighbors.shape[0]):
        traj = neighbors[idx]
        if np.abs(traj).sum() < 1e-6:
            continue

        curr = traj[-1]
        future = future_xy[idx]
        valid = (
            np.isfinite(future).all(axis=1)
            & (np.abs(future[:, 0]) <= view_range)
            & (np.abs(future[:, 1]) <= view_range)
        )
        future = future[valid]
        if len(future) == 0:
            continue

        line_label = "neighbor_future" if not future_label_used else None
        ax.plot(
            future[:, 0],
            future[:, 1],
            ":",
            color="#FFB3B3",
            linewidth=1.0,
            alpha=0.65,
            zorder=6,
            label=line_label,
        )
        future_label_used = True

        width = max(float(curr[6]), 1.5) if curr.shape[0] >= 8 else 2.0
        length = max(float(curr[7]), 3.5) if curr.shape[0] >= 8 else 4.5

        for step_idx in range(9, len(future), 10):
            fx, fy = float(future[step_idx, 0]), float(future[step_idx, 1])
            if curr.shape[0] >= 6:
                vx = float(curr[4])
                vy = float(curr[5])
                angle = np.degrees(np.arctan2(vy, vx)) if abs(vx) + abs(vy) > 1e-3 else 0.0
            elif curr.shape[0] >= 4:
                angle = np.degrees(np.arctan2(float(curr[3]), float(curr[2])))
            else:
                angle = 0.0

            rect = patches.Rectangle(
                (fx - length / 2, fy - width / 2),
                length,
                width,
                angle=angle,
                rotation_point="center",
                linewidth=0.8,
                edgecolor="#FFB3B3",
                facecolor="none",
                alpha=0.45,
                zorder=7,
            )
            ax.add_patch(rect)


def _load_records(meta_path: str) -> List[Dict[str, object]]:
    with open(meta_path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _select_examples(records: Sequence[Dict[str, object]], per_label: int) -> List[Dict[str, object]]:
    items = sorted(records, key=lambda record: float(record["score_gap"]))
    if not items:
        return []

    low_count = min(len(items), max(1, per_label // 2))
    selected = items[:low_count]

    if len(selected) >= per_label:
        return selected[:per_label]

    scores = np.asarray([float(record["score_gap"]) for record in items], dtype=np.float32)
    median = float(np.median(scores))
    mid_sorted = sorted(items, key=lambda record: abs(float(record["score_gap"]) - median))
    seen = {
        (str(record["scenario_id"]), int(record["chosen_idx"]), int(record["rejected_idx"]))
        for record in selected
    }
    for record in mid_sorted:
        key = (str(record["scenario_id"]), int(record["chosen_idx"]), int(record["rejected_idx"]))
        if key in seen:
            continue
        selected.append(record)
        seen.add(key)
        if len(selected) >= per_label:
            break
    return selected


def _candidate_npz_path(candidates_dir: str, scenario_id: str) -> Path:
    return Path(candidates_dir) / f"{scenario_id}_candidates.npz"


def _format_drop_summary(record: Dict[str, object]) -> str:
    score_drops = record["score_drops"]
    parts = [
        f"m={float(score_drops['margin']):+.2f}",
        f"p={float(score_drops['progress']):+.2f}",
        f"c={float(score_drops['comfort']):+.2f}",
        f"r={float(score_drops['route']):+.2f}",
        f"l={float(score_drops['legality']):+.2f}",
        f"s={float(score_drops['semantic']):+.2f}",
    ]
    return " ".join(parts)


def _primary_failure(record: Dict[str, object]) -> str:
    return str(record.get("failure_type", "unknown"))


def _render_pair(
    renderer: BEVRenderer,
    record: Dict[str, object],
    candidate_npz_path: Path,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = np.load(candidate_npz_path, allow_pickle=True)
    candidates = payload["candidates"]
    chosen_idx = int(record["chosen_idx"])
    rejected_idx = int(record["rejected_idx"])
    chosen = candidates[chosen_idx, :, :2]
    rejected = candidates[rejected_idx, :, :2]
    gt_future = payload["ego_agent_future"][:, :2]
    neighbors = payload["neighbor_agents_past"]
    lanes = payload["lanes"]
    neighbor_future = _extrapolate_neighbor_future(neighbors, future_steps=gt_future.shape[0])

    fig_w = renderer.image_size[0] / renderer.dpi
    fig_h = renderer.image_size[1] / renderer.dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=renderer.dpi)
    ax.set_facecolor(renderer.bg_color)
    fig.patch.set_facecolor(renderer.bg_color)

    vr = renderer.view_range
    ax.set_xlim(-vr, vr)
    ax.set_ylim(-vr, vr)
    ax.set_aspect("equal")
    ax.grid(True, color=renderer.grid_color, linewidth=0.5, alpha=0.3)
    ax.tick_params(colors="#444444", labelsize=6)

    renderer._draw_lanes(ax, lanes)
    renderer._draw_neighbors(ax, neighbors)
    _draw_neighbor_future(ax, neighbors, neighbor_future, view_range=renderer.view_range)

    # GT/reference in white dashed so we can see when route reward is just GT-matching.
    ax.plot(
        gt_future[:, 0],
        gt_future[:, 1],
        "--",
        color="white",
        linewidth=2.0,
        alpha=0.85,
        zorder=9,
        label="GT",
    )

    ax.plot(
        chosen[:, 0],
        chosen[:, 1],
        color="#00E676",
        linewidth=3.5,
        alpha=1.0,
        zorder=11,
        label="chosen",
    )
    ax.plot(
        rejected[:, 0],
        rejected[:, 1],
        color="#FF1744",
        linewidth=3.5,
        alpha=1.0,
        zorder=11,
        label="rejected",
    )

    ax.plot(chosen[::10, 0], chosen[::10, 1], "o", color="#00E676", markersize=4, alpha=0.8, zorder=12)
    ax.plot(rejected[::10, 0], rejected[::10, 1], "o", color="#FF1744", markersize=4, alpha=0.8, zorder=12)
    ax.plot(gt_future[::10, 0], gt_future[::10, 1], "o", color="white", markersize=3, alpha=0.7, zorder=10)
    ax.text(chosen[-1, 0] + 1.0, chosen[-1, 1] + 1.5, "chosen", color="#00E676", fontsize=10, fontweight="bold")
    ax.text(rejected[-1, 0] + 1.0, rejected[-1, 1] - 1.5, "rejected", color="#FF1744", fontsize=10, fontweight="bold")

    renderer._draw_ego(ax, (0.0, 0.0), 0.0)

    title = (
        f"label={record['dim_label']} | gap={float(record['score_gap']):.2f}\n"
        f"primary_failure={_primary_failure(record)} | {record['scenario_id']}\n"
        f"{_format_drop_summary(record)}"
    )
    ax.set_title(title, color="white", fontsize=9, pad=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.3, facecolor="black", edgecolor="gray", labelcolor="white")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=renderer.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_index(index_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    lines = [
        "# Pair Spot Check",
        "",
        "| image | dim_label | score_gap | primary_failure | scenario_id | drops |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['dim_label']} | {float(row['score_gap']):.4f} | "
            f"{row['primary_failure']} | {row['scenario_id']} | {row['drop_summary']} |"
        )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render spot-check BEV pairs for structured preferences")
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--candidates_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--labels", nargs="+", default=["route", "collision"])
    parser.add_argument("--per_label", type=int, default=10)
    parser.add_argument("--view_range", type=float, default=60.0)
    args = parser.parse_args()

    records = _load_records(args.meta_path)
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for label in args.labels:
        by_label[label] = [record for record in records if str(record["dim_label"]) == label]

    renderer = BEVRenderer(image_size=(900, 900), view_range=args.view_range)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, object]] = []
    for label in args.labels:
        selected = _select_examples(by_label.get(label, []), args.per_label)
        for idx, record in enumerate(selected, start=1):
            scenario_id = str(record["scenario_id"])
            candidate_npz_path = _candidate_npz_path(args.candidates_dir, scenario_id)
            image_name = f"{label}_{idx:02d}_{scenario_id}.png"
            save_path = output_dir / image_name
            _render_pair(renderer, record, candidate_npz_path, save_path)
            index_rows.append(
                {
                    "image": image_name,
                    "dim_label": label,
                    "score_gap": float(record["score_gap"]),
                    "primary_failure": _primary_failure(record),
                    "scenario_id": scenario_id,
                    "drop_summary": _format_drop_summary(record),
                }
            )

    _write_index(output_dir / "index.md", index_rows)
    print(f"Rendered {len(index_rows)} images to {output_dir}")


if __name__ == "__main__":
    main()

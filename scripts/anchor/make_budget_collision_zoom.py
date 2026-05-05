#!/usr/bin/env python3
"""Create a readable standalone collision-zoom figure for rescue cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GREEN = "#22c55e"
RED = "#ef4444"
GRAY = "#cbd5e1"
BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def _load_case(case_dir: Path):
    meta = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    trajs = np.load(case_dir / "trajectories.npz", allow_pickle=True)
    return meta, trajs


def _collision_window(meta: dict, traj_top1x9: np.ndarray) -> tuple[slice, int, int]:
    collision = meta["selected_top1x9"]["collision"]
    if collision is None:
        return slice(55, min(80, traj_top1x9.shape[0])), -1, -1
    t_hit = int(collision["t"])
    start = max(0, t_hit - 18)
    stop = min(traj_top1x9.shape[0], t_hit + 5)
    return slice(start, stop), int(collision["neighbor"]), t_hit


def render_case(case_dir: Path) -> Path:
    meta, trajs = _load_case(case_dir)

    traj_522 = np.asarray(trajs["traj_5_2_2"], dtype=np.float32)
    traj_top1 = np.asarray(trajs["traj_top1x9"], dtype=np.float32)
    neighbors = np.asarray(trajs["neighbor_future"], dtype=np.float32)

    window, neighbor_idx, t_hit = _collision_window(meta, traj_top1)
    seg_522 = traj_522[window, :2]
    seg_top1 = traj_top1[window, :2]
    seg_neighbor = (
        neighbors[neighbor_idx, window, :2]
        if neighbor_idx >= 0
        else np.zeros((len(seg_top1), 2), dtype=np.float32)
    )

    fig = plt.figure(figsize=(8.8, 7.4), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.7, 1.5], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cap = fig.add_subplot(gs[1, 0])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    cap.set_facecolor(BG)

    ax.plot(seg_522[:, 0], seg_522[:, 1], color=GREEN, linewidth=4.0, label="5-2-2 selected")
    ax.plot(seg_top1[:, 0], seg_top1[:, 1], color=RED, linewidth=4.0, label="top1x9 selected")
    if neighbor_idx >= 0:
        ax.plot(seg_neighbor[:, 0], seg_neighbor[:, 1], color=GRAY, linewidth=3.2, label="colliding neighbor")

    ax.scatter(seg_522[0, 0], seg_522[0, 1], s=110, color=GREEN, marker="o", edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter(seg_top1[0, 0], seg_top1[0, 1], s=110, color=RED, marker="o", edgecolors="white", linewidths=0.8, zorder=5)
    if neighbor_idx >= 0:
        ax.scatter(seg_neighbor[0, 0], seg_neighbor[0, 1], s=90, color=GRAY, marker="o", edgecolors="white", linewidths=0.8, zorder=5)

    if t_hit >= 0:
        hit_local = t_hit - window.start
        if 0 <= hit_local < len(seg_top1):
            ax.scatter(seg_top1[hit_local, 0], seg_top1[hit_local, 1], s=220, color=RED, marker="X", edgecolors="white", linewidths=0.9, zorder=6)
            if neighbor_idx >= 0 and 0 <= hit_local < len(seg_neighbor):
                ax.scatter(seg_neighbor[hit_local, 0], seg_neighbor[hit_local, 1], s=130, color=GRAY, marker="D", edgecolors="white", linewidths=0.8, zorder=6)
            ax.annotate(
                f"collision at t={t_hit}",
                xy=(seg_top1[hit_local, 0], seg_top1[hit_local, 1]),
                xytext=(14, 16),
                textcoords="offset points",
                color=TEXT,
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2),
            )

    all_xy = np.concatenate([seg_522, seg_top1, seg_neighbor], axis=0)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max(1.0, 0.18 * float(x_max - x_min))
    y_pad = max(0.8, 0.28 * float(y_max - y_min))
    ax.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
    ax.set_ylim(float(y_min - y_pad), float(y_max + y_pad))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, alpha=0.45, linewidth=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    ax.set_title(
        f"{meta['scene_name']} | local collision zoom",
        color=TEXT,
        fontsize=15,
        pad=12,
    )

    cap.axis("off")
    m_522 = meta["selected_5_2_2"]["metrics"]
    m_top1 = meta["selected_top1x9"]["metrics"]
    lines = [
        f"Anchor predictor top-3 probs: rank0={meta['anchor_topk_scores'][0]:.3f}, rank1={meta['anchor_topk_scores'][1]:.3f}, rank2={meta['anchor_topk_scores'][2]:.3f}",
        f"5-2-2 selects rank {meta['selected_5_2_2']['anchor_rank']} sample {meta['selected_5_2_2']['sample_i']} | logit {meta['selected_5_2_2']['selector_logit']:.3f} | safe | progress {m_522['progress_score']:.3f} | route {m_522['route_score']:.3f}",
        f"top1x9 selects rank {meta['selected_top1x9']['anchor_rank']} sample {meta['selected_top1x9']['sample_i']} | logit {meta['selected_top1x9']['selector_logit']:.3f} | collided | progress {m_top1['progress_score']:.3f} | route {m_top1['route_score']:.3f}",
    ]
    if meta["selected_top1x9"]["collision"] is not None:
        lines.append(
            f"Collision neighbor={meta['selected_top1x9']['collision']['neighbor']} | distance={meta['selected_top1x9']['collision']['distance']:.3f} m"
        )
    cap.text(
        0.02,
        0.92,
        "\n".join(lines),
        color=TEXT,
        fontsize=11,
        ha="left",
        va="top",
        family="monospace",
    )

    out_path = case_dir / "collision_zoom_readable.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+")
    args = parser.parse_args()
    for raw in args.case_dirs:
        print(render_case(Path(raw)))


if __name__ == "__main__":
    main()

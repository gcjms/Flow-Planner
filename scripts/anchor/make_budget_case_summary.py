#!/usr/bin/env python3
"""Create a more legible summary figure for budget-allocation rescue cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
SLATE = "#475569"
BLUE = "#38bdf8"
BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def _load_case(case_dir: Path):
    meta = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    trajs = np.load(case_dir / "trajectories.npz", allow_pickle=True)
    return meta, trajs


def _candidate_arrays(meta: dict):
    rows = meta["all_candidates"]
    anchor_rank = np.asarray([int(row["anchor_rank"]) for row in rows], dtype=np.int64)
    sample_i = np.asarray([int(row["sample_i"]) for row in rows], dtype=np.int64)
    logits = np.asarray([float(row["selector_logit"]) for row in rows], dtype=np.float32)
    x = np.arange(len(rows), dtype=np.int64)
    labels = [f"a{a}s{s}" for a, s in zip(anchor_rank, sample_i)]
    return x, anchor_rank, sample_i, logits, labels


def _eligible_masks(anchor_rank: np.ndarray, sample_i: np.ndarray):
    mask_522 = (
        ((anchor_rank == 0) & (sample_i <= 4))
        | ((anchor_rank == 1) & (sample_i <= 1))
        | ((anchor_rank == 2) & (sample_i <= 1))
    )
    mask_top1x9 = (anchor_rank == 0) & (sample_i <= 8)
    return mask_522, mask_top1x9


def _collision_window(meta: dict, traj_top1x9: np.ndarray):
    collision = meta["selected_top1x9"]["collision"]
    if collision is None:
        return slice(60, 80), np.empty((0, 2), dtype=np.float32), -1
    t_hit = int(collision["t"])
    start = max(0, t_hit - 18)
    stop = min(traj_top1x9.shape[0], t_hit + 4)
    neighbor_idx = int(collision["neighbor"])
    return slice(start, stop), neighbor_idx, t_hit


def _plot_score_panel(ax, meta: dict):
    scores = np.asarray(meta["anchor_topk_scores"], dtype=np.float32)
    idx = np.arange(len(scores))
    bars = ax.bar(idx, scores, color=[AMBER, "#fbbf24", "#fde68a"], width=0.65)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.006,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT,
        )
    ax.set_title("Anchor Predictor Top-k", color=TEXT, fontsize=13, pad=10)
    ax.set_xticks(idx, [f"rank {i}" for i in idx], color=TEXT)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_ylabel("prob", color=MUTED)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _plot_selector_panel(ax, meta: dict):
    x, anchor_rank, sample_i, logits, labels = _candidate_arrays(meta)
    mask_522, mask_top1x9 = _eligible_masks(anchor_rank, sample_i)

    colors = np.where(anchor_rank == 0, "#f59e0b", np.where(anchor_rank == 1, "#60a5fa", "#a78bfa"))
    ax.scatter(x, logits, c=colors, s=55, alpha=0.9, edgecolors="none")

    ax.scatter(x[mask_top1x9], logits[mask_top1x9], s=120, facecolors="none", edgecolors=RED, linewidths=1.2)
    ax.scatter(x[mask_522], logits[mask_522], s=120, facecolors="none", edgecolors=GREEN, linewidths=1.2)

    sel_522 = meta["selected_5_2_2"]
    sel_top1 = meta["selected_top1x9"]
    idx_522 = next(i for i, row in enumerate(meta["all_candidates"]) if int(row["anchor_rank"]) == int(sel_522["anchor_rank"]) and int(row["sample_i"]) == int(sel_522["sample_i"]))
    idx_top1 = next(i for i, row in enumerate(meta["all_candidates"]) if int(row["anchor_rank"]) == int(sel_top1["anchor_rank"]) and int(row["sample_i"]) == int(sel_top1["sample_i"]))

    ax.scatter([idx_522], [logits[idx_522]], s=180, color=GREEN, marker="*", edgecolors="white", linewidths=0.8, zorder=5)
    ax.scatter([idx_top1], [logits[idx_top1]], s=180, color=RED, marker="X", edgecolors="white", linewidths=0.8, zorder=5)

    ax.axhline(0.0, color=GRID, linewidth=1.0, alpha=0.8)
    ax.set_title("Selector Logits by Candidate", color=TEXT, fontsize=13, pad=10)
    ax.set_xticks(x, labels, rotation=50, ha="right", color=MUTED, fontsize=8)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_ylabel("logit", color=MUTED)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.text(0.01, 0.98, "green ring: eligible in 5-2-2\nred ring: eligible in top1x9", transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=9)


def _plot_zoom_panel(ax, meta: dict, trajs):
    traj_522 = np.asarray(trajs["traj_5_2_2"], dtype=np.float32)
    traj_top1 = np.asarray(trajs["traj_top1x9"], dtype=np.float32)
    anchor_522 = np.asarray(trajs["anchor_5_2_2"], dtype=np.float32)
    anchor_top1 = np.asarray(trajs["anchor_top1x9"], dtype=np.float32)
    neighbors = np.asarray(trajs["neighbor_future"], dtype=np.float32)

    window, neighbor_idx, t_hit = _collision_window(meta, traj_top1)
    if isinstance(neighbor_idx, np.ndarray):
        neighbor_traj = neighbor_idx
    else:
        neighbor_traj = neighbors[neighbor_idx, window, :2]

    ax.plot(anchor_522[window, 0], anchor_522[window, 1], color=GREEN, linestyle=":", linewidth=2.0, alpha=0.9, label="5-2-2 anchor")
    ax.plot(anchor_top1[window, 0], anchor_top1[window, 1], color=RED, linestyle=":", linewidth=2.0, alpha=0.9, label="top1x9 anchor")
    ax.plot(traj_522[window, 0], traj_522[window, 1], color=GREEN, linewidth=3.0, label="5-2-2 selected")
    ax.plot(traj_top1[window, 0], traj_top1[window, 1], color=RED, linewidth=3.0, label="top1x9 selected")

    if neighbor_traj.size:
        ax.plot(neighbor_traj[:, 0], neighbor_traj[:, 1], color=MUTED, linewidth=2.2, alpha=0.95, label="colliding neighbor")
    if t_hit >= 0:
        hit_local = t_hit - window.start
        if 0 <= hit_local < len(traj_top1[window]):
            ax.scatter([traj_top1[window][hit_local, 0]], [traj_top1[window][hit_local, 1]], s=170, color=RED, marker="X", zorder=6)
            if neighbor_traj.size and 0 <= hit_local < len(neighbor_traj):
                ax.scatter([neighbor_traj[hit_local, 0]], [neighbor_traj[hit_local, 1]], s=120, color=MUTED, marker="o", zorder=6)

    ax.set_title("Late-Stage Local Zoom", color=TEXT, fontsize=13, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    m_522 = meta["selected_5_2_2"]["metrics"]
    m_top1 = meta["selected_top1x9"]["metrics"]
    summary = (
        f"5-2-2: safe, rank {meta['selected_5_2_2']['anchor_rank']}, "
        f"logit {meta['selected_5_2_2']['selector_logit']:.3f}, "
        f"prog {m_522['progress_score']:.3f}, route {m_522['route_score']:.3f}\n"
        f"top1x9: collided, rank {meta['selected_top1x9']['anchor_rank']}, "
        f"logit {meta['selected_top1x9']['selector_logit']:.3f}, "
        f"prog {m_top1['progress_score']:.3f}, route {m_top1['route_score']:.3f}"
    )
    if t_hit >= 0:
        summary += f"\ncollision at t={t_hit}, dist={meta['selected_top1x9']['collision']['distance']:.3f} m"
    ax.text(0.01, 0.02, summary, transform=ax.transAxes, ha="left", va="bottom", color=TEXT, fontsize=9)


def render_case(case_dir: Path):
    meta, trajs = _load_case(case_dir)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=180)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{meta['scene_name']}  |  why 5-2-2 beats top1x9",
        color=TEXT,
        fontsize=15,
        y=0.98,
    )

    _plot_score_panel(axes[0], meta)
    _plot_selector_panel(axes[1], meta)
    _plot_zoom_panel(axes[2], meta, trajs)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = case_dir / "summary_explained.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+", help="Case directories containing metadata.json and trajectories.npz")
    args = parser.parse_args()

    for raw in args.case_dirs:
        out = render_case(Path(raw))
        print(out)


if __name__ == "__main__":
    main()

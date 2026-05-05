#!/usr/bin/env python3
"""Clearer rescue-case summary figure with per-budget 9-candidate panels."""

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
BLUE = "#60a5fa"
PURPLE = "#a78bfa"
BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def _load_case(case_dir: Path):
    meta = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    trajs = np.load(case_dir / "trajectories.npz", allow_pickle=True)
    return meta, trajs


def _candidate_rows(meta: dict):
    rows = list(meta["all_candidates"])
    for idx, row in enumerate(rows):
        row["_idx"] = idx
    return rows


def _panel_rows(meta: dict, mode: str):
    rows = _candidate_rows(meta)
    if mode == "top1x9":
        keep = [r for r in rows if int(r["anchor_rank"]) == 0 and int(r["sample_i"]) <= 8]
    elif mode == "5-2-2":
        keep = [
            r
            for r in rows
            if (
                (int(r["anchor_rank"]) == 0 and int(r["sample_i"]) <= 4)
                or (int(r["anchor_rank"]) == 1 and int(r["sample_i"]) <= 1)
                or (int(r["anchor_rank"]) == 2 and int(r["sample_i"]) <= 1)
            )
        ]
    else:
        raise ValueError(mode)
    return keep


def _anchor_color(rank: int) -> str:
    return {0: AMBER, 1: BLUE, 2: PURPLE}.get(rank, SLATE)


def _plot_anchor_scores(ax, meta: dict):
    scores = np.asarray(meta["anchor_topk_scores"], dtype=np.float32)
    idx = np.arange(len(scores))
    colors = [AMBER, BLUE, PURPLE][: len(scores)]
    bars = ax.bar(idx, scores, color=colors, width=0.65)
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
    ax.set_title("Anchor Predictor Top-3", color=TEXT, fontsize=13, pad=10)
    ax.set_xticks(idx, [f"rank {i}" for i in idx], color=TEXT)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_ylabel("prob", color=MUTED)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _plot_budget_panel(ax, meta: dict, mode: str):
    rows = _panel_rows(meta, mode)
    xs = np.arange(len(rows))
    logits = np.asarray([float(r["selector_logit"]) for r in rows], dtype=np.float32)
    ranks = [int(r["anchor_rank"]) for r in rows]
    labels = [f"a{int(r['anchor_rank'])}s{int(r['sample_i'])}" for r in rows]
    colors = [_anchor_color(rank) for rank in ranks]

    ax.scatter(xs, logits, c=colors, s=80, edgecolors="none", zorder=3)
    ax.axhline(0.0, color=GRID, linewidth=1.0, alpha=0.9)

    if mode == "top1x9":
        selected = meta["selected_top1x9"]
        sel_idx = next(i for i, r in enumerate(rows) if int(r["anchor_rank"]) == int(selected["anchor_rank"]) and int(r["sample_i"]) == int(selected["sample_i"]))
        ax.scatter([sel_idx], [logits[sel_idx]], s=230, color=RED, marker="X", edgecolors="white", linewidths=0.9, zorder=5)
        title = "top1x9 panel: exactly 9 candidates"
        note = (
            f"selected a{selected['anchor_rank']}s{selected['sample_i']} | "
            f"logit {selected['selector_logit']:.3f} | collided"
        )
    else:
        selected = meta["selected_5_2_2"]
        sel_idx = next(i for i, r in enumerate(rows) if int(r["anchor_rank"]) == int(selected["anchor_rank"]) and int(r["sample_i"]) == int(selected["sample_i"]))
        ax.scatter([sel_idx], [logits[sel_idx]], s=260, color=GREEN, marker="*", edgecolors="white", linewidths=0.9, zorder=5)
        title = "5-2-2 panel: exactly 9 candidates"
        note = (
            f"selected a{selected['anchor_rank']}s{selected['sample_i']} | "
            f"logit {selected['selector_logit']:.3f} | safe"
        )

    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.set_xticks(xs, labels, rotation=40, ha="right", color=MUTED, fontsize=8)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_ylabel("selector logit", color=MUTED)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID, alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.text(0.01, 0.98, note, transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=9)


def _collision_window(meta: dict, traj_top1x9: np.ndarray):
    collision = meta["selected_top1x9"]["collision"]
    if collision is None:
        return slice(60, 80), -1, -1
    t_hit = int(collision["t"])
    start = max(0, t_hit - 18)
    stop = min(traj_top1x9.shape[0], t_hit + 4)
    neighbor_idx = int(collision["neighbor"])
    return slice(start, stop), neighbor_idx, t_hit


def _plot_zoom(ax, meta: dict, trajs):
    traj_522 = np.asarray(trajs["traj_5_2_2"], dtype=np.float32)
    traj_top1 = np.asarray(trajs["traj_top1x9"], dtype=np.float32)
    anchor_522 = np.asarray(trajs["anchor_5_2_2"], dtype=np.float32)
    anchor_top1 = np.asarray(trajs["anchor_top1x9"], dtype=np.float32)
    neighbors = np.asarray(trajs["neighbor_future"], dtype=np.float32)

    window, neighbor_idx, t_hit = _collision_window(meta, traj_top1)
    neighbor_traj = neighbors[neighbor_idx, window, :2] if neighbor_idx >= 0 else np.empty((0, 2), dtype=np.float32)

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
                ax.scatter([neighbor_traj[hit_local, 0]], [neighbor_traj[hit_local, 1]], s=130, color=MUTED, marker="o", zorder=6)

    ax.set_title("Collision-zone local zoom", color=TEXT, fontsize=13, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    m_522 = meta["selected_5_2_2"]["metrics"]
    m_top1 = meta["selected_top1x9"]["metrics"]
    text = (
        f"5-2-2: rank {meta['selected_5_2_2']['anchor_rank']}, safe, "
        f"progress {m_522['progress_score']:.3f}, route {m_522['route_score']:.3f}\n"
        f"top1x9: rank {meta['selected_top1x9']['anchor_rank']}, collided, "
        f"progress {m_top1['progress_score']:.3f}, route {m_top1['route_score']:.3f}\n"
        f"collision at t={t_hit}, dist={meta['selected_top1x9']['collision']['distance']:.3f} m"
    )
    ax.text(0.01, 0.02, text, transform=ax.transAxes, ha="left", va="bottom", color=TEXT, fontsize=9)


def render_case(case_dir: Path):
    meta, trajs = _load_case(case_dir)
    fig = plt.figure(figsize=(13.5, 8.6), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"{meta['scene_name']} | top1x9 vs 5-2-2",
        color=TEXT,
        fontsize=15,
        y=0.98,
    )

    _plot_anchor_scores(ax0, meta)
    _plot_budget_panel(ax1, meta, "top1x9")
    _plot_budget_panel(ax2, meta, "5-2-2")
    _plot_zoom(ax3, meta, trajs)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = case_dir / "summary_explained_v2.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+")
    args = parser.parse_args()
    for raw in args.case_dirs:
        print(render_case(Path(raw)))


if __name__ == "__main__":
    main()

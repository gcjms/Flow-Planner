#!/usr/bin/env python3
"""Plot unique top-k anchors stored inside a candidate NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
COLORS = ["#f59e0b", "#60a5fa", "#a78bfa", "#34d399", "#f472b6"]


def render(candidate_npz: Path, out_path: Path) -> None:
    with np.load(candidate_npz, allow_pickle=True) as raw:
        anchors = np.asarray(raw["anchor_trajs"], dtype=np.float32)
        ranks = np.asarray(raw["anchor_ranks"], dtype=np.int64)
        idxs = np.asarray(raw["anchor_indices"], dtype=np.int64)

    unique = []
    seen = set()
    for i, rank in enumerate(ranks.tolist()):
        if rank in seen:
            continue
        seen.add(rank)
        unique.append((rank, int(idxs[i]), anchors[i]))
    unique.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(7.8, 7.0), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for color, (rank, vocab_idx, traj) in zip(COLORS, unique):
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=3.0, label=f"rank {rank} | vocab {vocab_idx}")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=90, color=color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(traj[-1, 0], traj[-1, 1], f"  r{rank}", color=TEXT, fontsize=10, va="center")

    ax.scatter([0.0], [0.0], s=120, color="#22c55e", marker="^", edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(0.2, 0.1, "ego start", color=TEXT, fontsize=10)

    all_xy = np.concatenate([traj[:, :2] for _, _, traj in unique], axis=0)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max(1.0, 0.12 * float(x_max - x_min))
    y_pad = max(1.0, 0.12 * float(y_max - y_min))
    ax.set_xlim(float(min(-0.5, x_min - x_pad)), float(x_max + x_pad))
    ax.set_ylim(float(min(-0.5, y_min - y_pad)), float(y_max + y_pad))

    ax.set_title(candidate_npz.stem.replace("_candidates", "") + " | cached top-k anchors", color=TEXT, fontsize=14, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, alpha=0.45, linewidth=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(Path(args.candidate_npz), Path(args.output))


if __name__ == "__main__":
    main()

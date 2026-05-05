#!/usr/bin/env python3
"""Plot the top-3 anchor templates for a rescue-case scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
COLORS = ["#f59e0b", "#60a5fa", "#a78bfa"]


def render_case(case_dir: Path, anchor_vocab_path: Path) -> Path:
    meta = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    vocab = np.load(anchor_vocab_path).astype(np.float32)
    idxs = [int(x) for x in meta["anchor_topk_indices"]]
    scores = [float(x) for x in meta["anchor_topk_scores"]]
    anchors = [vocab[idx] for idx in idxs]

    fig, ax = plt.subplots(figsize=(7.6, 7.2), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for rank, (anchor, score, vocab_idx, color) in enumerate(zip(anchors, scores, idxs, COLORS)):
        ax.plot(anchor[:, 0], anchor[:, 1], color=color, linewidth=3.2, label=f"rank {rank} | vocab {vocab_idx} | prob {score:.3f}")
        ax.scatter(anchor[-1, 0], anchor[-1, 1], s=90, color=color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(anchor[-1, 0], anchor[-1, 1], f"  r{rank}", color=TEXT, fontsize=10, va="center")

    ax.scatter([0.0], [0.0], s=120, color="#22c55e", marker="^", edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(0.2, 0.1, "ego start", color=TEXT, fontsize=10)

    ax.set_title(f"{meta['scene_name']} | top-3 anchor templates", color=TEXT, fontsize=15, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, alpha=0.45, linewidth=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    all_xy = np.concatenate([a[:, :2] for a in anchors], axis=0)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max(1.0, 0.12 * float(x_max - x_min))
    y_pad = max(1.0, 0.12 * float(y_max - y_min))
    ax.set_xlim(float(min(-0.5, x_min - x_pad)), float(x_max + x_pad))
    ax.set_ylim(float(min(-0.5, y_min - y_pad)), float(y_max + y_pad))

    out_path = case_dir / "top3_anchor_templates.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-vocab-path", required=True)
    parser.add_argument("case_dirs", nargs="+")
    args = parser.parse_args()

    vocab_path = Path(args.anchor_vocab_path)
    for raw in args.case_dirs:
        print(render_case(Path(raw), vocab_path))


if __name__ == "__main__":
    main()

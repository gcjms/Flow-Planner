#!/usr/bin/env python3
"""
Trajectory Anchor Visualization (Phase 0-3)
===========================================
对标 ``visualize_goals.py``，但可视化对象是 (K, T, 3) 轨迹模板而非 (K, 2) 点。

产出 4 张图（一个 PNG 网格 + 一个单 anchor 多子图）：
  1. overlay.png           : 所有 K 条 anchor 叠加在 ego frame 上
  2. overlay_by_size.png   : 同上，但 alpha/粗细按簇大小加权（看主流 mode）
  3. small_multiples.png   : 8x(K/8) 网格，每个子图画一条 anchor
  4. stats.png             : 终点分布 / 横向偏差 / 朝向变化直方图 + 文本统计

用法：
  python visualize_anchors.py \
      --vocab_path /root/Flow-Planner/anchor_vocab.npy \
      --meta_path  /root/Flow-Planner/anchor_vocab_meta.json \
      --output_dir /root/Flow-Planner/anchor_viz
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def _load_counts(meta_path: str | None, n_anchors: int) -> np.ndarray:
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)
        counts = meta.get("cluster_counts", [])
        if len(counts) == n_anchors:
            return np.asarray(counts, dtype=np.int64)
    return np.zeros((n_anchors,), dtype=np.int64)


def _plot_overlay(
    anchors: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    weight_by_size: bool,
) -> None:
    K, T, _ = anchors.shape
    cmap = get_cmap("tab20", max(K, 20))

    fig, ax = plt.subplots(figsize=(10, 10))
    max_count = int(counts.max()) if counts.max() > 0 else 1

    order = np.argsort(counts) if weight_by_size else np.arange(K)
    for k in order:
        xy = anchors[k, :, :2]
        if weight_by_size:
            w = counts[k] / max_count
            alpha = 0.15 + 0.75 * w
            lw = 0.8 + 2.5 * w
        else:
            alpha = 0.7
            lw = 1.2
        color = cmap(k % cmap.N)
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha, linewidth=lw)
        ax.plot(xy[-1, 0], xy[-1, 1], "o", color=color, alpha=alpha, markersize=3)

    ax.plot(0, 0, "k^", markersize=12, label="ego start")
    ax.axhline(0, color="k", linestyle="--", alpha=0.2)
    ax.axvline(0, color="k", linestyle="--", alpha=0.2)
    ax.set_xlabel("X (m) — forward")
    ax.set_ylabel("Y (m) — left(+) / right(-)")
    suffix = " [alpha∝cluster size]" if weight_by_size else ""
    ax.set_title(f"Anchor overlay (K={K}, T={T}){suffix}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _plot_small_multiples(
    anchors: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
    n_cols: int = 8,
) -> None:
    K, T, _ = anchors.shape
    n_rows = math.ceil(K / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 1.8),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    all_xy = anchors[:, :, :2]
    x_min, x_max = float(all_xy[:, :, 0].min()), float(all_xy[:, :, 0].max())
    y_min, y_max = float(all_xy[:, :, 1].min()), float(all_xy[:, :, 1].max())
    pad_x = 0.05 * (x_max - x_min + 1e-6)
    pad_y = 0.05 * (y_max - y_min + 1e-6)

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        if k < K:
            xy = anchors[k, :, :2]
            ax.plot(xy[:, 0], xy[:, 1], color="C0", linewidth=1.2)
            ax.plot(xy[-1, 0], xy[-1, 1], "o", color="C3", markersize=3)
            ax.plot(0, 0, "k^", markersize=3)
            ax.set_title(f"#{k} (n={counts[k]})", fontsize=7)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Anchor small multiples (K={anchors.shape[0]}, T={anchors.shape[1]})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _plot_stats(
    anchors: np.ndarray,
    counts: np.ndarray,
    out_path: Path,
) -> None:
    endpoints = anchors[:, -1, :2]                               # (K, 2)
    endpoint_norm = np.linalg.norm(endpoints, axis=-1)           # (K,)
    heading_end = anchors[:, -1, 2]                              # (K,)
    heading_delta = np.abs(
        np.arctan2(
            np.sin(heading_end - anchors[:, 0, 2]),
            np.cos(heading_end - anchors[:, 0, 2]),
        )
    )                                                            # (K,)
    lateral_peak = np.abs(anchors[:, :, 1]).max(axis=-1)         # (K,)
    curvature_proxy = np.abs(np.diff(anchors[:, :, 2], axis=-1)).sum(axis=-1)  # (K,)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    sc = ax.scatter(endpoints[:, 0], endpoints[:, 1],
                    c=counts, cmap="viridis", s=60, edgecolor="k")
    ax.plot(0, 0, "k^", markersize=10)
    ax.set_title("Endpoint scatter (color=cluster size)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    axes[0, 1].hist(endpoint_norm, bins=20, color="steelblue", edgecolor="k")
    axes[0, 1].set_title("Endpoint distance distribution")
    axes[0, 1].set_xlabel("dist (m)"); axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    axes[0, 2].hist(np.rad2deg(heading_delta), bins=20, color="indianred", edgecolor="k")
    axes[0, 2].set_title("|heading(T) - heading(0)| distribution")
    axes[0, 2].set_xlabel("deg"); axes[0, 2].set_ylabel("count")
    axes[0, 2].grid(True, alpha=0.3, axis="y")

    axes[1, 0].hist(lateral_peak, bins=20, color="seagreen", edgecolor="k")
    axes[1, 0].set_title("Peak |y| per anchor (lane change magnitude)")
    axes[1, 0].set_xlabel("m"); axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].hist(counts, bins=20, color="slateblue", edgecolor="k")
    axes[1, 1].set_title("Cluster size distribution")
    axes[1, 1].set_xlabel("#samples in cluster"); axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    axes[1, 2].axis("off")
    empty = int((counts == 0).sum())
    text = (
        f"Anchors K = {anchors.shape[0]}\n"
        f"Horizon T = {anchors.shape[1]} frames\n\n"
        f"Endpoint ||(x,y)|| (m):\n"
        f"  min / mean / max = "
        f"{endpoint_norm.min():.1f} / {endpoint_norm.mean():.1f} / {endpoint_norm.max():.1f}\n"
        f"|heading change| (deg):\n"
        f"  min / mean / max = "
        f"{np.rad2deg(heading_delta.min()):.1f} / "
        f"{np.rad2deg(heading_delta.mean()):.1f} / "
        f"{np.rad2deg(heading_delta.max()):.1f}\n"
        f"Peak |y| (m):\n"
        f"  min / mean / max = "
        f"{lateral_peak.min():.2f} / {lateral_peak.mean():.2f} / {lateral_peak.max():.2f}\n"
        f"Curvature proxy (sum |Δheading|):\n"
        f"  min / mean / max = "
        f"{curvature_proxy.min():.2f} / {curvature_proxy.mean():.2f} / {curvature_proxy.max():.2f}\n\n"
        f"Cluster sizes:\n"
        f"  min / median / max = "
        f"{counts.min()} / {int(np.median(counts))} / {counts.max()}\n"
        f"  empty clusters = {empty}\n"
    )
    axes[1, 2].text(0.02, 0.5, text, fontsize=11, family="monospace",
                    verticalalignment="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the trajectory anchor vocabulary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vocab_path", type=str, default="anchor_vocab.npy")
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Optional anchor_vocab_meta.json for cluster sizes.")
    parser.add_argument("--output_dir", type=str, default="anchor_viz")
    parser.add_argument("--small_multiples_cols", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = np.load(args.vocab_path)
    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError(
            f"Expected anchor vocab shape (K, T, 3); got {anchors.shape}"
        )
    K, T, _ = anchors.shape
    print(f"Loaded anchors from {args.vocab_path}: K={K}, T={T}")

    meta_path = args.meta_path or (
        str(Path(args.vocab_path).with_name(Path(args.vocab_path).stem + "_meta.json"))
    )
    counts = _load_counts(meta_path, K)
    if counts.sum() == 0:
        print(f"  [info] no cluster sizes found at {meta_path}; using uniform weights")

    print("\nRendering ...")
    _plot_overlay(anchors, counts, out_dir / "overlay.png", weight_by_size=False)
    _plot_overlay(anchors, counts, out_dir / "overlay_by_size.png", weight_by_size=True)
    _plot_small_multiples(anchors, counts, out_dir / "small_multiples.png",
                          n_cols=args.small_multiples_cols)
    _plot_stats(anchors, counts, out_dir / "stats.png")
    print(f"\nAll outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

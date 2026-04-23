#!/usr/bin/env python3
"""
Trajectory Anchor Vocabulary Construction (Phase 0-1)
=====================================================
对标 ``cluster_goals.py``，但聚类对象从 2D endpoint 升级为完整多秒轨迹。

核心区别：
- goal 版本：取 ``gt_future[goal_frame, :2]`` → 一个 (x, y) 点 → KMeans on (N, 2)
- anchor 版：取 ``gt_future[:T, :3]`` → 一条 (T, 3) 轨迹 → KMeans on (N, T*4)
  (heading 展开成 cos/sin 避免角度回绕；xy 和 trig 量级不一致通过权重归一化)

产出：
- ``anchor_vocab.npy``         shape ``(K, T, 3)``  每条 anchor 的 (x, y, heading)
- ``anchor_vocab_meta.json``    聚类配置 + 每簇样本数 + inertia

用法：
  python -m flow_planner.goal.cluster_trajectories \
      --data_dir /root/autodl-tmp/nuplan_npz \
      --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
      --output_path /root/Flow-Planner/anchor_vocab.npy \
      --n_anchors 128 \
      --traj_len 40 \
      --heading_weight 5.0 \
      --use_pca \
      --pca_dim 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np


def _load_trajectory(
    path: str,
    traj_len: int,
    max_endpoint_norm: float,
) -> np.ndarray | None:
    """Load one NPZ sample and return a (traj_len, 3) trajectory in (x, y, heading).

    Returns ``None`` if the sample is unusable (short / NaN / unrealistic).
    """
    data = np.load(path)
    future = data["ego_agent_future"]
    if future.ndim != 2 or future.shape[0] < traj_len or future.shape[1] < 3:
        return None

    traj = future[:traj_len, :3].astype(np.float32)
    if not np.isfinite(traj).all():
        return None
    if np.linalg.norm(traj[-1, :2]) > max_endpoint_norm:
        return None
    return traj


def _encode_features(
    trajs: np.ndarray,
    heading_weight: float,
) -> np.ndarray:
    """Flatten (N, T, 3) -> (N, T*4) with heading -> (cos, sin) * weight.

    The heading weight rescales cos/sin (∈ [-1, 1]) so they are not dominated by the
    (x, y) coordinates that can reach tens of meters.
    """
    xy = trajs[:, :, :2]                       # (N, T, 2)
    heading = trajs[:, :, 2]                   # (N, T)
    cos_h = np.cos(heading) * heading_weight   # (N, T)
    sin_h = np.sin(heading) * heading_weight   # (N, T)
    feats = np.concatenate(
        [xy, cos_h[..., None], sin_h[..., None]],
        axis=-1,
    )                                           # (N, T, 4)
    return feats.reshape(feats.shape[0], -1)    # (N, T*4)


def _maybe_pca(
    feats: np.ndarray,
    use_pca: bool,
    pca_dim: int,
) -> Tuple[np.ndarray, dict]:
    """Apply PCA if requested. Returns transformed array + a small info dict."""
    info = {"used": False, "dim": int(feats.shape[1])}
    if not use_pca:
        return feats, info
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [warn] sklearn not available; PCA disabled")
        return feats, info

    pca_dim = min(pca_dim, feats.shape[1], feats.shape[0])
    pca = PCA(n_components=pca_dim, random_state=42)
    reduced = pca.fit_transform(feats).astype(np.float32)
    info.update(
        used=True,
        dim=int(reduced.shape[1]),
        explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
    )
    print(
        f"  PCA {feats.shape[1]} -> {reduced.shape[1]} dims, "
        f"explained variance = {info['explained_variance_ratio']:.3f}"
    )
    return reduced, info


def _kmeans_fit(
    feats: np.ndarray,
    n_anchors: int,
    n_init: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_anchors,
        random_state=seed,
        n_init=n_init,
        verbose=0,
    )
    kmeans.fit(feats)
    return kmeans.labels_, float(kmeans.inertia_)


def _anchors_from_labels(
    trajs: np.ndarray,
    labels: np.ndarray,
    n_anchors: int,
) -> Tuple[np.ndarray, list[int]]:
    """Build anchor trajectories by averaging members of each KMeans cluster.

    For the heading dimension we average cos/sin and convert back via atan2 to avoid
    the ±π wrap bug you would get from a direct mean.
    """
    T = trajs.shape[1]
    anchors = np.zeros((n_anchors, T, 3), dtype=np.float32)
    counts: list[int] = []
    for k in range(n_anchors):
        mask = labels == k
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            continue
        members = trajs[mask]                              # (M, T, 3)
        xy_mean = members[:, :, :2].mean(axis=0)           # (T, 2)
        cos_mean = np.cos(members[:, :, 2]).mean(axis=0)   # (T,)
        sin_mean = np.sin(members[:, :, 2]).mean(axis=0)   # (T,)
        heading_mean = np.arctan2(sin_mean, cos_mean)      # (T,)
        anchors[k, :, :2] = xy_mean
        anchors[k, :, 2] = heading_mean
    return anchors, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster full GT trajectories into (K, T, 3) anchor vocabulary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training NPZ files")
    parser.add_argument("--data_list", type=str, required=True,
                        help="JSON file listing NPZ filenames (same as training data list)")
    parser.add_argument("--output_path", type=str, default="anchor_vocab.npy",
                        help="Where to save the anchor vocab (K, T, 3) numpy array")
    parser.add_argument("--n_anchors", type=int, default=128,
                        help="Number of anchor clusters (K). Plan default = 128.")
    parser.add_argument("--traj_len", type=int, default=40,
                        help="Future horizon in frames. 40 = 4s @ 10Hz (matches goal_frame=39)")
    parser.add_argument("--max_endpoint_norm", type=float, default=200.0,
                        help="Drop samples whose endpoint is farther than this (meters)")
    parser.add_argument("--heading_weight", type=float, default=5.0,
                        help="Rescale factor for cos/sin(heading) to balance xy magnitudes")
    parser.add_argument("--use_pca", action="store_true",
                        help="Apply PCA before KMeans (recommended for T*4 >= 64)")
    parser.add_argument("--pca_dim", type=int, default=16,
                        help="PCA target dimensionality")
    parser.add_argument("--n_init", type=int, default=10,
                        help="KMeans n_init")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.data_list, "r") as fp:
        file_list = json.load(fp)
    if args.max_samples:
        file_list = file_list[: args.max_samples]

    print(f"[1/4] Loading trajectories from {len(file_list)} files ...")
    t0 = time.time()
    trajs: list[np.ndarray] = []
    skipped = 0
    for i, fname in enumerate(file_list):
        path = os.path.join(args.data_dir, fname)
        try:
            traj = _load_trajectory(path, args.traj_len, args.max_endpoint_norm)
        except Exception as exc:
            traj = None
            if skipped < 5:
                print(f"  [warn] failed to load {fname}: {exc}")
        if traj is None:
            skipped += 1
            continue
        trajs.append(traj)
        if (i + 1) % 10_000 == 0:
            print(f"  [{i + 1}/{len(file_list)}] kept={len(trajs)} skipped={skipped}")

    if not trajs:
        print("[error] No valid trajectories collected. Check --data_dir / --data_list.")
        return 1

    trajs_arr = np.stack(trajs)  # (N, T, 3)
    print(
        f"  done in {time.time() - t0:.1f}s | kept={len(trajs_arr)} skipped={skipped}\n"
        f"  shape={trajs_arr.shape} "
        f"x=[{trajs_arr[:, -1, 0].min():.1f}, {trajs_arr[:, -1, 0].max():.1f}] "
        f"y=[{trajs_arr[:, -1, 1].min():.1f}, {trajs_arr[:, -1, 1].max():.1f}] "
        f"endpoint_norm_mean={np.linalg.norm(trajs_arr[:, -1, :2], axis=-1).mean():.1f}m"
    )

    print(f"\n[2/4] Encoding features (heading_weight={args.heading_weight}) ...")
    feats = _encode_features(trajs_arr, args.heading_weight)  # (N, T*4)
    feats, pca_info = _maybe_pca(feats, args.use_pca, args.pca_dim)
    print(f"  feature shape for KMeans: {feats.shape}")

    print(f"\n[3/4] Running KMeans K={args.n_anchors} n_init={args.n_init} ...")
    t0 = time.time()
    labels, inertia = _kmeans_fit(feats, args.n_anchors, args.n_init, args.seed)
    print(f"  done in {time.time() - t0:.1f}s | inertia={inertia:.1f}")

    print(f"\n[4/4] Building anchor trajectories (cluster-mean with cos/sin heading) ...")
    anchors, counts = _anchors_from_labels(trajs_arr, labels, args.n_anchors)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, anchors)
    print(f"  saved {out_path}  shape={anchors.shape}")

    meta = {
        "n_anchors": int(args.n_anchors),
        "traj_len": int(args.traj_len),
        "n_samples_used": int(len(trajs_arr)),
        "n_samples_skipped": int(skipped),
        "heading_weight": float(args.heading_weight),
        "pca": pca_info,
        "kmeans_inertia": inertia,
        "cluster_counts": counts,
        "seed": int(args.seed),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"  saved {meta_path}")

    counts_arr = np.asarray(counts)
    empty = int((counts_arr == 0).sum())
    print(
        f"\nCluster size stats: "
        f"min={counts_arr.min()} max={counts_arr.max()} "
        f"median={int(np.median(counts_arr))} empty_clusters={empty}"
    )
    if empty > 0:
        print(
            "  [warn] some clusters are empty; consider lowering --n_anchors "
            "or disabling --use_pca."
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

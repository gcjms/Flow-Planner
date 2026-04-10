#!/usr/bin/env python3
"""
Goal Point Vocabulary Construction (Step 1)
============================================
从训练数据的 GT 轨迹终点做 k-means 聚类，生成 goal vocabulary。

这个脚本只需要跑一次。输出一个 .npy 文件给后续训练和推理用。

用法:
  python -m flow_planner.goal.cluster_goals \
      --data_dir /root/autodl-tmp/nuplan_npz \
      --data_list /root/autodl-tmp/nuplan_npz/train_list.json \
      --output_path /root/Flow-Planner/goal_vocab.npy \
      --n_clusters 64
"""

import os
import sys
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Cluster GT trajectory endpoints to build goal vocabulary"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing training NPZ files",
    )
    parser.add_argument(
        "--data_list", type=str, required=True,
        help="JSON file listing NPZ filenames (same as training data list)",
    )
    parser.add_argument(
        "--output_path", type=str, default="goal_vocab.npy",
        help="Where to save the vocabulary (K, 2) numpy array",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=64,
        help="Number of goal clusters (K). 64 is a good starting point for nuPlan.",
    )
    parser.add_argument(
        "--goal_frame", type=int, default=39,
        help="Which future frame to use as goal point (0-indexed). "
             "Default 39 = 4s at 10Hz. Use -1 for last frame (8s).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples (for debugging)",
    )
    args = parser.parse_args()

    # ---- Load file list ----
    with open(args.data_list, "r") as f:
        file_list = json.load(f)

    if args.max_samples:
        file_list = file_list[: args.max_samples]

    print(f"Total files: {len(file_list)}")

    # ---- Collect GT endpoints ----
    endpoints = []
    skipped = 0
    for i, fname in enumerate(file_list):
        path = os.path.join(args.data_dir, fname)
        try:
            data = np.load(path)
            gt_future = data["ego_agent_future"]  # (T, D), D >= 2
            T = gt_future.shape[0]
            goal_idx = (T - 1) if args.goal_frame < 0 else min(args.goal_frame, T - 1)
            endpoint = gt_future[goal_idx, :2].astype(np.float32)

            if np.isnan(endpoint).any() or np.linalg.norm(endpoint) > 200:
                skipped += 1
                continue

            endpoints.append(endpoint)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: skip {fname}: {e}")

        if (i + 1) % 10000 == 0:
            print(f"  [{i + 1}/{len(file_list)}] collected {len(endpoints)}, skipped {skipped}")

    endpoints = np.stack(endpoints)  # (N, 2)
    print(f"\nCollected {len(endpoints)} endpoints (skipped {skipped})")
    print(f"  x range: [{endpoints[:, 0].min():.1f}, {endpoints[:, 0].max():.1f}]")
    print(f"  y range: [{endpoints[:, 1].min():.1f}, {endpoints[:, 1].max():.1f}]")
    print(f"  mean norm: {np.linalg.norm(endpoints, axis=-1).mean():.1f} m")

    # ---- K-Means Clustering ----
    from sklearn.cluster import KMeans

    print(f"\nRunning k-means with K={args.n_clusters} ...")
    kmeans = KMeans(
        n_clusters=args.n_clusters, random_state=42, n_init=10, verbose=0
    )
    kmeans.fit(endpoints)
    vocab = kmeans.cluster_centers_.astype(np.float32)  # (K, 2)

    # ---- Save ----
    np.save(args.output_path, vocab)
    print(f"\nSaved vocabulary to {args.output_path}")
    print(f"  Shape: {vocab.shape}")

    # ---- Print cluster stats ----
    labels = kmeans.labels_
    print(f"\nCluster distribution:")
    for i in range(args.n_clusters):
        count = (labels == i).sum()
        cx, cy = vocab[i]
        dist = np.linalg.norm(vocab[i])
        print(f"  [{i:3d}] center=({cx:6.1f}, {cy:6.1f})  dist={dist:5.1f}m  count={count}")

    print(f"\nInertia: {kmeans.inertia_:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find scenes whose predicted top-k anchors differ strongly laterally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from flow_planner.dpo.eval_multidim_utils import (
    load_anchor_predictor_model,
    load_planner_model,
    resolve_scene_files,
    scene_to_datasample,
)


BG = "#0f172a"
GRID = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
COLORS = ["#f59e0b", "#60a5fa", "#a78bfa", "#34d399", "#f472b6"]


def pairwise_stats(anchor_trajs: np.ndarray) -> Dict[str, float]:
    """Return simple pairwise spread metrics for top-k anchors."""
    k = int(anchor_trajs.shape[0])
    endpoint_deltas = []
    endpoint_dy = []
    endpoint_dx = []
    ade = []
    for i in range(k):
        for j in range(i + 1, k):
            diff_xy = anchor_trajs[i, :, :2] - anchor_trajs[j, :, :2]
            ade.append(float(np.linalg.norm(diff_xy, axis=-1).mean()))
            endpoint = anchor_trajs[i, -1, :2] - anchor_trajs[j, -1, :2]
            endpoint_deltas.append(float(np.linalg.norm(endpoint)))
            endpoint_dx.append(float(abs(endpoint[0])))
            endpoint_dy.append(float(abs(endpoint[1])))
    return {
        "pairwise_ade_mean": float(np.mean(ade)) if ade else 0.0,
        "pairwise_fde_mean": float(np.mean(endpoint_deltas)) if endpoint_deltas else 0.0,
        "pairwise_end_dx_max": float(np.max(endpoint_dx)) if endpoint_dx else 0.0,
        "pairwise_end_dy_max": float(np.max(endpoint_dy)) if endpoint_dy else 0.0,
        "endpoint_y_spread": float(anchor_trajs[:, -1, 1].max() - anchor_trajs[:, -1, 1].min()),
        "endpoint_x_spread": float(anchor_trajs[:, -1, 0].max() - anchor_trajs[:, -1, 0].min()),
    }


def render_anchor_plot(scene_name: str, anchor_indices: np.ndarray, anchor_scores: np.ndarray, anchor_trajs: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for rank, (idx, score, traj) in enumerate(zip(anchor_indices, anchor_scores, anchor_trajs)):
        color = COLORS[rank % len(COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=3.0, label=f"rank {rank} | vocab {int(idx)} | prob {float(score):.3f}")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=90, color=color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(traj[-1, 0], traj[-1, 1], f"  r{rank}", color=TEXT, fontsize=10, va="center")

    ax.scatter([0.0], [0.0], s=120, color="#22c55e", marker="^", edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(0.2, 0.1, "ego start", color=TEXT, fontsize=10)

    all_xy = anchor_trajs[:, :, :2].reshape(-1, 2)
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_pad = max(1.0, 0.12 * float(x_max - x_min))
    y_pad = max(1.0, 0.12 * float(y_max - y_min))
    ax.set_xlim(float(min(-0.5, x_min - x_pad)), float(x_max + x_pad))
    ax.set_ylim(float(min(-0.5, y_min - y_pad)), float(y_max + y_pad))

    ax.set_title(f"{scene_name} | top-k anchor spread", color=TEXT, fontsize=14, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, alpha=0.45, linewidth=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(loc="upper left", fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--anchor-vocab-path", required=True)
    parser.add_argument("--anchor-predictor-ckpt", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--scene-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-scenes", type=int, default=2000)
    parser.add_argument("--top-n-render", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_planner_model(
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        device=args.device,
        anchor_vocab_path=args.anchor_vocab_path,
    )
    predictor = load_anchor_predictor_model(
        model,
        ckpt_path=args.anchor_predictor_ckpt,
        device=args.device,
    )

    scene_files = resolve_scene_files(
        scene_dir=args.scene_dir,
        max_scenes=args.max_scenes,
        scene_manifest=args.scene_manifest,
        manifest_seed=None,
    )

    reports: List[Dict[str, object]] = []
    neighbor_limit = int(model.planner_params.get("neighbor_num", 32))
    for scene_idx, scene_file in enumerate(scene_files):
        with np.load(scene_file, allow_pickle=True) as raw:
            scene_data = {key: raw[key] for key in raw.files}
        data = scene_to_datasample(scene_data, device=args.device, neighbor_limit=neighbor_limit)
        prediction = predictor.predict_topk(data, top_k=args.top_k)
        anchor_indices = prediction["indices"][0].detach().cpu().numpy().astype(np.int64)
        anchor_scores = prediction["scores"][0].detach().cpu().numpy().astype(np.float32)
        anchor_trajs = prediction["anchor_trajs"][0].detach().cpu().numpy().astype(np.float32)
        stats = pairwise_stats(anchor_trajs)
        reports.append(
            {
                "scene_rank": scene_idx,
                "scene_name": Path(scene_file).stem,
                "scene_file": scene_file,
                "anchor_topk_indices": anchor_indices.tolist(),
                "anchor_topk_scores": [float(x) for x in anchor_scores.tolist()],
                **stats,
            }
        )

    reports.sort(key=lambda item: (item["pairwise_end_dy_max"], item["endpoint_y_spread"], item["pairwise_ade_mean"]), reverse=True)
    report_path = output_dir / "lateral_anchor_case_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for report in reports[: args.top_n_render]:
        with np.load(report["scene_file"], allow_pickle=True) as raw:
            scene_data = {key: raw[key] for key in raw.files}
        data = scene_to_datasample(scene_data, device=args.device, neighbor_limit=neighbor_limit)
        prediction = predictor.predict_topk(data, top_k=args.top_k)
        anchor_indices = prediction["indices"][0].detach().cpu().numpy().astype(np.int64)
        anchor_scores = prediction["scores"][0].detach().cpu().numpy().astype(np.float32)
        anchor_trajs = prediction["anchor_trajs"][0].detach().cpu().numpy().astype(np.float32)
        out_path = output_dir / f"{report['scene_name']}_topk_anchor_spread.png"
        render_anchor_plot(report["scene_name"], anchor_indices, anchor_scores, anchor_trajs, out_path)

    print(report_path)
    for report in reports[: min(10, len(reports))]:
        print(
            report["scene_name"],
            "dy_max=", round(float(report["pairwise_end_dy_max"]), 3),
            "y_spread=", round(float(report["endpoint_y_spread"]), 3),
            "ade=", round(float(report["pairwise_ade_mean"]), 3),
            "topk=", report["anchor_topk_indices"],
        )


if __name__ == "__main__":
    main()

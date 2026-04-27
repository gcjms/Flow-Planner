"""
Generate scored anchor-conditioned candidate sets for soft preference distillation.

For each scene, an AnchorPredictor proposes top-k anchors. The planner samples
several trajectories under each anchor condition. Unlike hard DPO mining, this
script keeps the full candidate set so downstream training can learn a soft
within-scene ranking instead of forcing every pair into chosen/rejected labels.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from flow_planner.dpo.anchor_candidate_scorer import (
    AnchorCandidateScoreWeights,
    build_candidate_record,
    json_ready_anchor_summaries,
    summarize_anchor_groups,
    summarize_scene,
)
from flow_planner.dpo.eval_multidim_utils import (
    evaluate_trajectory,
    infer_single_trajectory,
    load_anchor_predictor_model,
    load_planner_model,
    resolve_scene_files,
    scene_to_datasample,
)

logger = logging.getLogger(__name__)


def _load_scene(scene_file: str) -> Dict[str, np.ndarray]:
    with np.load(scene_file, allow_pickle=True) as raw:
        return {key: raw[key] for key in raw.files}


def _gt_errors(traj: np.ndarray, gt_future: np.ndarray) -> Dict[str, float]:
    if gt_future is None or len(gt_future) == 0:
        return {"ade": 0.0, "fde": 0.0}
    t = min(traj.shape[0], gt_future.shape[0])
    if t <= 0:
        return {"ade": 0.0, "fde": 0.0}
    pos_err = np.linalg.norm(traj[:t, :2] - gt_future[:t, :2], axis=-1)
    return {"ade": float(pos_err.mean()), "fde": float(pos_err[-1])}


def generate_anchor_softpref_candidates(args: argparse.Namespace) -> Dict[str, int]:
    device = args.device
    model = load_planner_model(
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
        device=device,
        anchor_vocab_path=args.anchor_vocab_path,
    )
    anchor_predictor = load_anchor_predictor_model(
        model,
        ckpt_path=args.anchor_predictor_ckpt,
        device=device,
    )

    scene_files = resolve_scene_files(
        scene_dir=args.scene_dir,
        max_scenes=args.max_scenes,
        scene_manifest=args.scene_manifest,
        manifest_seed=args.manifest_seed,
    )
    logger.info("Processing %d scenes", len(scene_files))

    candidates_dir = Path(args.output_dir) / "candidates"
    scored_dir = Path(args.output_dir) / "scored_dir"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    failures: List[Dict[str, str]] = []
    written = 0
    total_candidates = 0
    neighbor_limit = int(model.planner_params.get("neighbor_num", 32))
    score_weights = AnchorCandidateScoreWeights.from_args(args)

    for scene_i, scene_file in enumerate(scene_files):
        scenario_id = Path(scene_file).stem
        try:
            scene_data = _load_scene(scene_file)
            data = scene_to_datasample(scene_data, device=device, neighbor_limit=neighbor_limit)
            prediction = anchor_predictor.predict_topk(data, top_k=args.top_k)
            pred_indices = prediction["indices"][0].detach().cpu().numpy().astype(np.int64)
            pred_anchor_trajs = prediction["anchor_trajs"][0].detach().cpu().numpy().astype(np.float32)

            trajectories: List[np.ndarray] = []
            anchor_trajs: List[np.ndarray] = []
            anchor_indices: List[int] = []
            anchor_ranks: List[int] = []
            sample_ids: List[int] = []
            candidate_infos: List[Dict[str, object]] = []

            candidate_idx = 0
            for rank, (anchor_idx, anchor_np) in enumerate(zip(pred_indices, pred_anchor_trajs)):
                anchor_tensor = torch.from_numpy(anchor_np).unsqueeze(0).to(device)
                for sample_i in range(args.samples_per_anchor):
                    bon_seed = args.bon_seed_base + scene_i * 1000 + rank * 100 + sample_i
                    traj = infer_single_trajectory(
                        model,
                        data,
                        use_cfg=args.use_cfg,
                        cfg_weight=args.cfg_weight,
                        bon_seed=bon_seed,
                        anchor_traj=anchor_tensor,
                    )
                    metrics = evaluate_trajectory(
                        traj,
                        neighbor_future_gt=scene_data["neighbor_agents_future"],
                        route_lanes=scene_data.get("route_lanes"),
                        collision_dist=args.collision_dist,
                    )
                    gt_metrics = _gt_errors(traj, scene_data.get("ego_agent_future"))
                    metrics = {**metrics, **gt_metrics}

                    trajectories.append(traj[:, : args.state_dim].astype(np.float32))
                    anchor_trajs.append(anchor_np.astype(np.float32))
                    anchor_indices.append(int(anchor_idx))
                    anchor_ranks.append(int(rank))
                    sample_ids.append(int(sample_i))
                    candidate_infos.append(
                        build_candidate_record(
                            candidate_idx=candidate_idx,
                            anchor_index=int(anchor_idx),
                            anchor_rank=int(rank),
                            sample_i=int(sample_i),
                            seed=int(bon_seed),
                            metrics=metrics,
                            weights=score_weights,
                        )
                    )
                    candidate_idx += 1

            if len(trajectories) < args.min_candidates:
                continue

            candidate_npz = candidates_dir / f"{scenario_id}_candidates.npz"
            np.savez_compressed(
                candidate_npz,
                candidates=np.stack(trajectories, axis=0),
                anchor_trajs=np.stack(anchor_trajs, axis=0),
                anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
                anchor_ranks=np.asarray(anchor_ranks, dtype=np.int64),
                sample_ids=np.asarray(sample_ids, dtype=np.int64),
            )

            scored_payload = {
                "scenario_id": scenario_id,
                "scene_file": scene_file,
                "source_npz": str(candidate_npz),
                "top_k": args.top_k,
                "samples_per_anchor": args.samples_per_anchor,
                "candidates": candidate_infos,
                "score_config": score_weights.to_dict(),
                "scene_stats": summarize_scene(candidate_infos),
                "anchor_group_stats": json_ready_anchor_summaries(
                    summarize_anchor_groups(candidate_infos, method="mean")
                ),
            }
            (scored_dir / f"{scenario_id}.json").write_text(
                json.dumps(scored_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written += 1
            total_candidates += len(trajectories)

            if (scene_i + 1) % args.log_every == 0:
                logger.info(
                    "[%d/%d] written_scenes=%d candidates=%d failures=%d",
                    scene_i + 1,
                    len(scene_files),
                    written,
                    total_candidates,
                    len(failures),
                )
        except Exception as exc:  # noqa: BLE001 - keep long experiment alive.
            failures.append({"scene": Path(scene_file).name, "error": str(exc)})
            logger.warning("Failed scene %s: %s", scene_file, exc)

    meta = {
        "output_dir": args.output_dir,
        "candidates_dir": str(candidates_dir),
        "scored_dir": str(scored_dir),
        "scene_dir": args.scene_dir,
        "scene_manifest": args.scene_manifest,
        "max_scenes": args.max_scenes,
        "top_k": args.top_k,
        "samples_per_anchor": args.samples_per_anchor,
        "written_scenes": written,
        "total_candidates": total_candidates,
        "failures": failures,
    }
    Path(args.output_dir, "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "scenes": len(scene_files),
        "written_scenes": written,
        "total_candidates": total_candidates,
        "failures": len(failures),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anchor soft preference candidates")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--anchor_vocab_path", required=True)
    parser.add_argument("--anchor_predictor_ckpt", required=True)
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--scene_manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_scenes", type=int, default=100)
    parser.add_argument("--manifest_seed", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--samples_per_anchor", type=int, default=3)
    parser.add_argument("--min_candidates", type=int, default=2)
    parser.add_argument("--state_dim", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_weight", type=float, default=1.8)
    parser.add_argument("--bon_seed_base", type=int, default=61000)
    parser.add_argument("--collision_dist", type=float, default=2.0)
    parser.add_argument("--safety_weight", type=float, default=5.0)
    parser.add_argument("--route_weight", type=float, default=1.0)
    parser.add_argument("--progress_weight", type=float, default=1.0)
    parser.add_argument("--ttc_weight", type=float, default=0.1)
    parser.add_argument("--comfort_weight", type=float, default=0.05)
    parser.add_argument("--collision_weight", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    summary = generate_anchor_softpref_candidates(args)
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()

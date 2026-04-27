#!/usr/bin/env python3
"""Score saved anchor-conditioned candidate trajectory sets.

This is the explicit score stage in the anchor generate-then-score pipeline. It
can rescore existing ``*_candidates.npz`` files without rerunning the planner,
which keeps selector experiments reproducible when scorer weights change.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from flow_planner.dpo.anchor_candidate_scorer import (
    AnchorCandidateScoreWeights,
    build_candidate_record,
    json_ready_anchor_summaries,
    summarize_anchor_groups,
    summarize_scene,
)
from flow_planner.dpo.eval_multidim_utils import evaluate_trajectory

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as raw:
        return {key: raw[key] for key in raw.files}


def _candidate_scenario_id(path: Path) -> str:
    name = path.stem
    if name.endswith("_candidates"):
        return name[: -len("_candidates")]
    return name


def _resolve_scene_file(
    scenario_id: str,
    scene_dir: Optional[str],
    scene_file_by_id: Dict[str, str],
) -> str:
    if scenario_id in scene_file_by_id:
        return scene_file_by_id[scenario_id]
    if scene_dir is None:
        raise FileNotFoundError(
            f"No scene file known for scenario {scenario_id}; provide --scene_dir or --scene_manifest."
        )
    scene_file = Path(scene_dir) / f"{scenario_id}.npz"
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing scene file for {scenario_id}: {scene_file}")
    return str(scene_file)


def _read_scene_manifest(scene_manifest: Optional[str], scene_dir: Optional[str]) -> Dict[str, str]:
    if not scene_manifest:
        return {}
    manifest_path = Path(scene_manifest)
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    entries = json.loads(text) if manifest_path.suffix.lower() == ".json" else text.splitlines()
    out: Dict[str, str] = {}
    for entry in entries:
        entry = str(entry).strip()
        if not entry:
            continue
        path = Path(entry)
        if not path.is_absolute():
            if scene_dir is None:
                raise ValueError("Relative scene manifest entries require --scene_dir")
            path = Path(scene_dir) / path
        out[path.stem] = str(path)
    return out


def _gt_errors(traj: np.ndarray, gt_future: np.ndarray | None) -> Dict[str, float]:
    if gt_future is None or len(gt_future) == 0:
        return {"ade": 0.0, "fde": 0.0}
    t = min(traj.shape[0], gt_future.shape[0])
    if t <= 0:
        return {"ade": 0.0, "fde": 0.0}
    pos_err = np.linalg.norm(traj[:t, :2] - gt_future[:t, :2], axis=-1)
    return {"ade": float(pos_err.mean()), "fde": float(pos_err[-1])}


def score_candidate_file(
    candidate_path: Path,
    *,
    scene_file: str,
    output_dir: Path,
    weights: AnchorCandidateScoreWeights,
    collision_dist: float,
) -> Dict[str, object]:
    bundle = _load_npz(candidate_path)
    scene_data = _load_npz(Path(scene_file))

    candidates = np.asarray(bundle["candidates"])
    anchor_trajs = np.asarray(bundle["anchor_trajs"])
    anchor_indices = np.asarray(bundle["anchor_indices"], dtype=np.int64)
    anchor_ranks = np.asarray(bundle["anchor_ranks"], dtype=np.int64)
    sample_ids = np.asarray(bundle["sample_ids"], dtype=np.int64)

    if not (
        len(candidates)
        == len(anchor_trajs)
        == len(anchor_indices)
        == len(anchor_ranks)
        == len(sample_ids)
    ):
        raise ValueError(f"Candidate arrays have inconsistent lengths in {candidate_path}")

    scenario_id = _candidate_scenario_id(candidate_path)
    candidate_infos: List[Dict[str, object]] = []
    for candidate_idx, traj in enumerate(candidates):
        metrics = evaluate_trajectory(
            traj,
            neighbor_future_gt=scene_data["neighbor_agents_future"],
            route_lanes=scene_data.get("route_lanes"),
            collision_dist=collision_dist,
        )
        metrics = {**metrics, **_gt_errors(traj, scene_data.get("ego_agent_future"))}
        candidate_infos.append(
            build_candidate_record(
                candidate_idx=candidate_idx,
                anchor_index=int(anchor_indices[candidate_idx]),
                anchor_rank=int(anchor_ranks[candidate_idx]),
                sample_i=int(sample_ids[candidate_idx]),
                seed=-1,
                metrics=metrics,
                weights=weights,
            )
        )

    payload = {
        "scenario_id": scenario_id,
        "scene_file": scene_file,
        "source_npz": str(candidate_path),
        "candidates": candidate_infos,
        "score_config": weights.to_dict(),
        "scene_stats": summarize_scene(candidate_infos),
        "anchor_group_stats": json_ready_anchor_summaries(
            summarize_anchor_groups(candidate_infos, method="mean")
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scenario_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def score_anchor_candidates(args: argparse.Namespace) -> Dict[str, object]:
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    candidate_paths = sorted(candidate_dir.glob("*_candidates.npz"))
    if args.max_scenes is not None:
        candidate_paths = candidate_paths[: args.max_scenes]

    weights = AnchorCandidateScoreWeights.from_args(args)
    scene_file_by_id = _read_scene_manifest(args.scene_manifest, args.scene_dir)
    failures: List[Dict[str, str]] = []
    written = 0
    total_candidates = 0

    for idx, candidate_path in enumerate(candidate_paths):
        scenario_id = _candidate_scenario_id(candidate_path)
        try:
            scene_file = _resolve_scene_file(
                scenario_id,
                scene_dir=args.scene_dir,
                scene_file_by_id=scene_file_by_id,
            )
            payload = score_candidate_file(
                candidate_path,
                scene_file=scene_file,
                output_dir=output_dir,
                weights=weights,
                collision_dist=args.collision_dist,
            )
            written += 1
            total_candidates += len(payload["candidates"])
            if (idx + 1) % args.log_every == 0:
                logger.info(
                    "[%d/%d] written_scenes=%d candidates=%d failures=%d",
                    idx + 1,
                    len(candidate_paths),
                    written,
                    total_candidates,
                    len(failures),
                )
        except Exception as exc:  # noqa: BLE001 - keep batch scoring alive.
            failures.append({"candidate_file": str(candidate_path), "error": str(exc)})
            logger.warning("Failed scoring %s: %s", candidate_path, exc)

    meta = {
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "scene_dir": args.scene_dir,
        "scene_manifest": args.scene_manifest,
        "max_scenes": args.max_scenes,
        "written_scenes": written,
        "total_candidates": total_candidates,
        "score_config": weights.to_dict(),
        "failures": failures,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved anchor candidate NPZ files")
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--scene_manifest", default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
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
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    meta = score_anchor_candidates(parse_args())
    logger.info("Done: %s", meta)


if __name__ == "__main__":
    main()

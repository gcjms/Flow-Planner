#!/usr/bin/env python3
"""Recompute soft-preference candidate JSON scores from stored metrics.

This script intentionally reuses previously generated candidate NPZs and
per-candidate metrics, so we can isolate teacher-score logic fixes from
trajectory sampling noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def teacher_score(metrics: Dict[str, Any], score_cfg: Dict[str, Any]) -> float:
    safe = 1.0 - float(metrics["collided"])
    return (
        float(score_cfg.get("safety_weight", 5.0)) * safe
        + float(score_cfg.get("route_weight", 1.0)) * float(metrics["route_score"])
        + float(score_cfg.get("progress_weight", 1.0)) * float(metrics["progress_score"])
        + float(score_cfg.get("ttc_weight", 0.1)) * float(metrics["ttc_score"])
        + float(score_cfg.get("comfort_weight", 0.05)) * float(metrics["comfort_score"])
        + float(score_cfg.get("collision_weight", 0.05)) * float(metrics["collision_score"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Old candidate root with scored_dir/meta.json.")
    parser.add_argument("--output-root", required=True, help="New output root for rescored JSON/meta.")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_scored = source_root / "scored_dir"
    output_scored = output_root / "scored_dir"
    output_scored.mkdir(parents=True, exist_ok=True)

    meta_path = source_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing source meta.json: {meta_path}")
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    json_paths = sorted(source_scored.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"no json files found in {source_scored}")

    written = 0
    changed = 0
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        score_cfg = payload.get("score_config", {})
        for candidate in payload.get("candidates", []):
            old_score = float(candidate.get("total_score", 0.0))
            new_score = teacher_score(candidate["metrics"], score_cfg)
            candidate["total_score"] = float(new_score)
            if abs(new_score - old_score) > 1e-8:
                changed += 1
        payload["rescore_source_root"] = str(source_root)
        payload["rescore_note"] = (
            "Recomputed total_score from stored metrics after collision_score "
            "teacher-sign fix; candidate trajectories were reused unchanged."
        )
        (output_scored / json_path.name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written += 1

    output_meta = {
        "source_root": str(source_root),
        "source_scored_dir": str(source_scored),
        "source_candidates_dir": source_meta.get("candidates_dir"),
        "output_root": str(output_root),
        "output_scored_dir": str(output_scored),
        "scene_dir": source_meta.get("scene_dir"),
        "scene_manifest": source_meta.get("scene_manifest"),
        "max_scenes": source_meta.get("max_scenes"),
        "top_k": source_meta.get("top_k"),
        "samples_per_anchor": source_meta.get("samples_per_anchor"),
        "num_input_json": len(json_paths),
        "written_scenes": written,
        "rescored_candidates": changed,
        "note": (
            "Only scored_dir JSON files were regenerated. Existing candidate NPZs "
            "are reused via their original source_npz paths."
        ),
    }
    (output_root / "meta.json").write_text(
        json.dumps(output_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

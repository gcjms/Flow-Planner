#!/usr/bin/env python3
"""Build a tick-level closed-loop intervention manifest from selector trace.

The manifest is consumed by
``anchor_mode=predicted_anchor_candidate_selector_intervention``. Each selected
timestamp forces the planner to execute one candidate while all other ticks use
the unconditioned baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TIMESTAMP_METRICS = (
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _load_scene_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("scenes", payload.get("scenario_names", []))
        return {str(item) for item in payload}
    return {line.strip() for line in text.splitlines() if line.strip()}


def _iter_time_series(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, list):
        return [int(v) for v in values]
    try:
        if pd.isna(values):
            return []
    except ValueError:
        pass
    if hasattr(values, "tolist"):
        return [int(v) for v in values.tolist()]
    return []


def _build_timestamp_scene_map(metrics_run: Path | None) -> dict[int, tuple[str, str]]:
    if metrics_run is None:
        return {}
    metrics_dir = metrics_run / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")

    for metric_name in DEFAULT_TIMESTAMP_METRICS:
        path = metrics_dir / f"{metric_name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "time_series_timestamps" not in df.columns:
            continue
        timestamp_map: dict[int, tuple[str, str]] = {}
        for row in df.to_dict("records"):
            scenario_name = str(row.get("scenario_name", ""))
            log_name = str(row.get("log_name", ""))
            for ts in _iter_time_series(row.get("time_series_timestamps")):
                timestamp_map[ts] = (scenario_name, log_name)
        if timestamp_map:
            return timestamp_map

    raise FileNotFoundError(f"no usable time_series_timestamps found in: {metrics_dir}")


def _sort_key(row: dict[str, Any]) -> tuple[str, int]:
    return (
        str(row.get("scenario_name") or row.get("planner_instance_id") or ""),
        int(row.get("iteration_index", -1)),
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_jsonl(args.trace_jsonl)
    scenes = _load_scene_filter(args.scenes_file)
    timestamp_scene_map = _build_timestamp_scene_map(args.metrics_run)
    selected: list[dict[str, Any]] = []
    seen_timestamps: set[int] = set()

    for row in sorted(rows, key=_sort_key):
        timestamp = int(row["iteration_time_us"])
        mapped_scene = timestamp_scene_map.get(timestamp)
        scenario_name = row.get("scenario_name")
        log_name = row.get("log_name")
        if mapped_scene:
            scenario_name, log_name = mapped_scene
        if scenes is not None and str(scenario_name) not in scenes:
            continue
        if args.only_anchor_raw and row.get("raw_best_type") != "anchor":
            continue
        if args.min_iteration is not None and int(row.get("iteration_index", -1)) < args.min_iteration:
            continue
        if args.max_iteration is not None and int(row.get("iteration_index", -1)) > args.max_iteration:
            continue

        if timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)

        if args.force == "raw":
            candidate_type = row.get("raw_best_type")
            anchor_rank = row.get("raw_best_anchor_rank")
            sample_i = row.get("raw_best_sample_i")
            candidate_idx = row.get("raw_best_idx")
        elif args.force == "final":
            candidate_type = row.get("final_type")
            anchor_rank = row.get("final_anchor_rank")
            sample_i = row.get("final_sample_i")
            candidate_idx = row.get("final_idx")
        else:
            candidate_type = "unconditioned"
            anchor_rank = None
            sample_i = None
            candidate_idx = 0

        selected.append(
            {
                "scenario_name": scenario_name,
                "log_name": log_name,
                "iteration_index": int(row.get("iteration_index", -1)),
                "iteration_time_us": timestamp,
                "type": candidate_type,
                "anchor_rank": anchor_rank,
                "sample_i": sample_i,
                "candidate_idx": candidate_idx,
                "source_trace": str(args.trace_jsonl),
                "source_planner_instance_id": row.get("planner_instance_id"),
            }
        )
        if args.limit and len(selected) >= args.limit:
            break

    return {
        "schema_version": 1,
        "description": (
            "Closed-loop intervention manifest. Non-listed ticks should use "
            "baseline/unconditioned planning; listed ticks force the selected "
            "candidate for official rollout outcome collection."
        ),
        "source_trace": str(args.trace_jsonl),
        "force": args.force,
        "num_interventions": len(selected),
        "interventions": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-run", type=Path, default=None)
    parser.add_argument("--scenes-file", type=Path, default=None)
    parser.add_argument("--force", choices=("raw", "final", "baseline"), default="raw")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-iteration", type=int, default=None)
    parser.add_argument("--max-iteration", type=int, default=None)
    parser.add_argument("--only-anchor-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_interventions": manifest["num_interventions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

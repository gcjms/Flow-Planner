#!/usr/bin/env python3
"""Summarize candidate-selector JSONL trace by nuPlan scenario."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TIMESTAMP_METRICS = (
    "ego_progress_along_expert_route",
    "time_to_collision_within_bound",
    "driving_direction_compliance",
)


def _load_trace_rows(trace_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with trace_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {trace_jsonl}:{line_no}: {exc}") from exc
    return rows


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


def _build_timestamp_scene_map(metrics_run: Path) -> dict[int, tuple[str, str]]:
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


def _group_by_planner(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        planner_id = str(row.get("planner_instance_id", "unknown"))
        grouped[planner_id].append(row)
    for planner_rows in grouped.values():
        planner_rows.sort(key=lambda row: (int(row.get("iteration_index", -1)), int(row.get("iteration_time_us", -1))))
    return grouped


def _map_planners_to_scenes(
    grouped: dict[str, list[dict[str, Any]]],
    timestamp_scene_map: dict[int, tuple[str, str]],
) -> dict[str, tuple[str | None, str | None]]:
    mapping: dict[str, tuple[str | None, str | None]] = {}
    for planner_id, rows in grouped.items():
        votes: Counter[tuple[str, str]] = Counter()
        for row in rows:
            ts = row.get("iteration_time_us")
            if ts is None:
                continue
            scene = timestamp_scene_map.get(int(ts))
            if scene:
                votes[scene] += 1
        if votes:
            mapping[planner_id] = votes.most_common(1)[0][0]
        else:
            mapping[planner_id] = (None, None)
    return mapping


def _rank_key(row: dict[str, Any], prefix: str) -> str:
    rank = row.get(f"{prefix}_anchor_rank")
    if rank is None:
        return "unconditioned"
    return str(int(rank))


def _longest_run(indices: list[int]) -> int:
    if not indices:
        return 0
    indices = sorted(indices)
    best = 1
    cur = 1
    for prev, now in zip(indices, indices[1:]):
        if now == prev + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _summarize_group(
    planner_id: str,
    rows: list[dict[str, Any]],
    scenario_name: str | None,
    log_name: str | None,
) -> dict[str, Any]:
    ticks = len(rows)
    raw_anchor_rows = [row for row in rows if row.get("raw_best_type") == "anchor"]
    final_anchor_rows = [row for row in rows if row.get("final_type") == "anchor"]
    fallback_ticks = sum(1 for row in rows if bool(row.get("fallback_triggered")))
    final_anchor_iters = [int(row.get("iteration_index", -1)) for row in final_anchor_rows]
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("gate_reasons") or []:
            reason_counts[str(reason)] += 1

    return {
        "scenario_name": scenario_name,
        "log_name": log_name,
        "planner_instance_id": planner_id,
        "ticks": ticks,
        "raw_anchor_ticks": len(raw_anchor_rows),
        "raw_anchor_rate": len(raw_anchor_rows) / ticks if ticks else 0.0,
        "final_anchor_ticks": len(final_anchor_rows),
        "final_anchor_rate": len(final_anchor_rows) / ticks if ticks else 0.0,
        "fallback_ticks": fallback_ticks,
        "fallback_rate": fallback_ticks / ticks if ticks else 0.0,
        "first_final_anchor_iter": min(final_anchor_iters) if final_anchor_iters else None,
        "last_final_anchor_iter": max(final_anchor_iters) if final_anchor_iters else None,
        "longest_final_anchor_run": _longest_run(final_anchor_iters),
        "raw_rank_counts": dict(Counter(_rank_key(row, "raw_best") for row in rows)),
        "final_rank_counts": dict(Counter(_rank_key(row, "final") for row in rows)),
        "top_gate_reasons": reason_counts.most_common(),
    }


def _load_scene_delta(scene_delta_csv: Path | None) -> dict[str, dict[str, Any]]:
    if scene_delta_csv is None or not scene_delta_csv.exists():
        return {}
    with scene_delta_csv.open("r", encoding="utf-8") as f:
        return {row["scenario_name"]: row for row in csv.DictReader(f) if row.get("scenario_name")}


def _delta_value(row: dict[str, Any], metric_name: str) -> Any:
    for key in (f"delta_{metric_name}", f"delta.{metric_name}"):
        if key in row:
            return row[key]
    return None


def _merge_deltas(summary: dict[str, Any], scene_delta: dict[str, dict[str, Any]]) -> None:
    scenario_name = summary.get("scenario_name")
    if not scenario_name or scenario_name not in scene_delta:
        return
    row = scene_delta[scenario_name]
    for metric_name in (
        "no_ego_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_is_making_progress",
        "driving_direction_compliance",
        "ego_is_comfortable",
        "ego_progress_along_expert_route",
        "runtime_mean",
    ):
        value = _delta_value(row, metric_name)
        if value is not None:
            summary[f"delta_{metric_name}"] = _coerce_number(value)


def _coerce_number(value: Any) -> Any:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario_name",
        "log_name",
        "planner_instance_id",
        "ticks",
        "raw_anchor_ticks",
        "raw_anchor_rate",
        "final_anchor_ticks",
        "final_anchor_rate",
        "fallback_ticks",
        "fallback_rate",
        "first_final_anchor_iter",
        "last_final_anchor_iter",
        "longest_final_anchor_run",
        "raw_rank_counts",
        "final_rank_counts",
        "top_gate_reasons",
        "delta_no_ego_at_fault_collisions",
        "delta_ego_progress_along_expert_route",
        "delta_runtime_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            for key in ("raw_rank_counts", "final_rank_counts", "top_gate_reasons"):
                row[key] = _json_dumps(row.get(key))
            writer.writerow(row)


def _write_tick_csv(path: Path, rows: list[dict[str, Any]], planner_scene: dict[str, tuple[str | None, str | None]]) -> None:
    fieldnames = [
        "scenario_name",
        "log_name",
        "planner_instance_id",
        "iteration_index",
        "iteration_time_us",
        "raw_best_type",
        "raw_best_anchor_rank",
        "raw_best_sample_i",
        "final_type",
        "final_anchor_rank",
        "final_sample_i",
        "fallback_triggered",
        "gate_reasons",
        "raw_best_logit",
        "unconditioned_logit",
        "raw_best_rule_score",
        "unconditioned_rule_score",
        "raw_best_collision_score",
        "unconditioned_collision_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r.get("planner_instance_id")), int(r.get("iteration_index", -1)))):
            out = {key: row.get(key) for key in fieldnames}
            scenario_name, log_name = planner_scene.get(str(row.get("planner_instance_id")), (None, None))
            out["scenario_name"] = scenario_name
            out["log_name"] = log_name
            out["gate_reasons"] = _json_dumps(row.get("gate_reasons") or [])
            writer.writerow(out)


def _focus_case(
    focus_scene: str,
    summaries: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    planner_scene: dict[str, tuple[str | None, str | None]],
    last_n_ticks: int,
) -> dict[str, Any]:
    summary = next((row for row in summaries if row.get("scenario_name") == focus_scene), None)
    if summary is None:
        return {"focus_scene": focus_scene, "found": False}
    planner_id = str(summary["planner_instance_id"])
    rows = grouped.get(planner_id, [])
    tail_rows = []
    for row in rows[-last_n_ticks:]:
        tail_rows.append(
            {
                "iter": row.get("iteration_index"),
                "raw_type": row.get("raw_best_type"),
                "raw_rank": row.get("raw_best_anchor_rank"),
                "raw_sample": row.get("raw_best_sample_i"),
                "final_type": row.get("final_type"),
                "final_rank": row.get("final_anchor_rank"),
                "final_sample": row.get("final_sample_i"),
                "fallback": row.get("fallback_triggered"),
                "gate_reasons": row.get("gate_reasons") or [],
                "raw_logit": row.get("raw_best_logit"),
                "unconditioned_logit": row.get("unconditioned_logit"),
                "raw_rule_score": row.get("raw_best_rule_score"),
                "unconditioned_rule_score": row.get("unconditioned_rule_score"),
                "raw_collision_score": row.get("raw_best_collision_score"),
                "unconditioned_collision_score": row.get("unconditioned_collision_score"),
            }
        )
    final_iters = [
        int(row.get("iteration_index", -1))
        for row in rows
        if row.get("final_type") == "anchor"
    ]
    return {
        "focus_scene": focus_scene,
        "found": True,
        "summary": summary,
        "planner_scene": planner_scene.get(planner_id),
        "final_anchor_iters": final_iters,
        "last_ticks": tail_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", required=True, type=Path)
    parser.add_argument("--metrics-run", required=True, type=Path, help="Official eval run dir used to map trace timestamps to scenes")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene-delta-csv", type=Path, default=None)
    parser.add_argument("--focus-scene", default="71e4ce1d08e85a3c")
    parser.add_argument("--focus-last-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_trace_rows(args.trace_jsonl)
    grouped = _group_by_planner(rows)
    timestamp_scene_map = _build_timestamp_scene_map(args.metrics_run)
    planner_scene = _map_planners_to_scenes(grouped, timestamp_scene_map)
    scene_delta = _load_scene_delta(args.scene_delta_csv)

    summaries: list[dict[str, Any]] = []
    for planner_id, planner_rows in grouped.items():
        scenario_name, log_name = planner_scene.get(planner_id, (None, None))
        summary = _summarize_group(planner_id, planner_rows, scenario_name, log_name)
        _merge_deltas(summary, scene_delta)
        summaries.append(summary)
    summaries.sort(key=lambda row: (str(row.get("log_name")), str(row.get("scenario_name"))))

    summary_csv = args.output_dir / "candidate_trace_per_scene_summary.csv"
    summary_json = args.output_dir / "candidate_trace_per_scene_summary.json"
    ticks_csv = args.output_dir / "candidate_trace_ticks.csv"
    focus_json = args.output_dir / f"focus_scene_{args.focus_scene}.json"

    _write_csv(summary_csv, summaries)
    summary_json.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    _write_tick_csv(ticks_csv, rows, planner_scene)
    focus_json.write_text(
        json.dumps(
            _focus_case(args.focus_scene, summaries, grouped, planner_scene, args.focus_last_n),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    unmapped = sum(1 for scene, _ in planner_scene.values() if scene is None)
    print(summary_csv)
    print(summary_json)
    print(ticks_csv)
    print(focus_json)
    print(f"planner_instances={len(grouped)} rows={len(rows)} unmapped={unmapped}")


if __name__ == "__main__":
    main()

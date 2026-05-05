#!/usr/bin/env python3
"""Summarize single-tick closed-loop intervention rollouts.

Each intervention rollout forces one selected anchor candidate at one tick and
uses the unconditioned baseline for all other ticks. This script compares each
rollout against the matching baseline scene and emits a flat table that can be
used as the first closed-loop accept/reject label source.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.anchor.summarize_official_eval import NR_WEIGHTS


DEFAULT_CRITICAL_METRICS = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
)


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_trace_row(trace_path: Path, intervention: dict[str, Any]) -> dict[str, Any]:
    if not trace_path.exists():
        return {}
    target_ts = intervention.get("iteration_time_us")
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {trace_path}:{line_no}: {exc}") from exc
            if target_ts is not None and row.get("iteration_time_us") == target_ts:
                return row
            rows.append(row)
    return rows[0] if len(rows) == 1 else {}


def _load_scene_table(run_dir: Path) -> pd.DataFrame:
    report_path = run_dir / "runner_report.parquet"
    if not report_path.exists():
        raise FileNotFoundError(f"runner report not found: {report_path}")

    runner = pd.read_parquet(report_path)
    keep = [
        "scenario_name",
        "log_name",
        "succeeded",
        "error_message",
        "compute_trajectory_runtimes_mean",
        "compute_trajectory_runtimes_median",
        "compute_trajectory_runtimes_std",
        "duration",
    ]
    table = runner[[col for col in keep if col in runner.columns]].copy()

    for metric_name in NR_WEIGHTS:
        metric_path = run_dir / "metrics" / f"{metric_name}.parquet"
        if not metric_path.exists():
            continue
        metric_df = pd.read_parquet(metric_path)
        if "scenario_name" not in metric_df.columns or "metric_score" not in metric_df.columns:
            continue
        metric_df = metric_df[["scenario_name", "metric_score"]].copy()
        metric_df[metric_name] = metric_df["metric_score"].map(_to_float)
        metric_df = metric_df.drop(columns=["metric_score"])
        table = table.merge(metric_df, on="scenario_name", how="outer")

    table["weighted_score"] = table.apply(_weighted_score_for_row, axis=1)
    return table


def _weighted_score_for_row(row: pd.Series) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for metric_name, weight in NR_WEIGHTS.items():
        value = _to_float(row.get(metric_name))
        if math.isnan(value):
            continue
        weighted_sum += weight * value
        total_weight += weight
    return weighted_sum / total_weight * 100.0 if total_weight else math.nan


def _index_by_scene(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["scenario_name"]): row for row in table.to_dict("records") if row.get("scenario_name")}


def _find_intervention_runs(run_root: Path, batch_name: str) -> list[Path]:
    runs = [
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and path.name.startswith(f"{batch_name}_")
        and (path / "runner_report.parquet").exists()
    ]
    return sorted(runs, key=lambda path: path.name)


def _first_intervention(manifest_path: Path) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    interventions = payload.get("interventions") or []
    if len(interventions) != 1:
        raise ValueError(f"expected exactly one intervention in {manifest_path}, got {len(interventions)}")
    intervention = interventions[0]
    if not isinstance(intervention, dict):
        raise ValueError(f"invalid intervention object in {manifest_path}")
    return intervention


def _same_candidate(manifest: dict[str, Any], trace: dict[str, Any]) -> bool:
    if not trace:
        return False
    manifest_type = manifest.get("type")
    if manifest_type == "raw_best_anchor":
        logits = trace.get("logits") or []
        candidate_meta = trace.get("candidate_meta") or []
        best_idx: int | None = None
        best_logit: float | None = None
        for idx, meta in enumerate(candidate_meta):
            if not isinstance(meta, dict) or meta.get("type") != "anchor":
                continue
            if idx >= len(logits):
                continue
            logit = _to_float(logits[idx])
            if math.isnan(logit):
                continue
            if best_logit is None or logit > best_logit:
                best_idx = idx
                best_logit = logit
        return best_idx is not None and _nullable_int(trace.get("final_idx")) == best_idx
    if manifest_type != trace.get("final_type"):
        return False
    if manifest_type == "unconditioned":
        return True
    return (
        _nullable_int(manifest.get("anchor_rank")) == _nullable_int(trace.get("final_anchor_rank"))
        and _nullable_int(manifest.get("sample_i")) == _nullable_int(trace.get("final_sample_i"))
    )


def _nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], metric_name: str) -> float:
    return _to_float(candidate.get(metric_name)) - _to_float(baseline.get(metric_name))


def _make_label(
    row: dict[str, Any],
    *,
    critical_metrics: tuple[str, ...],
    min_progress_delta: float,
    min_weighted_delta: float,
    epsilon: float,
) -> tuple[str, str]:
    if not row.get("candidate_succeeded"):
        return "invalid", "candidate_failed"
    if not row.get("forced_matches_manifest"):
        return "invalid", "forced_candidate_mismatch"

    for metric_name in critical_metrics:
        delta = _to_float(row.get(f"delta_{metric_name}"))
        if math.isnan(delta):
            return "invalid", f"missing_{metric_name}"
        if delta < -epsilon:
            return "reject", f"critical_regression_{metric_name}"

    progress_delta = _to_float(row.get("delta_ego_progress_along_expert_route"))
    if not math.isnan(progress_delta) and progress_delta < min_progress_delta:
        return "reject", "progress_regression"

    weighted_delta = _to_float(row.get("delta_weighted_score"))
    if not math.isnan(weighted_delta) and weighted_delta < min_weighted_delta:
        return "reject", "weighted_regression"

    return "accept", "passes_default_policy"


def _build_row(
    run_dir: Path,
    intervention: dict[str, Any],
    trace: dict[str, Any],
    candidate_row: dict[str, Any],
    baseline_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_name": run_dir.name,
        "run_dir": str(run_dir),
        "scenario_name": intervention.get("scenario_name"),
        "log_name": intervention.get("log_name") or candidate_row.get("log_name"),
        "iteration_index": intervention.get("iteration_index"),
        "iteration_time_us": intervention.get("iteration_time_us"),
        "forced_type": intervention.get("type"),
        "forced_anchor_rank": intervention.get("anchor_rank"),
        "forced_sample_i": intervention.get("sample_i"),
        "forced_candidate_idx_source": intervention.get("candidate_idx"),
        "candidate_succeeded": bool(candidate_row.get("succeeded", False)),
        "candidate_error_message": candidate_row.get("error_message"),
        "forced_matches_manifest": _same_candidate(intervention, trace),
        "trace_final_type": trace.get("final_type"),
        "trace_final_anchor_rank": trace.get("final_anchor_rank"),
        "trace_final_sample_i": trace.get("final_sample_i"),
        "trace_final_idx": trace.get("final_idx"),
        "trace_raw_best_type": trace.get("raw_best_type"),
        "trace_raw_best_anchor_rank": trace.get("raw_best_anchor_rank"),
        "trace_raw_best_sample_i": trace.get("raw_best_sample_i"),
        "trace_raw_best_idx": trace.get("raw_best_idx"),
        "trace_raw_best_logit": trace.get("raw_best_logit"),
        "trace_unconditioned_logit": trace.get("unconditioned_logit"),
        "trace_gate_reasons": json.dumps(trace.get("gate_reasons") or [], sort_keys=True),
        "baseline_runtime_mean": baseline_row.get("compute_trajectory_runtimes_mean"),
        "candidate_runtime_mean": candidate_row.get("compute_trajectory_runtimes_mean"),
        "delta_runtime_mean": _to_float(candidate_row.get("compute_trajectory_runtimes_mean"))
        - _to_float(baseline_row.get("compute_trajectory_runtimes_mean")),
        "baseline_weighted_score": baseline_row.get("weighted_score"),
        "candidate_weighted_score": candidate_row.get("weighted_score"),
        "delta_weighted_score": _to_float(candidate_row.get("weighted_score"))
        - _to_float(baseline_row.get("weighted_score")),
    }

    for metric_name in NR_WEIGHTS:
        row[f"baseline_{metric_name}"] = baseline_row.get(metric_name)
        row[f"candidate_{metric_name}"] = candidate_row.get(metric_name)
        row[f"delta_{metric_name}"] = _metric_delta(candidate_row, baseline_row, metric_name)

    label, reason = _make_label(
        row,
        critical_metrics=tuple(args.critical_metrics),
        min_progress_delta=args.min_progress_delta,
        min_weighted_delta=args.min_weighted_delta,
        epsilon=args.epsilon,
    )
    row["closed_loop_label"] = label
    row["label_reason"] = reason
    return {key: _jsonable(value) for key, value in row.items()}


def _write_text_summary(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        f"runs {summary['num_runs']}",
        f"valid {summary['num_valid']} accept {summary['num_accept']} reject {summary['num_reject']} invalid {summary['num_invalid']}",
        f"mean_delta_weighted_score {summary['mean_delta_weighted_score']}",
        f"mean_delta_progress {summary['mean_delta_progress']}",
        f"label_reasons {json.dumps(summary['label_reason_counts'], sort_keys=True)}",
        "",
        "top_rows",
    ]
    for row in rows[:20]:
        lines.append(
            "  "
            f"{row['experiment_name']} "
            f"{row['closed_loop_label']} "
            f"{row['label_reason']} "
            f"dw={_fmt(row.get('delta_weighted_score'))} "
            f"dp={_fmt(row.get('delta_ego_progress_along_expert_route'))} "
            f"coll={_fmt(row.get('delta_no_ego_at_fault_collisions'))} "
            f"drv={_fmt(row.get('delta_drivable_area_compliance'))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = _to_float(value)
    return "nan" if math.isnan(number) else f"{number:+.4f}"


def _safe_mean(values: list[Any]) -> float | None:
    numeric = [_to_float(value) for value in values]
    numeric = [value for value in numeric if not math.isnan(value)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-progress-delta", type=float, default=-0.02)
    parser.add_argument("--min-weighted-delta", type=float, default=-2.0)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument(
        "--critical-metrics",
        default=",".join(DEFAULT_CRITICAL_METRICS),
        help="Comma-separated metrics that must not regress for an accept label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.critical_metrics = tuple(item.strip() for item in args.critical_metrics.split(",") if item.strip())
    output_dir = args.output_dir or args.run_root / f"{args.batch_name}_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_by_scene = _index_by_scene(_load_scene_table(args.baseline_run))
    rows: list[dict[str, Any]] = []
    for run_dir in _find_intervention_runs(args.run_root, args.batch_name):
        manifest_path = args.run_root / f"{args.batch_name}_manifests" / f"{run_dir.name}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found for {run_dir.name}: {manifest_path}")
        intervention = _first_intervention(manifest_path)
        scenario_name = str(intervention.get("scenario_name"))
        baseline_row = baseline_by_scene.get(scenario_name)
        if baseline_row is None:
            raise KeyError(f"baseline scene not found: {scenario_name}")
        candidate_table = _load_scene_table(run_dir)
        candidate_by_scene = _index_by_scene(candidate_table)
        candidate_row = candidate_by_scene.get(scenario_name)
        if candidate_row is None:
            raise KeyError(f"candidate scene not found in {run_dir}: {scenario_name}")
        trace = _load_trace_row(args.run_root / f"{run_dir.name}_trace.jsonl", intervention)
        rows.append(_build_row(run_dir, intervention, trace, candidate_row, baseline_row, args))

    rows = sorted(rows, key=lambda row: str(row["experiment_name"]))
    label_counts = Counter(str(row["closed_loop_label"]) for row in rows)
    reason_counts = Counter(str(row["label_reason"]) for row in rows)
    summary = {
        "run_root": str(args.run_root),
        "batch_name": args.batch_name,
        "baseline_run": str(args.baseline_run),
        "num_runs": len(rows),
        "num_valid": int(label_counts.get("accept", 0) + label_counts.get("reject", 0)),
        "num_accept": int(label_counts.get("accept", 0)),
        "num_reject": int(label_counts.get("reject", 0)),
        "num_invalid": int(label_counts.get("invalid", 0)),
        "label_counts": dict(label_counts),
        "label_reason_counts": dict(reason_counts),
        "critical_metrics": list(args.critical_metrics),
        "min_progress_delta": args.min_progress_delta,
        "min_weighted_delta": args.min_weighted_delta,
        "mean_delta_weighted_score": _safe_mean([row.get("delta_weighted_score") for row in rows]),
        "mean_delta_progress": _safe_mean([row.get("delta_ego_progress_along_expert_route") for row in rows]),
    }

    csv_path = output_dir / "closed_loop_intervention_labels.csv"
    json_path = output_dir / "closed_loop_intervention_summary.json"
    txt_path = output_dir / "closed_loop_intervention_summary.txt"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_text_summary(txt_path, rows, summary)

    print(txt_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

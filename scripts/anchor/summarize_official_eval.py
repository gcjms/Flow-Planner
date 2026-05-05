#!/usr/bin/env python3
"""Summarize nuPlan official closed-loop eval runs at per-scene level."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


NR_WEIGHTS = {
    "no_ego_at_fault_collisions": 5,
    "drivable_area_compliance": 5,
    "driving_direction_compliance": 5,
    "ego_is_comfortable": 2,
    "ego_is_making_progress": 5,
    "ego_progress_along_expert_route": 5,
    "time_to_collision_within_bound": 5,
    "speed_limit_compliance": 2,
}


def _read_runner_report(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "runner_report.parquet"
    if not path.exists():
        raise FileNotFoundError(f"runner report not found: {path}")
    df = pd.read_parquet(path)
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
    return df[[col for col in keep if col in df.columns]].copy()


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_metric_scores(run_dir: Path, metric_name: str) -> pd.DataFrame | None:
    path = run_dir / "metrics" / f"{metric_name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    keep = ["scenario_name", "log_name", "scenario_type", "metric_score"]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].copy()
    out[metric_name] = out["metric_score"].map(_to_float)
    drop_cols = ["metric_score"]
    return out.drop(columns=[col for col in drop_cols if col in out.columns])


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runner = _read_runner_report(run_dir)
    metrics: dict[str, pd.DataFrame] = {}
    for metric_name in NR_WEIGHTS:
        metric_df = _read_metric_scores(run_dir, metric_name)
        if metric_df is not None:
            metrics[metric_name] = metric_df
    if not metrics:
        raise FileNotFoundError(f"no supported metric parquet files found under: {run_dir / 'metrics'}")
    return runner, metrics


def _build_scene_table(run_dir: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    runner, metrics = _load_run(run_dir)
    table = runner.copy()
    for metric_name, metric_df in metrics.items():
        merge_cols = ["scenario_name"]
        value_cols = ["scenario_name", metric_name]
        if "scenario_type" in metric_df.columns and "scenario_type" not in table.columns:
            value_cols.append("scenario_type")
        if "log_name" in metric_df.columns and "log_name" not in table.columns:
            value_cols.append("log_name")
        table = table.merge(metric_df[value_cols], on=merge_cols, how="outer")

    summary = _summarize_single_run(table, metrics.keys())
    renamed = table.rename(columns={col: f"{col}_{label}" for col in table.columns if col != "scenario_name"})
    return renamed, summary


def _summarize_single_run(table: pd.DataFrame, metric_names: Any) -> dict[str, Any]:
    metric_names = [name for name in metric_names if name in table.columns]
    summary: dict[str, Any] = {
        "num_scenes": int(len(table)),
        "num_succeeded": int(table["succeeded"].fillna(False).astype(bool).sum()) if "succeeded" in table else None,
        "runtime_mean": _safe_mean(table.get("compute_trajectory_runtimes_mean")),
        "runtime_median": _safe_mean(table.get("compute_trajectory_runtimes_median")),
        "duration_mean": _safe_mean(table.get("duration")),
        "metrics": {},
    }

    weighted_sum = 0.0
    total_weight = 0
    product = 1.0
    for metric_name in metric_names:
        series = pd.to_numeric(table[metric_name], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        mean = float(valid.mean())
        pass_count = int((valid >= 1.0).sum())
        n = int(valid.shape[0])
        summary["metrics"][metric_name] = {
            "mean": mean,
            "pass_count": pass_count,
            "n": n,
            "pass_rate": pass_count / n if n else math.nan,
        }
        weight = NR_WEIGHTS.get(metric_name)
        if weight is not None:
            weighted_sum += weight * mean
            total_weight += weight
        product *= mean
    summary["weighted_score"] = weighted_sum / total_weight * 100.0 if total_weight else math.nan
    summary["product_score"] = product * 100.0 if summary["metrics"] else math.nan
    return summary


def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _merge_trace_summary(scene_delta: pd.DataFrame, trace_summary_csv: Path | None) -> pd.DataFrame:
    if trace_summary_csv is None or not trace_summary_csv.exists():
        return scene_delta
    trace = pd.read_csv(trace_summary_csv)
    if "scenario_name" not in trace.columns:
        raise ValueError(f"trace summary is missing scenario_name: {trace_summary_csv}")
    trace = trace.rename(columns={col: f"trace_{col}" for col in trace.columns if col != "scenario_name"})
    return scene_delta.merge(trace, on="scenario_name", how="left")


def _make_scene_delta(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
    trace_summary_csv: Path | None,
) -> pd.DataFrame:
    scene_delta = baseline.merge(candidate, on="scenario_name", how="outer")
    for metric_name in NR_WEIGHTS:
        base_col = f"{metric_name}_{baseline_label}"
        cand_col = f"{metric_name}_{candidate_label}"
        if base_col in scene_delta.columns and cand_col in scene_delta.columns:
            scene_delta[f"delta_{metric_name}"] = (
                pd.to_numeric(scene_delta[cand_col], errors="coerce")
                - pd.to_numeric(scene_delta[base_col], errors="coerce")
            )

    base_runtime = f"compute_trajectory_runtimes_mean_{baseline_label}"
    cand_runtime = f"compute_trajectory_runtimes_mean_{candidate_label}"
    if base_runtime in scene_delta.columns and cand_runtime in scene_delta.columns:
        scene_delta["delta_runtime_mean"] = (
            pd.to_numeric(scene_delta[cand_runtime], errors="coerce")
            - pd.to_numeric(scene_delta[base_runtime], errors="coerce")
        )
    return _merge_trace_summary(scene_delta, trace_summary_csv)


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _focus_case(scene_delta: pd.DataFrame, focus_scene: str) -> dict[str, Any]:
    rows = scene_delta[scene_delta["scenario_name"] == focus_scene]
    if rows.empty:
        return {"focus_scene": focus_scene, "found": False}
    row = rows.iloc[0].to_dict()
    return {
        "focus_scene": focus_scene,
        "found": True,
        "fields": {str(key): _jsonable(value) for key, value in row.items()},
    }


def _write_summary_text(
    output_path: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    scene_delta: pd.DataFrame,
    focus_scene: str,
) -> None:
    lines: list[str] = []
    lines.extend(_format_run_summary(baseline_label, baseline_summary))
    lines.append("")
    lines.extend(_format_run_summary(candidate_label, candidate_summary))
    lines.append("")
    lines.append(f"COMPARISON {candidate_label} vs {baseline_label}")
    lines.append(f"  weighted_score_delta {candidate_summary['weighted_score'] - baseline_summary['weighted_score']:+.4f}")
    lines.append(f"  product_score_delta {candidate_summary['product_score'] - baseline_summary['product_score']:+.4f}")
    lines.append(f"  runtime_mean_delta {_delta(candidate_summary.get('runtime_mean'), baseline_summary.get('runtime_mean')):+.4f}")
    for metric_name in NR_WEIGHTS:
        base_metric = baseline_summary["metrics"].get(metric_name)
        cand_metric = candidate_summary["metrics"].get(metric_name)
        if not base_metric or not cand_metric:
            continue
        mean_delta = cand_metric["mean"] - base_metric["mean"]
        delta_col = f"delta_{metric_name}"
        regressions = int((pd.to_numeric(scene_delta.get(delta_col), errors="coerce") < 0).sum()) if delta_col in scene_delta else 0
        improvements = int((pd.to_numeric(scene_delta.get(delta_col), errors="coerce") > 0).sum()) if delta_col in scene_delta else 0
        lines.append(
            f"  {metric_name} mean_delta {mean_delta:+.4f} "
            f"regressions {regressions} improvements {improvements}"
        )

    focus = _focus_case(scene_delta, focus_scene)
    lines.append("")
    lines.append(f"FOCUS_SCENE {focus_scene}")
    if not focus["found"]:
        lines.append("  not found")
    else:
        fields = focus["fields"]
        focus_keys = [
            f"log_name_{baseline_label}",
            f"log_name_{candidate_label}",
            "delta_no_ego_at_fault_collisions",
            "delta_drivable_area_compliance",
            "delta_time_to_collision_within_bound",
            "delta_ego_is_making_progress",
            "delta_ego_is_comfortable",
            "delta_ego_progress_along_expert_route",
            "delta_runtime_mean",
            "trace_raw_anchor_rate",
            "trace_final_anchor_rate",
            "trace_fallback_rate",
            "trace_top_gate_reasons",
        ]
        for key in focus_keys:
            if key in fields:
                lines.append(f"  {key} {fields[key]}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_run_summary(label: str, summary: dict[str, Any]) -> list[str]:
    lines = [f"RUN {label}"]
    lines.append(f"  scenes {summary['num_scenes']} success {summary['num_succeeded']}")
    lines.append(f"  runtime_mean {summary['runtime_mean']}")
    lines.append(f"  weighted_score {summary['weighted_score']:.4f}")
    lines.append(f"  product_score {summary['product_score']:.4f}")
    for metric_name in NR_WEIGHTS:
        metric = summary["metrics"].get(metric_name)
        if not metric:
            continue
        lines.append(
            f"  {metric_name} mean {metric['mean']:.4f} "
            f"pass {metric['pass_count']}/{metric['n']} pass_rate {metric['pass_rate']:.4f}"
        )
    return lines


def _delta(candidate: float | None, baseline: float | None) -> float:
    if candidate is None or baseline is None:
        return math.nan
    return candidate - baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path, help="Baseline official eval run directory")
    parser.add_argument("--candidate-run", required=True, type=Path, help="Candidate official eval run directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for summary outputs")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--focus-scene", default="71e4ce1d08e85a3c")
    parser.add_argument("--trace-summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_table, baseline_summary = _build_scene_table(args.baseline_run, args.baseline_label)
    candidate_table, candidate_summary = _build_scene_table(args.candidate_run, args.candidate_label)
    scene_delta = _make_scene_delta(
        baseline_table,
        candidate_table,
        args.baseline_label,
        args.candidate_label,
        args.trace_summary_csv,
    )
    scene_delta = scene_delta.sort_values("scenario_name")

    scene_csv = args.output_dir / "official_eval_scene_delta.csv"
    summary_txt = args.output_dir / "official_eval_summary.txt"
    summary_json = args.output_dir / "official_eval_summary.json"
    focus_json = args.output_dir / f"focus_scene_{args.focus_scene}.json"

    scene_delta.to_csv(scene_csv, index=False)
    summary_payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "focus_scene": args.focus_scene,
        "outputs": {
            "scene_delta_csv": str(scene_csv),
            "summary_txt": str(summary_txt),
            "summary_json": str(summary_json),
            "focus_json": str(focus_json),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    focus_json.write_text(json.dumps(_focus_case(scene_delta, args.focus_scene), indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_text(
        summary_txt,
        args.baseline_label,
        args.candidate_label,
        baseline_summary,
        candidate_summary,
        scene_delta,
        args.focus_scene,
    )

    print(summary_txt)
    print(scene_csv)
    print(focus_json)


if __name__ == "__main__":
    main()

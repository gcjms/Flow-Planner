#!/usr/bin/env python3
"""Create and optionally launch closed-loop intervention runs on AutoDL.

This script is intentionally deployment-oriented: it writes one manifest and
one shell script per intervention so each official rollout produces one clean
label candidate.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.anchor.build_closed_loop_intervention_manifest import (
    _build_timestamp_scene_map,
    _load_scene_filter,
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


def _select_ticks(rows: list[dict[str, Any]], ticks: list[int] | None) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: int(row.get("iteration_index", -1)))
    if not rows:
        return []
    if ticks is None:
        return [rows[0], rows[len(rows) // 2]]

    by_tick = {int(row.get("iteration_index", -1)): row for row in rows}
    selected: list[dict[str, Any]] = []
    for tick in ticks:
        if tick in by_tick:
            selected.append(by_tick[tick])
            continue
        nearest = min(rows, key=lambda row: abs(int(row.get("iteration_index", -1)) - tick))
        selected.append(nearest)
    return selected


def _select_tick_indices(num_ticks: int, ticks: list[int] | None) -> list[int]:
    if num_ticks <= 0:
        return []
    if ticks is None:
        return [0, num_ticks // 2]

    selected: list[int] = []
    for tick in ticks:
        clamped = max(0, min(int(tick), num_ticks - 1))
        selected.append(clamped)
    return selected


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


def _load_metric_tick_rows(metrics_run: Path, scenes: set[str]) -> dict[str, list[dict[str, Any]]]:
    metrics_dir = metrics_run / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics dir not found: {metrics_dir}")

    for metric_name in (
        "ego_progress_along_expert_route",
        "time_to_collision_within_bound",
        "driving_direction_compliance",
    ):
        path = metrics_dir / f"{metric_name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "time_series_timestamps" not in df.columns:
            continue
        rows_by_scene: dict[str, list[dict[str, Any]]] = {}
        for row in df.to_dict("records"):
            scene_name = str(row.get("scenario_name", ""))
            if scene_name not in scenes:
                continue
            timestamps = _iter_time_series(row.get("time_series_timestamps"))
            if not timestamps:
                continue
            log_name = str(row.get("log_name", ""))
            rows_by_scene[scene_name] = [
                {
                    "scenario_name": scene_name,
                    "log_name": log_name,
                    "iteration_index": idx,
                    "iteration_time_us": timestamp,
                }
                for idx, timestamp in enumerate(timestamps)
            ]
        if rows_by_scene:
            return rows_by_scene

    raise FileNotFoundError(f"no usable time_series_timestamps found in: {metrics_dir}")


def _write_run_script(
    path: Path,
    *,
    exp_name: str,
    run_root: Path,
    manifest_path: Path,
    scenario_name: str,
    log_name: str,
    project_root: Path,
    python_bin: Path,
    nuplan_script: Path,
    data_root: Path,
    maps_root: Path,
    planner_config: Path,
    planner_ckpt: Path,
    anchor_vocab: Path,
    anchor_predictor_ckpt: Path,
    candidate_selector_ckpt: Path,
    trace_training_payload: bool,
    scenario_filter: str,
) -> None:
    trace_path = run_root / f"{exp_name}_trace.jsonl"
    run_out = run_root / exp_name
    content = f"""#!/usr/bin/env bash
set -euo pipefail
export NUPLAN_DATA_ROOT={shlex.quote(str(data_root.parent.parent))}
export NUPLAN_MAPS_ROOT={shlex.quote(str(maps_root))}
export PYTHONPATH={shlex.quote(str(project_root))}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
cd {shlex.quote(str(project_root))}
RUN_OUT={shlex.quote(str(run_out))}
TRACE_PATH={shlex.quote(str(trace_path))}
MANIFEST={shlex.quote(str(manifest_path))}
mkdir -p "$RUN_OUT"
rm -f "$TRACE_PATH"
{shlex.quote(str(python_bin))} {shlex.quote(str(nuplan_script))} \\
  +simulation=closed_loop_nonreactive_agents \\
  planner=flow_planner \\
  planner.flow_planner.config_path={shlex.quote(str(planner_config))} \\
  planner.flow_planner.ckpt_path={shlex.quote(str(planner_ckpt))} \\
  planner.flow_planner.use_cfg=true \\
  planner.flow_planner.cfg_weight=1.8 \\
  planner.flow_planner.device=cuda \\
  scenario_builder=nuplan \\
  scenario_builder.data_root={shlex.quote(str(data_root))} \\
  scenario_filter={scenario_filter} \\
  scenario_filter.scenario_tokens=[{scenario_name}] \\
  scenario_filter.log_names=[{log_name}] \\
  scenario_filter.limit_total_scenarios=1 \\
  output_dir="$RUN_OUT" \\
  experiment_name={exp_name} \\
  verbose=false \\
  worker=single_machine_thread_pool \\
  worker.max_workers=1 \\
  enable_simulation_progress_bar=false \\
  exit_on_failure=false \\
  "hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \\
  planner.flow_planner.anchor_vocab_path={shlex.quote(str(anchor_vocab))} \\
  planner.flow_planner.anchor_mode=predicted_anchor_candidate_selector_intervention \\
  planner.flow_planner.anchor_predictor_ckpt={shlex.quote(str(anchor_predictor_ckpt))} \\
  planner.flow_planner.candidate_selector_ckpt={shlex.quote(str(candidate_selector_ckpt))} \\
  planner.flow_planner.anchor_top_k=3 \\
  planner.flow_planner.candidate_samples_per_anchor=3 \\
  planner.flow_planner.candidate_samples_per_anchor_list=[5,2,2] \\
  planner.flow_planner.candidate_intervention_manifest_path="$MANIFEST" \\
  planner.flow_planner.candidate_trace_path="$TRACE_PATH" \\
  planner.flow_planner.candidate_trace_training_payload={str(trace_training_payload).lower()} \\
  2>&1 | tee "$RUN_OUT/{exp_name}.log"
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--metrics-run", required=True, type=Path)
    parser.add_argument("--scenes-file", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--batch-name", default="batch1")
    parser.add_argument("--ticks", default="0,74")
    parser.add_argument("--force", choices=("trace_raw", "raw_best_anchor"), default="trace_raw")
    parser.add_argument("--launch-first", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/Flow-Planner-anchor-runtime"))
    parser.add_argument("--python-bin", type=Path, default=Path("/root/miniconda3/envs/flow_planner/bin/python"))
    parser.add_argument(
        "--nuplan-script",
        type=Path,
        default=Path("/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages/nuplan/planning/script/run_simulation.py"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/nuplan_official/data/cache/val"))
    parser.add_argument("--maps-root", type=Path, default=Path("/root/autodl-tmp/maps_raw/maps"))
    parser.add_argument("--planner-config", type=Path, default=Path("/root/autodl-tmp/Flow-Planner-anchor-runtime/flow_planner/script/anchor_finetune.yaml"))
    parser.add_argument("--planner-ckpt", type=Path, default=Path("/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth"))
    parser.add_argument("--anchor-vocab", type=Path, default=Path("/root/autodl-tmp/anchor_runs/anchor_vocab.npy"))
    parser.add_argument("--anchor-predictor-ckpt", type=Path, default=Path("/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth"))
    parser.add_argument(
        "--candidate-selector-ckpt",
        type=Path,
        default=Path("/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_clean_rootfix_20260505/anchor_candidate_selector_pairwise_best.pth"),
    )
    parser.add_argument("--trace-training-payload", action="store_true")
    parser.add_argument("--scenario-filter", default="val20_clean")
    args = parser.parse_args()

    scenes = _load_scene_filter(args.scenes_file)
    if not scenes:
        raise RuntimeError(f"no scenes loaded from {args.scenes_file}")
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.force == "trace_raw":
        if args.trace_jsonl is None:
            raise ValueError("--trace-jsonl is required when --force=trace_raw")
        timestamp_scene_map = _build_timestamp_scene_map(args.metrics_run)
        for row in _load_jsonl(args.trace_jsonl):
            scene_log = timestamp_scene_map.get(int(row["iteration_time_us"]))
            if not scene_log:
                continue
            scene_name, log_name = scene_log
            if scene_name not in scenes or row.get("raw_best_type") != "anchor":
                continue
            enriched = dict(row)
            enriched["scenario_name"] = scene_name
            enriched["log_name"] = log_name
            rows_by_scene[scene_name].append(enriched)
    else:
        rows_by_scene.update(_load_metric_tick_rows(args.metrics_run, scenes))

    ticks = None if args.ticks.lower() == "auto" else [int(v) for v in args.ticks.split(",") if v.strip()]
    manifest_dir = args.run_root / f"{args.batch_name}_manifests"
    script_dir = args.run_root / f"{args.batch_name}_scripts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    run_scripts: list[Path] = []
    idx = 0
    for scene_name in sorted(scenes):
        scene_rows = rows_by_scene.get(scene_name, [])
        if args.force == "trace_raw":
            selected_rows = _select_ticks(scene_rows, ticks)
        else:
            selected_rows = [
                scene_rows[tick_idx]
                for tick_idx in _select_tick_indices(len(scene_rows), ticks)
            ]
        for row in selected_rows:
            tick = int(row["iteration_index"])
            exp_name = f"{args.batch_name}_{idx:02d}_{scene_name}_tick{tick:03d}"
            intervention = {
                "scenario_name": scene_name,
                "log_name": row["log_name"],
                "iteration_index": tick,
                "iteration_time_us": int(row["iteration_time_us"]),
            }
            if args.force == "trace_raw":
                intervention.update(
                    {
                        "type": "anchor",
                        "anchor_rank": row.get("raw_best_anchor_rank"),
                        "sample_i": row.get("raw_best_sample_i"),
                        "candidate_idx": row.get("raw_best_idx"),
                        "source_trace": str(args.trace_jsonl),
                        "source_planner_instance_id": row.get("planner_instance_id"),
                    }
                )
            else:
                intervention.update(
                    {
                        "type": "raw_best_anchor",
                        "source_metrics_run": str(args.metrics_run),
                    }
                )
            manifest = {
                "schema_version": 1,
                "description": "Single-tick closed-loop intervention rollout.",
                "source_trace": str(args.trace_jsonl) if args.trace_jsonl else None,
                "source_metrics_run": str(args.metrics_run),
                "force": args.force,
                "num_interventions": 1,
                "interventions": [intervention],
            }
            manifest_path = manifest_dir / f"{exp_name}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            run_script = script_dir / f"run_{exp_name}.sh"
            _write_run_script(
                run_script,
                exp_name=exp_name,
                run_root=args.run_root,
                manifest_path=manifest_path,
                scenario_name=scene_name,
                log_name=str(row["log_name"]),
                project_root=args.project_root,
                python_bin=args.python_bin,
                nuplan_script=args.nuplan_script,
                data_root=args.data_root,
                maps_root=args.maps_root,
                planner_config=args.planner_config,
                planner_ckpt=args.planner_ckpt,
                anchor_vocab=args.anchor_vocab,
                anchor_predictor_ckpt=args.anchor_predictor_ckpt,
                candidate_selector_ckpt=args.candidate_selector_ckpt,
                trace_training_payload=args.trace_training_payload,
                scenario_filter=args.scenario_filter,
            )
            run_scripts.append(run_script)
            idx += 1

    launched: list[str] = []
    for run_script in run_scripts[: max(args.launch_first, 0)]:
        log_path = args.run_root / f"{run_script.stem}.nohup"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(run_script)], stdout=log, stderr=subprocess.STDOUT)
        launched.append(f"{run_script} pid={proc.pid}")

    print(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "num_scripts": len(run_scripts),
                "scripts": [str(path) for path in run_scripts],
                "launched": launched,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

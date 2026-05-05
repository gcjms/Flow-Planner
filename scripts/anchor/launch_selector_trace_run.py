#!/usr/bin/env python3
"""Create and optionally launch a selector trace-only closed-loop run."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def _read_scenes(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("scenes", payload.get("scenario_names", []))
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _quote_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--scenario-filter", default="val100_clean")
    parser.add_argument("--scenes-file", type=Path, default=None)
    parser.add_argument("--launch", action="store_true")
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
    parser.add_argument("--worker-max-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-weight", type=float, default=1.8)
    args = parser.parse_args()

    scenes = _read_scenes(args.scenes_file)
    args.run_root.mkdir(parents=True, exist_ok=True)
    run_out = args.run_root / args.experiment_name
    trace_path = args.run_root / "candidate_trace.jsonl"
    script_path = args.run_root / f"run_{args.experiment_name}.sh"
    monitor_path = args.run_root / "monitor.sh"
    meta_path = args.run_root / "launch_meta.txt"

    scenario_token_override = ""
    if scenes:
        scenario_token_override = (
            " \\\n"
            f"  scenario_filter.scenario_tokens={_quote_list(scenes)} \\\n"
            f"  scenario_filter.limit_total_scenarios={len(scenes)}"
        )

    content = f"""#!/usr/bin/env bash
set -euo pipefail
export NUPLAN_DATA_ROOT={shlex.quote(str(args.data_root.parent.parent))}
export NUPLAN_MAPS_ROOT={shlex.quote(str(args.maps_root))}
export PYTHONPATH={shlex.quote(str(args.project_root))}:${{PYTHONPATH:-}}
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
cd {shlex.quote(str(args.project_root))}
RUN_ROOT={shlex.quote(str(args.run_root))}
EXP={shlex.quote(args.experiment_name)}
RUN_OUT={shlex.quote(str(run_out))}
TRACE_PATH={shlex.quote(str(trace_path))}
mkdir -p "$RUN_OUT"
rm -f "$TRACE_PATH"
{shlex.quote(str(args.python_bin))} {shlex.quote(str(args.nuplan_script))} \\
  +simulation=closed_loop_nonreactive_agents \\
  planner=flow_planner \\
  planner.flow_planner.config_path={shlex.quote(str(args.planner_config))} \\
  planner.flow_planner.ckpt_path={shlex.quote(str(args.planner_ckpt))} \\
  planner.flow_planner.use_cfg=true \\
  planner.flow_planner.cfg_weight={args.cfg_weight} \\
  planner.flow_planner.device={shlex.quote(args.device)} \\
  scenario_builder=nuplan \\
  scenario_builder.data_root={shlex.quote(str(args.data_root))} \\
  scenario_filter={shlex.quote(args.scenario_filter)}{scenario_token_override} \\
  output_dir="$RUN_OUT" \\
  experiment_name="$EXP" \\
  verbose=false \\
  worker=single_machine_thread_pool \\
  worker.max_workers={args.worker_max_workers} \\
  enable_simulation_progress_bar=true \\
  exit_on_failure=false \\
  "hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \\
  planner.flow_planner.anchor_vocab_path={shlex.quote(str(args.anchor_vocab))} \\
  planner.flow_planner.anchor_mode=predicted_anchor_candidate_selector \\
  planner.flow_planner.anchor_predictor_ckpt={shlex.quote(str(args.anchor_predictor_ckpt))} \\
  planner.flow_planner.candidate_selector_ckpt={shlex.quote(str(args.candidate_selector_ckpt))} \\
  planner.flow_planner.anchor_top_k=3 \\
  planner.flow_planner.candidate_samples_per_anchor=3 \\
  planner.flow_planner.candidate_samples_per_anchor_list=[5,2,2] \\
  planner.flow_planner.candidate_trace_path="$TRACE_PATH" \\
  2>&1 | tee "$RUN_OUT/${{EXP}}.log"
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)

    monitor_content = f"""#!/usr/bin/env bash
set -euo pipefail
TRACE_PATH={shlex.quote(str(trace_path))}
EXP={shlex.quote(args.experiment_name)}
while true; do
  echo "===== $(date +%F_%T_%Z) ====="
  wc -l "$TRACE_PATH" 2>/dev/null || true
  ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd | grep -E "run_simulation.py|$EXP" | grep -v grep || true
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  free -h || true
  sleep 30
done
"""
    monitor_path.write_text(monitor_content, encoding="utf-8")
    monitor_path.chmod(0o755)

    meta_path.write_text(
        "\n".join(
            [
                f"experiment_name={args.experiment_name}",
                f"scenario_filter={args.scenario_filter}",
                f"num_scenes={len(scenes) if scenes else 'filter_default'}",
                f"scenes_file={args.scenes_file or ''}",
                f"trace_path={trace_path}",
                f"script_path={script_path}",
                f"worker_max_workers={args.worker_max_workers}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    launched = None
    if args.launch:
        log_path = args.run_root / f"run_{args.experiment_name}.nohup"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([str(script_path)], stdout=log, stderr=subprocess.STDOUT)
        launched = {"pid": proc.pid, "log_path": str(log_path)}

    print(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "script_path": str(script_path),
                "trace_path": str(trace_path),
                "monitor_path": str(monitor_path),
                "num_scenes": len(scenes) if scenes else None,
                "launched": launched,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

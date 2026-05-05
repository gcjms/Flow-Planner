#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}"

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/Flow-Planner-anchor-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/flow_planner/bin/python}"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/root/miniconda3/envs/flow_planner/lib/python3.9/site-packages}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/root/autodl-tmp/maps_raw/maps}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ -d /root/autodl-tmp/nuplan_official/data/cache/mini ]]; then
  SCENARIO_DATA_ROOT_DEFAULT="/root/autodl-tmp/nuplan_official/data/cache/mini"
elif [[ -d /root/autodl-tmp/nuplan_official/nuplan-v1.1/splits/mini ]]; then
  SCENARIO_DATA_ROOT_DEFAULT="/root/autodl-tmp/nuplan_official/nuplan-v1.1/splits/mini"
elif [[ -d /root/autodl-tmp/nuplan_official/public/nuplan-v1.1/splits/mini ]]; then
  SCENARIO_DATA_ROOT_DEFAULT="/root/autodl-tmp/nuplan_official/public/nuplan-v1.1/splits/mini"
else
  echo "Could not find extracted nuplan-v1.1 mini under /root/autodl-tmp/nuplan_official"
  exit 2
fi

SCENARIO_DATA_ROOT="${SCENARIO_DATA_ROOT:-$SCENARIO_DATA_ROOT_DEFAULT}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/root/autodl-tmp/nuplan_official}"
export NUPLAN_MAPS_ROOT
SCENARIO_FILTER="${SCENARIO_FILTER:-debug_2}"
SIMULATION_KIND="${SIMULATION_KIND:-closed_loop_nonreactive_agents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/anchor_runs/official_planner_anchor_smoke_20260430}"
THREADS_PER_NODE="${THREADS_PER_NODE:-2}"

PLANNER_CONFIG="${PLANNER_CONFIG:-$PROJECT_ROOT/flow_planner/script/anchor_finetune.yaml}"
PLANNER_CKPT="${PLANNER_CKPT:-/root/autodl-tmp/anchor_runs/planner_ft_sched_p0p5_20260426_1612/planner_anchor_best.pth}"
ANCHOR_VOCAB_PATH="${ANCHOR_VOCAB_PATH:-/root/autodl-tmp/anchor_runs/anchor_vocab.npy}"
ANCHOR_PREDICTOR_CKPT="${ANCHOR_PREDICTOR_CKPT:-/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth}"
CANDIDATE_SELECTOR_CKPT="${CANDIDATE_SELECTOR_CKPT:-/root/autodl-tmp/anchor_runs/anchor_candidate_selector_pairwise_sameanchor_allpairs_train2k_rescorefix_scenegroup_20260429_2306/anchor_candidate_selector_pairwise_best.pth}"
CFG_WEIGHT="${CFG_WEIGHT:-1.8}"
SMOKE_DEVICE="${SMOKE_DEVICE:-cpu}"

mkdir -p "$OUTPUT_ROOT" "$RAY_TMPDIR"
cd "$PROJECT_ROOT"

run_case() {
  local exp_name="$1"
  shift
  echo "=========================================================="
  echo "[$(date '+%F %T')] running $exp_name"
  echo "=========================================================="
  "$PYTHON_BIN" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
    +simulation="$SIMULATION_KIND" \
    planner=flow_planner \
    planner.flow_planner.config_path="$PLANNER_CONFIG" \
    planner.flow_planner.ckpt_path="$PLANNER_CKPT" \
    planner.flow_planner.use_cfg=true \
    planner.flow_planner.cfg_weight="$CFG_WEIGHT" \
    planner.flow_planner.device="$SMOKE_DEVICE" \
    scenario_builder=nuplan \
    scenario_builder.data_root="$SCENARIO_DATA_ROOT" \
    scenario_filter="$SCENARIO_FILTER" \
    output_dir="$OUTPUT_ROOT" \
    experiment_name="$exp_name" \
    verbose=true \
    worker=sequential \
    enable_simulation_progress_bar=true \
    exit_on_failure=false \
    "hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    "$@" 2>&1 | tee "$OUTPUT_ROOT/${exp_name}.log"
}

run_case "anchor_none_debug2" \
  +planner.flow_planner.anchor_vocab_path="$ANCHOR_VOCAB_PATH" \
  +planner.flow_planner.anchor_mode=none

run_case "anchor_selector_522_debug2" \
  +planner.flow_planner.anchor_vocab_path="$ANCHOR_VOCAB_PATH" \
  +planner.flow_planner.anchor_mode=predicted_anchor_candidate_selector \
  +planner.flow_planner.anchor_predictor_ckpt="$ANCHOR_PREDICTOR_CKPT" \
  +planner.flow_planner.candidate_selector_ckpt="$CANDIDATE_SELECTOR_CKPT" \
  +planner.flow_planner.anchor_top_k=3 \
  +planner.flow_planner.candidate_samples_per_anchor=3 \
  +planner.flow_planner.candidate_samples_per_anchor_list=5,2,2

echo "[$(date '+%F %T')] closed-loop smoke complete"

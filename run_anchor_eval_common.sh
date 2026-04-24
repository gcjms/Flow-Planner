#!/bin/bash
set -euo pipefail

anchor_eval_init() {
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-flow_planner}"

  FLOW_PLANNER_ROOT="${FLOW_PLANNER_ROOT:-/root/autodl-tmp/Flow-Planner}"
  CONFIG_PATH="${CONFIG_PATH:-flow_planner/script/anchor_finetune.yaml}"
  RAW_CKPT="${RAW_CKPT:-/root/autodl-tmp/ckpts/flowplanner_no_goal.pth}"
  PLANNER_FT_CKPT="${PLANNER_FT_CKPT:-/root/autodl-tmp/anchor_runs/planner_ft_run1/planner_anchor_best.pth}"
  ANCHOR_VOCAB_PATH="${ANCHOR_VOCAB_PATH:-/root/autodl-tmp/anchor_runs/anchor_vocab.npy}"
  ANCHOR_PREDICTOR_CKPT="${ANCHOR_PREDICTOR_CKPT:-/root/autodl-tmp/anchor_runs/anchor_predictor_run1/anchor_predictor_best.pth}"
  SCENE_DIR="${SCENE_DIR:-/root/autodl-tmp/nuplan_npz}"
  MANIFEST_PATH="${MANIFEST_PATH:-/root/autodl-tmp/anchor_runs/eval_manifest.json}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/anchor_runs/deploy_eval}"
  MAX_SCENES="${MAX_SCENES:-500}"
  MANIFEST_SEED="${MANIFEST_SEED:-3402}"
  DEVICE="${DEVICE:-cuda}"
  CFG_WEIGHT="${CFG_WEIGHT:-1.8}"
  BON_SEED="${BON_SEED:--1}"
  COLLISION_DIST="${COLLISION_DIST:-2.0}"

  export FLOW_PLANNER_ROOT
  export CONFIG_PATH
  export RAW_CKPT
  export PLANNER_FT_CKPT
  export ANCHOR_VOCAB_PATH
  export ANCHOR_PREDICTOR_CKPT
  export SCENE_DIR
  export MANIFEST_PATH
  export OUTPUT_ROOT
  export MAX_SCENES
  export MANIFEST_SEED
  export DEVICE
  export CFG_WEIGHT
  export BON_SEED
  export COLLISION_DIST

  mkdir -p "$OUTPUT_ROOT"
  cd "$FLOW_PLANNER_ROOT"
}

anchor_require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

anchor_require_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

anchor_validate_base_inputs() {
  anchor_require_file "$ANCHOR_VOCAB_PATH"
  anchor_require_dir "$SCENE_DIR"
}

anchor_print_summary() {
  local output_json="$1"
  local case_name="$2"
  CASE_NAME="$case_name" OUTPUT_JSON="$output_json" python - <<'PY'
import json
import os

case_name = os.environ["CASE_NAME"]
path = os.environ["OUTPUT_JSON"]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

summary = payload["summary"]
print("=== Summary ===")
print(f"case_name: {case_name}")
print(f"conditioning: {summary.get('conditioning_family', 'none')} / {summary.get('conditioning_mode', 'none')}")
for key in (
    "collision_rate",
    "avg_progress",
    "avg_route",
    "avg_collision_score",
    "avg_ttc",
    "avg_comfort",
    "scenes_evaluated",
    "scenes_failed",
):
    print(f"{key}: {summary[key]}")
print(f"summary_json: {path}")
PY
}

run_anchor_eval_case() {
  if [ "$#" -lt 3 ]; then
    echo "run_anchor_eval_case requires: <case_name> <ckpt_path> <anchor_mode> [extra args...]" >&2
    exit 1
  fi

  local case_name="$1"
  local ckpt_path="$2"
  local anchor_mode="$3"
  shift 3

  anchor_validate_base_inputs
  anchor_require_file "$ckpt_path"
  if [[ "$anchor_mode" == predicted_anchor* ]]; then
    anchor_require_file "$ANCHOR_PREDICTOR_CKPT"
  fi

  local output_dir="$OUTPUT_ROOT/$case_name"
  local output_json="$output_dir/${case_name}.json"
  local log_path="$output_dir/${case_name}.log"
  mkdir -p "$output_dir"

  local -a cmd=(
    python -m flow_planner.dpo.eval_multidim
    --config_path "$CONFIG_PATH"
    --ckpt_path "$ckpt_path"
    --anchor_vocab_path "$ANCHOR_VOCAB_PATH"
    --scene_dir "$SCENE_DIR"
    --max_scenes "$MAX_SCENES"
    --collision_dist "$COLLISION_DIST"
    --device "$DEVICE"
    --cfg_weight "$CFG_WEIGHT"
    --bon_seed "$BON_SEED"
    --anchor_mode "$anchor_mode"
    --output_json "$output_json"
  )

  local manifest_mode
  if [ -f "$MANIFEST_PATH" ]; then
    cmd+=(--scene_manifest "$MANIFEST_PATH")
    manifest_mode="reuse"
  else
    cmd+=(--write_scene_manifest "$MANIFEST_PATH" --manifest_seed "$MANIFEST_SEED")
    manifest_mode="create"
  fi

  if [[ "$anchor_mode" == predicted_anchor* ]]; then
    cmd+=(--anchor_predictor_ckpt "$ANCHOR_PREDICTOR_CKPT")
  fi

  if [ "$#" -gt 0 ]; then
    cmd+=("$@")
  fi

  {
    echo "=== Anchor eval case ==="
    echo "date: $(date)"
    echo "cwd: $(pwd)"
    echo "case_name: $case_name"
    echo "conditioning_mode: $anchor_mode"
    echo "config: $CONFIG_PATH"
    echo "ckpt: $ckpt_path"
    echo "anchor_vocab_path: $ANCHOR_VOCAB_PATH"
    echo "scene_dir: $SCENE_DIR"
    echo "manifest_path: $MANIFEST_PATH ($manifest_mode)"
    echo "output_json: $output_json"
    echo "device: $DEVICE"
    echo
    printf 'command:'
    printf ' %q' "${cmd[@]}"
    echo
    echo
  } | tee "$log_path"

  "${cmd[@]}" 2>&1 | tee -a "$log_path"
  anchor_print_summary "$output_json" "$case_name" | tee -a "$log_path"
}

print_anchor_eval_table() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi

  OUTPUT_ROOT="$OUTPUT_ROOT" python - "$@" <<'PY'
import json
import os
import sys

root = os.environ["OUTPUT_ROOT"]
cases = sys.argv[1:]
rows = []
for case_name in cases:
    path = os.path.join(root, case_name, f"{case_name}.json")
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]
    rows.append(
        (
            case_name,
            summary.get("conditioning_mode", "none"),
            summary["collision_rate"],
            summary["avg_progress"],
            summary["avg_route"],
        )
    )

if not rows:
    sys.exit(0)

print("=== Anchor Eval Table ===")
print(f"{'case_name':<28} {'conditioning_mode':<24} {'collision':>10} {'progress':>10} {'route':>10}")
for case_name, conditioning_mode, collision_rate, avg_progress, avg_route in rows:
    print(
        f"{case_name:<28} {conditioning_mode:<24} "
        f"{collision_rate:>9.1f}% {avg_progress:>10.4f} {avg_route:>10.4f}"
    )
PY
}

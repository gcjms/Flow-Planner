#!/bin/bash
set -euo pipefail

# One-click AutoDL helper for the missing "raw FlowPlanner" baseline.
#
# Default behavior:
#   1. Reuse the existing eval manifest if present.
#   2. Otherwise create a deterministic 500-scene manifest.
#   3. Evaluate `flowplanner_no_goal.pth` with `anchor_mode none`.
#
# You can override paths via environment variables:
#   FLOW_PLANNER_ROOT=/root/autodl-tmp/Flow-Planner \
#   RAW_CKPT=/root/autodl-tmp/ckpts/flowplanner_no_goal.pth \
#   ANCHOR_VOCAB_PATH=/root/autodl-tmp/anchor_runs/anchor_vocab.npy \
#   ./run_anchor_raw_no_goal_eval.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-flow_planner}"

FLOW_PLANNER_ROOT="${FLOW_PLANNER_ROOT:-/root/autodl-tmp/Flow-Planner}"
cd "$FLOW_PLANNER_ROOT"

CONFIG_PATH="${CONFIG_PATH:-flow_planner/script/anchor_finetune.yaml}"
RAW_CKPT="${RAW_CKPT:-/root/autodl-tmp/ckpts/flowplanner_no_goal.pth}"
ANCHOR_VOCAB_PATH="${ANCHOR_VOCAB_PATH:-/root/autodl-tmp/anchor_runs/anchor_vocab.npy}"
SCENE_DIR="${SCENE_DIR:-/root/autodl-tmp/nuplan_npz}"
MANIFEST_PATH="${MANIFEST_PATH:-/root/autodl-tmp/anchor_runs/eval_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/anchor_runs/raw_no_goal_eval}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/eval_raw_no_goal.json}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/eval_raw_no_goal.log}"

MAX_SCENES="${MAX_SCENES:-500}"
MANIFEST_SEED="${MANIFEST_SEED:-3402}"
DEVICE="${DEVICE:-cuda}"
CFG_WEIGHT="${CFG_WEIGHT:-1.8}"
BON_SEED="${BON_SEED:--1}"
COLLISION_DIST="${COLLISION_DIST:-2.0}"

mkdir -p "$OUTPUT_DIR"

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

require_file "$RAW_CKPT"
require_file "$ANCHOR_VOCAB_PATH"
require_dir "$SCENE_DIR"

CMD=(
  python -m flow_planner.dpo.eval_multidim
  --config_path "$CONFIG_PATH"
  --ckpt_path "$RAW_CKPT"
  --anchor_vocab_path "$ANCHOR_VOCAB_PATH"
  --scene_dir "$SCENE_DIR"
  --max_scenes "$MAX_SCENES"
  --collision_dist "$COLLISION_DIST"
  --device "$DEVICE"
  --cfg_weight "$CFG_WEIGHT"
  --bon_seed "$BON_SEED"
  --anchor_mode none
  --output_json "$OUTPUT_JSON"
)

if [ -f "$MANIFEST_PATH" ]; then
  CMD+=(--scene_manifest "$MANIFEST_PATH")
  MANIFEST_MODE="reuse"
else
  CMD+=(--write_scene_manifest "$MANIFEST_PATH" --manifest_seed "$MANIFEST_SEED")
  MANIFEST_MODE="create"
fi

{
  echo "=== Raw no-goal baseline eval ==="
  echo "date: $(date)"
  echo "cwd: $(pwd)"
  echo "config: $CONFIG_PATH"
  echo "ckpt: $RAW_CKPT"
  echo "anchor_vocab_path: $ANCHOR_VOCAB_PATH"
  echo "scene_dir: $SCENE_DIR"
  echo "manifest_path: $MANIFEST_PATH ($MANIFEST_MODE)"
  echo "output_json: $OUTPUT_JSON"
  echo "max_scenes: $MAX_SCENES"
  echo "device: $DEVICE"
  echo
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  echo
  echo
} | tee "$LOG_PATH"

"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"

OUTPUT_JSON="$OUTPUT_JSON" python - <<'PY' | tee -a "$LOG_PATH"
import json
import os

path = os.environ["OUTPUT_JSON"]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

summary = payload["summary"]
print("=== Summary ===")
for key in (
    "collision_rate",
    "avg_collision_score",
    "avg_ttc",
    "avg_comfort",
    "avg_progress",
    "avg_route",
    "scenes_evaluated",
    "scenes_failed",
):
    print(f"{key}: {summary[key]}")
print(f"summary_json: {path}")
PY

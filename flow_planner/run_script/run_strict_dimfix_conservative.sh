#!/usr/bin/env bash
set -euo pipefail

# Conservative strict_dimfix DPO pipeline:
#   1. 1-epoch adaptive DPO training on strict_dimfix preferences
#   2. fixed 1000-scene open-loop evaluation
#   3. write a compact text report
#   4. optionally shutdown the machine

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/Flow-Planner}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/flow_planner/bin/python}"

TRAIN_SCENE_DIR="${TRAIN_SCENE_DIR:-/root/autodl-tmp/hard_scenarios_v2}"
EVAL_SCENE_DIR="${EVAL_SCENE_DIR:-/root/autodl-tmp/hard_scenarios_v2}"
SCENE_MANIFEST="${SCENE_MANIFEST:-/root/autodl-tmp/eval_manifests/hard_1000_seed20260418.txt}"
PREFERENCE_PATH="${PREFERENCE_PATH:-/root/autodl-tmp/dpo_preferences_goal_structured/preferences_multi_strict_dimfix.npz}"

BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/checkpoints/model_goal.pth}"
BASE_CONFIG="${BASE_CONFIG:-$REPO_ROOT/checkpoints/config_goal.yaml}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-strict_dimfix_adaptive_b2.0_s1.0_r0.2_e1}"
OUTPUT_CKPT_DIR="${OUTPUT_CKPT_DIR:-$REPO_ROOT/checkpoints/dpo_goal_${EXPERIMENT_TAG}}"
OUTPUT_EVAL_DIR="${OUTPUT_EVAL_DIR:-/root/autodl-tmp/goal_dpo_eval_${EXPERIMENT_TAG}_1000}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
BETA="${BETA:-2.0}"
SFT_WEIGHT="${SFT_WEIGHT:-1.0}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-0.2}"
ADAPTIVE_EMA_DECAY="${ADAPTIVE_EMA_DECAY:-0.9}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.01}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-5.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_COMPONENT_GRAD_EVERY="${LOG_COMPONENT_GRAD_EVERY:-20}"

CFG_WEIGHT="${CFG_WEIGHT:-1.8}"
BON_SEED="${BON_SEED:-0}"
DEVICE="${DEVICE:-cuda}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-1}"

TRAIN_LOG="$OUTPUT_CKPT_DIR/train.log"
EVAL_LOG="$OUTPUT_EVAL_DIR/eval_multidim_none_1000.log"
EVAL_JSON="$OUTPUT_EVAL_DIR/eval_multidim_none_1000.json"
REPORT_TXT="$OUTPUT_EVAL_DIR/report.txt"

mkdir -p "$OUTPUT_CKPT_DIR" "$OUTPUT_EVAL_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "============================================================"
echo "strict_dimfix conservative pipeline"
echo "repo        : $REPO_ROOT"
echo "preferences : $PREFERENCE_PATH"
echo "train scenes: $TRAIN_SCENE_DIR"
echo "eval scenes : $EVAL_SCENE_DIR"
echo "manifest    : $SCENE_MANIFEST"
echo "output ckpt : $OUTPUT_CKPT_DIR"
echo "output eval : $OUTPUT_EVAL_DIR"
echo "============================================================"

"$PYTHON_BIN" -m flow_planner.dpo.train_dpo \
  --ckpt_path "$BASE_CKPT" \
  --config_path "$BASE_CONFIG" \
  --preference_path "$PREFERENCE_PATH" \
  --scene_dir "$TRAIN_SCENE_DIR" \
  --output_dir "$OUTPUT_CKPT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --beta "$BETA" \
  --sft_weight "$SFT_WEIGHT" \
  --adaptive_dpo_ratio_target "$ADAPTIVE_RATIO" \
  --adaptive_dpo_ema_decay "$ADAPTIVE_EMA_DECAY" \
  --adaptive_dpo_min "$ADAPTIVE_MIN" \
  --adaptive_dpo_max "$ADAPTIVE_MAX" \
  --num_workers "$NUM_WORKERS" \
  --log_component_grad_every "$LOG_COMPONENT_GRAD_EVERY" \
  --save_merged \
  2>&1 | tee "$TRAIN_LOG"

"$PYTHON_BIN" -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$OUTPUT_CKPT_DIR/model_dpo_merged.pth" \
  --config_path "$BASE_CONFIG" \
  --scene_dir "$EVAL_SCENE_DIR" \
  --scene_manifest "$SCENE_MANIFEST" \
  --max_scenes 1000 \
  --device "$DEVICE" \
  --use_cfg \
  --cfg_weight "$CFG_WEIGHT" \
  --bon_seed "$BON_SEED" \
  --output_json "$EVAL_JSON" \
  2>&1 | tee "$EVAL_LOG"

"$PYTHON_BIN" - <<'PY' "$EVAL_JSON" "$REPORT_TXT" "$EXPERIMENT_TAG" "$PREFERENCE_PATH" "$OUTPUT_CKPT_DIR/model_dpo_merged.pth"
import json
import sys
from pathlib import Path

eval_json, report_txt, tag, pref_path, ckpt_path = sys.argv[1:]
payload = json.loads(Path(eval_json).read_text(encoding="utf-8"))
summary = payload["summary"]
lines = [
    f"experiment_tag: {tag}",
    f"preference_path: {pref_path}",
    f"checkpoint: {ckpt_path}",
    f"collision_rate: {summary['collision_rate']:.1f}%",
    f"avg_progress: {summary['avg_progress']:.4f}",
    f"avg_route: {summary['avg_route']:.4f}",
    f"avg_comfort: {summary['avg_comfort']:.4f}",
    f"avg_ttc: {summary['avg_ttc']:.4f}",
    f"scenes_evaluated: {summary['scenes_evaluated']}",
    f"scenes_failed: {summary['scenes_failed']}",
    f"cfg_weight: {summary['cfg_weight']:.3f}",
    f"bon_seed: {summary['bon_seed']}",
    f"elapsed_minutes: {summary['elapsed_minutes']:.2f}",
]
Path(report_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(Path(report_txt).read_text(encoding="utf-8"))
PY

if [[ "$SHUTDOWN_WHEN_DONE" == "1" ]]; then
  echo "Pipeline finished. Shutting down machine..."
  shutdown -h now
fi

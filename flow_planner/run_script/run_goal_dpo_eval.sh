#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   1) Edit the variables below or override them via environment variables.
#   2) Run:
#        bash run_script/run_goal_dpo_eval.sh
#
# Example:
#   REPO_ROOT=/root/Flow-Planner \
#   SCENE_DIR=/root/autodl-tmp/hard_scenarios_v2 \
#   BASE_CKPT=/root/Flow-Planner/checkpoints/model.pth \
#   BASE_CONFIG=/root/Flow-Planner/checkpoints/config.yaml \
#   DPO_CKPT=/root/Flow-Planner/checkpoints/dpo_goal_tune_b3.0_s0.3_e1/model_dpo_merged.pth \
#   DPO_CONFIG=/root/Flow-Planner/checkpoints/dpo_goal_tune_b3.0_s0.3_e1/config.yaml \
#   GOAL_VOCAB=/root/Flow-Planner/goal_vocab.npy \
#   GOAL_PREDICTOR_CKPT=/root/Flow-Planner/checkpoints/goal_predictor.ckpt \
#   GOAL_PREDICTOR_HIDDEN_DIM=256 \
#   bash run_script/run_goal_dpo_eval.sh

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

REPO_ROOT="${REPO_ROOT:-/root/Flow-Planner}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SCENE_DIR="${SCENE_DIR:-/path/to/hard_scenarios_v2}"

BASE_CKPT="${BASE_CKPT:-/path/to/base/model.pth}"
BASE_CONFIG="${BASE_CONFIG:-/path/to/base/config.yaml}"

DPO_CKPT="${DPO_CKPT:-/path/to/dpo_goal_tune_b3.0_s0.3_e1/model_dpo_merged.pth}"
DPO_CONFIG="${DPO_CONFIG:-/path/to/dpo_goal_tune_b3.0_s0.3_e1/config.yaml}"

GOAL_VOCAB="${GOAL_VOCAB:-/path/to/goal_vocab.npy}"
GOAL_PREDICTOR_CKPT="${GOAL_PREDICTOR_CKPT:-}"
GOAL_PREDICTOR_HIDDEN_DIM="${GOAL_PREDICTOR_HIDDEN_DIM:-256}"
GOAL_PREDICTOR_DROPOUT="${GOAL_PREDICTOR_DROPOUT:-0.1}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/goal_dpo_eval}"
SMOKE_SCENES="${SMOKE_SCENES:-5}"
MAX_SCENES="${MAX_SCENES:-200}"
CFG_WEIGHT="${CFG_WEIGHT:-1.8}"
BON_SEED="${BON_SEED:--1}"
USE_CFG="${USE_CFG:-1}"
RUN_PREDICTED_GOAL="${RUN_PREDICTED_GOAL:-1}"
RUN_ORACLE_GOAL="${RUN_ORACLE_GOAL:-1}"

if [[ "$USE_CFG" == "1" ]]; then
  CFG_ARGS=(--use_cfg --cfg_weight "$CFG_WEIGHT")
else
  CFG_ARGS=(--no_cfg --cfg_weight "$CFG_WEIGHT")
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_and_log() {
  local name="$1"
  shift
  echo ""
  echo "============================================================"
  echo "Running: $name"
  echo "Command: $PYTHON_BIN $*"
  echo "============================================================"
  "$PYTHON_BIN" "$@" 2>&1 | tee "$OUTPUT_DIR/${name}.log"
}

echo "Repo root: $REPO_ROOT"
echo "Scene dir : $SCENE_DIR"
echo "Output dir: $OUTPUT_DIR"

# Smoke test first. If this fails, stop before the 200-scene jobs.
run_and_log smoke_eval_base \
  -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$BASE_CKPT" \
  --config_path "$BASE_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --device "$DEVICE" \
  --max_scenes "$SMOKE_SCENES" \
  --bon_seed "$BON_SEED" \
  "${CFG_ARGS[@]}" \
  --output_json "$OUTPUT_DIR/smoke_eval_base.json"

run_and_log eval_multidim_base \
  -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$BASE_CKPT" \
  --config_path "$BASE_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --device "$DEVICE" \
  --max_scenes "$MAX_SCENES" \
  --bon_seed "$BON_SEED" \
  "${CFG_ARGS[@]}" \
  --output_json "$OUTPUT_DIR/eval_multidim_base.json"

run_and_log eval_multidim_dpo_none \
  -m flow_planner.dpo.eval_multidim \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --device "$DEVICE" \
  --max_scenes "$MAX_SCENES" \
  --bon_seed "$BON_SEED" \
  "${CFG_ARGS[@]}" \
  --output_json "$OUTPUT_DIR/eval_multidim_dpo_none.json"

run_and_log eval_multidim_goal_none \
  -m flow_planner.dpo.eval_multidim_goal_ablation \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --device "$DEVICE" \
  --goal_mode none \
  --max_scenes "$MAX_SCENES" \
  --bon_seed "$BON_SEED" \
  "${CFG_ARGS[@]}" \
  --output_json "$OUTPUT_DIR/eval_multidim_goal_none.json"

run_and_log eval_multidim_goal_route \
  -m flow_planner.dpo.eval_multidim_goal_ablation \
  --ckpt_path "$DPO_CKPT" \
  --config_path "$DPO_CONFIG" \
  --scene_dir "$SCENE_DIR" \
  --device "$DEVICE" \
  --goal_mode route_goal \
  --goal_vocab_path "$GOAL_VOCAB" \
  --max_scenes "$MAX_SCENES" \
  --bon_seed "$BON_SEED" \
  "${CFG_ARGS[@]}" \
  --output_json "$OUTPUT_DIR/eval_multidim_goal_route.json"

if [[ "$RUN_PREDICTED_GOAL" == "1" ]]; then
  if [[ -z "$GOAL_PREDICTOR_CKPT" ]]; then
    echo "GOAL_PREDICTOR_CKPT is required when RUN_PREDICTED_GOAL=1" >&2
    exit 1
  fi

  run_and_log eval_multidim_goal_predicted \
    -m flow_planner.dpo.eval_multidim_goal_ablation \
    --ckpt_path "$DPO_CKPT" \
    --config_path "$DPO_CONFIG" \
    --scene_dir "$SCENE_DIR" \
    --device "$DEVICE" \
    --goal_mode predicted_goal \
    --goal_vocab_path "$GOAL_VOCAB" \
    --goal_predictor_ckpt "$GOAL_PREDICTOR_CKPT" \
    --goal_predictor_hidden_dim "$GOAL_PREDICTOR_HIDDEN_DIM" \
    --goal_predictor_dropout "$GOAL_PREDICTOR_DROPOUT" \
    --max_scenes "$MAX_SCENES" \
    --bon_seed "$BON_SEED" \
    "${CFG_ARGS[@]}" \
    --output_json "$OUTPUT_DIR/eval_multidim_goal_predicted.json"
else
  echo "Skipping predicted_goal ablation because RUN_PREDICTED_GOAL=$RUN_PREDICTED_GOAL"
fi

if [[ "$RUN_ORACLE_GOAL" == "1" ]]; then
  # Oracle (cheating) goal: snaps GT future endpoint to nearest goal-vocab
  # cluster. Not deployable, but gives the upper bound on decoder quality when
  # the "right" goal is supplied. Used to diagnose whether DPO training hurt
  # the decoder's ability to follow a correct goal.
  run_and_log eval_multidim_goal_oracle \
    -m flow_planner.dpo.eval_multidim_goal_ablation \
    --ckpt_path "$DPO_CKPT" \
    --config_path "$DPO_CONFIG" \
    --scene_dir "$SCENE_DIR" \
    --device "$DEVICE" \
    --goal_mode oracle_goal \
    --goal_vocab_path "$GOAL_VOCAB" \
    --max_scenes "$MAX_SCENES" \
    --bon_seed "$BON_SEED" \
    "${CFG_ARGS[@]}" \
    --output_json "$OUTPUT_DIR/eval_multidim_goal_oracle.json"
else
  echo "Skipping oracle_goal ablation because RUN_ORACLE_GOAL=$RUN_ORACLE_GOAL"
fi

echo ""
echo "All evaluation jobs completed. JSON summaries and logs are in: $OUTPUT_DIR"

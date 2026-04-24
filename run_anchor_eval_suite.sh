#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_anchor_eval_common.sh"

anchor_eval_init

usage() {
  cat <<'EOF'
Usage:
  bash run_anchor_eval_suite.sh
  bash run_anchor_eval_suite.sh raw_no_goal planner_ft_none oracle_anchor_rerank

Default (no args): run all anchor-branch eval cases needed for deployment/comparison.

Available cases:
  raw_no_goal
  planner_ft_none
  predicted_anchor
  predicted_anchor_rerank_a
  oracle_anchor
  oracle_anchor_rerank
EOF
}

run_named_case() {
  local case_name="$1"
  case "$case_name" in
    raw_no_goal)
      run_anchor_eval_case "raw_no_goal_baseline" "$RAW_CKPT" "none"
      ;;
    planner_ft_none)
      run_anchor_eval_case "planner_ft_none" "$PLANNER_FT_CKPT" "none"
      ;;
    predicted_anchor)
      run_anchor_eval_case "predicted_anchor_top1" "$PLANNER_FT_CKPT" "predicted_anchor"
      ;;
    predicted_anchor_rerank_a)
      run_anchor_eval_case \
        "predicted_anchor_rerank_a" \
        "$PLANNER_FT_CKPT" \
        "predicted_anchor_rerank" \
        --predicted_anchor_top_k 3 \
        --rerank_collision_weight 80 \
        --rerank_ttc_weight 15 \
        --rerank_route_weight 5 \
        --rerank_comfort_weight 0 \
        --rerank_progress_weight 0.0
      ;;
    oracle_anchor)
      run_anchor_eval_case "oracle_anchor" "$PLANNER_FT_CKPT" "oracle_anchor"
      ;;
    oracle_anchor_rerank)
      run_anchor_eval_case \
        "oracle_anchor_rerank" \
        "$PLANNER_FT_CKPT" \
        "oracle_anchor_rerank" \
        --predicted_anchor_top_k 3 \
        --rerank_collision_weight 80 \
        --rerank_ttc_weight 15 \
        --rerank_route_weight 5 \
        --rerank_comfort_weight 0 \
        --rerank_progress_weight 0.0
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown case: $case_name" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  set -- \
    raw_no_goal \
    planner_ft_none \
    predicted_anchor \
    predicted_anchor_rerank_a \
    oracle_anchor \
    oracle_anchor_rerank
fi

ran_cases=()
for case_name in "$@"; do
  run_named_case "$case_name"
  case "$case_name" in
    raw_no_goal) ran_cases+=("raw_no_goal_baseline") ;;
    planner_ft_none) ran_cases+=("planner_ft_none") ;;
    predicted_anchor) ran_cases+=("predicted_anchor_top1") ;;
    predicted_anchor_rerank_a) ran_cases+=("predicted_anchor_rerank_a") ;;
    oracle_anchor) ran_cases+=("oracle_anchor") ;;
    oracle_anchor_rerank) ran_cases+=("oracle_anchor_rerank") ;;
  esac
done

print_anchor_eval_table "${ran_cases[@]}"

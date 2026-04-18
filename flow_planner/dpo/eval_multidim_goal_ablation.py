"""
Goal Ablation for Multi-Dimensional Open-Loop Evaluation
========================================================
Run the same multi-dimensional evaluation pipeline under different goal modes:
  - none
  - route_goal
  - predicted_goal

Usage:
  python -m flow_planner.dpo.eval_multidim_goal_ablation \
      --ckpt_path checkpoints/dpo_goal_tune/model_dpo_merged.pth \
      --config_path checkpoints/config_goal.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --goal_mode route_goal \
      --goal_vocab_path /path/to/goal_vocab.npy \
      --max_scenes 200
"""

import argparse
import logging

from flow_planner.dpo.eval_multidim_utils import (
    load_goal_predictor_model,
    load_planner_model,
    log_summary,
    resolve_goal_vocab,
    run_multidim_evaluation,
    save_summary_json,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Goal ablation for multi-dimensional open-loop evaluation"
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument(
        "--goal_mode",
        type=str,
        choices=("none", "route_goal", "predicted_goal"),
        required=True,
    )
    parser.add_argument("--max_scenes", type=int, default=500)
    parser.add_argument("--collision_dist", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_cfg",
        action="store_true",
        default=True,
        help="Enable CFG during inference (default: True)",
    )
    parser.add_argument(
        "--no_cfg",
        dest="use_cfg",
        action="store_false",
        help="Disable CFG during inference",
    )
    parser.add_argument("--cfg_weight", type=float, default=1.8)
    parser.add_argument(
        "--bon_seed",
        type=int,
        default=-1,
        help="Deterministic inference seed. Use -1 for random sampling.",
    )
    parser.add_argument(
        "--goal_vocab_path",
        type=str,
        default=None,
        help="Optional override for model.goal_vocab_path or route-goal retrieval.",
    )
    parser.add_argument(
        "--goal_predictor_ckpt",
        type=str,
        default=None,
        help="Checkpoint for GoalPredictor. Required when goal_mode=predicted_goal.",
    )
    parser.add_argument("--goal_predictor_hidden_dim", type=int, default=256)
    parser.add_argument("--goal_predictor_dropout", type=float, default=0.1)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save a JSON summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    model = load_planner_model(
        args.config_path,
        args.ckpt_path,
        device=args.device,
        goal_vocab_path=args.goal_vocab_path,
    )
    logger.info("Model loaded successfully")

    if args.goal_mode != "none":
        goal_dim = int(getattr(model.model_decoder, "goal_dim", 0))
        if goal_dim <= 0:
            raise ValueError(
                "Selected goal_mode requires a goal-conditioned checkpoint "
                "(decoder.goal_dim > 0)."
            )

    goal_vocab = None
    if args.goal_mode in {"route_goal", "predicted_goal"}:
        goal_vocab = resolve_goal_vocab(model, args.goal_vocab_path)

    goal_predictor = None
    if args.goal_mode == "predicted_goal":
        if not args.goal_predictor_ckpt:
            raise ValueError(
                "--goal_predictor_ckpt is required when --goal_mode=predicted_goal"
            )
        goal_predictor = load_goal_predictor_model(
            model,
            args.goal_predictor_ckpt,
            device=args.device,
            hidden_dim=args.goal_predictor_hidden_dim,
            dropout=args.goal_predictor_dropout,
        )
        logger.info("Goal predictor loaded successfully")

    summary, failures = run_multidim_evaluation(
        model,
        scene_dir=args.scene_dir,
        device=args.device,
        max_scenes=args.max_scenes,
        collision_dist=args.collision_dist,
        use_cfg=args.use_cfg,
        cfg_weight=args.cfg_weight,
        bon_seed=args.bon_seed,
        goal_mode=args.goal_mode,
        goal_vocab=goal_vocab,
        goal_predictor=goal_predictor,
    )
    log_summary(summary, ckpt_path=args.ckpt_path)

    if args.output_json:
        save_summary_json(
            args.output_json,
            summary,
            failures,
            extra={
                "ckpt_path": args.ckpt_path,
                "config_path": args.config_path,
                "scene_dir": args.scene_dir,
                "goal_mode": args.goal_mode,
                "goal_vocab_path": args.goal_vocab_path,
                "goal_predictor_ckpt": args.goal_predictor_ckpt,
                "goal_predictor_hidden_dim": args.goal_predictor_hidden_dim,
                "goal_predictor_dropout": args.goal_predictor_dropout,
            },
        )


if __name__ == "__main__":
    main()

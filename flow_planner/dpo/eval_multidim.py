"""
Multi-Dimensional Open-Loop Evaluation
======================================
Evaluate a Flow-Planner checkpoint on hard scenarios across multiple
NR-CLS-aligned dimensions: collision rate, TTC, comfort, progress, route
consistency.

This version uses the current `NuPlanDataSample -> model(..., mode="inference")`
pipeline, so CFG and goal-conditioning stay aligned with the planner's
`forward_inference()` implementation.

Usage:
  python -m flow_planner.dpo.eval_multidim \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --max_scenes 200
"""

import argparse
import logging

from flow_planner.dpo.eval_multidim_utils import (
    load_planner_model,
    log_summary,
    run_multidim_evaluation,
    save_summary_json,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dimensional open-loop evaluation"
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument(
        "--scene_manifest",
        type=str,
        default=None,
        help="Optional .txt/.json manifest listing which scene files to evaluate.",
    )
    parser.add_argument(
        "--write_scene_manifest",
        type=str,
        default=None,
        help="Optional output .txt/.json path to save the resolved scene list.",
    )
    parser.add_argument(
        "--manifest_seed",
        type=int,
        default=None,
        help="If set and no manifest is provided, sample max_scenes deterministically with this seed.",
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
        help="Optional override for model.goal_vocab_path when loading config.",
    )
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

    summary, failures = run_multidim_evaluation(
        model,
        scene_dir=args.scene_dir,
        device=args.device,
        max_scenes=args.max_scenes,
        collision_dist=args.collision_dist,
        use_cfg=args.use_cfg,
        cfg_weight=args.cfg_weight,
        bon_seed=args.bon_seed,
        goal_mode="none",
        scene_manifest=args.scene_manifest,
        manifest_seed=args.manifest_seed,
        scene_manifest_out=args.write_scene_manifest,
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
                "scene_manifest": args.scene_manifest,
                "write_scene_manifest": args.write_scene_manifest,
                "manifest_seed": args.manifest_seed,
                "goal_vocab_path": args.goal_vocab_path,
            },
        )


if __name__ == "__main__":
    main()

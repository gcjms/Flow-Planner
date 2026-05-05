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
    load_anchor_predictor_model,
    load_candidate_selector_model,
    load_planner_model,
    log_summary,
    resolve_anchor_vocab,
    run_multidim_evaluation,
    save_summary_json,
)

logger = logging.getLogger(__name__)


def _parse_candidate_samples_per_anchor_list(raw):
    if raw is None or raw.strip() == "":
        return None
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError(
                f"Invalid --candidate_samples_per_anchor_list={raw!r}: empty item"
            )
        value = int(item)
        if value <= 0:
            raise ValueError(
                "--candidate_samples_per_anchor_list values must be positive"
            )
        values.append(value)
    return values


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
        "--anchor_vocab_path",
        type=str,
        default=None,
        help="Path to anchor_vocab.npy (K, T, 3). Required for any --anchor_mode != 'none'.",
    )
    parser.add_argument(
        "--anchor_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "route_anchor",
            "predicted_anchor",
            "predicted_anchor_candidate_selector",
            "predicted_anchor_rerank",
            "oracle_anchor",
            "oracle_anchor_rerank",
        ],
        help="Trajectory-anchor conditioning mode. Mutually exclusive with goal_mode.",
    )
    parser.add_argument(
        "--anchor_predictor_ckpt",
        type=str,
        default=None,
        help="Checkpoint of a trained AnchorPredictor (required for predicted_anchor / predicted_anchor_rerank / predicted_anchor_candidate_selector).",
    )
    parser.add_argument(
        "--candidate_selector_ckpt",
        type=str,
        default=None,
        help="Checkpoint of a trained CandidateSelector (required for predicted_anchor_candidate_selector).",
    )
    parser.add_argument(
        "--predicted_anchor_top_k",
        type=int,
        default=3,
        help="Used by candidate/rerank anchor modes: take top-k anchor candidates before selecting one trajectory.",
    )
    parser.add_argument(
        "--candidate_samples_per_anchor",
        type=int,
        default=3,
        help="Used by --anchor_mode=predicted_anchor_candidate_selector: sample this many trajectories per anchor.",
    )
    parser.add_argument(
        "--candidate_samples_per_anchor_list",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-anchor sample allocation for "
            "predicted_anchor_candidate_selector, e.g. '5,2,2'. "
            "Length must match --predicted_anchor_top_k."
        ),
    )
    parser.add_argument("--rerank_collision_weight", type=float, default=40.0)
    parser.add_argument("--rerank_ttc_weight", type=float, default=20.0)
    parser.add_argument("--rerank_route_weight", type=float, default=25.0)
    parser.add_argument("--rerank_comfort_weight", type=float, default=10.0)
    parser.add_argument(
        "--rerank_progress_weight",
        type=float,
        default=0.0,
        help="Default 0.0 on purpose: first ablation should not reward aggressive forward rush.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save a JSON summary.",
    )
    args = parser.parse_args()
    candidate_samples_per_anchor_list = _parse_candidate_samples_per_anchor_list(
        args.candidate_samples_per_anchor_list
    )
    if (
        candidate_samples_per_anchor_list is not None
        and len(candidate_samples_per_anchor_list) != args.predicted_anchor_top_k
    ):
        raise ValueError(
            "--candidate_samples_per_anchor_list length must match "
            "--predicted_anchor_top_k"
        )

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
        anchor_vocab_path=args.anchor_vocab_path,
    )
    logger.info("Model loaded successfully")

    anchor_vocab = None
    anchor_predictor = None
    candidate_selector = None
    if args.anchor_mode != "none":
        anchor_vocab = resolve_anchor_vocab(model, args.anchor_vocab_path)
        if args.anchor_mode in (
            "predicted_anchor",
            "predicted_anchor_rerank",
            "predicted_anchor_candidate_selector",
        ):
            if args.anchor_predictor_ckpt is None:
                raise ValueError(
                    "--anchor_mode=predicted_anchor / predicted_anchor_rerank / "
                    "predicted_anchor_candidate_selector "
                    "requires --anchor_predictor_ckpt"
                )
            anchor_predictor = load_anchor_predictor_model(
                model, args.anchor_predictor_ckpt, device=args.device
            )
        if args.anchor_mode == "predicted_anchor_candidate_selector":
            if args.candidate_selector_ckpt is None:
                raise ValueError(
                    "--anchor_mode=predicted_anchor_candidate_selector "
                    "requires --candidate_selector_ckpt"
                )
            candidate_selector = load_candidate_selector_model(
                model,
                args.candidate_selector_ckpt,
                device=args.device,
            )

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
        anchor_mode=args.anchor_mode,
        anchor_vocab=anchor_vocab,
        anchor_predictor=anchor_predictor,
        candidate_selector=candidate_selector,
        predicted_anchor_top_k=args.predicted_anchor_top_k,
        candidate_samples_per_anchor=args.candidate_samples_per_anchor,
        candidate_samples_per_anchor_list=candidate_samples_per_anchor_list,
        rerank_collision_weight=args.rerank_collision_weight,
        rerank_ttc_weight=args.rerank_ttc_weight,
        rerank_route_weight=args.rerank_route_weight,
        rerank_comfort_weight=args.rerank_comfort_weight,
        rerank_progress_weight=args.rerank_progress_weight,
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
                "anchor_vocab_path": args.anchor_vocab_path,
                "anchor_mode": args.anchor_mode,
                "anchor_predictor_ckpt": args.anchor_predictor_ckpt,
                "candidate_selector_ckpt": args.candidate_selector_ckpt,
                "predicted_anchor_top_k": args.predicted_anchor_top_k,
                "candidate_samples_per_anchor": args.candidate_samples_per_anchor,
                "candidate_samples_per_anchor_list": candidate_samples_per_anchor_list,
                "rerank_collision_weight": args.rerank_collision_weight,
                "rerank_ttc_weight": args.rerank_ttc_weight,
                "rerank_route_weight": args.rerank_route_weight,
                "rerank_comfort_weight": args.rerank_comfort_weight,
                "rerank_progress_weight": args.rerank_progress_weight,
            },
        )


if __name__ == "__main__":
    main()

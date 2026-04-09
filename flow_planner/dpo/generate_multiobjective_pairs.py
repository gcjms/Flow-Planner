"""
Multi-Objective DPO 偏好对生成器
================================
Phase 1: Multi-seed + Multi-CFG inference → 20 candidates/scene
Phase 2: TrajectoryScorer 5-dim scoring → TISA-DPO style targeted losers

对每个 hard scenario:
  1. 用 FlowPlanner 推理 N_seeds x N_cfgs 次 → K 条候选轨迹
  2. 用 TrajectoryScorer 打 5 维分 (collision, TTC, route, comfort, progress)
  3. Winner pool: composite score top-3
  4. Targeted losers per dimension:
     - Loser_TTC: good overall but low TTC
     - Loser_collision: close to winner but collides
     - Loser_comfort: safe but jerky
  5. 每个场景最多产生 3 对偏好对

用法:
  python -m flow_planner.dpo.generate_multiobjective_pairs \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --output_path dpo_data/multiobjective_pairs.npz \
      --max_scenes 5000 \
      --num_seeds 5 \
      --cfg_weights 0.5,1.0,1.8,3.0
"""

import os
import sys
import time
import glob
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DIMENSION_NAMES = ['collision', 'ttc', 'route', 'comfort', 'progress']

# NR-CLS weights for each dimension (used to prioritize pair selection)
NRCLS_WEIGHTS = {
    'collision': 5,
    'ttc': 5,
    'route': 5,
    'comfort': 2,
    'progress': 5,
}


# ==============================================================
# Model loading
# ==============================================================

def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)

    model = instantiate(cfg.model)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' in ckpt:
        sd = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    state_dict = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    return model


# ==============================================================
# Multi-seed + Multi-CFG inference
# ==============================================================

def _build_scene_inputs(scene_data: dict, device: str) -> dict:
    """Convert raw scene npz data to model input tensors (batch=1)."""
    return {
        'neighbor_past': torch.from_numpy(scene_data['neighbor_agents_past']).float().unsqueeze(0).to(device),
        'lanes': torch.from_numpy(scene_data['lanes']).float().unsqueeze(0).to(device),
        'lanes_speedlimit': torch.from_numpy(scene_data['lanes_speed_limit']).float().unsqueeze(0).to(device),
        'lanes_has_speedlimit': torch.from_numpy(scene_data['lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'routes': torch.from_numpy(scene_data['route_lanes']).float().unsqueeze(0).to(device),
        'map_objects': torch.from_numpy(scene_data['static_objects']).float().unsqueeze(0).to(device),
        'ego_current': torch.from_numpy(scene_data['ego_current_state']).float().unsqueeze(0).to(device),
    }


def _prepare_decoder_inputs_cfg(model, inputs: dict, device: str) -> Tuple[dict, dict]:
    """
    Prepare decoder inputs with proper CFG doubling.

    Returns:
        decoder_inputs_cfg:   batch=2 (conditional + unconditional) for use_cfg=True
        decoder_inputs_nocfg: batch=1 (conditional only) for use_cfg=False
    """
    # --- No-CFG path: batch=1, conditional only ---
    inputs_nocfg = {**inputs, 'cfg_flags': torch.ones(1, device=device, dtype=torch.int32)}
    enc_in_nocfg = model.extract_encoder_inputs(inputs_nocfg)
    enc_out_nocfg = model.encoder(**enc_in_nocfg)
    dec_nocfg = model.extract_decoder_inputs(enc_out_nocfg, inputs_nocfg)

    # --- CFG path: batch=2, [conditional, unconditional] ---
    cfg_flags = torch.tensor([1, 0], device=device, dtype=torch.int32)

    inputs_cfg = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs_cfg[k] = v.repeat(2, *([1] * (v.dim() - 1)))
        else:
            inputs_cfg[k] = v
    inputs_cfg['cfg_flags'] = cfg_flags

    cfg_type = getattr(model, 'cfg_type', 'neighbors')
    if cfg_type == 'neighbors' and 'neighbor_past' in inputs_cfg:
        nb = inputs_cfg['neighbor_past']  # (2, M, T, D)
        neighbor_num = nb.shape[1]
        cfg_neighbor_num = min(
            model.planner_params.get('cfg_neighbor_num', 0), neighbor_num
        )
        mask = cfg_flags.float().view(2, *([1] * (nb.dim() - 1)))
        mask = mask.expand_as(nb).clone()
        mask[:, cfg_neighbor_num:, :, :] = 1.0
        inputs_cfg['neighbor_past'] = nb * mask
    elif cfg_type == 'lanes' and 'lanes' in inputs_cfg:
        lanes = inputs_cfg['lanes']
        mask = cfg_flags.float().view(2, *([1] * (lanes.dim() - 1)))
        inputs_cfg['lanes'] = lanes * mask

    enc_in_cfg = model.extract_encoder_inputs(inputs_cfg)
    enc_out_cfg = model.encoder(**enc_in_cfg)
    dec_cfg = model.extract_decoder_inputs(enc_out_cfg, inputs_cfg)

    return dec_cfg, dec_nocfg


def run_inference_multi_cfg(
    model, scene_data: dict, device: str,
    num_seeds: int = 5,
    cfg_weights: List[float] = None,
    use_sde: bool = False,
    sigma_base: float = 0.3,
    sde_steps: int = 20,
) -> List[np.ndarray]:
    """
    对单个场景推理 num_seeds x len(cfg_weights) 次。
    Encoder 跑两次：一次 CFG 模式 (batch=2)、一次 no-CFG 模式 (batch=1)。
    Decoder 用不同 seed 和 CFG weight 跑多次。

    SDE 模式下使用 generate_sde() 替代 generate()，在 ODE 每步注入可控噪声。
    """
    if cfg_weights is None:
        cfg_weights = [0.5, 1.0, 1.8, 3.0]

    inputs = _build_scene_inputs(scene_data, device)
    trajectories = []

    with torch.no_grad():
        decoder_inputs_cfg, decoder_inputs_nocfg = _prepare_decoder_inputs_cfg(
            model, inputs, device
        )

        from flow_planner.model.model_utils.traj_tool import assemble_actions

        B = 1
        saved_cfg_weight = model.flow_ode.cfg_weight

        for w in cfg_weights:
            use_cfg = abs(w - 1.0) > 0.01
            decoder_inputs = decoder_inputs_cfg if use_cfg else decoder_inputs_nocfg

            # Set CFG weight so both ODE and SDE paths use the correct w
            model.flow_ode.cfg_weight = w

            for s in range(num_seeds):
                torch.manual_seed(s * 12345 + int(time.time()) % 10000)
                x_init = torch.randn(
                    (B, model.action_num, model.planner_params['action_len'],
                     model.planner_params['state_dim']),
                    device=device
                )

                if use_sde:
                    sample = model.flow_ode.generate_sde(
                        x_init, model.decoder, model._model_type,
                        use_cfg=use_cfg, cfg_weight=w,
                        sigma_base=sigma_base, sde_steps=sde_steps,
                        **decoder_inputs
                    )
                else:
                    sample = model.flow_ode.generate(
                        x_init, model.decoder, model._model_type,
                        use_cfg=use_cfg,
                        **decoder_inputs
                    )

                sample = assemble_actions(
                    sample, model.planner_params['future_len'],
                    model.planner_params['action_len'],
                    model.planner_params['action_overlap'],
                    model.planner_params['state_dim'],
                    model.assemble_method
                )
                sample = model.data_processor.state_postprocess(sample)

                result = sample.squeeze(0).cpu().numpy()
                if result.ndim == 3:
                    result = result[0]
                trajectories.append(result)

        model.flow_ode.cfg_weight = saved_cfg_weight

    return trajectories


# ==============================================================
# Multi-objective scoring
# ==============================================================

def score_candidates_multidim(
    trajectories: List[np.ndarray],
    neighbor_past: np.ndarray,
    route_lanes: np.ndarray,
    future_steps: int = 80,
    dt: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Score K candidate trajectories on 5 dimensions using TrajectoryScorer.
    Uses extrapolated neighbor future (not raw neighbor_past).

    Returns dict of {dim_name: (K,) scores} plus 'composite'.
    """
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    K = len(trajectories)
    T = min(t.shape[0] for t in trajectories)
    D = min(t.shape[1] for t in trajectories)

    traj_tensor = torch.from_numpy(
        np.stack([t[:T, :D] for t in trajectories], axis=0)
    ).float()  # (K, T, D)

    scorer = TrajectoryScorer(
        collision_weight=1.0, ttc_weight=1.0, route_weight=1.0,
        comfort_weight=1.0, progress_weight=1.0,
        collision_threshold=2.0, ttc_threshold=3.0, dt=dt,
    )

    # Extrapolate neighbor future from neighbor_past
    nb_past_tensor = torch.from_numpy(neighbor_past).float()  # (M, T_p, D_n)
    nb_future = TrajectoryScorer.extrapolate_neighbor_future(
        nb_past_tensor, future_steps=future_steps, dt=0.5
    )

    # Route tensor
    route_tensor = None
    if route_lanes is not None and route_lanes.size > 0:
        rl = route_lanes
        if rl.ndim == 3:
            valid_mask = np.abs(rl).sum(axis=-1).sum(axis=-1) > 1e-6
            if valid_mask.any():
                rl_flat = rl[valid_mask].reshape(-1, rl.shape[-1])
                route_tensor = torch.from_numpy(rl_flat[:, :2]).float()
        elif rl.ndim == 2:
            route_tensor = torch.from_numpy(rl[:, :2]).float()

    # Per-dimension scores
    collision_scores = scorer._collision_score(traj_tensor, nb_future).numpy()
    ttc_scores = scorer._ttc_score(traj_tensor, nb_future).numpy()
    route_scores = scorer._route_score(traj_tensor, route_tensor).numpy()
    comfort_scores = scorer._comfort_score(traj_tensor).numpy()
    progress_scores = scorer._progress_score(traj_tensor).numpy()

    # Composite with NR-CLS-aligned weights
    composite = (
        NRCLS_WEIGHTS['collision'] * collision_scores +
        NRCLS_WEIGHTS['ttc'] * ttc_scores +
        NRCLS_WEIGHTS['route'] * route_scores +
        NRCLS_WEIGHTS['comfort'] * comfort_scores +
        NRCLS_WEIGHTS['progress'] * progress_scores
    )

    return {
        'collision': collision_scores,
        'ttc': ttc_scores,
        'route': route_scores,
        'comfort': comfort_scores,
        'progress': progress_scores,
        'composite': composite,
    }


# ==============================================================
# TISA-DPO style pair construction
# ==============================================================

def construct_targeted_pairs(
    trajectories: List[np.ndarray],
    dim_scores: Dict[str, np.ndarray],
    target_dims: List[str] = None,
    score_gap_threshold: float = 0.15,
) -> List[Dict]:
    """
    Construct targeted preference pairs following TISA-DPO methodology.

    For each target dimension:
      - chosen = trajectory from the winner pool (top-3 composite)
      - rejected = trajectory that's decent overall but specifically bad on this dimension
        (DriveDPO insight: hard negatives that look good but fail subtly)

    Returns list of pair dicts with keys: chosen, rejected, dimension, chosen_idx, rejected_idx.
    """
    if target_dims is None:
        target_dims = ['collision', 'ttc', 'comfort']

    composite = dim_scores['composite']
    K = len(composite)
    if K < 3:
        return []

    # Winner pool: top-3 by composite score
    top_indices = np.argsort(composite)[::-1][:3]
    winner_idx = int(top_indices[0])

    pairs = []
    T = min(t.shape[0] for t in trajectories)
    D = min(t.shape[1] for t in trajectories)

    for dim in target_dims:
        dim_score = dim_scores[dim]

        # Check if there's enough variance on this dimension to learn from
        score_range = dim_score.max() - dim_score.min()
        if score_range < score_gap_threshold:
            continue

        winner_dim_score = dim_score[winner_idx]

        # Find targeted loser: highest composite score among candidates
        # whose dimension-specific score is substantially worse than the winner's.
        # This implements the DriveDPO "hard negative" idea.
        best_loser_idx = -1
        best_loser_composite = -float('inf')

        for j in range(K):
            if j in top_indices:
                continue
            gap = winner_dim_score - dim_score[j]
            if gap < score_gap_threshold:
                continue
            if composite[j] > best_loser_composite:
                best_loser_composite = composite[j]
                best_loser_idx = j

        if best_loser_idx < 0:
            continue

        chosen_traj = trajectories[winner_idx][:T, :D]
        rejected_traj = trajectories[best_loser_idx][:T, :D]

        pairs.append({
            'chosen': chosen_traj,
            'rejected': rejected_traj,
            'dimension': dim,
            'chosen_idx': winner_idx,
            'rejected_idx': best_loser_idx,
            'chosen_dim_score': float(dim_score[winner_idx]),
            'rejected_dim_score': float(dim_score[best_loser_idx]),
            'chosen_composite': float(composite[winner_idx]),
            'rejected_composite': float(composite[best_loser_idx]),
        })

    return pairs


# ==============================================================
# Main
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Multi-Objective DPO preference pairs (TISA-DPO style)')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Directory containing hard scenario NPZ files')
    parser.add_argument('--output_path', type=str,
                        default='dpo_data/multiobjective_pairs.npz')
    parser.add_argument('--max_scenes', type=int, default=5000)
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of random seeds per CFG weight')
    parser.add_argument('--cfg_weights', type=str, default='0.5,1.0,1.8,3.0',
                        help='Comma-separated CFG weights to try')
    parser.add_argument('--target_dims', type=str, default='collision,ttc,comfort',
                        help='Comma-separated dimensions for targeted losers')
    parser.add_argument('--score_gap_threshold', type=float, default=0.15,
                        help='Min dimension score gap to form a pair')
    parser.add_argument('--sde', action='store_true',
                        help='Use SDE sampling for trajectory diversity')
    parser.add_argument('--sigma_base', type=float, default=0.3,
                        help='SDE noise strength (only used when --sde is set)')
    parser.add_argument('--sde_steps', type=int, default=20,
                        help='SDE integration steps (only used when --sde is set)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    cfg_weights = [float(w) for w in args.cfg_weights.split(',')]
    target_dims = [d.strip() for d in args.target_dims.split(',')]
    total_candidates = args.num_seeds * len(cfg_weights)
    logger.info(f"Config: {args.num_seeds} seeds x {len(cfg_weights)} CFG weights "
                f"= {total_candidates} candidates/scene")
    if args.sde:
        logger.info(f"SDE mode: sigma_base={args.sigma_base}, sde_steps={args.sde_steps}")
    else:
        logger.info("ODE mode (deterministic)")
    logger.info(f"Target dimensions: {target_dims}")

    model = load_model(args.config_path, args.ckpt_path, args.device)
    logger.info("Model loaded successfully")

    all_scene_files = sorted(glob.glob(os.path.join(args.scene_dir, '*.npz')))
    random.seed(42)
    if len(all_scene_files) > args.max_scenes:
        scene_files = random.sample(all_scene_files, args.max_scenes)
    else:
        scene_files = all_scene_files
    logger.info(f"Processing {len(scene_files)} scenes "
                f"(sampled from {len(all_scene_files)} total)")

    # Statistics
    total = 0
    per_dim_pair_count = {d: 0 for d in target_dims}
    scenes_with_pairs = 0
    diversity_stats = []

    chosen_list = []
    rejected_list = []
    dim_labels = []
    scenario_ids = []

    start_time = time.time()

    for i, scene_file in enumerate(scene_files):
        try:
            scene_data = dict(np.load(scene_file, allow_pickle=True))

            # Phase 1: Multi-seed + Multi-CFG inference (ODE or SDE)
            trajectories = run_inference_multi_cfg(
                model, scene_data, args.device,
                num_seeds=args.num_seeds, cfg_weights=cfg_weights,
                use_sde=args.sde, sigma_base=args.sigma_base,
                sde_steps=args.sde_steps,
            )

            # Trajectory diversity (pairwise RMSE of first 5)
            if len(trajectories) >= 2:
                dists = []
                for a in range(min(5, len(trajectories))):
                    for b in range(a + 1, min(5, len(trajectories))):
                        T = min(trajectories[a].shape[0], trajectories[b].shape[0])
                        diff = trajectories[a][:T, :2] - trajectories[b][:T, :2]
                        dists.append(np.sqrt(np.mean(diff**2)))
                diversity_stats.append(np.mean(dists))

            # Phase 2: Multi-objective scoring
            dim_scores = score_candidates_multidim(
                trajectories,
                neighbor_past=scene_data['neighbor_agents_past'],
                route_lanes=scene_data.get('route_lanes', None),
                future_steps=trajectories[0].shape[0],
            )

            # Phase 3: Targeted pair construction
            pairs = construct_targeted_pairs(
                trajectories, dim_scores,
                target_dims=target_dims,
                score_gap_threshold=args.score_gap_threshold,
            )

            total += 1
            if pairs:
                scenes_with_pairs += 1

            sid = str(scene_data.get('token', Path(scene_file).stem))
            for pair in pairs:
                chosen_list.append(pair['chosen'])
                rejected_list.append(pair['rejected'])
                dim_labels.append(pair['dimension'])
                scenario_ids.append(sid)
                per_dim_pair_count[pair['dimension']] += 1

            # Progress report
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                n_pairs = len(chosen_list)
                avg_div = np.mean(diversity_stats[-50:]) if diversity_stats else 0
                dim_str = ' '.join(
                    f"{d}={per_dim_pair_count[d]}" for d in target_dims
                )
                logger.info(
                    f"[{i+1}/{len(scene_files)}] "
                    f"Pairs: {n_pairs} ({dim_str}) | "
                    f"ScenesWithPairs: {scenes_with_pairs}/{total} | "
                    f"AvgDiv: {avg_div:.3f}m | "
                    f"{rate:.1f} scenes/s | "
                    f"ETA: {(len(scene_files)-i-1)/rate/60:.1f}min"
                )

        except Exception as e:
            logger.warning(f"Scene {scene_file}: {e}")
            continue

    # Save results
    elapsed = time.time() - start_time
    n_pairs = len(chosen_list)

    logger.info("=" * 60)
    logger.info(f"Multi-Objective Mining Complete!")
    logger.info(f"  Total scenes processed: {total}")
    logger.info(f"  Scenes with pairs: {scenes_with_pairs} ({scenes_with_pairs/max(total,1)*100:.1f}%)")
    logger.info(f"  Total pairs: {n_pairs}")
    for d in target_dims:
        logger.info(f"    {d}: {per_dim_pair_count[d]}")
    if diversity_stats:
        logger.info(f"  Avg trajectory diversity: {np.mean(diversity_stats):.4f}m")
    logger.info(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")

    if n_pairs > 0:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

        chosen_arr = np.stack(chosen_list, axis=0)
        rejected_arr = np.stack(rejected_list, axis=0)

        np.savez(
            args.output_path,
            chosen=chosen_arr,
            rejected=rejected_arr,
            scenario_ids=np.array(scenario_ids),
            dim_labels=np.array(dim_labels),
            metadata=np.array({
                'total_scenes': total,
                'scenes_with_pairs': scenes_with_pairs,
                'total_pairs': n_pairs,
                'per_dim_pairs': per_dim_pair_count,
                'num_seeds': args.num_seeds,
                'cfg_weights': cfg_weights,
                'target_dims': target_dims,
                'score_gap_threshold': args.score_gap_threshold,
                'avg_diversity_m': float(np.mean(diversity_stats)) if diversity_stats else 0,
                'use_sde': args.sde,
                'sigma_base': args.sigma_base if args.sde else 0.0,
                'sde_steps': args.sde_steps if args.sde else 0,
            }),
        )
        logger.info(f"Saved {n_pairs} multi-objective pairs to {args.output_path}")
        logger.info(f"  chosen: {chosen_arr.shape}, rejected: {rejected_arr.shape}")
        logger.info(f"  dim_labels: {np.unique(dim_labels, return_counts=True)}")
    else:
        logger.warning("No valid multi-objective pairs generated!")
        logger.warning("Try lowering --score_gap_threshold or increasing --num_seeds.")


if __name__ == '__main__':
    main()

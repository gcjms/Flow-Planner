"""
Multi-Dimensional Open-Loop Evaluation
=======================================
Evaluate a Flow-Planner checkpoint on hard scenarios across multiple NR-CLS-aligned
dimensions: collision rate, TTC, comfort, progress, route consistency.

Uses TrajectoryScorer with extrapolated neighbor futures (not raw past data).

用法:
  python -m flow_planner.dpo.eval_multidim \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --max_scenes 500
"""

import os
import sys
import time
import glob
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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


def run_inference(model, scene_data: dict, device: str = 'cuda') -> np.ndarray:
    inputs = {
        'neighbor_past': torch.from_numpy(scene_data['neighbor_agents_past']).float().unsqueeze(0).to(device),
        'lanes': torch.from_numpy(scene_data['lanes']).float().unsqueeze(0).to(device),
        'lanes_speedlimit': torch.from_numpy(scene_data['lanes_speed_limit']).float().unsqueeze(0).to(device),
        'lanes_has_speedlimit': torch.from_numpy(scene_data['lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'routes': torch.from_numpy(scene_data['route_lanes']).float().unsqueeze(0).to(device),
        'map_objects': torch.from_numpy(scene_data['static_objects']).float().unsqueeze(0).to(device),
        'ego_current': torch.from_numpy(scene_data['ego_current_state']).float().unsqueeze(0).to(device),
        'cfg_flags': torch.ones(1, device=device, dtype=torch.int32),
    }

    with torch.no_grad():
        encoder_inputs = model.extract_encoder_inputs(inputs)
        encoder_outputs = model.encoder(**encoder_inputs)
        decoder_inputs = model.extract_decoder_inputs(encoder_outputs, inputs)

        from flow_planner.model.model_utils.traj_tool import assemble_actions

        B = 1
        x_init = torch.randn(
            (B, model.action_num, model.planner_params['action_len'],
             model.planner_params['state_dim']),
            device=device
        )

        sample = model.flow_ode.generate(
            x_init, model.decoder, model._model_type,
            use_cfg=True, cfg_weight=1.8,
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
    return result


def evaluate_trajectory(
    traj: np.ndarray,
    neighbor_past: np.ndarray,
    neighbor_future_gt: np.ndarray,
    route_lanes: np.ndarray,
    collision_dist: float = 2.0,
) -> Dict[str, float]:
    """Evaluate a single trajectory on multiple dimensions."""
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    T = traj.shape[0]

    # Use GT neighbor future for accurate collision/TTC evaluation
    # (different from training-time scorer which uses extrapolation)
    nb_future = torch.from_numpy(neighbor_future_gt[:, :, :2]).float()

    traj_tensor = torch.from_numpy(traj).float().unsqueeze(0)  # (1, T, D)

    scorer = TrajectoryScorer(
        collision_threshold=collision_dist, ttc_threshold=3.0, dt=0.1
    )

    collision_score = scorer._collision_score(traj_tensor, nb_future).item()
    ttc_score = scorer._ttc_score(traj_tensor, nb_future).item()
    comfort_score = scorer._comfort_score(traj_tensor).item()
    progress_score = scorer._progress_score(traj_tensor).item()

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
    route_score = scorer._route_score(traj_tensor, route_tensor).item()

    # Binary collision check (for collision rate reporting)
    collided = False
    M = neighbor_future_gt.shape[0]
    T_nb = neighbor_future_gt.shape[1]
    T_check = min(T, T_nb)
    for t in range(T_check):
        ex, ey = float(traj[t, 0]), float(traj[t, 1])
        for m in range(M):
            nx, ny = float(neighbor_future_gt[m, t, 0]), float(neighbor_future_gt[m, t, 1])
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                continue
            dist = ((ex - nx)**2 + (ey - ny)**2) ** 0.5
            if dist < collision_dist:
                collided = True
                break
        if collided:
            break

    return {
        'collided': 1.0 if collided else 0.0,
        'collision_score': collision_score,
        'ttc_score': ttc_score,
        'comfort_score': comfort_score,
        'progress_score': progress_score,
        'route_score': route_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-dimensional open-loop evaluation')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scene_dir', type=str, required=True)
    parser.add_argument('--max_scenes', type=int, default=500)
    parser.add_argument('--collision_dist', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    model = load_model(args.config_path, args.ckpt_path, args.device)
    logger.info("Model loaded successfully")

    scene_files = sorted(glob.glob(os.path.join(args.scene_dir, '*.npz')))
    if args.max_scenes:
        scene_files = scene_files[:args.max_scenes]
    logger.info(f"Evaluating {len(scene_files)} scenes from {args.scene_dir}")

    all_metrics = {
        'collided': [], 'collision_score': [], 'ttc_score': [],
        'comfort_score': [], 'progress_score': [], 'route_score': [],
    }

    start_time = time.time()

    for i, scene_file in enumerate(scene_files):
        try:
            scene_data = dict(np.load(scene_file, allow_pickle=True))
            pred_traj = run_inference(model, scene_data, args.device)
            nb_future = scene_data['neighbor_agents_future']

            metrics = evaluate_trajectory(
                pred_traj, scene_data['neighbor_agents_past'],
                nb_future, scene_data.get('route_lanes', None),
                collision_dist=args.collision_dist,
            )

            for k, v in metrics.items():
                all_metrics[k].append(v)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                cr = np.mean(all_metrics['collided']) * 100
                logger.info(
                    f"[{i+1}/{len(scene_files)}] "
                    f"collision_rate={cr:.1f}% | "
                    f"{rate:.1f} scenes/s | "
                    f"ETA: {(len(scene_files)-i-1)/rate/60:.1f}min"
                )

        except Exception as e:
            logger.warning(f"Scene {scene_file}: {e}")
            continue

    elapsed = time.time() - start_time
    total = len(all_metrics['collided'])

    collision_rate = np.mean(all_metrics['collided']) * 100
    avg_collision_score = np.mean(all_metrics['collision_score'])
    avg_ttc = np.mean(all_metrics['ttc_score'])
    avg_comfort = np.mean(all_metrics['comfort_score'])
    avg_progress = np.mean(all_metrics['progress_score'])
    avg_route = np.mean(all_metrics['route_score'])

    logger.info("=" * 60)
    logger.info("SUMMARY — Multi-Dimensional Open-Loop Evaluation")
    logger.info(f"  Checkpoint: {args.ckpt_path}")
    logger.info(f"  Scenes evaluated: {total}")
    logger.info(f"  collision_rate: {collision_rate:.1f}%")
    logger.info(f"  avg_collision_score: {avg_collision_score:.4f}")
    logger.info(f"  avg_ttc: {avg_ttc:.4f}")
    logger.info(f"  avg_comfort: {avg_comfort:.4f}")
    logger.info(f"  avg_progress: {avg_progress:.4f}")
    logger.info(f"  avg_route: {avg_route:.4f}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

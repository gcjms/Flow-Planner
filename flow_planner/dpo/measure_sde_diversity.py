"""
SDE 多样性验证脚本
==================
对比 ODE 和 SDE 采样在同一场景下产生的轨迹多样性。
验证 SDE 扰动能否为 DPO/GRPO 提供足够的候选差异。

输出指标:
  - pairwise_rmse: 候选轨迹间的平均 RMSE (米)
  - endpoint_spread: 终点位置的标准差 (米)
  - max_lateral_spread: 最大横向分散度 (米)
  - reward_std: TrajectoryScorer 打分的标准差

用法:
  python -m flow_planner.dpo.measure_sde_diversity \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --num_scenes 100 \
      --num_samples 20 \
      --sigma_base 0.3 \
      --sde_steps 20
"""

import os
import sys
import time
import glob
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)

    model = instantiate(cfg.model)
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


def prepare_inputs(model, scene_data: dict, device: str):
    """
    Run encoder with proper CFG doubling.
    Returns decoder_inputs with batch=2 (conditional + unconditional) for CFG.
    """
    inputs = {
        'neighbor_past': torch.from_numpy(scene_data['neighbor_agents_past']).float().unsqueeze(0).to(device),
        'lanes': torch.from_numpy(scene_data['lanes']).float().unsqueeze(0).to(device),
        'lanes_speedlimit': torch.from_numpy(scene_data['lanes_speed_limit']).float().unsqueeze(0).to(device),
        'lanes_has_speedlimit': torch.from_numpy(scene_data['lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'routes': torch.from_numpy(scene_data['route_lanes']).float().unsqueeze(0).to(device),
        'map_objects': torch.from_numpy(scene_data['static_objects']).float().unsqueeze(0).to(device),
        'ego_current': torch.from_numpy(scene_data['ego_current_state']).float().unsqueeze(0).to(device),
    }

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
        nb = inputs_cfg['neighbor_past']
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

    encoder_inputs = model.extract_encoder_inputs(inputs_cfg)
    encoder_outputs = model.encoder(**encoder_inputs)
    decoder_inputs = model.extract_decoder_inputs(encoder_outputs, inputs_cfg)
    return decoder_inputs


def sample_trajectories(model, decoder_inputs, device, num_samples,
                        use_sde=False, sigma_base=0.3, sde_steps=20,
                        cfg_weight=1.8):
    """Generate num_samples trajectories using ODE or SDE."""
    from flow_planner.model.model_utils.traj_tool import assemble_actions

    B = 1
    trajectories = []

    saved_cfg_weight = model.flow_ode.cfg_weight
    model.flow_ode.cfg_weight = cfg_weight

    for s in range(num_samples):
        torch.manual_seed(s * 7919 + 42)
        x_init = torch.randn(
            (B, model.action_num, model.planner_params['action_len'],
             model.planner_params['state_dim']),
            device=device,
        )

        with torch.no_grad():
            if use_sde:
                sample = model.flow_ode.generate_sde(
                    x_init, model.decoder, model._model_type,
                    use_cfg=True, cfg_weight=cfg_weight,
                    sigma_base=sigma_base, sde_steps=sde_steps,
                    **decoder_inputs,
                )
            else:
                sample = model.flow_ode.generate(
                    x_init, model.decoder, model._model_type,
                    use_cfg=True,
                    **decoder_inputs,
                )

        sample = assemble_actions(
            sample, model.planner_params['future_len'],
            model.planner_params['action_len'],
            model.planner_params['action_overlap'],
            model.planner_params['state_dim'],
            model.assemble_method,
        )
        sample = model.data_processor.state_postprocess(sample)
        traj = sample.squeeze(0).cpu().numpy()
        if traj.ndim == 3:
            traj = traj[0]
        trajectories.append(traj)

    model.flow_ode.cfg_weight = saved_cfg_weight

    return trajectories


def compute_diversity_metrics(trajectories):
    """Compute diversity metrics for a set of candidate trajectories."""
    K = len(trajectories)
    if K < 2:
        return {}

    T = min(t.shape[0] for t in trajectories)
    xy = np.stack([t[:T, :2] for t in trajectories], axis=0)  # (K, T, 2)

    # 1. Pairwise RMSE (meters)
    dists = []
    for a in range(K):
        for b in range(a + 1, K):
            diff = xy[a] - xy[b]
            dists.append(np.sqrt(np.mean(diff ** 2)))
    pairwise_rmse = np.mean(dists)

    # 2. Endpoint spread: std of final positions
    endpoints = xy[:, -1, :]  # (K, 2)
    endpoint_spread = np.sqrt(np.var(endpoints[:, 0]) + np.var(endpoints[:, 1]))

    # 3. Max lateral spread at any timestep
    lateral_spreads = []
    for t_idx in range(T):
        positions = xy[:, t_idx, :]  # (K, 2)
        spread = np.sqrt(np.var(positions[:, 0]) + np.var(positions[:, 1]))
        lateral_spreads.append(spread)
    max_lateral_spread = np.max(lateral_spreads)

    # 4. Midpoint spread (t=T//2)
    mid = T // 2
    mid_positions = xy[:, mid, :]
    midpoint_spread = np.sqrt(np.var(mid_positions[:, 0]) + np.var(mid_positions[:, 1]))

    return {
        'pairwise_rmse': pairwise_rmse,
        'endpoint_spread': endpoint_spread,
        'max_lateral_spread': max_lateral_spread,
        'midpoint_spread': midpoint_spread,
    }


def compute_reward_variance(trajectories, scene_data):
    """Compute TrajectoryScorer reward variance across candidates."""
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    K = len(trajectories)
    T = min(t.shape[0] for t in trajectories)
    D = min(t.shape[1] for t in trajectories)
    traj_tensor = torch.from_numpy(
        np.stack([t[:T, :D] for t in trajectories], axis=0)
    ).float()

    scorer = TrajectoryScorer(
        collision_threshold=2.0, ttc_threshold=3.0, dt=0.1,
    )

    nb_past = torch.from_numpy(scene_data['neighbor_agents_past']).float()
    nb_future = TrajectoryScorer.extrapolate_neighbor_future(
        nb_past, future_steps=T, dt=0.5,
    )

    collision_scores = scorer._collision_score(traj_tensor, nb_future).numpy()
    ttc_scores = scorer._ttc_score(traj_tensor, nb_future).numpy()
    comfort_scores = scorer._comfort_score(traj_tensor).numpy()
    progress_scores = scorer._progress_score(traj_tensor).numpy()

    composite = (5 * collision_scores + 5 * ttc_scores +
                 2 * comfort_scores + 5 * progress_scores)

    return {
        'reward_std': float(np.std(composite)),
        'reward_range': float(np.max(composite) - np.min(composite)),
        'collision_std': float(np.std(collision_scores)),
        'ttc_std': float(np.std(ttc_scores)),
        'comfort_std': float(np.std(comfort_scores)),
        'progress_std': float(np.std(progress_scores)),
        'num_distinct_collision': int(np.sum(collision_scores < 0.5)),
    }


def main():
    parser = argparse.ArgumentParser(description='Measure SDE vs ODE trajectory diversity')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scene_dir', type=str, required=True)
    parser.add_argument('--num_scenes', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Trajectories to generate per scene per method')
    parser.add_argument('--sigma_base', type=str, default='0.1,0.3,0.5',
                        help='Comma-separated sigma values to test')
    parser.add_argument('--sde_steps', type=int, default=20)
    parser.add_argument('--cfg_weight', type=float, default=1.8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    sigma_values = [float(s) for s in args.sigma_base.split(',')]
    logger.info(f"Testing sigma values: {sigma_values}")
    logger.info(f"SDE steps: {args.sde_steps}, samples/scene: {args.num_samples}")

    model = load_model(args.config_path, args.ckpt_path, args.device)
    logger.info("Model loaded")

    scene_files = sorted(glob.glob(os.path.join(args.scene_dir, '*.npz')))
    random.seed(42)
    if len(scene_files) > args.num_scenes:
        scene_files = random.sample(scene_files, args.num_scenes)
    logger.info(f"Testing on {len(scene_files)} scenes")

    # method → list of per-scene metrics
    results = {'ode': []}
    for sigma in sigma_values:
        results[f'sde_σ={sigma}'] = []

    reward_results = {'ode': []}
    for sigma in sigma_values:
        reward_results[f'sde_σ={sigma}'] = []

    start = time.time()

    for i, sf in enumerate(scene_files):
        try:
            scene_data = dict(np.load(sf, allow_pickle=True))

            with torch.no_grad():
                decoder_inputs = prepare_inputs(model, scene_data, args.device)

            # ODE baseline
            ode_trajs = sample_trajectories(
                model, decoder_inputs, args.device,
                num_samples=args.num_samples, use_sde=False,
                cfg_weight=args.cfg_weight,
            )
            ode_div = compute_diversity_metrics(ode_trajs)
            results['ode'].append(ode_div)

            ode_rew = compute_reward_variance(ode_trajs, scene_data)
            reward_results['ode'].append(ode_rew)

            # SDE with different sigma values
            for sigma in sigma_values:
                sde_trajs = sample_trajectories(
                    model, decoder_inputs, args.device,
                    num_samples=args.num_samples, use_sde=True,
                    sigma_base=sigma, sde_steps=args.sde_steps,
                    cfg_weight=args.cfg_weight,
                )
                key = f'sde_σ={sigma}'
                sde_div = compute_diversity_metrics(sde_trajs)
                results[key].append(sde_div)

                sde_rew = compute_reward_variance(sde_trajs, scene_data)
                reward_results[key].append(sde_rew)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                # Quick preview
                ode_rmse = np.mean([r['pairwise_rmse'] for r in results['ode']])
                sde_preview = []
                for sigma in sigma_values:
                    key = f'sde_σ={sigma}'
                    avg = np.mean([r['pairwise_rmse'] for r in results[key]])
                    sde_preview.append(f"σ={sigma}:{avg:.3f}")
                logger.info(
                    f"[{i+1}/{len(scene_files)}] "
                    f"ODE_rmse={ode_rmse:.3f}m | SDE: {' '.join(sde_preview)} | "
                    f"{rate:.1f} scenes/s"
                )

        except Exception as e:
            logger.warning(f"Scene {sf}: {e}")
            continue

    # === Final Report ===
    elapsed = time.time() - start
    print("\n" + "=" * 72)
    print("  SDE Diversity Measurement Report")
    print("=" * 72)
    print(f"  Scenes: {len(results['ode'])}  |  Samples/scene: {args.num_samples}")
    print(f"  SDE steps: {args.sde_steps}  |  CFG weight: {args.cfg_weight}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print()

    # Geometry diversity table
    header = f"{'Method':<16} {'RMSE(m)':>8} {'EndPt(m)':>9} {'MaxLat(m)':>10} {'MidPt(m)':>9}"
    print(header)
    print("-" * len(header))

    for method in results:
        metrics_list = results[method]
        if not metrics_list:
            continue
        avg_rmse = np.mean([m['pairwise_rmse'] for m in metrics_list])
        avg_end = np.mean([m['endpoint_spread'] for m in metrics_list])
        avg_lat = np.mean([m['max_lateral_spread'] for m in metrics_list])
        avg_mid = np.mean([m['midpoint_spread'] for m in metrics_list])
        print(f"{method:<16} {avg_rmse:>8.4f} {avg_end:>9.4f} {avg_lat:>10.4f} {avg_mid:>9.4f}")

    print()

    # Reward variance table
    header2 = (f"{'Method':<16} {'Rew_std':>8} {'Rew_rng':>8} "
               f"{'Col_std':>8} {'TTC_std':>8} {'Cmf_std':>8} {'Prg_std':>8} {'#Collide':>8}")
    print(header2)
    print("-" * len(header2))

    for method in reward_results:
        rlist = reward_results[method]
        if not rlist:
            continue
        print(
            f"{method:<16} "
            f"{np.mean([r['reward_std'] for r in rlist]):>8.4f} "
            f"{np.mean([r['reward_range'] for r in rlist]):>8.4f} "
            f"{np.mean([r['collision_std'] for r in rlist]):>8.4f} "
            f"{np.mean([r['ttc_std'] for r in rlist]):>8.4f} "
            f"{np.mean([r['comfort_std'] for r in rlist]):>8.4f} "
            f"{np.mean([r['progress_std'] for r in rlist]):>8.4f} "
            f"{np.mean([r['num_distinct_collision'] for r in rlist]):>8.1f}"
        )

    print()
    print("=" * 72)

    # Decision guidance
    if not results['ode']:
        print("  ERROR: No scenes processed successfully. Check data paths and model.")
        print("=" * 72)
        return

    ode_rmse = np.mean([m['pairwise_rmse'] for m in results['ode']])
    best_sigma = None
    best_improvement = 0
    for sigma in sigma_values:
        key = f'sde_σ={sigma}'
        sde_rmse = np.mean([m['pairwise_rmse'] for m in results[key]])
        improvement = sde_rmse / max(ode_rmse, 1e-6)
        sde_rew_std = np.mean([r['reward_std'] for r in reward_results[key]])
        ode_rew_std = np.mean([r['reward_std'] for r in reward_results['ode']])

        if sde_rew_std > ode_rew_std and improvement > best_improvement:
            best_improvement = improvement
            best_sigma = sigma

    if best_sigma is not None:
        print(f"  RECOMMENDATION: Use sigma_base={best_sigma}")
        print(f"  Diversity improvement: {best_improvement:.1f}x over ODE")
        key = f'sde_σ={best_sigma}'
        sde_rew_std = np.mean([r['reward_std'] for r in reward_results[key]])
        ode_rew_std = np.mean([r['reward_std'] for r in reward_results['ode']])
        print(f"  Reward std: ODE={ode_rew_std:.4f} → SDE={sde_rew_std:.4f}")
        print(f"  → Proceed to DPO pair mining with --sde --sigma_base {best_sigma}")
    else:
        print("  WARNING: No sigma value improved reward variance over ODE.")
        print("  SDE may not be effective for this model/scenario set.")
        print("  Consider: increase sde_steps, try different sigma range, or explore GRPO.")
    print("=" * 72)


if __name__ == '__main__':
    main()

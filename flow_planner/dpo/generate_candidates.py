#!/usr/bin/env python3
"""
DPO Candidate Generation Script (AutoDL)
=========================================
对筛选后的 NPZ 场景生成 K 条候选轨迹，保存为偏好对生成的原材料。

用法 (在 AutoDL 上):
  python /root/scripts/generate_candidates.py \
      --data_dir /root/autodl-tmp/dpo_mining \
      --config_path /root/Flow-Planner/checkpoints/config.yaml \
      --ckpt_path /root/Flow-Planner/checkpoints/model.pth \
      --output_dir /root/autodl-tmp/dpo_candidates \
      --num_candidates 5
"""

import os
import sys
import glob
import json
import argparse
import logging
import time
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, '/root/Flow-Planner')

logger = logging.getLogger(__name__)


def load_model(config_path, ckpt_path, device='cuda'):
    """Load FlowPlanner model from config + checkpoint."""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)

    model = instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'ema_state_dict' in ckpt:
        sd = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    state_dict = {k.replace('module.', ''): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing {len(missing)} keys: {missing[:3]}")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:3]}")

    model = model.to(device).eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model


def npz_to_datasample(npz_path, device='cuda'):
    """Convert a raw NPZ file to NuPlanDataSample (batched=True, B=1)."""
    from flow_planner.data.dataset.nuplan import NuPlanDataSample

    data = np.load(npz_path)

    return NuPlanDataSample(
        batched=True,
        ego_past=torch.from_numpy(data['ego_agent_past']).unsqueeze(0).to(device),
        ego_current=torch.from_numpy(data['ego_current_state']).unsqueeze(0).to(device),
        ego_future=torch.from_numpy(data['ego_agent_future']).float().unsqueeze(0).to(device),
        neighbor_past=torch.from_numpy(data['neighbor_agents_past'][:32]).unsqueeze(0).to(device),
        neighbor_future=torch.zeros(1, 0, 80, 3, device=device),  # empty (predicted_neighbor_num=0)
        neighbor_future_observed=torch.from_numpy(data['neighbor_agents_future']).unsqueeze(0).to(device),
        lanes=torch.from_numpy(data['lanes']).unsqueeze(0).to(device),
        lanes_speedlimit=torch.from_numpy(data['lanes_speed_limit']).unsqueeze(0).to(device),
        lanes_has_speedlimit=torch.from_numpy(data['lanes_has_speed_limit']).unsqueeze(0).to(device),
        routes=torch.from_numpy(data['route_lanes']).unsqueeze(0).to(device),
        routes_speedlimit=torch.from_numpy(data['route_lanes_speed_limit']).unsqueeze(0).to(device),
        routes_has_speedlimit=torch.from_numpy(data['route_lanes_has_speed_limit']).unsqueeze(0).to(device),
        map_objects=torch.from_numpy(data['static_objects']).unsqueeze(0).to(device),
    )


def generate_candidates(model, npz_path, num_candidates=5, device='cuda'):
    """Generate K candidate trajectories for one scenario."""
    data = npz_to_datasample(npz_path, device=device)

    with torch.no_grad():
        # use_cfg=False to get diverse trajectories
        candidates = model(
            data, mode='inference',
            use_cfg=False, cfg_weight=0.0,
            num_candidates=num_candidates,
            return_all_candidates=True,
            bon_seed=42,
        )
        # candidates: (B=1, N, T, D) → (N, T, D)
        candidates = candidates.squeeze(0).cpu().numpy()

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--config_path', type=str,
                        default='/root/Flow-Planner/checkpoints/config.yaml')
    parser.add_argument('--ckpt_path', type=str,
                        default='/root/Flow-Planner/checkpoints/model.pth')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save candidate trajectories')
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--max_scenarios', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all NPZ files
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    if args.max_scenarios:
        npz_files = npz_files[:args.max_scenarios]
    logger.info(f"Found {len(npz_files)} NPZ files")

    # Load model
    model = load_model(args.config_path, args.ckpt_path)

    # Generate candidates
    t0 = time.time()
    success = 0
    fail = 0
    for i, npz_path in enumerate(npz_files):
        fname = os.path.basename(npz_path).replace('.npz', '')
        out_path = os.path.join(args.output_dir, fname + '_candidates.npz')

        if os.path.exists(out_path):
            success += 1
            continue

        try:
            candidates = generate_candidates(
                model, npz_path,
                num_candidates=args.num_candidates,
            )

            # Also save the condition data for later BEV rendering
            raw = np.load(npz_path)
            np.savez_compressed(
                out_path,
                candidates=candidates,  # (K, T, D)
                ego_agent_past=raw['ego_agent_past'],
                ego_current_state=raw['ego_current_state'],
                ego_agent_future=raw['ego_agent_future'],
                neighbor_agents_past=raw['neighbor_agents_past'],
                lanes=raw['lanes'],
                token=raw['token'],
            )
            success += 1

        except Exception as e:
            logger.warning(f"[{i}] Failed on {fname}: {e}")
            fail += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(npz_files) - i - 1) / rate
            logger.info(
                f"[{i+1}/{len(npz_files)}] "
                f"success={success} fail={fail} "
                f"rate={rate:.1f} it/s ETA={eta/60:.0f}min"
            )

    elapsed = time.time() - t0
    logger.info(f"Done: {success} success, {fail} fail, {elapsed/60:.1f} min total")


if __name__ == '__main__':
    main()

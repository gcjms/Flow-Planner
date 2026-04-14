#!/usr/bin/env python3
"""
DPO Candidate Generation with Goal-Diverse Sampling (Step 3)
=============================================================
用不同的 goal point 生成决策级不同的候选轨迹，用于 DPO 偏好对构建。

与原版 generate_candidates.py 的区别:
  - 原版: 5 条轨迹只靠不同随机种子 → 几乎一样
  - 本版: 5 条轨迹各给不同的 goal point → 左绕/右绕/刹停等不同行为

前提: 需要先完成 Step 1 (cluster_goals.py) 和 Step 2 (重训带 goal 的模型)

用法:
  python -m flow_planner.dpo.generate_candidates_goal \
      --data_dir /root/autodl-tmp/dpo_mining \
      --config_path /root/Flow-Planner/checkpoints/config_goal.yaml \
      --ckpt_path /root/Flow-Planner/checkpoints/model_goal.pth \
      --vocab_path /root/Flow-Planner/goal_vocab.npy \
      --output_dir /root/autodl-tmp/dpo_candidates_goal \
      --num_candidates 5
"""

import os
import sys
import glob
import argparse
import logging
import time
import torch
import numpy as np

sys.path.insert(0, '/root/Flow-Planner')

from flow_planner.goal.goal_utils import load_goal_vocab, select_diverse_goals

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
        logger.warning(f"Missing {len(missing)} keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys: {unexpected[:5]}")

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
        neighbor_future=torch.zeros(1, 0, 80, 3, device=device),
        neighbor_future_observed=torch.from_numpy(data['neighbor_agents_future']).unsqueeze(0).to(device),
        lanes=torch.from_numpy(data['lanes']).unsqueeze(0).to(device),
        lanes_speedlimit=torch.from_numpy(data['lanes_speed_limit']).unsqueeze(0).to(device),
        lanes_has_speedlimit=torch.from_numpy(data['lanes_has_speed_limit']).unsqueeze(0).to(device),
        routes=torch.from_numpy(data['route_lanes']).unsqueeze(0).to(device),
        routes_speedlimit=torch.from_numpy(data['route_lanes_speed_limit']).unsqueeze(0).to(device),
        routes_has_speedlimit=torch.from_numpy(data['route_lanes_has_speed_limit']).unsqueeze(0).to(device),
        map_objects=torch.from_numpy(data['static_objects']).unsqueeze(0).to(device),
    )


def generate_candidates_with_goals(
    model, npz_path, vocab, num_candidates=5,
    device='cuda', use_cfg=True, cfg_weight=1.8,
):
    """
    用不同的 goal point 生成 K 条候选轨迹。

    Returns:
        candidates: (K, T, D) numpy array
        goal_labels: (K, 2) numpy array — 每条轨迹对应的 goal point
    """
    data = npz_to_datasample(npz_path, device=device)

    # 选择 K 个多样化的 goal point
    _, goals = select_diverse_goals(vocab, n_goals=num_candidates)
    # goals: (K, 2)

    all_trajs = []
    with torch.no_grad():
        for i in range(num_candidates):
            gp = torch.from_numpy(goals[i:i+1]).float().to(device)  # (1, 2)

            traj = model(
                data, mode='inference',
                use_cfg=use_cfg, cfg_weight=cfg_weight,
                num_candidates=1,
                return_all_candidates=False,
                bon_seed=42 + i,
                goal_point=gp,
            )
            # traj: (B=1, T, D)
            all_trajs.append(traj.squeeze(0).cpu().numpy())

    candidates = np.stack(all_trajs, axis=0)  # (K, T, D)
    return candidates, goals


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO candidates with goal-diverse sampling"
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to goal_vocab.npy from Step 1')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_candidates', type=int, default=5)
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--use_cfg', action='store_true', default=True,
                        help='Enable CFG (default: True)')
    parser.add_argument('--no_cfg', dest='use_cfg', action='store_false',
                        help='Disable CFG')
    parser.add_argument('--cfg_weight', type=float, default=1.8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load vocab
    vocab = load_goal_vocab(args.vocab_path)
    logger.info(f"Goal vocabulary: {vocab.shape[0]} clusters")

    # Find NPZ files
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    if args.max_scenarios:
        npz_files = npz_files[:args.max_scenarios]
    logger.info(f"Found {len(npz_files)} NPZ files")

    # Load model
    model = load_model(args.config_path, args.ckpt_path)

    # Generate
    t0 = time.time()
    success, fail = 0, 0
    for i, npz_path in enumerate(npz_files):
        fname = os.path.basename(npz_path).replace('.npz', '')
        out_path = os.path.join(args.output_dir, fname + '_candidates.npz')

        if os.path.exists(out_path):
            success += 1
            continue

        try:
            candidates, goal_labels = generate_candidates_with_goals(
                model, npz_path, vocab,
                num_candidates=args.num_candidates,
                use_cfg=args.use_cfg, cfg_weight=args.cfg_weight,
            )

            raw = np.load(npz_path)
            np.savez_compressed(
                out_path,
                candidates=candidates,        # (K, T, D)
                goal_labels=goal_labels,       # (K, 2) — 每条轨迹的 goal point
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

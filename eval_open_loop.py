"""
Open-Loop Evaluation Script for Flow-Planner
Computes ADE, FDE, and other metrics on preprocessed npz data.

Usage:
    python eval_open_loop.py \
        --config_path <path_to_.hydra/config.yaml> \
        --ckpt_path <path_to_checkpoint.pth> \
        --data_dir <path_to_npz_data> \
        --data_list <path_to_json_list> \
        --max_samples 200
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import omegaconf
from hydra.utils import instantiate

from flow_planner.data.data_process.utils import convert_to_model_inputs
from flow_planner.data.dataset.nuplan import NuPlanDataSample


def load_model(config_path, ckpt_path, device='cuda', use_ema=True):
    config = omegaconf.OmegaConf.load(config_path)
    model = instantiate(config.model)

    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    if use_ema:
        sd = state_dict['ema_state_dict']
    else:
        sd = state_dict.get('model', state_dict)

    model_sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
    if not model_sd:
        model_sd = sd
    model.load_state_dict(model_sd)
    model.eval()
    model = model.to(device)
    return model, config


def npz_to_data_sample(npz, device='cuda'):
    """Convert npz data to NuPlanDataSample, matching observation_adapter output format."""
    raw = {
        "ego_agent_past": npz['ego_agent_past'],
        "ego_current_state": npz['ego_current_state'],
        "neighbor_agents_past": npz['neighbor_agents_past'],
        "static_objects": npz['static_objects'],
        "lanes": npz['lanes'],
        "lanes_speed_limit": npz['lanes_speed_limit'],
        "lanes_has_speed_limit": npz['lanes_has_speed_limit'],
        "route_lanes": npz['route_lanes'],
        "route_lanes_speed_limit": npz['route_lanes_speed_limit'],
        "route_lanes_has_speed_limit": npz['route_lanes_has_speed_limit'],
    }
    m = convert_to_model_inputs(raw, device)

    return NuPlanDataSample(
        batched=(m['ego_current_state'].dim() > 1),
        ego_past=m['ego_agent_past'],
        ego_current=m['ego_current_state'],
        neighbor_past=m['neighbor_agents_past'],
        lanes=m['lanes'],
        lanes_speedlimit=m['lanes_speed_limit'],
        lanes_has_speedlimit=m['lanes_has_speed_limit'],
        routes=m['route_lanes'],
        routes_speedlimit=m['route_lanes_speed_limit'],
        routes_has_speedlimit=m['route_lanes_has_speed_limit'],
        map_objects=m['static_objects'],
    )


@torch.no_grad()
def evaluate(model, data_dir, data_list, device='cuda', max_samples=None,
             use_cfg=True, cfg_weight=1.8):
    with open(data_list) as f:
        files = json.load(f)
    if max_samples:
        files = files[:max_samples]

    all_ade, all_fde = [], []
    all_ade_1s, all_ade_3s = [], []
    all_heading_err = []
    all_lateral_err, all_longitudinal_err = [], []

    for fname in tqdm(files, desc="Evaluating"):
        npz = np.load(os.path.join(data_dir, fname))
        gt_future = npz['ego_agent_future']  # (80, 3): x, y, heading

        data = npz_to_data_sample(npz, device)

        outputs = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight)
        pred = outputs[0, 0].cpu().numpy()  # (80, 4): x, y, cos_h, sin_h

        # Convert prediction to (x, y, heading)
        pred_xy = pred[:, :2]
        pred_heading = np.arctan2(pred[:, 3], pred[:, 2])

        gt_xy = gt_future[:, :2]
        gt_heading = gt_future[:, 2]

        # ADE & FDE
        errors = np.linalg.norm(pred_xy - gt_xy, axis=-1)
        ade = errors.mean()
        fde = errors[-1]
        all_ade.append(ade)
        all_fde.append(fde)

        # ADE @1s (first 10 frames) and @3s (first 30 frames)
        all_ade_1s.append(errors[:10].mean())
        all_ade_3s.append(errors[:30].mean())

        # Heading error (rad)
        h_err = np.abs(pred_heading - gt_heading)
        h_err = np.minimum(h_err, 2 * np.pi - h_err)
        all_heading_err.append(h_err.mean())

        # Lateral / Longitudinal error (decompose along GT heading)
        diff = pred_xy - gt_xy
        cos_h = np.cos(gt_heading)
        sin_h = np.sin(gt_heading)
        longitudinal = diff[:, 0] * cos_h + diff[:, 1] * sin_h
        lateral = -diff[:, 0] * sin_h + diff[:, 1] * cos_h
        all_lateral_err.append(np.abs(lateral).mean())
        all_longitudinal_err.append(np.abs(longitudinal).mean())

    results = {
        'num_samples': len(files),
        'ADE (m)': np.mean(all_ade),
        'FDE (m)': np.mean(all_fde),
        'ADE@1s (m)': np.mean(all_ade_1s),
        'ADE@3s (m)': np.mean(all_ade_3s),
        'Heading Error (rad)': np.mean(all_heading_err),
        'Heading Error (deg)': np.degrees(np.mean(all_heading_err)),
        'Lateral Error (m)': np.mean(all_lateral_err),
        'Longitudinal Error (m)': np.mean(all_longitudinal_err),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Flow-Planner Open-Loop Evaluation')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to .hydra/config.yaml from training')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npz files')
    parser.add_argument('--data_list', type=str, required=True,
                        help='Path to JSON file listing npz filenames')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_ema', action='store_true',
                        help='Use raw model weights instead of EMA')
    parser.add_argument('--no_cfg', action='store_true',
                        help='Disable classifier-free guidance')
    parser.add_argument('--cfg_weight', type=float, default=1.8)
    args = parser.parse_args()

    print("Loading model...")
    model, config = load_model(args.config_path, args.ckpt_path,
                               args.device, use_ema=not args.no_ema)
    print(f"Model loaded from {args.ckpt_path}")

    print("Running evaluation...")
    results = evaluate(model, args.data_dir, args.data_list,
                       device=args.device, max_samples=args.max_samples,
                       use_cfg=not args.no_cfg, cfg_weight=args.cfg_weight)

    print("\n" + "=" * 50)
    print("Open-Loop Evaluation Results")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50)


if __name__ == '__main__':
    main()

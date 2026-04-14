"""
On-Policy DPO 偏好对生成器
===========================
对每个训练场景:
  1. 用 FlowPlanner 推理 N 次（不同噪声 seed）
  2. 碰撞检测每条轨迹
  3. 筛选 "borderline" 场景: 有些次碰撞、有些次不碰撞
  4. chosen = 模型自己的不碰撞轨迹, rejected = 模型自己的碰撞轨迹
  → 纯 on-policy 偏好对

用法:
  python -m flow_planner.dpo.generate_onpolicy_pairs \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --output_path dpo_data/onpolicy_pairs.npz \
      --max_scenes 5000 \
      --num_samples 20
"""

import os
import sys
import time
import glob
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==============================================================
# 碰撞检测 (复用 generate_oracle_pairs.py)
# ==============================================================

def check_collision(
    ego_traj: np.ndarray,       # (T, D) 其中 D>=2, [:, :2] = x, y
    neighbor_future: np.ndarray, # (M, T_n, D_n) 其中 [:, :, :2] = x, y
    collision_dist: float = 2.0,
) -> Tuple[bool, int]:
    T_ego = ego_traj.shape[0]
    M, T_nb, _ = neighbor_future.shape
    T = min(T_ego, T_nb)
    
    for t in range(T):
        ex, ey = float(ego_traj[t, 0]), float(ego_traj[t, 1])
        for m in range(M):
            nx, ny = float(neighbor_future[m, t, 0]), float(neighbor_future[m, t, 1])
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                continue
            dist = ((ex - nx)**2 + (ey - ny)**2) ** 0.5
            if dist < collision_dist:
                return True, t
    return False, -1


# ==============================================================
# 模型加载
# ==============================================================

def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    
    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    
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


def run_inference_multi(
    model, scene_data: dict, device: str, num_samples: int = 20
) -> List[np.ndarray]:
    """对单个场景推理 num_samples 次，每次用不同随机噪声"""
    
    # 构建输入
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
    
    trajectories = []
    
    with torch.no_grad():
        # Encoder 只跑一次
        encoder_inputs = model.extract_encoder_inputs(inputs)
        encoder_outputs = model.encoder(**encoder_inputs)
        decoder_inputs = model.extract_decoder_inputs(encoder_outputs, inputs)
        
        from flow_planner.model.model_utils.traj_tool import assemble_actions
        
        B = 1
        for s in range(num_samples):
            # 每次用不同的随机种子生成初始噪声
            torch.manual_seed(s * 12345 + int(time.time()) % 10000)
            x_init = torch.randn(
                (B, model.action_num, model.planner_params['action_len'],
                 model.planner_params['state_dim']),
                device=device
            )
            
            sample = model.flow_ode.generate(
                x_init, model.decoder, model._model_type,
                use_cfg=False, cfg_weight=1.0,
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
            trajectories.append(result)  # (T, D)
    
    return trajectories


# ==============================================================
# 主流程
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate On-Policy DPO preference pairs')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scene_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='dpo_data/onpolicy_pairs.npz')
    parser.add_argument('--max_scenes', type=int, default=5000)
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of inference runs per scene with different noise seeds')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # 加载模型
    model = load_model(args.config_path, args.ckpt_path, args.device)
    logger.info("Model loaded successfully")
    
    # 获取场景文件列表并随机采样
    all_scene_files = sorted(glob.glob(os.path.join(args.scene_dir, '*.npz')))
    random.seed(42)
    if len(all_scene_files) > args.max_scenes:
        scene_files = random.sample(all_scene_files, args.max_scenes)
    else:
        scene_files = all_scene_files
    logger.info(f"Processing {len(scene_files)} scenes (sampled from {len(all_scene_files)} total)")
    
    # 统计
    total = 0
    borderline_count = 0
    all_collide_count = 0
    none_collide_count = 0
    
    chosen_list = []
    rejected_list = []
    scenario_ids = []
    diversity_stats = []  # 记录每个场景的轨迹多样性
    
    start_time = time.time()
    
    for i, scene_file in enumerate(scene_files):
        try:
            scene_data = dict(np.load(scene_file, allow_pickle=True))
            nb_future = scene_data['neighbor_agents_future']  # (M, T, 3)
            
            # 推理 N 次
            trajectories = run_inference_multi(
                model, scene_data, args.device, args.num_samples
            )
            
            # 碰撞检测每条轨迹
            collided_trajs = []
            safe_trajs = []
            for traj in trajectories:
                hit, step = check_collision(traj, nb_future)
                if hit:
                    collided_trajs.append((traj, step))
                else:
                    safe_trajs.append(traj)
            
            # 计算轨迹多样性 (pairwise RMSE)
            if len(trajectories) >= 2:
                dists = []
                for a in range(min(5, len(trajectories))):
                    for b in range(a+1, min(5, len(trajectories))):
                        T = min(trajectories[a].shape[0], trajectories[b].shape[0])
                        diff = trajectories[a][:T, :2] - trajectories[b][:T, :2]
                        rmse = np.sqrt(np.mean(diff**2))
                        dists.append(rmse)
                avg_diversity = np.mean(dists)
                diversity_stats.append(avg_diversity)
            
            total += 1
            n_collide = len(collided_trajs)
            n_safe = len(safe_trajs)
            
            if n_collide > 0 and n_safe > 0:
                # Borderline 场景！构建偏好对
                borderline_count += 1
                
                # 选最好的 safe 轨迹（离 GT 最近的）
                gt_traj = scene_data['ego_agent_future'][:, :2]
                best_safe = None
                best_dist = float('inf')
                for st in safe_trajs:
                    T = min(st.shape[0], gt_traj.shape[0])
                    d = np.mean(np.sqrt(np.sum((st[:T, :2] - gt_traj[:T])**2, axis=-1)))
                    if d < best_dist:
                        best_dist = d
                        best_safe = st
                
                # 选最"接近安全"的碰撞轨迹（碰撞时间步最晚的）
                worst_collide = max(collided_trajs, key=lambda x: x[1])
                
                T = min(best_safe.shape[0], worst_collide[0].shape[0])
                D = min(best_safe.shape[1], worst_collide[0].shape[1])
                
                chosen_list.append(best_safe[:T, :D])
                rejected_list.append(worst_collide[0][:T, :D])
                
                sid = str(scene_data.get('token', Path(scene_file).stem))
                scenario_ids.append(sid)
                
            elif n_collide == len(trajectories):
                all_collide_count += 1
            else:
                none_collide_count += 1
            
            # 进度报告
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                n_pairs = len(chosen_list)
                avg_div = np.mean(diversity_stats[-50:]) if diversity_stats else 0
                logger.info(
                    f"[{i+1}/{len(scene_files)}] "
                    f"Borderline: {borderline_count} | "
                    f"AllCollide: {all_collide_count} | "
                    f"NoneCollide: {none_collide_count} | "
                    f"Pairs: {n_pairs} | "
                    f"AvgDiversity(last50): {avg_div:.4f}m | "
                    f"Speed: {rate:.1f} scenes/s | "
                    f"ETA: {(len(scene_files)-i-1)/rate/60:.1f}min"
                )
                
        except Exception as e:
            logger.warning(f"Scene {scene_file}: {e}")
            continue
    
    # 保存结果
    elapsed = time.time() - start_time
    n_pairs = len(chosen_list)
    
    logger.info("=" * 60)
    logger.info(f"On-Policy Mining Complete!")
    logger.info(f"  Total scenes processed: {total}")
    logger.info(f"  Borderline scenes (some collide, some don't): {borderline_count} ({borderline_count/max(total,1)*100:.1f}%)")
    logger.info(f"  All collide (no safe trajectory found): {all_collide_count} ({all_collide_count/max(total,1)*100:.1f}%)")
    logger.info(f"  None collide (all safe): {none_collide_count} ({none_collide_count/max(total,1)*100:.1f}%)")
    logger.info(f"  Valid on-policy pairs: {n_pairs}")
    logger.info(f"  Avg trajectory diversity: {np.mean(diversity_stats):.4f}m" if diversity_stats else "  No diversity data")
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
            metadata=np.array({
                'total_scenes': total,
                'borderline_scenes': borderline_count,
                'all_collide_scenes': all_collide_count,
                'none_collide_scenes': none_collide_count,
                'valid_pairs': n_pairs,
                'num_samples_per_scene': args.num_samples,
                'avg_diversity_m': float(np.mean(diversity_stats)) if diversity_stats else 0,
            }),
        )
        logger.info(f"Saved {n_pairs} on-policy pairs to {args.output_path}")
        logger.info(f"  chosen: {chosen_arr.shape}")
        logger.info(f"  rejected: {rejected_arr.shape}")
    else:
        logger.warning("No valid on-policy pairs generated!")
        logger.warning("This likely means trajectory diversity is too low.")
        logger.warning("Consider: increasing num_samples, or using SDE sampling.")


if __name__ == '__main__':
    main()

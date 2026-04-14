"""
Oracle DPO 偏好对生成器
========================
对每个 hard scenario:
  1. 加载场景 NPZ
  2. 用 FlowPlanner 跑一次开环推理 → 模型轨迹
  3. 碰撞检测：模型轨迹 vs 邻居 GT 未来位置
  4. 如果碰撞：GT轨迹=chosen，模型轨迹=rejected

用法：
  python -m flow_planner.dpo.generate_oracle_pairs \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --scene_dir /path/to/hard_scenarios_v2 \
      --output_path dpo_data/oracle_pairs.npz \
      --max_scenes 5000
"""

import os
import sys
import time
import glob
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==============================================================
# 碰撞检测
# ==============================================================

def check_collision(
    ego_traj: np.ndarray,       # (T, D) 其中 D>=2, [:, :2] = x, y
    neighbor_future: np.ndarray, # (M, T_n, D_n) 其中 [:, :, :2] = x, y
    collision_dist: float = 2.0,  # 两车中心距小于此值视为碰撞
) -> Tuple[bool, int]:
    """
    欧氏距离碰撞检测。
    
    两车中心距 < collision_dist 即碰撞。
    3.0m ≈ 一个车长的安全距离。
    
    Returns:
        collided: 是否碰撞
        collision_step: 第一次碰撞的时间步 (-1 if no collision)
    """
    T_ego = ego_traj.shape[0]
    M, T_nb, _ = neighbor_future.shape
    T = min(T_ego, T_nb)
    
    for t in range(T):
        ex, ey = float(ego_traj[t, 0]), float(ego_traj[t, 1])
        
        for m in range(M):
            nx, ny = float(neighbor_future[m, t, 0]), float(neighbor_future[m, t, 1])
            
            # 跳过无效邻居 (全 0 表示不存在)
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                continue
            
            # 欧氏距离
            dist = ((ex - nx)**2 + (ey - ny)**2) ** 0.5
            
            if dist < collision_dist:
                return True, t
    
    return False, -1


# ==============================================================
# 模型加载 & 推理
# ==============================================================

def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    """加载 FlowPlanner 模型"""
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    
    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)
    
    model = instantiate(cfg.model)
    
    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
    """
    对单个场景跑模型推理，返回预测轨迹 (T, D)。
    
    使用与 train_dpo.py 相同的 encoder → decoder 流程。
    """
    # 构建输入 dict（模拟 NuPlanDataSample 的结构）
    class FakeData:
        pass
    
    data = FakeData()
    data.ego_current = torch.from_numpy(scene_data['ego_current_state']).float().unsqueeze(0).to(device)
    data.neighbor_past = torch.from_numpy(scene_data['neighbor_agents_past']).float().unsqueeze(0).to(device)
    data.ego_past = torch.from_numpy(scene_data['ego_agent_past']).float().unsqueeze(0).to(device)
    
    # 构建 model_inputs dict
    inputs = {
        'neighbor_past': data.neighbor_past,
        'lanes': torch.from_numpy(scene_data['lanes']).float().unsqueeze(0).to(device),
        'lanes_speedlimit': torch.from_numpy(scene_data['lanes_speed_limit']).float().unsqueeze(0).to(device),
        'lanes_has_speedlimit': torch.from_numpy(scene_data['lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'routes': torch.from_numpy(scene_data['route_lanes']).float().unsqueeze(0).to(device),
        'map_objects': torch.from_numpy(scene_data['static_objects']).float().unsqueeze(0).to(device),
        'ego_current': data.ego_current,
        'cfg_flags': torch.ones(1, device=device, dtype=torch.int32),  # conditioned
    }
    
    with torch.no_grad():
        encoder_inputs = model.extract_encoder_inputs(inputs)
        encoder_outputs = model.encoder(**encoder_inputs)
        decoder_inputs = model.extract_decoder_inputs(encoder_outputs, inputs)
        
        # 生成初始噪声
        B = 1
        x_init = torch.randn(
            (B, model.action_num, model.planner_params['action_len'],
             model.planner_params['state_dim']),
            device=device
        )
        
        # ODE 采样
        from flow_planner.model.model_utils.traj_tool import assemble_actions
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
        # 反归一化到真实坐标
        sample = model.data_processor.state_postprocess(sample)
    
    result = sample.squeeze(0).cpu().numpy()  # (T, D)
    # 确保是 2D
    if result.ndim == 3:
        result = result[0]
    return result  # (T, D)


# ==============================================================
# 主流程
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Oracle DPO preference pairs')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Directory containing hard scenario NPZ files')
    parser.add_argument('--output_path', type=str, default='dpo_data/oracle_pairs.npz')
    parser.add_argument('--max_scenes', type=int, default=5000,
                        help='Maximum number of scenes to process')
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
    
    # 获取场景文件列表
    scene_files = sorted(glob.glob(os.path.join(args.scene_dir, '*.npz')))
    if args.max_scenes:
        scene_files = scene_files[:args.max_scenes]
    logger.info(f"Processing {len(scene_files)} scenes from {args.scene_dir}")
    
    # 统计
    total = 0
    collisions = 0
    gt_collisions = 0
    
    chosen_list = []
    rejected_list = []
    scenario_ids = []
    collision_steps = []
    
    start_time = time.time()
    
    for i, scene_file in enumerate(scene_files):
        try:
            scene_data = dict(np.load(scene_file, allow_pickle=True))
            
            # 模型推理
            pred_traj = run_inference(model, scene_data, args.device)  # (T, D)
            
            # GT 轨迹
            gt_traj = scene_data['ego_agent_future']  # (T, 3) = (x, y, heading)
            
            # 邻居未来轨迹
            nb_future = scene_data['neighbor_agents_future']  # (M, T, 3)
            
            # 碰撞检测：模型轨迹 vs 邻居
            pred_collided, pred_step = check_collision(pred_traj, nb_future)
            
            # 碰撞检测：GT 轨迹 vs 邻居 (sanity check)
            gt_collided, _ = check_collision(gt_traj, nb_future)
            
            total += 1
            if pred_collided:
                collisions += 1
            if gt_collided:
                gt_collisions += 1
            
            # 只有模型碰撞 + GT 不碰撞时，才构成有效偏好对
            if pred_collided and not gt_collided:
                # 取相同长度
                T = min(pred_traj.shape[0], gt_traj.shape[0])
                
                # chosen = GT, 确保维度匹配
                gt_for_pair = gt_traj[:T, :]  # (T, 3)
                pred_for_pair = pred_traj[:T, :]  # (T, D)
                
                # 对齐维度（GT 可能是 3 维，pred 可能是 4 维）
                D = min(gt_for_pair.shape[1], pred_for_pair.shape[1])
                gt_for_pair = gt_for_pair[:, :D]
                pred_for_pair = pred_for_pair[:, :D]
                
                chosen_list.append(gt_for_pair)
                rejected_list.append(pred_for_pair)
                
                sid = str(scene_data.get('token', Path(scene_file).stem))
                scenario_ids.append(sid)
                collision_steps.append(pred_step)
            
            # 进度报告
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                n_pairs = len(chosen_list)
                logger.info(
                    f"[{i+1}/{len(scene_files)}] "
                    f"碰撞率: {collisions}/{total}={collisions/total*100:.1f}% | "
                    f"GT碰撞: {gt_collisions} | "
                    f"有效对: {n_pairs} | "
                    f"速度: {rate:.1f} scenes/s | "
                    f"ETA: {(len(scene_files)-i-1)/rate/60:.1f}min"
                )
                
        except Exception as e:
            logger.warning(f"Scene {scene_file}: {e}")
            continue
    
    # 保存结果
    elapsed = time.time() - start_time
    n_pairs = len(chosen_list)
    
    logger.info("=" * 60)
    logger.info(f"Processing complete!")
    logger.info(f"  Total scenes: {total}")
    logger.info(f"  Model collisions: {collisions} ({collisions/max(total,1)*100:.1f}%)")
    logger.info(f"  GT collisions: {gt_collisions} ({gt_collisions/max(total,1)*100:.1f}%)")
    logger.info(f"  Valid Oracle pairs: {n_pairs}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    
    if n_pairs > 0:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        
        chosen_arr = np.stack(chosen_list, axis=0)    # (N, T, D)
        rejected_arr = np.stack(rejected_list, axis=0)  # (N, T, D)
        
        np.savez(
            args.output_path,
            chosen=chosen_arr,
            rejected=rejected_arr,
            scenario_ids=np.array(scenario_ids),
            collision_steps=np.array(collision_steps),
            metadata=np.array({
                'total_scenes': total,
                'model_collisions': collisions,
                'gt_collisions': gt_collisions,
                'valid_pairs': n_pairs,
            }),
        )
        logger.info(f"Saved {n_pairs} Oracle pairs to {args.output_path}")
        logger.info(f"  chosen: {chosen_arr.shape}")
        logger.info(f"  rejected: {rejected_arr.shape}")
    else:
        logger.warning("No valid Oracle pairs generated!")
    

if __name__ == '__main__':
    main()

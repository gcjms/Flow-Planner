"""
Risk Feature Extraction for Flow Planner
==========================================
从 NuPlanDataSample 或 .npz 文件中提取驾驶风险特征。

特征列表：
    - min_dist:         最近邻车辆距离 (m)
    - n_neighbors:      有效邻居数量
    - ego_speed:        自车速度 (m/s)
    - max_delta_v:      与邻居的最大速度差 (m/s)
    - ttc:              Time-to-Collision (s), 无碰撞风险时 = 999
    - thw:              Time Headway (s), 前方无车时 = 999
    - drac:             Deceleration Rate to Avoid Crash (m/s²)
    - mean_neighbor_speed: 邻居平均速度 (m/s)
    - speed_variance:   邻居速度方差
    - front_vehicle_dist: 前方最近车辆距离 (m)
    - lateral_min_dist: 横向最近车辆距离 (m)
    - rear_approach_speed: 后方最快接近速度 (m/s)

输出: shape (12,) 的特征向量
"""

import torch
import numpy as np
from typing import Union, Optional
import os


# ============================================================
# 特征名列表（用于可视化和分析）
# ============================================================
RISK_FEATURE_NAMES = [
    'min_dist',           # 0: 最近邻距离
    'n_neighbors',        # 1: 有效邻居数
    'ego_speed',          # 2: 自车速度
    'max_delta_v',        # 3: 最大速度差
    'ttc',                # 4: 碰撞时间
    'thw',                # 5: 跟车时距
    'drac',               # 6: 避碰减速度
    'mean_neighbor_speed',# 7: 邻居平均速度
    'speed_variance',     # 8: 邻居速度方差
    'front_vehicle_dist', # 9: 前方最近车距
    'lateral_min_dist',   # 10: 横向最近距离
    'rear_approach_speed',# 11: 后方最快接近速度
]

NUM_RISK_FEATURES = len(RISK_FEATURE_NAMES)

# 截断常量：当没有风险时使用的默认值
_NO_RISK_TTC = 999.0
_NO_RISK_THW = 999.0
_NO_RISK_DIST = 999.0


def compute_speed_from_trajectory(traj: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    从轨迹点序列计算速度。
    
    Args:
        traj: (T, D) 轨迹，D >= 2 (x, y, ...)
        dt: 时间步长 (nuPlan 默认 10Hz = 0.1s)
    
    Returns:
        speeds: (T-1,) 各时间步的速度 (m/s)
    """
    if traj.shape[0] < 2:
        return np.array([0.0])
    
    diffs = np.diff(traj[:, :2], axis=0)  # (T-1, 2)
    speeds = np.linalg.norm(diffs, axis=1) / dt  # (T-1,)
    return speeds


def extract_risk_features_from_npz(npz_path: str) -> np.ndarray:
    """
    从单个 .npz 文件提取风险特征。
    
    Args:
        npz_path: .npz 文件路径
    
    Returns:
        features: (NUM_RISK_FEATURES,) 风险特征向量
    """
    data = np.load(npz_path, allow_pickle=True)
    
    ego_current = data['ego_current_state']     # (D,) 当前ego状态
    ego_past = data['ego_agent_past']            # (T_past, D) ego过去轨迹
    neighbor_past = data['neighbor_agents_past']  # (N, T_past, D) 邻居过去轨迹
    
    return _compute_risk_features(ego_current, ego_past, neighbor_past)


def extract_risk_features_from_sample(
    ego_current: Union[torch.Tensor, np.ndarray],
    ego_past: Union[torch.Tensor, np.ndarray],
    neighbor_past: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    从 NuPlanDataSample 的字段提取风险特征。
    
    Args:
        ego_current: (D,) 或 (B, D)
        ego_past: (T, D) 或 (B, T, D) 
        neighbor_past: (N, T, D) 或 (B, N, T, D)
    
    Returns:
        features: (NUM_RISK_FEATURES,) 或 (B, NUM_RISK_FEATURES)
    """
    if isinstance(ego_current, torch.Tensor):
        ego_current = ego_current.detach().cpu().numpy()
    if isinstance(ego_past, torch.Tensor):
        ego_past = ego_past.detach().cpu().numpy()
    if isinstance(neighbor_past, torch.Tensor):
        neighbor_past = neighbor_past.detach().cpu().numpy()
    
    # Handle batched input
    if ego_current.ndim == 2:
        batch_features = []
        for i in range(ego_current.shape[0]):
            feat = _compute_risk_features(
                ego_current[i], ego_past[i], neighbor_past[i]
            )
            batch_features.append(feat)
        return np.stack(batch_features, axis=0)
    
    return _compute_risk_features(ego_current, ego_past, neighbor_past)


def _compute_risk_features(
    ego_current: np.ndarray,
    ego_past: np.ndarray,
    neighbor_past: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    核心风险特征计算函数。
    
    Args:
        ego_current: (D,) ego 当前状态 [x, y, cos_h, sin_h, ...]
        ego_past: (T_past, D) ego 过去轨迹
        neighbor_past: (N, T_past, D) 邻居过去轨迹
    
    Returns:
        features: (NUM_RISK_FEATURES,) 风险特征向量
    
    注意: 所有坐标都在 ego-centric 坐标系下，
         即 ego 当前位置为原点 (0, 0)，朝向为 x 轴正方向。
    """
    # ego 位置和朝向（在 ego-centric 下为原点）
    ego_pos = np.array([0.0, 0.0])  # ego-centric 坐标系下 ego 在原点
    
    # ---- 计算 ego 速度 ----
    ego_speeds = compute_speed_from_trajectory(ego_past, dt)
    ego_speed = ego_speeds[-1] if len(ego_speeds) > 0 else 0.0
    
    # ---- 识别有效邻居 ----
    # 邻居如果全为0则无效（padding）
    N = neighbor_past.shape[0]
    valid_mask = np.any(np.abs(neighbor_past[:, -1, :2]) > 0.01, axis=1)  # (N,)
    n_neighbors = int(valid_mask.sum())
    
    if n_neighbors == 0:
        # 没有邻居 → 无风险
        return np.array([
            _NO_RISK_DIST,  # min_dist
            0.0,            # n_neighbors
            ego_speed,      # ego_speed
            0.0,            # max_delta_v
            _NO_RISK_TTC,   # ttc
            _NO_RISK_THW,   # thw
            0.0,            # drac
            0.0,            # mean_neighbor_speed
            0.0,            # speed_variance
            _NO_RISK_DIST,  # front_vehicle_dist
            _NO_RISK_DIST,  # lateral_min_dist
            0.0,            # rear_approach_speed
        ], dtype=np.float32)
    
    # ---- 邻居位置和速度 ----
    valid_neighbors = neighbor_past[valid_mask]  # (M, T, D)
    
    # 邻居当前位置（最后一帧）
    neighbor_positions = valid_neighbors[:, -1, :2]  # (M, 2)
    
    # 邻居速度
    neighbor_speeds = np.array([
        compute_speed_from_trajectory(valid_neighbors[i], dt)[-1]
        for i in range(n_neighbors)
    ])  # (M,)
    
    # ---- 基础距离指标 ----
    distances = np.linalg.norm(neighbor_positions - ego_pos, axis=1)  # (M,)
    min_dist = float(distances.min())
    
    # ---- 速度差 ----
    delta_v = np.abs(ego_speed - neighbor_speeds)  # (M,)
    max_delta_v = float(delta_v.max())
    mean_neighbor_speed = float(neighbor_speeds.mean())
    speed_variance = float(neighbor_speeds.var())
    
    # ---- 纵向分析（前方/后方车辆）----
    # 在 ego-centric 坐标系下，x>0 是前方
    longitudinal = neighbor_positions[:, 0]  # x 坐标
    lateral = neighbor_positions[:, 1]       # y 坐标
    
    # 前方车辆：x > 0 且 |y| < 3m（大致在同车道）
    front_mask = (longitudinal > 0) & (np.abs(lateral) < 3.0)
    if front_mask.any():
        front_dists = distances[front_mask]
        front_vehicle_dist = float(front_dists.min())
        
        # THW: 跟车时距 = 前方距离 / ego速度
        thw = front_vehicle_dist / max(ego_speed, 0.1)
        thw = min(thw, _NO_RISK_THW)
    else:
        front_vehicle_dist = _NO_RISK_DIST
        thw = _NO_RISK_THW
    
    # ---- 横向最近距离 ----
    lateral_dists = np.abs(lateral)
    lateral_min_dist = float(lateral_dists.min())
    
    # ---- TTC: Time-to-Collision ----
    # 简化计算：只考虑接近中的邻居
    # 方法：比较当前帧和前一帧的距离变化
    if valid_neighbors.shape[1] >= 2:
        prev_positions = valid_neighbors[:, -2, :2]  # (M, 2)
        prev_distances = np.linalg.norm(prev_positions - ego_past[-2, :2] if ego_past.shape[0] >= 2 else ego_pos, axis=1)
        approach_speeds = (prev_distances - distances) / dt  # 正值=在接近
        
        approaching = approach_speeds > 0.1  # 只看在接近的车
        if approaching.any():
            ttc_values = distances[approaching] / approach_speeds[approaching]
            ttc = float(ttc_values.min())
            ttc = min(ttc, _NO_RISK_TTC)
        else:
            ttc = _NO_RISK_TTC
    else:
        ttc = _NO_RISK_TTC
    
    # ---- DRAC: Deceleration Rate to Avoid Crash ----
    # DRAC = approach_speed² / (2 × distance)
    if min_dist > 0.1 and max_delta_v > 0.1:
        drac = max_delta_v ** 2 / (2 * min_dist)
    else:
        drac = 0.0
    
    # ---- 后方接近速度 ----
    rear_mask = (longitudinal < -1.0)  # 后方车辆
    if rear_mask.any() and valid_neighbors.shape[1] >= 2:
        rear_approach = approach_speeds[rear_mask] if 'approach_speeds' in dir() else np.array([0.0])
        rear_approach_speed = float(max(rear_approach.max(), 0.0))
    else:
        rear_approach_speed = 0.0
    
    # ---- 构建特征向量 ----
    features = np.array([
        min_dist,              # 0
        float(n_neighbors),    # 1
        ego_speed,             # 2
        max_delta_v,           # 3
        ttc,                   # 4
        thw,                   # 5
        drac,                  # 6
        mean_neighbor_speed,   # 7
        speed_variance,        # 8
        front_vehicle_dist,    # 9
        lateral_min_dist,      # 10
        rear_approach_speed,   # 11
    ], dtype=np.float32)
    
    return features


def normalize_features(features: np.ndarray, stats: dict = None) -> tuple:
    """
    对风险特征做归一化（用于训练 Risk Network）。
    
    Args:
        features: (N, NUM_RISK_FEATURES) 原始特征
        stats: 可选，预计算的统计量 {'mean': ..., 'std': ...}
    
    Returns:
        normalized: (N, NUM_RISK_FEATURES) 归一化后的特征
        stats: {'mean': ..., 'std': ...}
    """
    if stats is None:
        # 对 TTC/THW/dist 中的大值做 clip，避免影响统计
        clipped = features.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, 100)   # min_dist
        clipped[:, 4] = np.clip(clipped[:, 4], 0, 30)    # ttc
        clipped[:, 5] = np.clip(clipped[:, 5], 0, 30)    # thw
        clipped[:, 9] = np.clip(clipped[:, 9], 0, 100)   # front_vehicle_dist
        clipped[:, 10] = np.clip(clipped[:, 10], 0, 50)  # lateral_min_dist
        
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        std[std < 1e-6] = 1.0  # 避免除零
        stats = {'mean': mean, 'std': std}
    
    # Clip then normalize
    normalized = features.copy()
    normalized[:, 0] = np.clip(normalized[:, 0], 0, 100)
    normalized[:, 4] = np.clip(normalized[:, 4], 0, 30)
    normalized[:, 5] = np.clip(normalized[:, 5], 0, 30)
    normalized[:, 9] = np.clip(normalized[:, 9], 0, 100)
    normalized[:, 10] = np.clip(normalized[:, 10], 0, 50)
    
    normalized = (normalized - stats['mean']) / stats['std']
    
    return normalized, stats


# ============================================================
# CLI: 批量提取风险特征
# ============================================================
if __name__ == '__main__':
    import argparse
    import glob
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Extract risk features from npz data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .npz files')
    parser.add_argument('--output', type=str, required=True, help='Output .npz path for extracted features')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    args = parser.parse_args()
    
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    if args.max_samples:
        npz_files = npz_files[:args.max_samples]
    
    print(f"Processing {len(npz_files)} files from {args.data_dir}")
    
    all_features = []
    all_filenames = []
    failed = 0
    
    for npz_path in tqdm(npz_files, desc="Extracting risk features"):
        try:
            features = extract_risk_features_from_npz(npz_path)
            all_features.append(features)
            all_filenames.append(os.path.basename(npz_path))
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Failed: {os.path.basename(npz_path)}: {e}")
    
    features_array = np.stack(all_features, axis=0)  # (N, NUM_RISK_FEATURES)
    
    # Print statistics
    print(f"\nExtracted {features_array.shape[0]} samples ({failed} failed)")
    print(f"\nFeature statistics:")
    for i, name in enumerate(RISK_FEATURE_NAMES):
        vals = features_array[:, i]
        print(f"  {name:25s}: mean={vals.mean():8.2f}, std={vals.std():8.2f}, "
              f"min={vals.min():8.2f}, max={vals.max():8.2f}")
    
    # Save
    np.savez(args.output,
             features=features_array,
             feature_names=np.array(RISK_FEATURE_NAMES),
             filenames=np.array(all_filenames))
    print(f"\nSaved to {args.output}")

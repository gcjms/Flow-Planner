"""
Grid Search: 找每个场景的最优 CFG 权重 w
==============================================
使用 Open-Loop 评估（ADE/FDE）作为代理指标，
对每个 Val 场景遍历 w 候选值，找到使 ADE 最小的 w。

用法:
    python -m flow_planner.risk.grid_search_w \
        --checkpoint /path/to/model.pth \
        --data_dir /path/to/val_npz/ \
        --data_list /path/to/flow_planner_training.json \
        --output /path/to/risk_dataset.npz \
        --max_samples 500
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from flow_planner.risk.risk_features import extract_risk_features_from_npz, NUM_RISK_FEATURES, RISK_FEATURE_NAMES


def run_grid_search(
    model,
    dataloader,
    data_dir: str,
    data_list: list,
    w_candidates: list = None,
    device: str = 'cuda',
    max_samples: int = None,
):
    """
    对每个场景，遍历 w 候选值，找到使 ADE 最小的 w。
    
    Args:
        model: FlowPlanner 模型（已加载权重）
        dataloader: Val 数据集的 DataLoader
        data_dir: npz 文件目录（用于提取风险特征）
        data_list: npz 文件名列表
        w_candidates: w 候选值列表
        device: 设备
        max_samples: 最大样本数
    
    Returns:
        results: dict with 'features', 'optimal_w', 'ade_matrix', 'filenames'
    """
    if w_candidates is None:
        w_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    model.eval()
    
    all_features = []
    all_optimal_w = []
    all_ade_per_w = []
    all_filenames = []
    
    n_processed = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Grid Search")):
            if max_samples and n_processed >= max_samples:
                break
            
            data = data.to(device)
            B = data.ego_current.shape[0]
            gt_future = data.ego_future  # (B, T, D) 或类似
            
            ade_per_w = np.zeros((B, len(w_candidates)))
            
            for w_idx, w in enumerate(w_candidates):
                # 推理
                pred = model(data, mode='inference', use_cfg=True, cfg_weight=w)
                # pred: (B, T, D) 预测轨迹
                
                # 计算 ADE
                if gt_future.shape[1] > pred.shape[1]:
                    gt_trimmed = gt_future[:, :pred.shape[1], :2]
                else:
                    gt_trimmed = gt_future[:, :, :2]
                    pred = pred[:, :gt_trimmed.shape[1], :]
                
                ade = torch.mean(
                    torch.norm(pred[:, :, :2] - gt_trimmed, dim=-1), dim=-1
                )  # (B,)
                
                ade_per_w[:, w_idx] = ade.cpu().numpy()
            
            # 找每个样本的最优 w
            optimal_w_indices = np.argmin(ade_per_w, axis=1)
            optimal_w = np.array([w_candidates[i] for i in optimal_w_indices])
            
            # 提取风险特征
            start_idx = batch_idx * dataloader.batch_size
            for i in range(B):
                global_idx = start_idx + i
                if global_idx >= len(data_list):
                    break
                
                npz_path = os.path.join(data_dir, data_list[global_idx])
                try:
                    features = extract_risk_features_from_npz(npz_path)
                    all_features.append(features)
                    all_optimal_w.append(optimal_w[i])
                    all_ade_per_w.append(ade_per_w[i])
                    all_filenames.append(data_list[global_idx])
                except Exception as e:
                    print(f"  Skip {data_list[global_idx]}: {e}")
            
            n_processed += B
    
    results = {
        'features': np.stack(all_features, axis=0),
        'optimal_w': np.array(all_optimal_w),
        'ade_matrix': np.stack(all_ade_per_w, axis=0),
        'w_candidates': np.array(w_candidates),
        'filenames': np.array(all_filenames),
        'feature_names': np.array(RISK_FEATURE_NAMES),
    }
    
    return results


def run_grid_search_simple(
    model,
    dataset,
    data_dir: str,
    w_candidates: list = None,
    device: str = 'cuda',
    max_samples: int = None,
):
    """
    简化版 Grid Search：逐个样本处理，不需要 DataLoader。
    适合在内存有限的情况下使用。
    """
    if w_candidates is None:
        w_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    model.eval()
    
    n_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    all_features = []
    all_optimal_w = []
    all_ade_per_w = []
    
    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Grid Search"):
            data = dataset[idx]
            # Add batch dimension
            data = data.repeat(1) if hasattr(data, 'repeat') else data
            data = data.to(device)
            
            ade_values = []
            for w in w_candidates:
                pred = model(data, mode='inference', use_cfg=True, cfg_weight=w)
                gt = data.ego_future
                
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                
                min_T = min(pred.shape[1], gt.shape[1])
                ade = torch.mean(
                    torch.norm(pred[:, :min_T, :2] - gt[:, :min_T, :2], dim=-1)
                ).item()
                ade_values.append(ade)
            
            # 提取风险特征
            features = extract_risk_features_from_npz(
                os.path.join(data_dir, dataset.data_list[idx])
            )
            
            optimal_w = w_candidates[np.argmin(ade_values)]
            
            all_features.append(features)
            all_optimal_w.append(optimal_w)
            all_ade_per_w.append(ade_values)
    
    return {
        'features': np.stack(all_features),
        'optimal_w': np.array(all_optimal_w),
        'ade_matrix': np.array(all_ade_per_w),
        'w_candidates': np.array(w_candidates),
        'feature_names': np.array(RISK_FEATURE_NAMES),
    }


def analyze_grid_search_results(results: dict):
    """打印 Grid Search 结果统计"""
    optimal_w = results['optimal_w']
    w_candidates = results['w_candidates']
    features = results['features']
    
    print(f"\n{'='*60}")
    print(f"Grid Search Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(optimal_w)}")
    print(f"W candidates: {w_candidates.tolist()}")
    
    print(f"\nOptimal w distribution:")
    for w in w_candidates:
        count = (optimal_w == w).sum()
        pct = count / len(optimal_w) * 100
        bar = '█' * int(pct / 2)
        print(f"  w={w:.1f}: {count:5d} ({pct:5.1f}%) {bar}")
    
    print(f"\nOptimal w statistics:")
    print(f"  Mean:   {optimal_w.mean():.2f}")
    print(f"  Std:    {optimal_w.std():.2f}")
    print(f"  Median: {np.median(optimal_w):.2f}")
    
    # Feature correlation with optimal w
    print(f"\nFeature correlation with optimal w:")
    for i, name in enumerate(RISK_FEATURE_NAMES):
        valid = features[:, i] < 900  # exclude placeholder values
        if valid.sum() > 10:
            corr = np.corrcoef(features[valid, i], optimal_w[valid])[0, 1]
            print(f"  {name:25s}: r = {corr:+.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid Search for optimal CFG weight per scenario')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Val npz data directory')
    parser.add_argument('--data_list', type=str, required=True, help='Val data list JSON')
    parser.add_argument('--output', type=str, required=True, help='Output risk_dataset.npz path')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("Grid Search requires GPU and trained model.")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"This script should be run on AutoDL or a GPU machine.")
    print(f"\nTo run on AutoDL, use:")
    print(f"  python -m flow_planner.risk.grid_search_w \\")
    print(f"    --checkpoint <path_to_checkpoint> \\")
    print(f"    --data_dir <val_npz_dir> \\")
    print(f"    --data_list <val_json> \\")
    print(f"    --output risk_dataset.npz \\")
    print(f"    --max_samples 500")

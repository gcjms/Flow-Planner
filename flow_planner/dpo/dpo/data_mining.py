"""
DPO Data Mining: 从训练集 NPZ 中挖掘高价值交互场景
==================================================

扫描本地 processed/ 目录下的 .npz 文件，用纯几何启发式快速筛选出
适合做 DPO 偏好学习的"高交互、高难度"关键帧。

筛选标准（满足任一即入选）：
  1. 无保护转弯 (Unprotected Turn):
     - GT 轨迹终点相对起点的航向变化 > 45°
  2. 近距离交互 (Close Interaction):
     - 存在邻居车辆当前帧距离 ego < 10m
  3. 高横向加速度 (High Lateral Acceleration):
     - GT 轨迹的横向偏移标准差 > 2m（说明有变道/绕行）
  4. 紧急制动 (Emergency Braking):
     - GT 轨迹前半段和后半段的纵向位移比 < 0.3（急刹特征）

用法：
  python -m flow_planner.dpo.data_mining \
      --data_dir /mnt/d/flow_planner_backup/processed \
      --output_path dpo_data/hard_scenarios.json \
      --max_scenarios 5000

输出：
  JSON 文件，包含被选中场景的文件路径列表和筛选理由。
"""

import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def analyze_scenario(npz_path: str) -> Optional[Dict]:
    """
    分析单个 NPZ 场景，判断是否为高价值交互帧。

    Returns:
        dict with scenario info and tags, or None if not interesting.
    """
    try:
        npz = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    tags = []
    scores = {}

    # --- 1. 提取 GT 未来轨迹 ---
    if 'ego_agent_future' not in npz:
        return None
    ego_future = npz['ego_agent_future']  # (T, D) where D >= 2, typically (80, 4)
    if ego_future.shape[0] < 10:
        return None

    x = ego_future[:, 0]  # 纵向
    y = ego_future[:, 1]  # 横向

    # --- 2. 无保护转弯检测 ---
    # 用轨迹终点 vs 起点的方位角变化
    dx_total = x[-1] - x[0]
    dy_total = y[-1] - y[0]
    final_heading = np.arctan2(dy_total, dx_total)  # 弧度

    # 轨迹前 10% 的局部方向
    n_early = max(5, len(x) // 10)
    dx_early = x[n_early] - x[0]
    dy_early = y[n_early] - y[0]
    early_heading = np.arctan2(dy_early, dx_early)

    heading_change = abs(np.degrees(final_heading - early_heading))
    # 处理 ±180° 跳变
    if heading_change > 180:
        heading_change = 360 - heading_change

    scores['heading_change_deg'] = float(heading_change)
    if heading_change > 45:
        tags.append('unprotected_turn')

    # --- 3. 近距离交互检测 ---
    if 'neighbor_agents_past' in npz:
        neighbors = npz['neighbor_agents_past']  # (M, T_p, D)
        if neighbors.ndim == 3 and neighbors.shape[0] > 0:
            # 取最后一帧（当前时刻）的位置
            curr_positions = neighbors[:, -1, :2]  # (M, 2)
            # 过滤掉全零的无效邻居
            valid = np.abs(curr_positions).sum(axis=1) > 1e-3
            if valid.any():
                dists = np.linalg.norm(curr_positions[valid], axis=1)
                min_dist = float(dists.min())
                scores['min_neighbor_dist'] = min_dist
                scores['n_close_neighbors'] = int((dists < 15).sum())
                if min_dist < 10:
                    tags.append('close_interaction')

    # --- 4. 高横向加速度 / 变道检测 ---
    lateral_std = float(np.std(y))
    scores['lateral_std'] = lateral_std
    if lateral_std > 2.0:
        tags.append('high_lateral')

    # --- 5. 紧急制动检测 ---
    half = len(x) // 2
    dist_first_half = np.sqrt((x[half] - x[0])**2 + (y[half] - y[0])**2)
    dist_second_half = np.sqrt((x[-1] - x[half])**2 + (y[-1] - y[half])**2)
    total_dist = dist_first_half + dist_second_half
    if total_dist > 1e-3:
        decel_ratio = dist_second_half / total_dist
        scores['decel_ratio'] = float(decel_ratio)
        if decel_ratio < 0.3:
            tags.append('emergency_braking')

    # --- 6. 高速场景 ---
    total_travel = float(np.sqrt(dx_total**2 + dy_total**2))
    scores['total_travel_m'] = total_travel
    if total_travel > 50:
        tags.append('high_speed')

    # 没有任何标签 → 不感兴趣
    if not tags:
        return None

    return {
        'path': npz_path,
        'filename': os.path.basename(npz_path),
        'tags': tags,
        'scores': scores,
    }


def mine_scenarios(
    data_dir: str,
    max_scenarios: int = 5000,
    num_workers: int = 8,
) -> List[Dict]:
    """
    并行扫描数据目录下所有 NPZ 文件，挖掘高价值场景。

    Args:
        data_dir: 数据根目录（包含 train_boston/, train_pittsburgh/ 等子目录）
        max_scenarios: 最多返回多少个场景
        num_workers: 并行工作进程数

    Returns:
        selected: 筛选后的场景列表，按综合分数降序排列
    """
    # 收集所有 NPZ 文件
    npz_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))

    logger.info(f"Found {len(npz_files)} NPZ files in {data_dir}")

    # 并行扫描
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_scenario, f): f for f in npz_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mining"):
            result = future.result()
            if result is not None:
                results.append(result)

    logger.info(f"Found {len(results)} interesting scenarios out of {len(npz_files)}")

    # 统计各标签的数量
    tag_counts = {}
    for r in results:
        for tag in r['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    logger.info(f"Tag distribution: {tag_counts}")

    # 按"有多少个标签"和综合分数排序（多标签 = 更有价值）
    def score_scenario(s):
        n_tags = len(s['tags'])
        heading = s['scores'].get('heading_change_deg', 0)
        dist = 100 - s['scores'].get('min_neighbor_dist', 100)  # 越近越高
        lateral = s['scores'].get('lateral_std', 0)
        return n_tags * 100 + heading + dist + lateral * 10

    results.sort(key=score_scenario, reverse=True)

    # 截取 top N
    selected = results[:max_scenarios]
    logger.info(f"Selected top {len(selected)} scenarios")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Mine high-value interactive scenarios for DPO training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/d/flow_planner_backup/processed',
                        help='Root directory of processed NPZ data')
    parser.add_argument('--output_path', type=str,
                        default='dpo_data/hard_scenarios.json',
                        help='Output JSON file path')
    parser.add_argument('--max_scenarios', type=int, default=5000,
                        help='Maximum number of scenarios to select')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    selected = mine_scenarios(
        data_dir=args.data_dir,
        max_scenarios=args.max_scenarios,
        num_workers=args.num_workers,
    )

    # 保存结果
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    output = {
        'total_scanned': len(list(Path(args.data_dir).rglob('*.npz'))),
        'total_selected': len(selected),
        'tag_summary': {},
        'scenarios': selected,
    }

    # 统计标签
    for s in selected:
        for tag in s['tags']:
            output['tag_summary'][tag] = output['tag_summary'].get(tag, 0) + 1

    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {args.output_path}")
    logger.info(f"Tag summary: {output['tag_summary']}")

    # 打印 top 10 样例
    print("\n" + "="*70)
    print(f"  Top 10 Hardest Scenarios (out of {len(selected)})")
    print("="*70)
    for i, s in enumerate(selected[:10]):
        print(f"  {i+1}. {s['filename']}")
        print(f"     Tags: {', '.join(s['tags'])}")
        print(f"     Scores: {s['scores']}")
        print()


if __name__ == '__main__':
    main()

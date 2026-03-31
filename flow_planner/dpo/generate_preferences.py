"""
离线偏好对数据生成器
===================
从 nuPlan 数据集中离线采样轨迹，用物理打分器或 VLM 生成 (chosen, rejected) 偏好对。

使用方法：
  python -m flow_planner.dpo.generate_preferences \
      --config_path checkpoints/model_config.yaml \
      --ckpt_path checkpoints/model.pth \
      --data_root /path/to/val/data \
      --output_dir dpo_data \
      --num_candidates 10 \
      --scorer_mode physical   # 或 vlm
"""

import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

# ============================================================
# 方案 B（物理打分器）：直接用 TrajectoryScorer 排序
# 不需要 VLM，纯数学计算，100% 客观
# ============================================================

def generate_preferences_with_scorer(
    model,
    dataloader,
    scorer,
    num_candidates: int = 10,
    output_dir: str = "dpo_data",
    device: str = "cuda",
) -> List[Dict]:
    """
    用物理打分器生成偏好对。

    流程：
      1. 对每个场景，用 Flow-Planner 生成 K 条候选轨迹
      2. 用 TrajectoryScorer 对每条轨迹打分
      3. 最高分 → chosen，最低分 → rejected

    Args:
        model: Flow-Planner 模型
        dataloader: nuPlan 数据加载器
        scorer: TrajectoryScorer 实例
        num_candidates: 每个场景采样多少条候选
        output_dir: 输出目录
        device: 计算设备

    Returns:
        preferences: list of dicts, 每个包含 chosen/rejected 轨迹
    """
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    preferences = []
    total = len(dataloader)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx % 50 == 0:
                logger.info(f"Processing scenario {idx}/{total}")

            # 将数据移到 GPU
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 生成 K 条候选轨迹
            candidates = []
            for k in range(num_candidates):
                torch.manual_seed(k * 1000 + idx)
                pred = model(data, mode='inference',
                           use_cfg=True, cfg_weight=1.8,
                           num_candidates=1)
                candidates.append(pred['trajectory'].squeeze(0))  # (T, D)

            # Stack: (K, T, D)
            candidates = torch.stack(candidates, dim=0)

            # 用打分器评分
            # 外推邻居未来轨迹
            neighbor_past = data.get('neighbor_past', None)
            if neighbor_past is not None and neighbor_past.dim() == 3:
                neighbor_future = TrajectoryScorer.extrapolate_neighbor_future(
                    neighbor_past[0] if neighbor_past.dim() == 4 else neighbor_past,
                    future_steps=candidates.shape[1],
                    dt=0.5,
                )
            else:
                neighbor_future = None

            scores = scorer.score_trajectories(
                trajectories=candidates,
                neighbors=neighbor_future,
                route=None,
            )

            # 选 chosen 和 rejected
            best_idx = scores.argmax().item()
            worst_idx = scores.argmin().item()

            # 如果最好和最差是同一条（所有分数相同），跳过
            if best_idx == worst_idx:
                continue

            score_gap = scores[best_idx].item() - scores[worst_idx].item()

            # 只保留分差 > 2.0 的有意义偏好对
            if score_gap < 2.0:
                continue

            preference = {
                'scenario_idx': idx,
                'chosen': candidates[best_idx].cpu().numpy(),       # (T, D)
                'rejected': candidates[worst_idx].cpu().numpy(),    # (T, D)
                'chosen_score': scores[best_idx].item(),
                'rejected_score': scores[worst_idx].item(),
                'score_gap': score_gap,
                'condition': {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()
                    if k in ['ego_current', 'neighbor_past', 'lane']
                },
            }
            preferences.append(preference)

    # 保存
    output_path = os.path.join(output_dir, 'preferences.npz')
    np.savez_compressed(
        output_path,
        preferences=np.array(preferences, dtype=object),
        num_pairs=len(preferences),
    )
    logger.info(f"Generated {len(preferences)} preference pairs → {output_path}")

    return preferences


# ============================================================
# 方案 A（VLM 评价）：让 Gemini 看 BEV 图排名
# ============================================================

def generate_preferences_with_vlm(
    model,
    dataloader,
    bev_renderer,
    api_key: str,
    num_candidates: int = 10,
    output_dir: str = "dpo_data",
    device: str = "cuda",
) -> List[Dict]:
    """
    用 Gemini VLM 生成偏好对。

    流程：
      1. 生成 K 条候选轨迹
      2. 渲染 BEV 图
      3. 发送给 Gemini API 评价排名
      4. 排名第 1 → chosen，排名最后 → rejected
    """
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    vlm = genai.GenerativeModel("gemini-1.5-pro")

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    preferences = []

    prompt = """你是一名经验丰富的自动驾驶安全评审员。
下面是一张自动驾驶车辆的鸟瞰图（BEV），图中标注了多条候选行驶轨迹（用不同颜色和编号表示）。
蓝色三角形是自车，红色方块是周围车辆，灰色线是车道线。

请根据以下标准对轨迹进行排名（从最优到最差）：
1. 安全性：是否会与周围车辆发生碰撞
2. 合规性：是否遵守车道线
3. 平滑性：轨迹是否自然流畅

请只输出轨迹编号的排名，格式如：1, 3, 2, 5, 4"""

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx % 50 == 0:
                logger.info(f"[VLM] Processing scenario {idx}/{len(dataloader)}")

            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 生成 K 条候选
            candidates = []
            for k in range(num_candidates):
                torch.manual_seed(k * 1000 + idx)
                pred = model(data, mode='inference',
                           use_cfg=True, cfg_weight=1.8,
                           num_candidates=1)
                candidates.append(pred['trajectory'].squeeze(0))
            candidates = torch.stack(candidates, dim=0)

            # 渲染 BEV 图
            bev_path = os.path.join(output_dir, 'bev_images', f'scenario_{idx}.png')
            os.makedirs(os.path.dirname(bev_path), exist_ok=True)
            bev_renderer.render(
                candidates=candidates.cpu().numpy(),
                neighbors=data.get('neighbor_past', None),
                lanes=data.get('lane', None),
                save_path=bev_path,
            )

            # 调用 Gemini API
            try:
                img = Image.open(bev_path)
                response = vlm.generate_content([prompt, img])
                ranking_text = response.text.strip()

                # 解析排名 "1, 3, 2, 5, 4" → [0, 2, 1, 4, 3]
                ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
                best_idx = ranking[0]
                worst_idx = ranking[-1]

                if best_idx != worst_idx and best_idx < len(candidates) and worst_idx < len(candidates):
                    preferences.append({
                        'scenario_idx': idx,
                        'chosen': candidates[best_idx].cpu().numpy(),
                        'rejected': candidates[worst_idx].cpu().numpy(),
                        'vlm_ranking': ranking,
                    })
            except Exception as e:
                logger.warning(f"VLM failed for scenario {idx}: {e}")
                continue

    output_path = os.path.join(output_dir, 'preferences_vlm.npz')
    np.savez_compressed(output_path, preferences=np.array(preferences, dtype=object))
    logger.info(f"Generated {len(preferences)} VLM preference pairs → {output_path}")

    return preferences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate DPO preference pairs')
    parser.add_argument('--scorer_mode', choices=['physical', 'vlm'], default='physical')
    parser.add_argument('--num_candidates', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='dpo_data')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (only for vlm mode)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Mode: {args.scorer_mode}, K={args.num_candidates}")
    logger.info("Note: Run with model and dataloader initialized externally.")

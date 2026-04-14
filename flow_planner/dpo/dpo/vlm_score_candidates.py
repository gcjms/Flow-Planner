#!/usr/bin/env python3
"""
VLM Preference Pair Scoring (Local)
====================================
从 AutoDL 生成的候选轨迹 NPZ 文件中：
  1. 渲染 BEV 鸟瞰图（5 条候选轨迹 + 邻居 + 车道线）
  2. 调 Gemini 2.5 Flash API 打分排名
  3. 构建 (chosen, rejected) 偏好对
  4. 输出 DPO 训练用的 preferences.npz

用法：
  python -m flow_planner.dpo.vlm_score_candidates \
      --candidates_dir dpo_data/candidates \
      --output_dir dpo_data/preferences \
      --api_key YOUR_GEMINI_API_KEY \
      --max_scenarios 5000 \
      --workers 4
"""

import os
import json
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ============================================================
# BEV Rendering
# ============================================================

def render_bev(candidate_npz_path: str, output_path: str) -> str:
    """
    Render BEV image with 5 candidate trajectories for VLM scoring.
    
    Returns: path to saved PNG image.
    """
    from flow_planner.dpo.bev_renderer import BEVRenderer

    data = np.load(candidate_npz_path, allow_pickle=True)
    candidates = data['candidates']        # (K, B, T, D) or (K, T, D)
    if candidates.ndim == 4:
        candidates = candidates.squeeze(1)  # (K, T, D)
    ego_past = data['ego_agent_past']      # (T_p, 14)
    neighbors = data['neighbor_agents_past']  # (M, T_p, 11)
    lanes = data.get('lanes', None)        # (N_l, P, 12) if available

    renderer = BEVRenderer(image_size=(800, 800), view_range=60.0)
    renderer.render_scenario(
        candidates=candidates,
        neighbors=neighbors,
        lanes=lanes,
        save_path=output_path
    )
    return output_path


# ============================================================
# VLM Scoring via Gemini
# ============================================================

VLM_PROMPT = """你是一名高度严谨的自动驾驶安全评审专家。
下面是一张自动驾驶车辆的鸟瞰图（BEV），图中标注了 5 条候选行驶轨迹（用不同颜色和编号 1-5 指定，结尾处标注了对应编号）。
蓝色三角形是自车，红色方块是周围车辆，灰色线是车道线。

【视觉信息】：上方提供了场景的路口和邻居车辆鸟瞰图，用于宏观战术理解（直行、过路口等）。
{metrics_text}

请严格结合“图景”与“物理数据”，对 5 条轨迹进行排名（从最优到最差）。
评估标准（核心优先级）：
1. 绝对安全：不能与车辆发生碰撞（数据中距离最近邻近车辆过近的直接垫底，这是最高红线）！
2. 合规居中：保持在车道内，距离车道线点太近说明有压线风险。
3. 平稳前进：轨迹合理舒展。

请严格按以下 JSON 格式输出，不要添加任何其他内容：
{{"ranking": [最优轨迹编号, ..., 最差轨迹编号], "reason": "结合物理距离数据解释：如轨迹X因为距离车辆仅0.X米存在碰撞风险而垫底，轨迹Y安全居中..."}}
"""

def compute_metrics_text(candidate_npz_path: str) -> str:
    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = data['candidates'] # (5, B, T, D) or (5, T, D)
    if cands.ndim == 4: cands = cands.squeeze(1)
    
    traj_xy = cands[:, :, :2]
    
    curr_n = data['neighbor_agents_past'][:, -1, :2]
    # Filter out padding neighbors (all-zero position)
    valid_n = curr_n[(np.abs(curr_n).sum(axis=1) > 1e-6)]
    
    l_pts = data.get('lanes', None)
    if l_pts is not None and l_pts.shape[0] > 0:
        l_pts = l_pts[:, :, :2].reshape(-1, 2)
        
    metrics_text = "【物理数据补充】\n"
    for k in range(5):
        traj = traj_xy[k]
        
        obs_dist = 99.0
        if len(valid_n) > 0:
            diff = traj[:, None, :] - valid_n[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            obs_dist = np.min(dist)
            
        lane_dist = 99.0
        if l_pts is not None:
            diff2 = traj[:, None, :] - l_pts[None, :, :]
            dist2 = np.linalg.norm(diff2, axis=-1)
            lane_dist = np.min(dist2)
            
        metrics_text += f"- 候选轨迹 #{k+1}：距离最近邻近车辆最小间距 {obs_dist:.2f} 米，距离最近车道边界锚点最小间距 {lane_dist:.2f} 米。\n"
    return metrics_text



def score_with_vlm(
    image_path: str,
    client,
    model_name: str = "gemini-3.1-pro",
    prompt: str = "",
    max_retries: int = 3,
) -> Optional[Dict]:
    """
    Send BEV image + prompt to Gemini and get trajectory ranking.
    """
    from PIL import Image

    img = Image.open(image_path)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
            )
            text = response.text.strip()

            # Try to parse JSON from response
            # Handle cases where model wraps in ```json ... ```
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            result = json.loads(text)

            if 'ranking' in result and len(result['ranking']) >= 2:
                return result

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {image_path}: {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    return None


def build_preference_pair(
    candidate_npz_path: str,
    ranking: Dict,
) -> Dict:
    """
    Given VLM ranking, build a (chosen, rejected) preference pair.
    
    chosen = best ranked trajectory
    rejected = worst ranked trajectory
    """
    data = np.load(candidate_npz_path, allow_pickle=True)
    candidates = data['candidates']  # (K, B, T, D) or (K, T, D)
    if candidates.ndim == 4:
        candidates = candidates.squeeze(1)

    best_idx = ranking['ranking'][0] - 1  # 1-indexed → 0-indexed
    worst_idx = ranking['ranking'][-1] - 1

    return {
        'scenario_id': os.path.basename(candidate_npz_path).replace('_candidates.npz', ''),
        'chosen': candidates[best_idx],      # (T, D)
        'rejected': candidates[worst_idx],    # (T, D)
        'chosen_idx': best_idx,
        'rejected_idx': worst_idx,
        'ranking': ranking['ranking'],
        'reason': ranking.get('reason', ''),
        # Condition data for DPO training
        'ego_agent_past': data['ego_agent_past'],
        'ego_current_state': data['ego_current_state'],
        'ego_agent_future': data['ego_agent_future'],
        'neighbor_agents_past': data['neighbor_agents_past'],
        'lanes': data.get('lanes', np.array([])),
    }


# ============================================================
# Main Pipeline
# ============================================================

def process_one_scenario(
    npz_path: str,
    bev_dir: str,
    client,
    model_name: str,
) -> Optional[Dict]:
    """Process a single scenario: render → VLM score → build pair."""
    basename = os.path.basename(npz_path).replace('.npz', '')
    bev_path = os.path.join(bev_dir, basename + '.png')

    # 1. Render BEV
    try:
        render_bev(npz_path, bev_path)
    except Exception as e:
        logger.warning(f"Render failed for {basename}: {e}")
        return None

    # 1.5 Calculate Mathematical text constraints
    metrics_text = compute_metrics_text(npz_path)
    prompt = VLM_PROMPT.format(metrics_text=metrics_text)

    # 2. VLM score
    ranking = score_with_vlm(bev_path, client, model_name, prompt=prompt)
    if ranking is None:
        logger.warning(f"VLM failed for {basename}")
        return None

    # 3. Build pair
    pair = build_preference_pair(npz_path, ranking)
    return pair


def main():
    parser = argparse.ArgumentParser(
        description='Score candidate trajectories with VLM and build preference pairs'
    )
    parser.add_argument('--candidates_dir', type=str, required=True,
                        help='Directory with *_candidates.npz files')
    parser.add_argument('--output_dir', type=str, default='dpo_data/preferences',
                        help='Output directory for preference pairs')
    parser.add_argument('--api_key', type=str, required=True,
                        help='Gemini API key')
    parser.add_argument('--model_name', type=str, default='gemini-3.1-pro-preview')
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel VLM API workers (be cautious with rate limits)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, 'bev_images')
    os.makedirs(bev_dir, exist_ok=True)

    # Find candidate files
    npz_files = sorted(Path(args.candidates_dir).glob('*_candidates.npz'))
    if args.max_scenarios:
        npz_files = npz_files[:args.max_scenarios]
    logger.info(f"Found {len(npz_files)} candidate files")

    # Init Gemini client
    from google import genai
    client = genai.Client(api_key=args.api_key)

    # Process all scenarios
    preferences = []
    success = 0
    fail = 0
    t0 = time.time()

    for i, npz_path in enumerate(npz_files):
        pair = process_one_scenario(
            str(npz_path), bev_dir, client, args.model_name
        )
        if pair:
            preferences.append(pair)
            success += 1
        else:
            fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            logger.info(
                f"[{i+1}/{len(npz_files)}] "
                f"success={success} fail={fail} "
                f"rate={rate:.0f}/hr"
            )

    elapsed = time.time() - t0
    logger.info(f"Done: {success} pairs, {fail} failed, {elapsed/60:.1f} min")

    # Save as NPZ for DPO training
    output_path = os.path.join(args.output_dir, 'preferences.npz')
    chosen_trajs = np.array([p['chosen'] for p in preferences])
    rejected_trajs = np.array([p['rejected'] for p in preferences])

    # Save condition data
    np.savez_compressed(
        output_path,
        chosen=chosen_trajs,
        rejected=rejected_trajs,
        scenario_ids=[p['scenario_id'] for p in preferences],
        rankings=[p['ranking'] for p in preferences],
        reasons=[p['reason'] for p in preferences],
    )
    logger.info(f"Saved {len(preferences)} pairs to {output_path}")

    # Also save full details as JSON
    json_path = os.path.join(args.output_dir, 'preference_details.json')
    details = [{
        'scenario_id': p['scenario_id'],
        'chosen_idx': int(p['chosen_idx']),
        'rejected_idx': int(p['rejected_idx']),
        'ranking': p['ranking'],
        'reason': p['reason'],
    } for p in preferences]
    with open(json_path, 'w') as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved details to {json_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Hybrid Rule + VLM Preference Scoring Pipeline
==============================================
Tier 1 (ALL scenes): Pure rule scoring = -FDE + collision penalty
Tier 2 (high-spread scenes): VLM fused prompt with GT + image + data

Output: preferences.npz + preference_details.json for DPO training.
"""

import os, json, argparse, logging, time, glob
import numpy as np
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ============================================================
# Tier 1: Rule-Based Scoring
# ============================================================

def _extrapolate_neighbor_future(neighbor_past: np.ndarray, future_steps: int, dt: float = 0.1) -> np.ndarray:
    """
    用恒速模型外推邻居未来位置。
    Args:
        neighbor_past: (M, T_p, 11) — 邻居历史 [x, y, cos_h, sin_h, vx, vy, w, l, type*3]
        future_steps: ego 轨迹的时间步数
        dt: 每步时间间隔 (秒)
    Returns:
        neighbor_future: (M, future_steps, 2) — 外推的 (x, y)
    """
    M = neighbor_past.shape[0]
    future = np.zeros((M, future_steps, 2), dtype=np.float32)
    for m in range(M):
        last = neighbor_past[m, -1]
        x, y = last[0], last[1]
        vx, vy = last[4], last[5]
        for t in range(future_steps):
            future[m, t, 0] = x + vx * dt * (t + 1)
            future[m, t, 1] = y + vy * dt * (t + 1)
    return future


def rule_score(candidate_npz_path: str) -> Dict:
    """Score candidates by FDE + collision distance (with extrapolated neighbor future)."""
    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = data['candidates']
    if cands.ndim == 4: cands = cands.squeeze(1)  # (K, T, D)
    num_candidates = cands.shape[0]

    gt_future = data['ego_agent_future']  # (T, 3+)
    gt_end = gt_future[-1, :2]

    neighbors = data['neighbor_agents_past']  # (M, T_p, 11)
    valid_mask = np.abs(neighbors[:, -1, :2]).sum(axis=1) > 1e-6
    valid_neighbors = neighbors[valid_mask]  # (M', T_p, 11)

    T_ego = cands.shape[1]
    neighbor_future = None
    if len(valid_neighbors) > 0:
        neighbor_future = _extrapolate_neighbor_future(valid_neighbors, T_ego)  # (M', T_ego, 2)

    scores = []
    details = []
    for k in range(num_candidates):
        traj = cands[k, :, :2]  # (T, 2)

        fde = float(np.linalg.norm(traj[-1] - gt_end))

        min_len = min(len(traj), len(gt_future))
        ade = float(np.mean(np.linalg.norm(traj[:min_len] - gt_future[:min_len, :2], axis=-1)))

        # 跟外推的邻居未来位置比，而不是当前位置
        obs_dist = 99.0
        if neighbor_future is not None:
            # traj: (T, 2), neighbor_future: (M', T, 2) → 逐时间步比较
            min_t = min(T_ego, neighbor_future.shape[1])
            dists = np.linalg.norm(
                traj[:min_t, None, :] - neighbor_future[:, :min_t, :].transpose(1, 0, 2),
                axis=-1,
            )  # (min_t, M')
            obs_dist = float(dists.min())

        collision_penalty = max(0, 3.0 - obs_dist) * 10.0

        score = -fde - 0.5 * ade - collision_penalty
        scores.append(score)
        details.append({'fde': fde, 'ade': ade, 'obs_dist': obs_dist})

    ranking_idx = np.argsort(scores)[::-1]
    ranking = [int(i + 1) for i in ranking_idx]

    chosen_idx = int(ranking_idx[0])
    rejected_idx = int(ranking_idx[-1])

    d_c = details[chosen_idx]
    d_r = details[rejected_idx]
    reason = (f"规则打分: chosen=#{chosen_idx+1}(FDE={d_c['fde']:.1f}m, "
              f"ADE={d_c['ade']:.1f}m, obs={d_c['obs_dist']:.1f}m) vs "
              f"rejected=#{rejected_idx+1}(FDE={d_r['fde']:.1f}m, "
              f"ADE={d_r['ade']:.1f}m, obs={d_r['obs_dist']:.1f}m)")

    return {
        'chosen_idx': chosen_idx,
        'rejected_idx': rejected_idx,
        'ranking': ranking,
        'reason': reason,
        'method': 'rule',
        'scores': scores,
    }


def compute_lateral_spread(candidate_npz_path: str) -> float:
    """Compute lateral spread of 5 candidates in ego frame."""
    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = data['candidates']
    if cands.ndim == 4: cands = cands.squeeze(1)

    ego = data['ego_agent_past']
    cos_h, sin_h = ego[-1, 2], ego[-1, 3]
    heading = np.arctan2(sin_h, cos_h)
    R = np.array([[np.cos(-heading), -np.sin(-heading)],
                  [np.sin(-heading),  np.cos(-heading)]])

    xy = cands[:, :, :2]
    xy_rot = np.einsum('ij,ktj->kti', R, xy)

    max_lateral = 0.0
    for t in [19, 39, 59, 79]:
        if t < xy_rot.shape[1]:
            y_vals = xy_rot[:, t, 1]
            spread = y_vals.max() - y_vals.min()
            max_lateral = max(max_lateral, spread)
    return max_lateral


# ============================================================
# Tier 2: VLM Scoring (for high-spread scenes)
# ============================================================

def vlm_score(candidate_npz_path: str, bev_dir: str, client, model_name: str) -> Optional[Dict]:
    """Render BEV with GT, compute metrics, call VLM with fused prompt."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from flow_planner.dpo.bev_renderer import TRAJECTORY_COLORS
    from PIL import Image

    data = np.load(candidate_npz_path, allow_pickle=True)
    cands = data['candidates']
    if cands.ndim == 4: cands = cands.squeeze(1)
    ego_future = data['ego_agent_future']
    neighbors = data['neighbor_agents_past']
    lanes = data.get('lanes', None)
    gt_end = ego_future[-1, :2]

    # Determine GT direction
    gt_dir = np.degrees(np.arctan2(gt_end[1], gt_end[0]))
    if abs(gt_dir) < 30: direction = "直行"
    elif gt_dir < -30: direction = "右转"
    else: direction = "左转"

    # --- Render BEV with GT ---
    basename = os.path.basename(candidate_npz_path).replace('.npz', '')
    bev_path = os.path.join(bev_dir, basename + '.png')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax.set_facecolor('#1a1a2e'); fig.patch.set_facecolor('#1a1a2e')
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_aspect('equal')
    ax.grid(True, color='#333333', linewidth=0.5, alpha=0.3)

    if lanes is not None:
        for l in range(lanes.shape[0]):
            lane = lanes[l]
            valid = np.abs(lane).sum(axis=1) > 1e-6
            if valid.sum() > 1:
                ax.plot(lane[valid, 0], lane[valid, 1], '-', color='#555555', linewidth=0.8, alpha=0.6)

    for m in range(neighbors.shape[0]):
        traj = neighbors[m]
        if np.abs(traj).sum() < 1e-6: continue
        curr = traj[-1]; nx, ny = curr[0], curr[1]
        if abs(nx) > 60 or abs(ny) > 60: continue
        D = traj.shape[1]
        w = max(float(curr[6]), 1.5) if D >= 8 else 2.0
        l_ = max(float(curr[7]), 3.5) if D >= 8 else 4.5
        angle = np.degrees(np.arctan2(curr[3], curr[2])) if D >= 4 else 0.0
        rect = patches.Rectangle((nx-l_/2, ny-w/2), l_, w, angle=angle,
            rotation_point='center', linewidth=1, edgecolor='#FF6B6B',
            facecolor='#FF6B6B', alpha=0.5, zorder=8)
        ax.add_patch(rect)

    ax.plot(ego_future[:, 0], ego_future[:, 1], '--', color='white', linewidth=2.5, alpha=0.8, zorder=9)
    ax.text(ego_future[-1, 0]+1, ego_future[-1, 1]+1, 'GT', color='white', fontsize=9,
            fontweight='bold', zorder=15, bbox=dict(facecolor='black', alpha=0.7, pad=2))

    label_offsets = np.linspace(-3, 3, cands.shape[0])
    for k in range(cands.shape[0]):
        traj = cands[k, :, :2]; color = TRAJECTORY_COLORS[k % len(TRAJECTORY_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.9, zorder=10)
        ax.text(traj[-1, 0]+1, traj[-1, 1]+label_offsets[k], f'#{k+1}', color=color,
                fontsize=11, fontweight='bold', zorder=15,
                bbox=dict(facecolor='#1a1a2e', edgecolor='none', alpha=0.7, pad=0.5))

    from matplotlib.patches import Polygon as MplPolygon
    ax.add_patch(MplPolygon([(2.7,0),(-1.8,1),(-1.8,-1)], closed=True,
        facecolor='#448AFF', edgecolor='white', linewidth=1.5, zorder=20))

    plt.tight_layout()
    fig.savefig(bev_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- Compute metrics ---
    traj_xy = cands[:, :, :2]
    num_cands = cands.shape[0]

    valid_mask = np.abs(neighbors[:, -1, :2]).sum(axis=1) > 1e-6
    valid_nb = neighbors[valid_mask]
    T_ego = traj_xy.shape[1]
    nb_future = None
    if len(valid_nb) > 0:
        nb_future = _extrapolate_neighbor_future(valid_nb, T_ego)

    lines = []
    for k in range(num_cands):
        traj = traj_xy[k]
        obs_dist = 99.0
        if nb_future is not None:
            min_t = min(T_ego, nb_future.shape[1])
            dists = np.linalg.norm(
                traj[:min_t, None, :] - nb_future[:, :min_t, :].transpose(1, 0, 2), axis=-1
            )
            obs_dist = float(dists.min())
        fde = float(np.linalg.norm(traj[-1] - gt_end))
        lines.append(f"- 轨迹#{k+1}: 距最近车辆 {obs_dist:.1f}m, 终点距GT终点 {fde:.1f}m")

    prompt = f"""你是自动驾驶安全评审专家。

图中用不同颜色标注了5条候选轨迹（编号1-5），白色虚线是专家参考轨迹（GT），蓝色三角形是自车，红色方块是周围车辆。

【物理数据】
{chr(10).join(lines)}
参考轨迹终点: ({gt_end[0]:.1f}, {gt_end[1]:.1f}), 大致方向为{direction}。

【请你结合图像和数据综合判断】
1. 哪些轨迹驶出了结构化道路或驶入了不合理的区域？（看图）
2. 哪些轨迹与参考轨迹方向一致？（看数据+图）
3. 哪些轨迹存在碰撞风险？（看数据）

请按综合评分从最优到最差排序，严格输出JSON：
{{"ranking": [最优, ..., 最差], "reason": "综合分析..."}}
"""

    # --- Call VLM ---
    img = Image.open(bev_path)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name, contents=[prompt, img])
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            result = json.loads(text)
            if 'ranking' in result and len(result['ranking']) >= 2:
                chosen_idx = result['ranking'][0] - 1
                rejected_idx = result['ranking'][-1] - 1
                return {
                    'chosen_idx': chosen_idx,
                    'rejected_idx': rejected_idx,
                    'ranking': result['ranking'],
                    'reason': result.get('reason', ''),
                    'method': 'vlm',
                }
        except Exception as e:
            logger.warning(f"VLM attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    return None  # VLM failed, will fall back to rule


# ============================================================
# Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Hybrid Rule+VLM preference scoring')
    parser.add_argument('--candidates_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='dpo_data/preferences_final')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Gemini API key (required for VLM tier)')
    parser.add_argument('--model_name', type=str, default='gemini-3.1-pro-preview')
    parser.add_argument('--spread_threshold', type=float, default=5.0,
                        help='Lateral spread threshold for VLM scoring (meters)')
    parser.add_argument('--max_scenarios', type=int, default=None)
    parser.add_argument('--skip_vlm', action='store_true',
                        help='Skip VLM scoring entirely, use only rules')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)
    bev_dir = os.path.join(args.output_dir, 'bev_images')
    os.makedirs(bev_dir, exist_ok=True)

    npz_files = sorted(Path(args.candidates_dir).glob('*_candidates.npz'))
    if args.max_scenarios:
        npz_files = npz_files[:args.max_scenarios]
    logger.info(f"Found {len(npz_files)} candidate files")

    # Init VLM client if needed
    client = None
    if args.api_key and not args.skip_vlm:
        from google import genai
        client = genai.Client(api_key=args.api_key)

    # --- Phase 1: Scan for high-spread scenes ---
    logger.info("Phase 1: Scanning lateral spread...")
    spreads = {}
    for npz_path in npz_files:
        try:
            spread = compute_lateral_spread(str(npz_path))
            spreads[str(npz_path)] = spread
        except Exception as e:
            logger.warning(f"Spread computation failed for {npz_path}: {e}")
            spreads[str(npz_path)] = 0.0

    high_spread = {k for k, v in spreads.items() if v >= args.spread_threshold}
    logger.info(f"High-spread scenes (>= {args.spread_threshold}m): {len(high_spread)}")

    # --- Phase 2: Score all scenes ---
    logger.info("Phase 2: Scoring...")
    preferences = []
    stats = {'rule': 0, 'vlm': 0, 'vlm_fail': 0}
    t0 = time.time()

    for i, npz_path in enumerate(npz_files):
        npz_str = str(npz_path)
        scenario_id = npz_path.stem.replace('_candidates', '')

        use_vlm = npz_str in high_spread and client is not None

        if use_vlm:
            result = vlm_score(npz_str, bev_dir, client, args.model_name)
            if result is None:
                result = rule_score(npz_str)
                result['method'] = 'rule_fallback'
                stats['vlm_fail'] += 1
            else:
                stats['vlm'] += 1
        else:
            result = rule_score(npz_str)
            stats['rule'] += 1

        # Load trajectory data for output
        data = np.load(npz_str, allow_pickle=True)
        cands = data['candidates']
        if cands.ndim == 4: cands = cands.squeeze(1)

        preferences.append({
            'scenario_id': scenario_id,
            'chosen': cands[result['chosen_idx']],
            'rejected': cands[result['rejected_idx']],
            'chosen_idx': result['chosen_idx'],
            'rejected_idx': result['rejected_idx'],
            'ranking': result['ranking'],
            'reason': result['reason'],
            'method': result['method'],
            'lateral_spread': spreads.get(npz_str, 0.0),
            # Condition data for DPO training
            'ego_agent_past': data['ego_agent_past'],
            'ego_current_state': data['ego_current_state'],
            'ego_agent_future': data['ego_agent_future'],
            'neighbor_agents_past': data['neighbor_agents_past'],
            'lanes': data.get('lanes', np.array([])),
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info(f"[{i+1}/{len(npz_files)}] rule={stats['rule']} vlm={stats['vlm']} "
                        f"vlm_fail={stats['vlm_fail']} rate={rate:.1f}/s "
                        f"ETA={((len(npz_files)-i-1)/rate)/60:.1f}min")

    elapsed = time.time() - t0
    logger.info(f"Done: {len(preferences)} pairs in {elapsed/60:.1f}min")
    logger.info(f"Stats: rule={stats['rule']} vlm={stats['vlm']} vlm_fail={stats['vlm_fail']}")

    # --- Save outputs ---
    output_path = os.path.join(args.output_dir, 'preferences.npz')
    np.savez_compressed(output_path,
        chosen=np.array([p['chosen'] for p in preferences]),
        rejected=np.array([p['rejected'] for p in preferences]),
        scenario_ids=[p['scenario_id'] for p in preferences],
        rankings=[p['ranking'] for p in preferences],
        reasons=[p['reason'] for p in preferences],
    )
    logger.info(f"Saved {len(preferences)} pairs to {output_path}")

    json_path = os.path.join(args.output_dir, 'preference_details.json')
    details = [{
        'scenario_id': p['scenario_id'],
        'chosen_idx': int(p['chosen_idx']),
        'rejected_idx': int(p['rejected_idx']),
        'ranking': p['ranking'],
        'reason': p['reason'],
        'method': p['method'],
        'lateral_spread': round(float(p['lateral_spread']), 2),
    } for p in preferences]
    with open(json_path, 'w') as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved details to {json_path}")


if __name__ == '__main__':
    main()

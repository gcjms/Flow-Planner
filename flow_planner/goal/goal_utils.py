"""
Goal Point Utilities
====================
加载 goal vocabulary、查找最近聚类中心、选择多样化 goal 等工具函数。
"""

import numpy as np
import torch


def load_goal_vocab(path: str) -> np.ndarray:
    """加载 goal vocabulary, 返回 (K, 2) numpy array."""
    vocab = np.load(path)
    assert vocab.ndim == 2 and vocab.shape[1] == 2, \
        f"Expected shape (K, 2), got {vocab.shape}"
    return vocab


def find_nearest_goal(endpoints: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """
    为每个 endpoint 找 vocabulary 中最近的 goal point。

    Args:
        endpoints: (B, 2)
        vocab: (K, 2)
    Returns:
        indices: (B,) int array, 每个元素是 vocab 中的索引
    """
    # (B, 1, 2) - (1, K, 2) → (B, K)
    dists = np.linalg.norm(
        endpoints[:, None, :] - vocab[None, :, :], axis=-1
    )
    return dists.argmin(axis=1)


def extract_valid_route_points(route_lanes: np.ndarray) -> np.ndarray:
    """Extract valid XY route points from raw route lanes."""
    points = np.asarray(route_lanes)[..., :2].reshape(-1, 2)
    finite_mask = np.isfinite(points).all(axis=1)
    nonzero_mask = np.linalg.norm(points, axis=1) > 1e-3
    return points[finite_mask & nonzero_mask]


def select_goal_from_route(
    route_lanes: np.ndarray,
    vocab: np.ndarray,
    min_forward: float = 2.0,
    max_goal_dist: float = 45.0,
    target_progress: float = 30.0,
) -> np.ndarray:
    """
    Select a non-oracle goal point using route geometry only.

    Strategy:
      1. Extract valid route points in ego frame.
      2. Pick a forward route anchor near a target progress distance.
      3. Retrieve the goal-cluster center that best matches the route polyline
         and stays close to the forward anchor.
    """
    route_points = extract_valid_route_points(route_lanes)
    goal_points = np.asarray(vocab, dtype=np.float32)

    if goal_points.ndim != 2 or goal_points.shape[1] != 2:
        raise ValueError(f"Expected vocab shape (K, 2), got {goal_points.shape}")

    goal_norm = np.linalg.norm(goal_points, axis=-1)
    goal_mask = (
        (goal_points[:, 0] > -1.0)
        & (goal_norm >= min_forward)
        & (goal_norm <= max_goal_dist)
    )
    valid_goals = goal_points[goal_mask]
    if len(valid_goals) == 0:
        valid_goals = goal_points

    if len(route_points) == 0:
        target_idx = np.argmin(np.abs(np.linalg.norm(valid_goals, axis=-1) - target_progress))
        return valid_goals[target_idx]

    route_norm = np.linalg.norm(route_points, axis=-1)
    forward_mask = (
        (route_points[:, 0] > 0.0)
        & (route_norm >= min_forward)
        & (route_norm <= max_goal_dist + 10.0)
    )
    forward_points = route_points[forward_mask]
    if len(forward_points) == 0:
        forward_points = route_points

    forward_norm = np.linalg.norm(forward_points, axis=-1)
    anchor_idx = np.argmin(np.abs(forward_norm - target_progress))
    anchor = forward_points[anchor_idx]

    route_dist = np.linalg.norm(
        valid_goals[:, None, :] - route_points[None, :, :],
        axis=-1,
    ).min(axis=1)
    anchor_dist = np.linalg.norm(valid_goals - anchor[None, :], axis=-1)
    progress_bias = np.abs(np.linalg.norm(valid_goals, axis=-1) - np.linalg.norm(anchor))

    # Route matching dominates; anchor/progress stabilize retrieval on turns.
    score = route_dist + 0.35 * anchor_dist + 0.05 * progress_bias
    return valid_goals[np.argmin(score)]


def find_nearest_goal_torch(endpoints: torch.Tensor, vocab: torch.Tensor) -> torch.Tensor:
    """
    Torch 版本，用于训练循环中 (GPU 上直接算，不走 numpy)。

    Args:
        endpoints: (B, 2) tensor
        vocab: (K, 2) tensor (应提前 .to(device))
    Returns:
        indices: (B,) long tensor
    """
    # (B, 1, 2) - (1, K, 2) → (B, K)
    dists = torch.linalg.norm(
        endpoints.unsqueeze(1) - vocab.unsqueeze(0), dim=-1
    )
    return dists.argmin(dim=1)


def select_diverse_goals(
    vocab: np.ndarray,
    n_goals: int,
    max_dist: float = 40.0,
    min_dist: float = 1.0,
) -> tuple:
    """
    从 vocabulary 中选出 n_goals 个最大化多样性的 goal point。

    用于 DPO 候选生成：挑差异最大的 goal，生成决策级不同的轨迹。

    Args:
        vocab: (K, 2)
        n_goals: 要选几个
        max_dist: 最远距离 (ego-centric 坐标, 4s horizon ~40m is reasonable)
        min_dist: 最近距离 (太近 = 已经到了, 没意义)

    Returns:
        indices: (n_goals,) int array
        goals: (n_goals, 2) selected goal points
    """
    dists = np.linalg.norm(vocab, axis=-1)
    valid_mask = (dists >= min_dist) & (dists <= max_dist) & (vocab[:, 0] > 0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        valid_indices = np.arange(len(vocab))

    if len(valid_indices) <= n_goals:
        selected = valid_indices
        if len(selected) < n_goals:
            extra = np.random.choice(selected, n_goals - len(selected), replace=True)
            selected = np.concatenate([selected, extra])
        return selected, vocab[selected]

    # Greedy farthest-point sampling
    valid_points = vocab[valid_indices]
    first_idx = valid_points[:, 0].argmax()  # 最远前方的点先选
    selected_local = [first_idx]

    for _ in range(n_goals - 1):
        sel_pts = valid_points[selected_local]
        remaining = np.ones(len(valid_points), dtype=bool)
        remaining[selected_local] = False
        rem_idx = np.where(remaining)[0]

        if len(rem_idx) == 0:
            break

        min_dists = np.min(
            np.linalg.norm(
                valid_points[rem_idx, None, :] - sel_pts[None, :, :],
                axis=-1,
            ),
            axis=1,
        )
        best = rem_idx[min_dists.argmax()]
        selected_local.append(best)

    selected_global = valid_indices[np.array(selected_local)]
    return selected_global, vocab[selected_global]

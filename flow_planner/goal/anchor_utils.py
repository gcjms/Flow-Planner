"""
Trajectory Anchor Utilities (Phase 1)
=====================================
对标 ``goal_utils.py``，但操作对象是完整轨迹 anchor ``(K, T, 3)``
而非 endpoint ``(K, 2)``。

关键设计：
- 距离度量用 **weighted L2 over (x, y, cos_h, sin_h)**，避免角度环绕 (±π)。
- ``find_nearest_anchor*`` 接受 GT 轨迹 (B, T, 3) 找最近 anchor。
- ``select_diverse_anchors`` 做 farthest-point sampling，用于 DPO 候选挖掘。
- ``select_anchor_from_route`` 给无 GT 的在线场景提供纯几何 fallback。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def load_anchor_vocab(path: str) -> np.ndarray:
    """Load anchor vocab and sanity-check shape.

    Returns:
        anchors: (K, T, 3) float32 numpy array — (x, y, heading) per frame.
    """
    vocab = np.load(path)
    if vocab.ndim != 3 or vocab.shape[-1] != 3:
        raise ValueError(
            f"Expected anchor vocab shape (K, T, 3); got {vocab.shape}. "
            "Did you point --anchor_vocab_path at a legacy (K, 2) goal vocab?"
        )
    return vocab.astype(np.float32)


def _expand_features_np(trajs: np.ndarray, heading_weight: float) -> np.ndarray:
    """(N, T, 3) -> (N, T*4) with heading replaced by (cos, sin) * heading_weight."""
    xy = trajs[..., :2]
    h = trajs[..., 2]
    cos_h = np.cos(h) * heading_weight
    sin_h = np.sin(h) * heading_weight
    feats = np.concatenate([xy, cos_h[..., None], sin_h[..., None]], axis=-1)
    return feats.reshape(feats.shape[0], -1)


def _expand_features_torch(trajs: torch.Tensor, heading_weight: float) -> torch.Tensor:
    xy = trajs[..., :2]
    h = trajs[..., 2]
    cos_h = torch.cos(h) * heading_weight
    sin_h = torch.sin(h) * heading_weight
    feats = torch.cat([xy, cos_h.unsqueeze(-1), sin_h.unsqueeze(-1)], dim=-1)
    return feats.reshape(feats.shape[0], -1)


def find_nearest_anchor(
    trajs: np.ndarray,
    vocab: np.ndarray,
    heading_weight: float = 5.0,
) -> np.ndarray:
    """Find the nearest anchor index for each input trajectory (numpy).

    Args:
        trajs: (B, T, 3) or (T, 3) ground-truth / query trajectories
        vocab: (K, T, 3) anchor vocabulary (same T as ``trajs``)
        heading_weight: cos/sin rescale factor so orientation is not drowned by xy

    Returns:
        indices: (B,) int64 array of anchor ids (or scalar if input was (T, 3))
    """
    single = (trajs.ndim == 2)
    if single:
        trajs = trajs[None, ...]
    if trajs.shape[1:] != vocab.shape[1:]:
        raise ValueError(
            f"Traj shape {trajs.shape} and vocab shape {vocab.shape} must share (T, 3)."
        )

    traj_feats = _expand_features_np(trajs, heading_weight)       # (B, T*4)
    vocab_feats = _expand_features_np(vocab, heading_weight)      # (K, T*4)
    dists = np.linalg.norm(
        traj_feats[:, None, :] - vocab_feats[None, :, :],
        axis=-1,
    )                                                             # (B, K)
    idx = dists.argmin(axis=1)
    return int(idx[0]) if single else idx


def find_nearest_anchor_torch(
    trajs: torch.Tensor,
    vocab: torch.Tensor,
    heading_weight: float = 5.0,
) -> torch.Tensor:
    """Torch counterpart of :func:`find_nearest_anchor` (runs on GPU)."""
    if trajs.dim() != 3 or trajs.size(-1) != 3:
        raise ValueError(f"Expected (B, T, 3); got {tuple(trajs.shape)}")
    if vocab.dim() != 3 or vocab.size(-1) != 3:
        raise ValueError(f"Expected vocab (K, T, 3); got {tuple(vocab.shape)}")
    if trajs.shape[1:] != vocab.shape[1:]:
        raise ValueError(
            f"Traj shape {tuple(trajs.shape)} vs vocab {tuple(vocab.shape)} "
            "must share (T, 3). Regenerate anchor_vocab.npy with matching --traj_len."
        )

    traj_feats = _expand_features_torch(trajs.float(), heading_weight)       # (B, T*4)
    vocab_feats = _expand_features_torch(vocab.float(), heading_weight)      # (K, T*4)
    dists = torch.linalg.norm(
        traj_feats.unsqueeze(1) - vocab_feats.unsqueeze(0),
        dim=-1,
    )                                                                         # (B, K)
    return dists.argmin(dim=1)


def lookup_anchor_traj(
    indices: torch.Tensor,
    vocab: torch.Tensor,
) -> torch.Tensor:
    """Gather full anchor trajectories given cluster indices.

    Args:
        indices: (B,) long tensor
        vocab:   (K, T, 3) float tensor on any device
    Returns:
        anchors: (B, T, 3) float tensor on the same device as ``indices``
    """
    return vocab.to(indices.device)[indices]


def select_diverse_anchors(
    vocab: np.ndarray,
    n_anchors: int,
    heading_weight: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Farthest-point sampling on anchor vocab (DPO candidate generation).

    Returns:
        indices: (n,) int64
        anchors: (n, T, 3)
    """
    vocab = np.asarray(vocab, dtype=np.float32)
    K = vocab.shape[0]
    if K <= n_anchors:
        return np.arange(K), vocab.copy()

    feats = _expand_features_np(vocab, heading_weight)  # (K, T*4)

    first = int(np.argmax(np.linalg.norm(feats - feats.mean(0, keepdims=True), axis=-1)))
    selected = [first]
    min_dist = np.linalg.norm(feats - feats[first:first + 1], axis=-1)
    for _ in range(n_anchors - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        new_d = np.linalg.norm(feats - feats[nxt:nxt + 1], axis=-1)
        min_dist = np.minimum(min_dist, new_d)

    idx = np.asarray(selected, dtype=np.int64)
    return idx, vocab[idx]


def _extract_valid_route_points(route_lanes: np.ndarray) -> np.ndarray:
    points = np.asarray(route_lanes)[..., :2].reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    nonzero = np.linalg.norm(points, axis=1) > 1e-3
    return points[finite & nonzero]


def select_anchor_from_route(
    route_lanes: np.ndarray,
    vocab: np.ndarray,
    target_progress: float = 30.0,
    heading_weight: float = 5.0,
) -> np.ndarray:
    """Pick an anchor whose endpoint/shape best matches the route polyline.

    This is the anchor-world analog of :func:`goal_utils.select_goal_from_route`.
    Strategy:
      1. Score anchors by how close their points lie to the route polyline.
      2. Prefer anchors whose endpoint is near a ``target_progress`` meters anchor point.

    Returns:
        anchor: (T, 3) numpy array
    """
    vocab = np.asarray(vocab, dtype=np.float32)
    if vocab.ndim != 3 or vocab.shape[-1] != 3:
        raise ValueError(f"Expected vocab (K, T, 3); got {vocab.shape}")

    route_points = _extract_valid_route_points(route_lanes)
    endpoints = vocab[:, -1, :2]
    endpoint_norm = np.linalg.norm(endpoints, axis=-1)

    if len(route_points) == 0:
        idx = int(np.argmin(np.abs(endpoint_norm - target_progress)))
        return vocab[idx]

    anchor_xy = vocab[:, :, :2].reshape(vocab.shape[0], -1, 2)                       # (K, T, 2)
    flat_xy = anchor_xy.reshape(-1, 2)                                               # (K*T, 2)
    route_dist = np.linalg.norm(
        flat_xy[:, None, :] - route_points[None, :, :], axis=-1
    ).min(axis=1).reshape(vocab.shape[0], -1).mean(axis=-1)                          # (K,)

    forward_norm = np.linalg.norm(route_points, axis=-1)
    forward_mask = (route_points[:, 0] > 0.0) & (forward_norm >= 2.0)
    forward_points = route_points[forward_mask] if forward_mask.any() else route_points
    anchor_pt = forward_points[np.argmin(np.abs(np.linalg.norm(forward_points, axis=-1) - target_progress))]
    endpoint_dist = np.linalg.norm(endpoints - anchor_pt[None, :], axis=-1)
    progress_bias = np.abs(endpoint_norm - np.linalg.norm(anchor_pt))

    score = route_dist + 0.35 * endpoint_dist + 0.05 * progress_bias
    return vocab[int(np.argmin(score))]

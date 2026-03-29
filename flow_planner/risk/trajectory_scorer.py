"""
Trajectory Scorer: 安全评分函数
============================
对候选轨迹进行多维度安全评分，用于 Best-of-N 轨迹筛选。

评分维度：
  1. 碰撞检测：轨迹是否与邻居碰撞（权重 50）
  2. TTC（碰撞时间）：越大越安全（权重 20）
  3. 路线一致性：轨迹终点与参考路线的偏差（权重 15）
  4. 舒适度：加速度/曲率变化率（权重 5）
  5. 进度：沿路线方向的前进距离（权重 10）

注意：推理时 neighbor_future 不可用，需用 extrapolate_neighbors()
从 neighbor_past 线性外推得到邻居预测未来位置。
"""

import torch
import numpy as np
from typing import Optional


class TrajectoryScorer:
    """Rule-based 轨迹评分器，对齐 nuPlan 闭环评分标准"""

    def __init__(
        self,
        collision_weight: float = 50.0,
        ttc_weight: float = 20.0,
        route_weight: float = 15.0,
        comfort_weight: float = 5.0,
        progress_weight: float = 10.0,
        collision_threshold: float = 2.0,     # 碰撞距离阈值 (m), 车宽约2m
        ttc_threshold: float = 3.0,            # TTC 安全阈值 (s)
        dt: float = 0.1,                       # 时间步长 (s)
    ):
        self.collision_weight = collision_weight
        self.ttc_weight = ttc_weight
        self.route_weight = route_weight
        self.comfort_weight = comfort_weight
        self.progress_weight = progress_weight
        self.collision_threshold = collision_threshold
        self.ttc_threshold = ttc_threshold
        self.dt = dt

    @staticmethod
    def extrapolate_neighbors(
        neighbor_past: torch.Tensor,
        T: int,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        从邻居历史轨迹线性外推未来位置。
        
        推理时 neighbor_future 不可用，用 neighbor_past 最后两帧的
        速度做恒速线性外推，预测未来 T 步的位置。

        Args:
            neighbor_past: (M, T_past, D) 邻居历史轨迹，D >= 2 (x, y, ...)
            T: int 需要外推的未来步数
            dt: float 时间步长

        Returns:
            neighbor_future: (M, T, 2) 外推的邻居未来 xy 位置
        """
        if neighbor_past is None or neighbor_past.numel() == 0:
            return None
        
        M = neighbor_past.shape[0]
        device = neighbor_past.device
        
        if neighbor_past.shape[1] < 2:
            # 只有一帧，假设静止
            last_pos = neighbor_past[:, -1, :2]  # (M, 2)
            return last_pos.unsqueeze(1).expand(M, T, 2).clone()
        
        # 用最后两帧计算速度
        last_pos = neighbor_past[:, -1, :2]    # (M, 2)
        prev_pos = neighbor_past[:, -2, :2]    # (M, 2)
        velocity = (last_pos - prev_pos) / dt  # (M, 2) m/s
        
        # 线性外推 T 步
        steps = torch.arange(1, T + 1, device=device, dtype=torch.float32)  # (T,)
        # (M, 1, 2) + (1, T, 1) * (M, 1, 2) * dt
        future_pos = last_pos.unsqueeze(1) + steps.unsqueeze(0).unsqueeze(-1) * velocity.unsqueeze(1) * dt
        
        return future_pos  # (M, T, 2)

    def score_trajectories(
        self,
        trajectories: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None,
        route: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对 N 条候选轨迹打分。

        Args:
            trajectories: (N, T, D) N 条候选轨迹, D >= 2 (x, y, ...)
            neighbors: (M, T_n, D_n) 邻居轨迹（如有），用于碰撞/TTC 计算
            route: (T_r, 2) 规划路线参考点（如有）

        Returns:
            scores: (N,) 每条轨迹的综合评分，越高越好
        """
        N = trajectories.shape[0]
        device = trajectories.device
        scores = torch.zeros(N, device=device)

        # 1. 碰撞评分
        collision_scores = self._collision_score(trajectories, neighbors)
        scores += self.collision_weight * collision_scores

        # 2. TTC 评分
        ttc_scores = self._ttc_score(trajectories, neighbors)
        scores += self.ttc_weight * ttc_scores

        # 3. 路线一致性评分
        route_scores = self._route_score(trajectories, route)
        scores += self.route_weight * route_scores

        # 4. 舒适度评分
        comfort_scores = self._comfort_score(trajectories)
        scores += self.comfort_weight * comfort_scores

        # 5. 前进进度评分
        progress_scores = self._progress_score(trajectories)
        scores += self.progress_weight * progress_scores

        return scores

    def _collision_score(
        self,
        trajectories: torch.Tensor,
        neighbors: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """碰撞评分：与邻居最近距离越大越好"""
        N = trajectories.shape[0]
        device = trajectories.device

        if neighbors is None or neighbors.shape[0] == 0:
            return torch.ones(N, device=device)  # 没有邻居 → 满分

        # trajectories: (N, T, D), neighbors: (M, T_n, D_n)
        traj_xy = trajectories[:, :, :2]  # (N, T, 2)
        T = traj_xy.shape[1]

        # 使用邻居的当前位置（假设静态或线性外推）
        neigh_xy = neighbors[:, :, :2]  # (M, T_n, 2)
        T_n = neigh_xy.shape[1]
        min_T = min(T, T_n)

        # 计算每条轨迹与每个邻居的最近距离
        # traj_xy: (N, min_T, 2) → (N, 1, min_T, 2)
        # neigh_xy: (M, min_T, 2) → (1, M, min_T, 2)
        traj_exp = traj_xy[:, :min_T, :].unsqueeze(1)
        neigh_exp = neigh_xy[:, :min_T, :].unsqueeze(0)

        # 距离: (N, M, min_T)
        dists = torch.norm(traj_exp - neigh_exp, dim=-1)

        # 每条轨迹的最小距离
        min_dists, _ = dists.reshape(N, -1).min(dim=-1)  # (N,)

        # 评分：距离 < threshold → 0 分，> 2*threshold → 满分
        scores = torch.clamp(
            (min_dists - self.collision_threshold) / self.collision_threshold,
            min=0.0, max=1.0
        )

        return scores

    def _ttc_score(
        self,
        trajectories: torch.Tensor,
        neighbors: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """TTC 评分：碰撞时间越大越好"""
        N = trajectories.shape[0]
        device = trajectories.device

        if neighbors is None or neighbors.shape[0] == 0:
            return torch.ones(N, device=device)

        traj_xy = trajectories[:, :, :2]  # (N, T, 2)
        neigh_xy = neighbors[:, :, :2]  # (M, T_n, 2)
        T = traj_xy.shape[1]
        T_n = neigh_xy.shape[1]
        min_T = min(T, T_n)

        # 计算速度
        traj_vel = (traj_xy[:, 1:min_T, :] - traj_xy[:, :min_T-1, :]) / self.dt  # (N, min_T-1, 2)
        neigh_vel = (neigh_xy[:, 1:min_T, :] - neigh_xy[:, :min_T-1, :]) / self.dt  # (M, min_T-1, 2)

        # 相对位置和速度
        # (N, 1, min_T-1, 2) - (1, M, min_T-1, 2) → (N, M, min_T-1, 2)
        rel_pos = traj_xy[:, :min_T-1, :].unsqueeze(1) - neigh_xy[:, :min_T-1, :].unsqueeze(0)
        rel_vel = traj_vel.unsqueeze(1) - neigh_vel.unsqueeze(0)

        # TTC = -dot(rel_pos, rel_vel) / |rel_vel|^2
        dot = (rel_pos * rel_vel).sum(dim=-1)  # (N, M, min_T-1)
        vel_sq = (rel_vel ** 2).sum(dim=-1) + 1e-8
        ttc = -dot / vel_sq  # (N, M, min_T-1)

        # 只关注正向 TTC（接近中的目标）
        ttc = torch.where(ttc > 0, ttc, torch.tensor(float('inf'), device=device))

        # 每条轨迹的最小 TTC
        min_ttc, _ = ttc.reshape(N, -1).min(dim=-1)  # (N,)

        # 评分：TTC < threshold → 0 分，> 2*threshold → 满分
        scores = torch.clamp(
            (min_ttc - self.ttc_threshold) / self.ttc_threshold,
            min=0.0, max=1.0
        )

        return scores

    def _route_score(
        self,
        trajectories: torch.Tensor,
        route: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """路线一致性：轨迹终点与路线的偏差越小越好"""
        N = trajectories.shape[0]
        device = trajectories.device

        if route is None:
            return torch.ones(N, device=device)

        # 轨迹终点
        traj_end = trajectories[:, -1, :2]  # (N, 2)

        # 找路线上最近的点
        dists = torch.norm(traj_end.unsqueeze(1) - route.unsqueeze(0), dim=-1)  # (N, T_r)
        min_route_dist, _ = dists.min(dim=-1)  # (N,)

        # 评分：100m尺度归一化，偏差越小分数越高
        scores = torch.exp(-min_route_dist / 10.0)

        return scores

    def _comfort_score(
        self, trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """舒适度：加速度/曲率变化平滑"""
        N = trajectories.shape[0]
        traj_xy = trajectories[:, :, :2]  # (N, T, 2)

        # 速度
        vel = (traj_xy[:, 1:, :] - traj_xy[:, :-1, :]) / self.dt  # (N, T-1, 2)

        # 加速度
        acc = (vel[:, 1:, :] - vel[:, :-1, :]) / self.dt  # (N, T-2, 2)
        acc_mag = torch.norm(acc, dim=-1)  # (N, T-2)

        # 最大加速度
        max_acc, _ = acc_mag.max(dim=-1)  # (N,)

        # Jerk（加速度变化率）
        jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / self.dt  # (N, T-3, 2)
        jerk_mag = torch.norm(jerk, dim=-1)
        max_jerk, _ = jerk_mag.max(dim=-1)  # (N,)

        # 评分：加速度 < 4m/s² 且 jerk < 8m/s³ → 满分
        acc_score = torch.clamp(1.0 - (max_acc - 4.0) / 4.0, min=0.0, max=1.0)
        jerk_score = torch.clamp(1.0 - (max_jerk - 8.0) / 8.0, min=0.0, max=1.0)

        return 0.5 * acc_score + 0.5 * jerk_score

    def _progress_score(
        self, trajectories: torch.Tensor,
    ) -> torch.Tensor:
        """前进进度：沿纵向的位移越大越好（鼓励前行而非停滞）"""
        N = trajectories.shape[0]
        traj_xy = trajectories[:, :, :2]  # (N, T, 2)

        # 纵向位移（x方向，在自车坐标系下就是前进方向）
        forward_dist = traj_xy[:, -1, 0]  # (N,)

        # 归一化
        scores = torch.clamp(forward_dist / 60.0, min=0.0, max=1.0)

        return scores


def select_best_trajectory(
    trajectories: torch.Tensor,
    scorer: TrajectoryScorer,
    neighbors: Optional[torch.Tensor] = None,
    route: Optional[torch.Tensor] = None,
) -> tuple:
    """
    从 N 条候选轨迹中选择最优。

    Args:
        trajectories: (N, T, D) N 条候选轨迹
        scorer: TrajectoryScorer 实例
        neighbors: (M, T_n, D_n) 邻居轨迹
        route: (T_r, 2) 路线参考点

    Returns:
        best_traj: (1, T, D) 最优轨迹
        best_idx: int 最优轨迹索引
        scores: (N,) 所有轨迹的分数
    """
    scores = scorer.score_trajectories(trajectories, neighbors, route)
    best_idx = scores.argmax().item()
    best_traj = trajectories[best_idx:best_idx+1]
    return best_traj, best_idx, scores

"""
Trajectory Scorer: 安全评分函数
============================
对候选轨迹进行多维度安全评分，用于 Best-of-N 轨迹筛选。

评分维度：
  1. 碰撞检测：轨迹是否与邻居碰撞
  2. TTC（碰撞时间）：越大越安全
  3. 路线一致性：轨迹终点与参考路线的偏差
  4. 舒适度：加速度/曲率变化率
  5. 进度：沿路线方向的前进距离
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TrajectoryScorer:
    """Rule-based 轨迹评分器，对齐 nuPlan 闭环评分标准"""

    def __init__(
        self,
        collision_weight: float = 40.0,
        ttc_weight: float = 20.0,
        route_weight: float = 15.0,
        comfort_weight: float = 10.0,
        progress_weight: float = 15.0,
        collision_threshold: float = 2.0,     # 碰撞距离阈值 (m)
        ttc_threshold: float = 3.0,            # TTC 安全阈值 (s)
        dt: float = 0.1,                       # 时间步长 (s)
        verbose: bool = False,                 # 是否输出详细打分日志
    ):
        self.collision_weight = collision_weight
        self.ttc_weight = ttc_weight
        self.route_weight = route_weight
        self.comfort_weight = comfort_weight
        self.progress_weight = progress_weight
        self.collision_threshold = collision_threshold
        self.ttc_threshold = ttc_threshold
        self.dt = dt
        self.verbose = verbose

    @staticmethod
    def extrapolate_neighbor_future(
        neighbor_past: torch.Tensor,
        future_steps: int,
        dt: float = 0.5,
    ) -> torch.Tensor:
        """
        用邻居最后一帧的速度做匀速直线外推，生成未来轨迹。

        Args:
            neighbor_past: (M, T_p, D) 邻居历史轨迹
                D 维度: [x, y, cos_h, sin_h, vx, vy, width, length, type×3]
            future_steps: 外推的未来时间步数
            dt: 每步时间间隔 (秒)

        Returns:
            neighbor_future: (M, future_steps, 2) 外推的未来 xy 位置
        """
        if neighbor_past is None or neighbor_past.numel() == 0:
            return None

        M = neighbor_past.shape[0]
        device = neighbor_past.device

        # 提取最后一帧的位置和速度
        last_pos = neighbor_past[:, -1, :2]   # (M, 2) x, y
        last_vel = neighbor_past[:, -1, 4:6]  # (M, 2) vx, vy

        # 检查是否有全零的无效邻居（padding），保留 mask
        valid_mask = (neighbor_past[:, -1, :6].abs().sum(dim=-1) > 0)  # (M,)

        # 匀速外推: pos_t = pos_0 + v * dt * t
        time_offsets = torch.arange(1, future_steps + 1, device=device, dtype=last_pos.dtype)  # (T,)
        # (M, 1, 2) + (M, 1, 2) * (1, T, 1) -> (M, T, 2)
        future_pos = last_pos.unsqueeze(1) + last_vel.unsqueeze(1) * (time_offsets.unsqueeze(0).unsqueeze(-1) * dt)

        # 把无效邻居的外推结果清零
        future_pos = future_pos * valid_mask.unsqueeze(-1).unsqueeze(-1).float()

        return future_pos  # (M, future_steps, 2)

    def score_trajectories(
        self,
        trajectories: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None,
        route: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对 N 条候选轨迹打分。

        Args:
            trajectories: (N, T, D) 或者是 (N, 1, T, D)
            neighbors: (M, T_n, 2) 邻居未来轨迹 (外推后的 xy 坐标)
            route: (T_r, 2) 规划路线参考点
        """
        if trajectories.dim() == 4 and trajectories.shape[1] == 1:
            trajectories = trajectories.squeeze(1)
            
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

        if self.verbose:
            logger.info(
                f"Scorer [N={N}] "
                f"collision={collision_scores.cpu().numpy().round(4)} "
                f"ttc={ttc_scores.cpu().numpy().round(4)} "
                f"route={route_scores.cpu().numpy().round(4)} "
                f"comfort={comfort_scores.cpu().numpy().round(4)} "
                f"progress={progress_scores.cpu().numpy().round(4)} "
                f"total={scores.cpu().numpy().round(4)}"
            )

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

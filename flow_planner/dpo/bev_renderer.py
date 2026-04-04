"""
BEV 可视化渲染器
=================
用 matplotlib 渲染鸟瞰图 (Bird's Eye View)，展示候选轨迹和场景上下文。
用于 VLM (Gemini) 评价候选轨迹的质量。

用法：
    renderer = BEVRenderer(image_size=(800, 800), view_range=50.0)
    renderer.render_scenario(
        ego_pos=(0, 0), ego_heading=0,
        candidates=candidates_np,       # (K, T, 2)
        neighbors=neighbor_past_np,     # (M, T_p, D)
        lanes=lanes_np,                 # (L, P, 2)
        save_path='scenario_001.png',
        chosen_idx=0, rejected_idx=3,
    )
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# 色盲友好的调色板 (最多 10 条候选)
TRAJECTORY_COLORS = [
    '#2196F3',  # Blue
    '#FF5722',  # Deep Orange
    '#4CAF50',  # Green
    '#9C27B0',  # Purple
    '#FF9800',  # Orange
    '#00BCD4',  # Cyan
    '#E91E63',  # Pink
    '#795548',  # Brown
    '#607D8B',  # Blue Grey
    '#CDDC39',  # Lime
]


class BEVRenderer:
    """
    鸟瞰图渲染器。

    坐标约定：
      - 自车位于原点 (0, 0)
      - x 轴正方向为自车前进方向
      - y 轴正方向为左侧
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 800),
        view_range: float = 50.0,
        dpi: int = 100,
        bg_color: str = '#1a1a2e',
        grid_color: str = '#16213e',
    ):
        """
        Args:
            image_size: 输出图片大小 (width, height) in pixels
            view_range: 显示范围 (米)，以自车为中心的正方形
            dpi: 图片 DPI
            bg_color: 背景色
            grid_color: 网格线颜色
        """
        self.image_size = image_size
        self.view_range = view_range
        self.dpi = dpi
        self.bg_color = bg_color
        self.grid_color = grid_color

    def render_scenario(
        self,
        candidates: np.ndarray,
        ego_pos: Tuple[float, float] = (0.0, 0.0),
        ego_heading: float = 0.0,
        neighbors: Optional[np.ndarray] = None,
        lanes: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        chosen_idx: Optional[int] = None,
        rejected_idx: Optional[int] = None,
        title: Optional[str] = None,
        scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        渲染一个场景的 BEV 图。

        Args:
            candidates: (K, T, >=2) 候选轨迹，至少包含 x, y 坐标
            ego_pos: 自车位置 (x, y)
            ego_heading: 自车朝向 (弧度)
            neighbors: (M, T_p, D) 邻居历史轨迹
                       D >= 2: [x, y, ...]
                       D >= 4: [x, y, cos_h, sin_h, ...]
                       D >= 8: [x, y, cos_h, sin_h, vx, vy, width, length, ...]
            lanes: (L, P, >=2) 车道线点序列
            save_path: 保存路径，None 则 show
            chosen_idx: chosen 轨迹索引（绿色高亮）
            rejected_idx: rejected 轨迹索引（红色高亮）
            title: 图片标题
            scores: (K,) 每条轨迹的分数
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyArrowPatch

        fig_w = self.image_size[0] / self.dpi
        fig_h = self.image_size[1] / self.dpi
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=self.dpi)

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)

        vr = self.view_range
        ax.set_xlim(-vr, vr)
        ax.set_ylim(-vr, vr)
        ax.set_aspect('equal')
        ax.grid(True, color=self.grid_color, linewidth=0.5, alpha=0.3)

        # 隐藏刻度（VLM 不需要坐标轴数值）
        ax.tick_params(colors='#444444', labelsize=6)

        # ===== 1. 绘制车道线 =====
        if lanes is not None:
            self._draw_lanes(ax, lanes)

        # ===== 2. 绘制邻居 =====
        if neighbors is not None:
            self._draw_neighbors(ax, neighbors)

        # ===== 3. 绘制候选轨迹 =====
        K = candidates.shape[0]
        # Spread labels to avoid overlapping
        label_offsets_y = np.linspace(-3, 3, K) if K > 1 else [0]
        
        for k in range(K):
            traj = candidates[k]  # (T, >=2)
            color = TRAJECTORY_COLORS[k % len(TRAJECTORY_COLORS)]
            linewidth = 1.0  # Thinner line so overlaps don't become blobs
            alpha = 0.9      # High alpha to be identifiable
            label = f"#{k+1}"

            # 高亮 chosen / rejected
            if k == chosen_idx:
                color = '#00E676'  # 亮绿
                linewidth = 3.5
                alpha = 1.0
                label = f"#{k+1} ✓ CHOSEN"
            elif k == rejected_idx:
                color = '#FF1744'  # 亮红
                linewidth = 3.5
                alpha = 1.0
                label = f"#{k+1} ✗ REJECTED"

            # 绘制轨迹线
            ax.plot(traj[:, 0], traj[:, 1],
                    color=color, linewidth=linewidth, alpha=alpha,
                    zorder=10, solid_capstyle='round')

            # Add markers every 10 timesteps to show longitudinal speed
            marker_indices = range(10, len(traj), 10)
            ax.plot(traj[marker_indices, 0], traj[marker_indices, 1], 'o', color=color, markersize=4, alpha=0.8, zorder=12)

            # Draw labels at the end of trajectory, staggered
            if scores is not None and k < len(scores):
                label += f" ({scores[k]:.1f})"
            
            tx = traj[-1, 0] + 1.0
            ty = traj[-1, 1] + label_offsets_y[k]
            ax.text(tx, ty, label, color=color, fontsize=10, fontweight='bold', zorder=15,
                    bbox=dict(facecolor=self.bg_color, edgecolor='none', alpha=0.7, pad=0.5))

            # 起点小圆点
            ax.plot(traj[0, 0], traj[0, 1], 'o',
                    color=color, markersize=4, zorder=11)

        # ===== 4. 绘制自车 =====
        self._draw_ego(ax, ego_pos, ego_heading)

        # ===== 5. 标题 =====
        if title:
            ax.set_title(title, color='white', fontsize=10, pad=10)

        # ===== 6. 图例 =====
        ax.legend(loc='upper right', fontsize=6, framealpha=0.3,
                  facecolor='black', edgecolor='gray', labelcolor='white')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
            logger.debug(f"BEV saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    def _draw_ego(self, ax, pos, heading):
        """绘制自车：蓝色三角形 + 朝向"""
        from matplotlib.patches import Polygon as MplPolygon

        ex, ey = pos
        # 自车三角形（长 4.5m, 宽 2m 的简化表示）
        length = 4.5
        width = 2.0

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # 三角形三个顶点（车头、左后、右后）
        front = (ex + length * 0.6 * cos_h, ey + length * 0.6 * sin_h)
        rear_left = (ex - length * 0.4 * cos_h + width * 0.5 * sin_h,
                     ey - length * 0.4 * sin_h - width * 0.5 * cos_h)
        rear_right = (ex - length * 0.4 * cos_h - width * 0.5 * sin_h,
                      ey - length * 0.4 * sin_h + width * 0.5 * cos_h)

        triangle = MplPolygon(
            [front, rear_left, rear_right],
            closed=True,
            facecolor='#448AFF',
            edgecolor='white',
            linewidth=1.5,
            zorder=20,
            label='Ego'
        )
        ax.add_patch(triangle)

        # 朝向箭头
        arrow_len = 3.0
        ax.annotate('', xy=(ex + arrow_len * cos_h, ey + arrow_len * sin_h),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                    zorder=21)

    def _draw_neighbors(self, ax, neighbors):
        """
        绘制邻居车辆。

        neighbors: (M, T_p, D) 其中 D 至少为 2
          D=2:  [x, y]
          D=4:  [x, y, cos_h, sin_h]
          D>=7: [x, y, cos_h, sin_h, vx, vy, width, length, ...]
        """
        import matplotlib.patches as patches

        M = neighbors.shape[0]
        D = neighbors.shape[2] if neighbors.ndim == 3 else neighbors.shape[1]

        for m in range(M):
            if neighbors.ndim == 3:
                traj = neighbors[m]  # (T_p, D)
            else:
                traj = neighbors[m:m+1]

            # 检查是否全零（无效邻居）
            if np.abs(traj).sum() < 1e-6:
                continue

            # 当前位置（最后一帧）
            curr = traj[-1]
            nx, ny = curr[0], curr[1]

            # 检查是否在视野范围内
            if abs(nx) > self.view_range or abs(ny) > self.view_range:
                continue

            # 绘制历史轨迹（虚线）
            if traj.shape[0] > 1:
                ax.plot(traj[:, 0], traj[:, 1], '--',
                       color='#FF6B6B', linewidth=1.0, alpha=0.4, zorder=5)

            # 绘制车辆矩形
            if D >= 8:
                w = max(float(curr[6]), 1.5)
                l = max(float(curr[7]), 3.5)
            else:
                w, l = 2.0, 4.5

            if D >= 4:
                cos_h, sin_h = curr[2], curr[3]
                angle = np.degrees(np.arctan2(sin_h, cos_h))
            else:
                angle = 0.0

            rect = patches.Rectangle(
                (nx - l / 2, ny - w / 2), l, w,
                angle=angle,
                rotation_point='center',
                linewidth=1.0,
                edgecolor='#FF6B6B',
                facecolor='#FF6B6B',
                alpha=0.5,
                zorder=8,
            )
            ax.add_patch(rect)

    def _draw_lanes(self, ax, lanes):
        """
        绘制车道线。

        lanes: (L, P, >=2) 车道线点序列
          或 (L*P, >=2) 展平后的点云
        """
        if lanes.ndim == 2:
            # 展平的点云，按距离聚类不太现实，直接画散点
            valid = np.abs(lanes).sum(axis=-1) > 1e-6
            ax.scatter(lanes[valid, 0], lanes[valid, 1],
                      s=1, c='#666666', alpha=0.3, zorder=2)
            return

        L = lanes.shape[0]
        for l in range(L):
            lane = lanes[l]  # (P, >=2)
            # 过滤全零点
            valid = np.abs(lane).sum(axis=-1) > 1e-6
            if valid.sum() < 2:
                continue

            ax.plot(lane[valid, 0], lane[valid, 1],
                   color='#666666', linewidth=1.0, alpha=0.5, zorder=2,
                   solid_capstyle='round')


def render_preference_pair(
    renderer: BEVRenderer,
    preference: dict,
    save_path: str,
):
    """
    渲染一对偏好对的 BEV 对比图。

    Args:
        renderer: BEVRenderer 实例
        preference: 包含 chosen, rejected, condition 的字典
        save_path: 保存路径
    """
    chosen = preference['chosen']       # (T, D)
    rejected = preference['rejected']   # (T, D)

    candidates = np.stack([chosen, rejected], axis=0)  # (2, T, D)

    # 从 condition 中提取上下文
    condition = preference.get('condition', {})
    neighbors = condition.get('neighbor_past', None)
    lanes = condition.get('lane', None)

    # 确保是 numpy
    if hasattr(candidates, 'numpy'):
        candidates = candidates.numpy()

    renderer.render_scenario(
        candidates=candidates[:, :, :2],
        neighbors=neighbors,
        lanes=lanes,
        chosen_idx=0,
        rejected_idx=1,
        save_path=save_path,
        title=f"Score gap: {preference.get('score_gap', 'N/A'):.2f}"
              if 'score_gap' in preference else None,
    )

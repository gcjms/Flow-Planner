"""
DPO Loss for Flow Matching
==========================
实现 Direct Preference Optimization 损失函数，适配 Flow Matching 的连续速度场生成。

核心思想：
  - 给定 (chosen, rejected) 偏好对
  - 计算两条轨迹在 Flow Matching 框架下的对数似然
  - 用 DPO Loss 让模型更倾向于生成 chosen 轨迹

参考：
  - Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FlowMatchingDPOLoss(nn.Module):
    """Flow Matching 框架下的 DPO 损失函数"""

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: DPO 温度系数，控制偏好强度。
                  越大 → 对偏好差异越敏感
                  越小 → 更保守，不会偏离参考模型太远
        """
        super().__init__()
        self.beta = beta

    def compute_log_prob(
        self,
        model: nn.Module,
        trajectory: torch.Tensor,
        condition: dict,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算模型对给定轨迹的对数似然近似。

        原理：
          1. 给定目标轨迹 y 和随机噪声 x0
          2. 计算中间状态: x_t = (1-t)*x0 + t*y
          3. 模型预测速度: v_pred = model(x_t, t, condition)
          4. 真实速度: v_true = y - x0
          5. log P(y|x) ≈ -||v_pred - v_true||²

        Args:
            model: Flow-Planner 模型（带或不带 LoRA）
            trajectory: (B, T, 2) 轨迹坐标
            condition: dict, 场景条件（包含 neighbor, lane 等编码后的特征）
            t: (B, 1, 1) 采样的时间步
            x0: (B, T, 2) 采样的噪声

        Returns:
            log_prob: (B,) 每个样本的对数似然
        """
        B = trajectory.shape[0]

        # 中间状态：线性插值
        x_t = (1 - t) * x0 + t * trajectory  # (B, T, 2)

        # 真实速度场：直线的斜率
        v_true = trajectory - x0  # (B, T, 2)

        # 模型预测速度场
        v_pred = model.predict_velocity(x_t, t.squeeze(-1).squeeze(-1), condition)  # (B, T, 2)

        # 对数似然 ≈ 负 MSE（逐样本求和）
        mse = (v_pred - v_true).pow(2).sum(dim=(-1, -2))  # (B,)
        log_prob = -mse

        return log_prob

    def forward(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        condition: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DPO Loss。

        Args:
            model: 正在训练的 Flow-Planner（带 LoRA）
            ref_model: 冻结的参考 Flow-Planner（原始权重）
            chosen: (B, T, 2) 好轨迹
            rejected: (B, T, 2) 坏轨迹
            condition: dict, 编码后的场景条件

        Returns:
            loss: scalar, DPO 损失
            metrics: dict, 训练指标
        """
        B = chosen.shape[0]
        device = chosen.device

        # 采样共享的噪声和时间（chosen 和 rejected 用同一组）
        x0 = torch.randn_like(chosen)
        t = torch.rand(B, 1, 1, device=device)

        # 计算训练模型的 log prob
        log_pi_w = self.compute_log_prob(model, chosen, condition, t, x0)
        log_pi_l = self.compute_log_prob(model, rejected, condition, t, x0)

        # 计算参考模型的 log prob（不计算梯度）
        with torch.no_grad():
            log_ref_w = self.compute_log_prob(ref_model, chosen, condition, t, x0)
            log_ref_l = self.compute_log_prob(ref_model, rejected, condition, t, x0)

        # DPO 核心公式
        # Δ = (log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))
        delta = (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)

        # Loss = -log σ(β · Δ)
        loss = -F.logsigmoid(self.beta * delta).mean()

        # 训练指标
        metrics = {
            "dpo/loss": loss.item(),
            "dpo/delta_mean": delta.mean().item(),
            "dpo/chosen_reward": (log_pi_w - log_ref_w).mean().item(),
            "dpo/rejected_reward": (log_pi_l - log_ref_l).mean().item(),
            "dpo/accuracy": (delta > 0).float().mean().item(),  # 模型是否正确偏好 chosen
        }

        return loss, metrics

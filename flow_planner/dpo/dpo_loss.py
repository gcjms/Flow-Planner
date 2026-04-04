"""
DPO Loss for Flow Matching
==========================
实现 Direct Preference Optimization 损失函数，适配 Flow-Planner 的连续速度场生成。

核心思想：
  - 给定 (chosen, rejected) 偏好对
  - 计算两条轨迹在 Flow Matching 框架下的对数似然
  - 用 DPO Loss 让模型更倾向于生成 chosen 轨迹

参考：
  - Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023)
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)

适配说明：
  Flow-Planner 的 decoder 接口:
    decoder(x, t, **model_extra)
      x: (B, P, action_len, state_dim) — action tokens
      t: (B,) — flow matching 时间步
      model_extra: encoder 输出 (encodings, masks, routes_cond, token_dist, cfg_flags)
    → prediction: (B, P, action_len, state_dim)

  对数似然计算:
    给定目标轨迹 y, 随机噪声 x0, 时间步 t:
    1. 中间状态: x_t = (1-t)*x0 + t*y   (线性插值路径)
    2. 真实速度: v_true = y - x0         (直线路径的斜率)
    3. 模型预测: v_pred = decoder(x_t, t, cond)
    4. log P(y|cond) ≈ -||v_pred - v_true||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from flow_planner.model.model_utils.traj_tool import traj_chunking


class FlowMatchingDPOLoss(nn.Module):
    """Flow Matching 框架下的 DPO 损失函数"""

    def __init__(self, beta: float = 0.1, sft_weight: float = 0.1):
        """
        Args:
            beta: DPO 温度系数，控制偏好强度。
                  越大 → 对偏好差异越敏感
                  越小 → 更保守，不会偏离参考模型太远
            sft_weight: SFT 损失的权重，控制锚点效应。
        """
        super().__init__()
        self.beta = beta
        self.sft_weight = sft_weight

    def compute_log_prob(
        self,
        model: nn.Module,
        trajectory: torch.Tensor,
        encoder_outputs: dict,
        t: torch.Tensor,
        x0: torch.Tensor,
        action_len: int,
        action_overlap: int,
        data_processor=None,
    ) -> torch.Tensor:
        """
        计算模型对给定轨迹的对数似然近似。

        流程：
          1. 归一化轨迹（raw space → model space）
          2. 将轨迹分块为 action tokens (与训练一致)
          3. 对噪声也做相同分块
          4. 计算中间状态和真实速度（在 token 空间）
          5. 用 decoder 预测速度
          6. log P(y|x) ≈ -||v_pred - v_true||²

        Args:
            model: FlowPlanner 模型 (调用 model.decoder)
            trajectory: (B, T, D) 轨迹坐标, D=state_dim (raw space)
            encoder_outputs: decoder 所需的条件信息 (来自 model.extract_decoder_inputs)
            t: (B,) 采样的时间步, 范围 [0, 1]
            x0: (B, T, D) 采样的高斯噪声
            action_len: action token 长度 (如 20)
            action_overlap: action token 重叠量 (如 10)
            data_processor: ModelInputProcessor (用于归一化)

        Returns:
            log_prob: (B,) 每个样本的对数似然
        """
        B = trajectory.shape[0]

        # 归一化：raw coordinates → model normalized space
        if data_processor is not None:
            trajectory = data_processor.state_preprocess(trajectory)

        # 将轨迹和噪声分块为 action tokens
        # Training code convention: traj_chunking expects (B, P, T, D) with P=1
        # traj_chunking: (B, 1, T, D) → list of (B, 1, action_len, D)
        traj_tokens = traj_chunking(trajectory.unsqueeze(1), action_len, action_overlap)
        traj_tokens = torch.cat(traj_tokens, dim=1)   # (B, P, action_len, D)

        noise_tokens = traj_chunking(x0.unsqueeze(1), action_len, action_overlap)
        noise_tokens = torch.cat(noise_tokens, dim=1)  # (B, P, action_len, D)

        # 中间状态 x_t = (1-t)*x0 + t*y (在 token 空间)
        t_expand = t.view(B, 1, 1, 1)  # 广播到 (B, P, action_len, D)
        x_t = (1 - t_expand) * noise_tokens + t_expand * traj_tokens

        # 真实速度场 v_true = y - x0
        v_true = traj_tokens - noise_tokens

        # 模型预测速度场
        v_pred = model.decoder(x_t, t, **encoder_outputs)  # (B, P, action_len, D)

        # 对数似然 ≈ -MSE (逐样本求和)
        mse = (v_pred - v_true).pow(2).sum(dim=(-1, -2, -3))  # (B,)
        log_prob = -mse

        return log_prob

    def forward(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        encoder_outputs: dict,
        action_len: int,
        action_overlap: int,
        data_processor=None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DPO Loss。

        Args:
            model: 正在训练的 Flow-Planner（带 LoRA 的 decoder）
            ref_model: 冻结的参考 Flow-Planner（原始权重）
            chosen: (B, T, D) 好轨迹 (raw space)
            rejected: (B, T, D) 坏轨迹 (raw space)
            encoder_outputs: 编码后的场景条件 (来自共享 encoder)
            action_len: action token 长度
            action_overlap: action token 重叠量
            data_processor: ModelInputProcessor (用于归一化)
            sft_weight: SFT 损失的权重 (防止模式崩溃和遗忘)

        Returns:
            loss: scalar, DPO 损失
            metrics: dict, 训练指标
        """
        B = chosen.shape[0]
        device = chosen.device

        # 采样共享的噪声和时间（chosen 和 rejected 用同一组）
        x0 = torch.randn_like(chosen)
        t = torch.rand(B, device=device)

        # 计算训练模型 (policy) 的 log prob
        log_pi_w = self.compute_log_prob(
            model, chosen, encoder_outputs, t, x0,
            action_len, action_overlap, data_processor
        )
        log_pi_l = self.compute_log_prob(
            model, rejected, encoder_outputs, t, x0,
            action_len, action_overlap, data_processor
        )

        # 计算参考模型的 log prob（不计算梯度）
        with torch.no_grad():
            log_ref_w = self.compute_log_prob(
                ref_model, chosen, encoder_outputs, t, x0,
                action_len, action_overlap, data_processor
            )
            log_ref_l = self.compute_log_prob(
                ref_model, rejected, encoder_outputs, t, x0,
                action_len, action_overlap, data_processor
            )

        # 核心 DPO 公式
        # Δ = (log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))
        delta = (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)

        # 1. DPO Loss = -log σ(β · Δ)
        dpo_loss = -F.logsigmoid(self.beta * delta).mean()

        # 2. SFT Loss = -log π_θ(y_w) (也就是 chosen 轨迹的 MSE)
        sft_loss = -log_pi_w.mean()

        # 总损失
        sft_weight = getattr(self, 'sft_weight', 0.0) # fallback if not set at init
        loss = dpo_loss + sft_weight * sft_loss

        # 训练指标
        metrics = {
            "dpo/loss": loss.item(),
            "dpo/loss_dpo_only": dpo_loss.item(),
            "dpo/loss_sft": sft_loss.item(),
            "dpo/delta_mean": delta.mean().item(),
            "dpo/chosen_reward": (log_pi_w - log_ref_w).mean().item(),
            "dpo/rejected_reward": (log_pi_l - log_ref_l).mean().item(),
            "dpo/accuracy": (delta > 0).float().mean().item(),  # 模型是否正确偏好 chosen
            "dpo/log_pi_chosen": log_pi_w.mean().item(),
            "dpo/log_pi_rejected": log_pi_l.mean().item(),
        }

        return loss, metrics

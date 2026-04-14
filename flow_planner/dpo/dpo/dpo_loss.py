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

    def __init__(self, beta: float = 0.1, sft_weight: float = 0.1, num_t_samples: int = 16):
        """
        Args:
            beta: DPO 温度系数，控制偏好强度。
            sft_weight: SFT 损失的权重。
            num_t_samples: 每次 forward 采样多少个时间步来估计 log_prob。
                          设为 1 = 原来的行为（方差极大）；
                          设为 16 = 对 16 个不同 t 求平均，方差降低 ~16 倍。
        """
        super().__init__()
        self.beta = beta
        self.sft_weight = sft_weight
        self.num_t_samples = num_t_samples

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
        计算模型对给定轨迹的对数似然近似（单时间步版本）。

        Args:
            model: FlowPlanner 模型
            trajectory: (B, T, D) 轨迹坐标 (raw space)
            encoder_outputs: decoder 条件
            t: (B,) 时间步 [0, 1]
            x0: (B, T, D) 高斯噪声
            action_len, action_overlap: token 分块参数
            data_processor: 归一化器

        Returns:
            log_prob: (B,)
        """
        B = trajectory.shape[0]

        # 归一化
        if data_processor is not None:
            trajectory = data_processor.state_preprocess(trajectory)

        # 分块为 action tokens
        traj_tokens = traj_chunking(trajectory.unsqueeze(1), action_len, action_overlap)
        traj_tokens = torch.cat(traj_tokens, dim=1)

        noise_tokens = traj_chunking(x0.unsqueeze(1), action_len, action_overlap)
        noise_tokens = torch.cat(noise_tokens, dim=1)

        # 中间状态 & 真实速度
        t_expand = t.view(B, 1, 1, 1)
        x_t = (1 - t_expand) * noise_tokens + t_expand * traj_tokens
        v_true = traj_tokens - noise_tokens

        # 模型预测
        v_pred = model.decoder(x_t, t, **encoder_outputs)

        # log_prob = -MSE (per-sample mean)
        mse = (v_pred - v_true).pow(2).mean(dim=(-1, -2, -3))
        return -mse

    def compute_log_prob_multi_t(
        self,
        model: nn.Module,
        trajectory: torch.Tensor,
        encoder_outputs: dict,
        action_len: int,
        action_overlap: int,
        data_processor=None,
        num_samples: int = 16,
    ) -> torch.Tensor:
        """
        多次采样 t，取平均来稳定 log_prob 估计。

        单次采样方差极大（t≈0 时 chosen/rejected 不可区分，t≈1 时差异明显），
        平均 K 次后方差降低 K 倍，DPO 梯度才有稳定的方向。

        Args:
            num_samples: 采样时间步的数量（K）

        Returns:
            log_prob: (B,) 平均后的 log_prob
        """
        B = trajectory.shape[0]
        device = trajectory.device

        # 归一化只做一次
        if data_processor is not None:
            trajectory = data_processor.state_preprocess(trajectory)

        # 分块只做一次
        traj_tokens = traj_chunking(trajectory.unsqueeze(1), action_len, action_overlap)
        traj_tokens = torch.cat(traj_tokens, dim=1)

        log_probs = []
        for _ in range(num_samples):
            x0 = torch.randn_like(traj_tokens)
            t = torch.rand(B, device=device)

            t_expand = t.view(B, 1, 1, 1)
            x_t = (1 - t_expand) * x0 + t_expand * traj_tokens
            v_true = traj_tokens - x0

            v_pred = model.decoder(x_t, t, **encoder_outputs)
            mse = (v_pred - v_true).pow(2).mean(dim=(-1, -2, -3))
            log_probs.append(-mse)

        # 平均 K 个采样的 log_prob
        return torch.stack(log_probs, dim=0).mean(dim=0)  # (B,)

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
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DPO Loss（多时间步采样版本）。

        Args:
            sample_weights: (B,) optional per-sample weights for multi-objective DPO.
                           Allows weighting TTC pairs higher than comfort pairs, etc.
                           If None, all samples weighted equally.
        """
        B = chosen.shape[0]
        K = self.num_t_samples

        log_pi_w = self.compute_log_prob_multi_t(
            model, chosen, encoder_outputs,
            action_len, action_overlap, data_processor, K
        )
        log_pi_l = self.compute_log_prob_multi_t(
            model, rejected, encoder_outputs,
            action_len, action_overlap, data_processor, K
        )

        with torch.no_grad():
            log_ref_w = self.compute_log_prob_multi_t(
                ref_model, chosen, encoder_outputs,
                action_len, action_overlap, data_processor, K
            )
            log_ref_l = self.compute_log_prob_multi_t(
                ref_model, rejected, encoder_outputs,
                action_len, action_overlap, data_processor, K
            )

        delta = (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)

        per_sample_dpo = -F.logsigmoid(self.beta * delta)

        if sample_weights is not None:
            w = sample_weights.to(per_sample_dpo.device)
            w = w / (w.sum() + 1e-8) * B
            dpo_loss = (per_sample_dpo * w).mean()
        else:
            dpo_loss = per_sample_dpo.mean()

        sft_loss = -log_pi_w.mean()

        sft_weight = getattr(self, 'sft_weight', 0.0)
        loss = dpo_loss + sft_weight * sft_loss

        metrics = {
            "dpo/loss": loss.item(),
            "dpo/loss_dpo_only": dpo_loss.item(),
            "dpo/loss_sft": sft_loss.item(),
            "dpo/delta_mean": delta.mean().item(),
            "dpo/chosen_reward": (log_pi_w - log_ref_w).mean().item(),
            "dpo/rejected_reward": (log_pi_l - log_ref_l).mean().item(),
            "dpo/accuracy": (delta > 0).float().mean().item(),
            "dpo/log_pi_chosen": log_pi_w.mean().item(),
            "dpo/log_pi_rejected": log_pi_l.mean().item(),
        }

        return loss, metrics

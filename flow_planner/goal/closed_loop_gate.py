from __future__ import annotations

import torch
import torch.nn as nn

from flow_planner.goal.candidate_selector import CandidateSelector


class ClosedLoopGate(nn.Module):
    """Binary gate for whether a selected anchor candidate should take over."""

    def __init__(
        self,
        planner_backbone,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = planner_backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.scorer = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def extract_scene_features(self, data) -> torch.Tensor:
        batch_size = data.ego_current.shape[0]
        cfg_flags = torch.ones((batch_size,), dtype=torch.int32, device=self.backbone.device)

        with torch.set_grad_enabled(not self.freeze_backbone):
            model_inputs, _ = self.backbone.prepare_model_input(
                cfg_flags,
                data,
                use_cfg=False,
                is_training=False,
            )
            encoder_inputs = self.backbone.extract_encoder_inputs(model_inputs)
            encoder_outputs = self.backbone.encoder(**encoder_inputs)

            agents_tokens, lane_tokens = encoder_outputs["encodings"]
            agents_mask, lane_mask = encoder_outputs["masks"]
            route_cond = encoder_outputs["routes_cond"]

            pooled_agents = self._masked_mean(agents_tokens, agents_mask)
            pooled_lanes = self._masked_mean(lane_tokens, lane_mask)
            ego_current = model_inputs["ego_current"].float()

            features = torch.cat(
                [route_cond.float(), pooled_agents.float(), pooled_lanes.float(), ego_current],
                dim=-1,
            )
        return features

    def score_features(
        self,
        scene_features: torch.Tensor,
        candidate_trajs: torch.Tensor,
        anchor_trajs: torch.Tensor,
    ) -> torch.Tensor:
        """Return accept logits with shape ``(B,)`` or ``(B, N)``."""
        features = CandidateSelector.build_candidate_features(
            scene_features,
            candidate_trajs,
            anchor_trajs,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    def forward(
        self,
        data,
        candidate_trajs: torch.Tensor,
        anchor_trajs: torch.Tensor,
    ) -> torch.Tensor:
        scene_features = self.extract_scene_features(data)
        return self.score_features(scene_features, candidate_trajs, anchor_trajs)

from __future__ import annotations

import torch
import torch.nn as nn


class CandidateSelector(nn.Module):
    """Scene-conditioned scorer over generated anchor/candidate trajectories.

    The scorer reuses the planner backbone's scene encoder, then ranks a
    variable number of candidates within the same scene:

        scene + anchor_traj + candidate_traj -> scalar score
    """

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

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def extract_scene_features(self, data) -> torch.Tensor:
        """Pool the same scene tokens used by goal/anchor predictors."""
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

    def forward(
        self,
        data,
        candidate_trajs: torch.Tensor,
        anchor_trajs: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-candidate logits with shape ``(B, N)``."""
        if candidate_trajs.ndim != 4:
            raise ValueError(
                f"candidate_trajs must have shape (B, N, T, D), got {tuple(candidate_trajs.shape)}"
            )
        if anchor_trajs.ndim != 4:
            raise ValueError(
                f"anchor_trajs must have shape (B, N, T, D), got {tuple(anchor_trajs.shape)}"
            )
        if candidate_trajs.shape[:2] != anchor_trajs.shape[:2]:
            raise ValueError(
                "candidate_trajs and anchor_trajs must share batch/candidate dims, "
                f"got {tuple(candidate_trajs.shape[:2])} vs {tuple(anchor_trajs.shape[:2])}"
            )

        scene_features = self.extract_scene_features(data)
        batch_size, num_candidates = candidate_trajs.shape[:2]

        candidate_flat = candidate_trajs.reshape(batch_size, num_candidates, -1).float()
        anchor_flat = anchor_trajs.reshape(batch_size, num_candidates, -1).float()

        overlap_dim = min(candidate_trajs.shape[-1], anchor_trajs.shape[-1])
        delta_flat = (
            candidate_trajs[..., :overlap_dim] - anchor_trajs[..., :overlap_dim]
        ).reshape(batch_size, num_candidates, -1).float()

        scene_expanded = scene_features.unsqueeze(1).expand(-1, num_candidates, -1)
        features = torch.cat(
            [scene_expanded, candidate_flat, anchor_flat, delta_flat],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)

        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
        return logits

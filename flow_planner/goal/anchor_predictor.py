"""
AnchorPredictor (Phase 1)
=========================
对标 ``goal_predictor.GoalPredictor``，但预测对象从 endpoint 聚类 ``(K,)``
升级为轨迹 anchor 聚类 ``(K,)``，label 来自 backbone 的
``_get_anchor_for_gt`` → ``find_nearest_anchor_torch``。

Phase 1 只保留单头 imitation CE（学 "最像 GT 的 anchor"）。Phase 2 会在
同一个 head 旁边加上 metric heads (safety/progress/comfort...)
做 Hydra-MDP 风格的多 teacher soft distillation，复用本类的
``extract_scene_features`` 不变。
"""

from __future__ import annotations

import torch
import torch.nn as nn



class AnchorPredictor(nn.Module):
    """Scene-conditioned classifier over the trajectory anchor vocabulary.

    The backbone provides:
      - normalized model inputs
      - scene encoder outputs
      - anchor vocabulary (backbone._anchor_vocab_tensor)

    The predictor itself is a light MLP on pooled scene features.
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

        if getattr(self.backbone, "_anchor_vocab_tensor", None) is None:
            raise ValueError(
                "planner_backbone must be initialized with anchor_vocab_path "
                "(FlowPlanner(..., anchor_vocab_path=...))"
            )

        self.num_anchors = int(self.backbone._anchor_vocab_tensor.shape[0])
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_anchors),
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
        """Same pooled scene features as GoalPredictor — intentionally shared."""
        B = data.ego_current.shape[0]
        cfg_flags = torch.ones((B,), dtype=torch.int32, device=self.backbone.device)

        with torch.set_grad_enabled(not self.freeze_backbone):
            model_inputs, _ = self.backbone.prepare_model_input(
                cfg_flags, data, use_cfg=False, is_training=False
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

    def get_anchor_labels(self, data) -> torch.Tensor:
        """Nearest-anchor indices for a batch (used as classification target).

        Delegates to the backbone's index lookup so we don't do
        ``find_nearest(find_nearest(gt))`` (which was idempotent but wasted a
        full (B, K) distance matrix every step).
        """
        indices = self.backbone._get_anchor_index_for_gt(data)
        if indices is None:
            raise ValueError(
                "anchor vocabulary is required to build training labels "
                "(backbone._get_anchor_index_for_gt returned None)"
            )
        return indices

    def forward(self, data) -> torch.Tensor:
        features = self.extract_scene_features(data)
        return self.head(features)

    @torch.no_grad()
    def predict_topk(self, data, top_k: int = 5):
        """Return top-k anchor ids + full trajectory templates."""
        logits = self.forward(data)
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        scores, indices = torch.topk(probs, k=k, dim=-1)                # (B, k)
        vocab = self.backbone._anchor_vocab_tensor.to(indices.device)   # (K, T, 3)
        B, kk = indices.shape
        anchor_trajs = vocab[indices.reshape(-1)].reshape(B, kk, *vocab.shape[1:])
        return {
            "logits": logits,
            "probs": probs,
            "scores": scores,
            "indices": indices,
            "anchor_trajs": anchor_trajs,
        }

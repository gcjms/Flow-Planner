import torch
import torch.nn as nn


class GoalPredictor(nn.Module):
    """
    Lightweight scene-conditioned goal classifier on top of a pretrained
    FlowPlanner backbone.

    The backbone provides:
      - normalized model inputs
      - scene encoder outputs
      - goal vocabulary used to build training labels

    The predictor itself is a small MLP that maps scene features to logits
    over the global goal vocabulary.
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

        if self.backbone._goal_vocab is None:
            raise ValueError("planner_backbone must be initialized with goal_vocab_path")

        self.num_goals = int(self.backbone._goal_vocab.shape[0])
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
            nn.Linear(hidden_dim, self.num_goals),
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

    def extract_scene_features(self, data):
        """
        Build a scene-level feature vector from the pretrained planner encoder.
        """
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

    def get_goal_labels(self, data):
        goal_points = self.backbone._get_goal_for_gt(data)
        if goal_points is None:
            raise ValueError("goal vocabulary is required to build training labels")
        vocab = self.backbone._goal_vocab_tensor.to(goal_points.device)
        # Exact match is expected because _get_goal_for_gt selects directly from vocab.
        diff = goal_points.unsqueeze(1) - vocab.unsqueeze(0)
        indices = torch.argmin(torch.linalg.norm(diff, dim=-1), dim=1)
        return indices

    def forward(self, data):
        features = self.extract_scene_features(data)
        return self.head(features)

    @torch.no_grad()
    def predict_topk(self, data, top_k: int = 5):
        logits = self.forward(data)
        probs = torch.softmax(logits, dim=-1)
        scores, indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        goal_points = self.backbone._goal_vocab_tensor.to(indices.device)[indices]
        return {
            "logits": logits,
            "probs": probs,
            "scores": scores,
            "indices": indices,
            "goal_points": goal_points,
        }

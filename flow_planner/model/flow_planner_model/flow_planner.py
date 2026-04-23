import re
import os
import sys
from typing import Literal, Callable, Any, Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from flow_planner.model.model_base import DiffusionADPlanner
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class FlowPlanner(DiffusionADPlanner):

    def __init__(
        self,
        model_encoder,
        model_decoder,

        flow_ode,
        
        model_type: Literal['x_start', 'noise', 'velocity'] = 'x_start',
        kinematic: Literal["waypoints", "velocity", "acceleration"] = 'waypoints',
    
        assemble_method='linear',
        
        data_processor: ModelInputProcessor = None,
        
        goal_vocab_path: str = None,
        anchor_vocab_path: str = None,

        device='cuda',
        **planner_params
    ):
        
        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        self.device = device
        
        self.flow_ode = flow_ode # including flow matching path and ode solver
        self.cfg_prob = planner_params['cfg_prob']
        self.cfg_weight = planner_params['cfg_weight']
        self.cfg_type = planner_params['cfg_type']

        self.kinematic = kinematic
        
        self.assemble_method = assemble_method
        
        self.data_processor = data_processor

        # Risk-Aware Adaptive CFG
        self.risk_network = None
        self.adaptive_ode_steps = None

        self.planner_params = planner_params # including the action_len, future_len etc.
        self.action_num = (self.planner_params['future_len'] - self.planner_params['action_overlap']) // (self.planner_params['action_len'] - self.planner_params['action_overlap'])
        
        self.basic_loss = nn.MSELoss(reduction='none')

        # Goal Point Conditioning (legacy)
        self._goal_vocab = None
        self._goal_vocab_tensor = None
        if goal_vocab_path is not None:
            import numpy as np
            self._goal_vocab = np.load(goal_vocab_path).astype(np.float32)
            self._goal_vocab_tensor = torch.from_numpy(self._goal_vocab)

        # Trajectory Anchor Conditioning (Phase 1)
        # anchor_vocab_path points at a (K, T, 3) numpy array produced by
        # cluster_trajectories.py. Goal and anchor vocabs are mutually exclusive.
        self._anchor_vocab = None
        self._anchor_vocab_tensor = None
        if anchor_vocab_path is not None:
            if goal_vocab_path is not None:
                raise ValueError(
                    "goal_vocab_path and anchor_vocab_path are mutually exclusive; "
                    "set exactly one."
                )
            import numpy as np
            self._anchor_vocab = np.load(anchor_vocab_path).astype(np.float32)
            if self._anchor_vocab.ndim != 3 or self._anchor_vocab.shape[-1] != 3:
                raise ValueError(
                    f"anchor_vocab must be (K, T, 3); got {self._anchor_vocab.shape}"
                )
            self._anchor_vocab_tensor = torch.from_numpy(self._anchor_vocab)
        
    def prepare_model_input(self, cfg_flags, data: NuPlanDataSample, use_cfg, is_training):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == 'neighbors':
                neighbor_num = self.planner_params['neighbor_num']
                cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                mask_flags = cfg_flags.view(B, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                data.neighbor_past *= mask_flags
            elif cfg_type == 'lanes':
                data.lanes = data.lanes * cfg_flags.view(B, *([1] * (data.lanes.dim()-1)))

        else:
            if use_cfg:
                data = data.repeat(2)
                cfg_type = self.cfg_type
                if cfg_type == 'neighbors':
                    neighbor_num = self.planner_params['neighbor_num']
                    cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                    mask_flags = cfg_flags.view(B * 2, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    data.neighbor_past *= mask_flags
                elif cfg_type == 'lanes':
                    data.lanes = data.lanes * cfg_flags.view(B * 2, *([1] * (data.lanes.dim()-1)))
           
        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        )
            
        model_inputs.update({'cfg_flags': cfg_flags})
        
        return model_inputs, gt
        
    def extract_encoder_inputs(self, inputs):
        
        encoder_inputs = {
            'neighbors': inputs['neighbor_past'],
            'lanes': inputs['lanes'],
            'lanes_speed_limit': inputs['lanes_speedlimit'],
            'lanes_has_speed_limit': inputs['lanes_has_speedlimit'],
            'static': inputs['map_objects'],
            'routes': inputs['routes']
        }
        return encoder_inputs
    
    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(cfg_flags=inputs['cfg_flags'] if 'cfg_flags' in inputs.keys() else None,)
        model_extra.update(encoder_outputs)
        return model_extra
    
    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)
    
    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)
        
    def forward(self, data: NuPlanDataSample, mode='train', **params):
        if mode == 'train':
            return self.forward_train(data)
        elif mode == 'inference':
            return self.forward_inference(
                data, params.get('use_cfg', True), params.get('cfg_weight', None),
                num_candidates=params.get('num_candidates', 1),
                bon_seed=params.get('bon_seed', -1),
                return_all_candidates=params.get('return_all_candidates', False),
                goal_point=params.get('goal_point', None),
                anchor_traj=params.get('anchor_traj', None),
            )
    
    def _get_goal_for_gt(self, data: NuPlanDataSample):
        """从 GT 轨迹的 goal_frame 处查 vocabulary 中最近的 goal point, 返回 (B, 2) tensor."""
        if self._goal_vocab_tensor is None:
            return None
        from flow_planner.goal.goal_utils import find_nearest_goal_torch
        goal_frame = self.planner_params.get('goal_frame', 39)  # 39 = 4s@10Hz
        T = data.ego_future.shape[1]
        idx_t = (T - 1) if goal_frame < 0 else min(goal_frame, T - 1)
        raw_point = data.ego_future[:, idx_t, :2].float().to(self.device)  # (B, 2)
        vocab = self._goal_vocab_tensor.to(self.device)  # (K, 2)
        idx = find_nearest_goal_torch(raw_point, vocab)  # (B,)
        return vocab[idx]  # (B, 2)

    def _get_anchor_index_for_gt(self, data: NuPlanDataSample):
        """Return the nearest-anchor index (B,) for each GT future.

        Aligned to the **last** ``T_anchor`` frames of ``data.ego_future``,
        which matches what the model actually predicts (see
        ``ModelInputProcessor`` using ``ego_future[..., -future_len:, :3]``).
        This also tolerates raw ego_future being longer than ``T_anchor``
        without asserting, as long as it has at least ``T_anchor`` frames.

        Returns ``None`` if the anchor vocab is not configured.
        """
        if self._anchor_vocab_tensor is None:
            return None
        from flow_planner.goal.anchor_utils import find_nearest_anchor_torch

        vocab = self._anchor_vocab_tensor.to(self.device)  # (K, T_anchor, 3)
        _, T_anchor, _ = vocab.shape

        gt_future = data.ego_future  # (B, T_future, D); D >= 3
        if gt_future.shape[-1] < 3:
            raise ValueError(
                f"ego_future needs at least 3 channels (x, y, heading); got "
                f"{gt_future.shape[-1]}. The anchor path requires heading."
            )
        T_future = gt_future.shape[1]
        if T_future < T_anchor:
            raise ValueError(
                f"ego_future T={T_future} is shorter than anchor horizon T={T_anchor}. "
                f"Regenerate anchor_vocab.npy with --traj_len={T_future} or extend "
                "the dataset future window."
            )

        # Align to the LAST T_anchor frames (same slice the model actually predicts).
        gt_traj = gt_future[:, -T_anchor:, :3].float().to(self.device)  # (B, T_anchor, 3)
        return find_nearest_anchor_torch(gt_traj, vocab)                # (B,)

    def _get_anchor_for_gt(self, data: NuPlanDataSample):
        """Return the nearest anchor trajectory (B, T_anchor, 3) for each GT
        future in the batch, or ``None`` if the anchor vocab is not configured.
        """
        idx = self._get_anchor_index_for_gt(data)
        if idx is None:
            return None
        return self._anchor_vocab_tensor.to(idx.device)[idx]  # (B, T_anchor, 3)

    def forward_train(self, data: NuPlanDataSample):
        '''
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss
        
        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        '''
        B = data.ego_current.shape[0]

        # Conditioning: pick goal (legacy) or anchor (Phase 1) BEFORE normalization.
        # Exactly one branch is active (vocab constructors enforce exclusivity).
        goal_point = self._get_goal_for_gt(data)
        anchor_traj = self._get_anchor_for_gt(data)

        roll_dice = torch.rand((B, 1))
        cfg_flags = (roll_dice > self.cfg_prob).to(torch.int32).to(self.device) # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        model_inputs, gt = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True) # note that the cfg_flags are packed into the model_inputs
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)

        if goal_point is not None:
            decoder_model_extra['goal_point'] = goal_point
        if anchor_traj is not None:
            decoder_model_extra['anchor_traj'] = anchor_traj

        B, P, T_, D = gt.shape
        
        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type)
        noised_traj_tokens = traj_chunking(noised_traj, self.planner_params['action_len'], self.planner_params['action_overlap'])
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(target, self.planner_params['action_len'], self.planner_params['action_overlap'])
        target_tokens = torch.cat(target_tokens, dim=1)
        
        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
        
        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict['batch_loss'] = batch_loss
        
        loss = torch.sum(batch_loss, dim=-1) # (B, action_num, action_length, dim)
        loss_dict['ego_planning_loss'] = loss.mean()

        if self.planner_params['action_overlap'] > 0:
            consistency_loss = [torch.mean(torch.sum(self.basic_loss(prediction[:, i:i+1, -self.planner_params['action_overlap']:, :], prediction[:, i+1:i+2, :self.planner_params['action_overlap'], :]), dim=-1)) for i in range(0, prediction.shape[1]-2)]
            loss_dict['consistency_loss'] = sum(consistency_loss) / len(consistency_loss)
        else:
            loss_dict['consistency_loss'] = torch.tensor(0.0, device=loss.device)
        
        assert not torch.isnan(loss).sum(), f"loss is NaN"
        
        return prediction, loss_dict
    
    def forward_inference(self, data: NuPlanDataSample, use_cfg=True, cfg_weight=None,
                          num_candidates: int = 1, return_all_candidates: bool = False,
                          bon_seed: int = -1, goal_point=None, anchor_traj=None):
        """
        Forward inference with optional Best-of-N trajectory selection.

        Args:
            data: input data sample
            use_cfg: whether to use classifier-free guidance
            cfg_weight: CFG weight (None = use default)
            num_candidates: number of candidate trajectories to generate (Best-of-N)
            return_all_candidates: if True, return all N candidates instead of best one
            bon_seed: if >= 0, use deterministic seeds for candidate generation
                      (seed + i for candidate i). Set to -1 for random.
            goal_point: (B, 2) tensor — legacy goal conditioning. If None, no goal used.
            anchor_traj: (B, T, 3) tensor — trajectory anchor conditioning (Phase 1).
                          Mutually exclusive with goal_point.

        Returns:
            sample: (B, T, D) best trajectory, or (B, N, T, D) if return_all_candidates
        """
        B = data.ego_current.shape[0]
        _ode_steps_modified = False

        # ---- Adaptive ODE Steps (scene complexity) ----
        if self.adaptive_ode_steps is not None:
            n_neighbors = data.neighbor_past.shape[1] if data.neighbor_past is not None else 0
            # Simple complexity: more neighbors → more steps
            if n_neighbors <= 5:
                adaptive_steps = 2
            elif n_neighbors <= 15:
                adaptive_steps = 4
            else:
                adaptive_steps = 6
            self._original_steps = self.flow_ode.sample_params.get('sample_steps', 4)
            self.flow_ode.sample_params['sample_steps'] = adaptive_steps
            _ode_steps_modified = True

        # ---- Legacy Risk Network CFG (kept for backward compat) ----
        if cfg_weight is None and self.risk_network is not None:
            from flow_planner.risk.risk_features import extract_risk_features_from_sample
            import numpy as np
            risk_features = extract_risk_features_from_sample(
                ego_current=data.ego_current,
                ego_past=data.ego_past,
                neighbor_past=data.neighbor_past,
            )
            risk_features_tensor = torch.from_numpy(risk_features).float().to(self.device)
            if risk_features_tensor.ndim == 1:
                risk_features_tensor = risk_features_tensor.unsqueeze(0)
            with torch.no_grad():
                risk_output = self.risk_network(risk_features_tensor)
                cfg_weight = risk_output['w'].mean().item()

        # ---- Encoder (run once, shared across all candidates) ----
        if use_cfg:
            cfg_flags = torch.cat([torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)], dim=0).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)

        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)
        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)

        # ---- Conditioning injection (goal legacy OR anchor Phase 1) ----
        if goal_point is not None and anchor_traj is not None:
            raise ValueError(
                "goal_point and anchor_traj are mutually exclusive at inference."
            )
        if goal_point is not None:
            gp = goal_point.to(self.device).float()  # (B, 2)
            if use_cfg:
                decoder_model_extra['goal_point'] = torch.cat(
                    [gp, torch.zeros_like(gp)], dim=0
                )  # (2B, 2)
            else:
                decoder_model_extra['goal_point'] = gp
        if anchor_traj is not None:
            at = anchor_traj.to(self.device).float()  # (B, T, 3)
            if at.dim() != 3 or at.size(-1) != 3:
                raise ValueError(f"anchor_traj must be (B, T, 3); got {tuple(at.shape)}")
            if use_cfg:
                decoder_model_extra['anchor_traj'] = torch.cat(
                    [at, torch.zeros_like(at)], dim=0
                )  # (2B, T, 3)
            else:
                decoder_model_extra['anchor_traj'] = at

        # ---- Generate N candidate trajectories ----
        all_candidates = []
        for i in range(num_candidates):
            # Deterministic seed for reproducible candidate generation
            if bon_seed >= 0:
                torch.manual_seed(bon_seed + i)

            x_init = torch.randn(
                (B, self.action_num, self.planner_params['action_len'],
                 self.planner_params['state_dim']),
                device=self.device
            )
            sample = self.flow_ode.generate(
                x_init, self.decoder, self._model_type,
                use_cfg=use_cfg, cfg_weight=cfg_weight,
                **decoder_model_extra
            )
            sample = assemble_actions(
                sample, self.planner_params['future_len'],
                self.planner_params['action_len'],
                self.planner_params['action_overlap'],
                self.planner_params['state_dim'],
                self.assemble_method
            )
            sample = self.data_processor.state_postprocess(sample)
            all_candidates.append(sample)

        # Restore original ODE steps
        if _ode_steps_modified:
            self.flow_ode.sample_params['sample_steps'] = self._original_steps

        # ---- Single candidate: return directly ----
        if num_candidates == 1:
            return all_candidates[0]

        # ---- Best-of-N: score and select ----
        # Stack candidates: (N, B, T, D) → for each batch, pick best among N
        candidates = torch.stack(all_candidates, dim=0)  # (N, B, T, D)

        if return_all_candidates:
            # candidates: (N, B, ...) → (B, N, ...) regardless of ndim
            return torch.movedim(candidates, 0, 1)

        # Score each candidate using safety scorer
        from flow_planner.risk.trajectory_scorer import TrajectoryScorer
        scorer = TrajectoryScorer(verbose=(num_candidates > 1))

        best_trajs = []
        for b in range(B):
            # Get N candidates for this batch element
            batch_candidates = candidates[:, b, :, :]  # (N, T, D)

            # ---- Extrapolate neighbor future using constant-velocity model ----
            # neighbor_past shape: (B, M, T_p, 11) or (M, T_p, 11)
            # 维度: [x, y, cos_h, sin_h, vx, vy, width, length, type×3]
            neighbors_future = None
            if hasattr(data, 'neighbor_past') and data.neighbor_past is not None and data.neighbor_past.numel() > 0:
                if data.neighbor_past.dim() == 3:
                    nb_past = data.neighbor_past  # (M, T_p, D)
                else:
                    nb_past = data.neighbor_past[b]  # (M, T_p, D)

                T_ego = batch_candidates.shape[1]  # ego 轨迹的时间步数
                neighbors_future = TrajectoryScorer.extrapolate_neighbor_future(
                    nb_past, future_steps=T_ego, dt=0.1
                )

            # Route reference points for route consistency scoring
            # data.routes: (B, R, T_r, D_r) 路线车道点
            route = None
            if hasattr(data, 'routes') and data.routes is not None and data.routes.numel() > 0:
                route_data = data.routes[b]  # (R, T_r, D_r)
                route = route_data[:, :, :2].reshape(-1, 2)  # flatten to (N_points, 2)

            scores = scorer.score_trajectories(
                batch_candidates, neighbors=neighbors_future, route=route
            )
            best_idx = scores.argmax().item()
            best_trajs.append(batch_candidates[best_idx:best_idx+1])

        result = torch.cat(best_trajs, dim=0)  # (B, T, D)
        return result
    
    def load_risk_network(self, checkpoint_path: str):
        """加载训练好的 Risk Network 用于自适应 CFG 推理"""
        from flow_planner.risk.risk_network import load_risk_network, AdaptiveODESteps
        
        self.risk_network = load_risk_network(checkpoint_path, device=self.device)
        self.adaptive_ode_steps = AdaptiveODESteps(min_steps=2, max_steps=6)
        
        print(f"Risk Network loaded from {checkpoint_path}")
        print(f"  Parameters: {self.risk_network.num_parameters}")
        print(f"  w range: [{self.risk_network.w_min}, {self.risk_network.w_max}]")
        print(f"  Adaptive ODE steps: [{self.adaptive_ode_steps.min_steps}, {self.adaptive_ode_steps.max_steps}]")
    
    @property
    def model_type(self,):
        return self._model_type
    
    def get_optimizer_params(self):
        return [
            {'params': self.model_encoder.parameters()},
            {'params': self.model_decoder.parameters()}
        ]
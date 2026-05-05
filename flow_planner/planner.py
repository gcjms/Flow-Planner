import json
import os
import uuid
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Type
import hydra
from hydra.utils import instantiate
import omegaconf
from flow_planner.dpo.config_utils import load_composed_config

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from flow_planner.data.data_process.data_processor import DataProcessor
from flow_planner.data.dataset.nuplan import NuPlanDataSample

def identity(ego_state, predictions):
    return predictions


class FlowPlanner(AbstractPlanner):
    def __init__(
            self,
            config_path,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            use_cfg: bool = True,
            cfg_weight: float = 1.0,
            num_candidates: int = 1,
            bon_seed: int = -1,
            anchor_vocab_path: Optional[str] = None,
            anchor_mode: str = "none",
            anchor_predictor_ckpt: Optional[str] = None,
            candidate_selector_ckpt: Optional[str] = None,
            closed_loop_gate_ckpt: Optional[str] = None,
            closed_loop_gate_threshold: float = 0.5,
            anchor_top_k: int = 3,
            candidate_samples_per_anchor: int = 3,
            candidate_samples_per_anchor_list: Optional[str] = None,
            candidate_trace_path: Optional[str] = None,
            candidate_trace_training_payload: bool = False,
            candidate_intervention_manifest_path: Optional[str] = None,
            anchor_state_dim: int = 3,
            anchor_token_num: int = 4,
            anchor_attn_heads: int = 8,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        config = load_composed_config(config_path)
        valid_anchor_modes = {
            "none",
            "predicted_anchor",
            "predicted_anchor_rerank",
            "predicted_anchor_candidate_selector",
            "predicted_anchor_candidate_selector_hybrid",
            "predicted_anchor_candidate_selector_hybrid_gate",
            "predicted_anchor_candidate_selector_strict_gate",
            "predicted_anchor_candidate_selector_closed_loop_gate",
            "predicted_anchor_candidate_selector_intervention",
        }
        if anchor_mode not in valid_anchor_modes:
            raise ValueError(f"unsupported anchor_mode {anchor_mode!r}; expected one of {sorted(valid_anchor_modes)}")
        if anchor_vocab_path is not None:
            omegaconf.OmegaConf.update(config, "model.anchor_vocab_path", anchor_vocab_path, force_add=True)
            future_len = omegaconf.OmegaConf.select(config, "model.future_len")
            if future_len is None:
                raise ValueError("model.future_len missing; cannot configure anchor planner")
            dec_anchor_state = omegaconf.OmegaConf.select(config, "model.model_decoder.anchor_state_dim")
            if dec_anchor_state in (None, 0):
                omegaconf.OmegaConf.update(config, "model.model_decoder.goal_dim", 0, force_add=True)
                omegaconf.OmegaConf.update(config, "model.model_decoder.anchor_state_dim", anchor_state_dim, force_add=True)
                omegaconf.OmegaConf.update(config, "model.model_decoder.anchor_len", int(future_len), force_add=True)
                omegaconf.OmegaConf.update(config, "model.model_decoder.anchor_token_num", anchor_token_num, force_add=True)
                omegaconf.OmegaConf.update(config, "model.model_decoder.anchor_attn_heads", anchor_attn_heads, force_add=True)
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = instantiate(config.model)

        self.core = instantiate(config.core)

        self.data_processor = DataProcessor(None)

        self.use_cfg = use_cfg

        self.cfg_weight = cfg_weight
        
        self.num_candidates = num_candidates
        self.bon_seed = bon_seed
        self.anchor_vocab_path = anchor_vocab_path
        self.anchor_mode = anchor_mode
        self.anchor_predictor_ckpt = anchor_predictor_ckpt
        self.candidate_selector_ckpt = candidate_selector_ckpt
        self.closed_loop_gate_ckpt = closed_loop_gate_ckpt
        self.closed_loop_gate_threshold = float(closed_loop_gate_threshold)
        self.anchor_top_k = anchor_top_k
        self.candidate_samples_per_anchor = candidate_samples_per_anchor
        self.candidate_trace_path = candidate_trace_path
        self.candidate_trace_training_payload = bool(candidate_trace_training_payload)
        self.candidate_intervention_manifest_path = candidate_intervention_manifest_path
        self._candidate_interventions = self._load_candidate_interventions(
            candidate_intervention_manifest_path
        )
        self._planner_instance_id = f"{os.getpid()}-{id(self):x}-{uuid.uuid4().hex[:8]}"
        self.candidate_samples_per_anchor_list = None
        if candidate_samples_per_anchor_list:
            if isinstance(candidate_samples_per_anchor_list, (list, tuple)):
                self.candidate_samples_per_anchor_list = [int(value) for value in candidate_samples_per_anchor_list]
            else:
                raw_values = str(candidate_samples_per_anchor_list).strip()
                raw_values = raw_values.strip('[]')
                self.candidate_samples_per_anchor_list = [
                    int(value.strip())
                    for value in raw_values.split(",")
                    if value.strip()
                ]
        self.anchor_predictor = None
        self.candidate_selector = None
        self.closed_loop_gate = None

    @staticmethod
    def _load_candidate_interventions(path: Optional[str]) -> Dict[int, Dict[str, Any]]:
        """Load per-timestamp closed-loop intervention directives."""
        if not path:
            return {}
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_items = payload.get("interventions", payload.get("events", []))
        else:
            raw_items = payload
        if not isinstance(raw_items, list):
            raise ValueError(
                "candidate intervention manifest must be a list or contain an "
                "'interventions' list"
            )

        interventions: Dict[int, Dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError(f"invalid intervention entry: {item!r}")
            timestamp = item.get("iteration_time_us", item.get("time_us"))
            if timestamp is None:
                raise ValueError(f"intervention entry missing iteration_time_us: {item!r}")
            timestamp_int = int(timestamp)
            if timestamp_int in interventions:
                raise ValueError(f"duplicate intervention timestamp: {timestamp_int}")
            interventions[timestamp_int] = dict(item)
        return interventions
        
    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            ckpt = torch.load(self._ckpt_path, weights_only=False, map_location=self._device)

            # Support multiple checkpoint formats used across training/eval codepaths.
            state_dict = ckpt
            if isinstance(ckpt, dict):
                if self._ema_enabled and isinstance(ckpt.get("ema_state_dict"), dict):
                    state_dict = ckpt["ema_state_dict"]
                else:
                    for key in ("state_dict", "model_state_dict", "model"):
                        value = ckpt.get(key)
                        if isinstance(value, dict):
                            state_dict = value
                            break

            if not isinstance(state_dict, dict):
                raise TypeError(f"Unsupported checkpoint format: {type(state_dict)!r}")

            if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
                model_state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            else:
                model_state_dict = state_dict
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._planner.device = self._device

        if self.anchor_mode != "none":
            if self.anchor_vocab_path is None:
                raise ValueError("anchor_mode requires anchor_vocab_path")
            if self.anchor_predictor_ckpt is None:
                raise ValueError("anchor_mode requires anchor_predictor_ckpt")
            from flow_planner.dpo.eval_multidim_utils import load_anchor_predictor_model
            self.anchor_predictor = load_anchor_predictor_model(
                self._planner, self.anchor_predictor_ckpt, device=self._device
            )
            if self.anchor_mode in {
                "predicted_anchor_candidate_selector",
                "predicted_anchor_candidate_selector_hybrid",
                "predicted_anchor_candidate_selector_hybrid_gate",
                "predicted_anchor_candidate_selector_strict_gate",
                "predicted_anchor_candidate_selector_closed_loop_gate",
                "predicted_anchor_candidate_selector_intervention",
            }:
                if self.candidate_selector_ckpt is None:
                    raise ValueError("predicted_anchor_candidate_selector requires candidate_selector_ckpt")
                from flow_planner.dpo.eval_multidim_utils import load_candidate_selector_model
                self.candidate_selector = load_candidate_selector_model(
                    self._planner, self.candidate_selector_ckpt, device=self._device
                )
                if self.anchor_mode == "predicted_anchor_candidate_selector_closed_loop_gate":
                    if self.closed_loop_gate_ckpt is None:
                        raise ValueError("closed-loop gate mode requires closed_loop_gate_ckpt")
                    from flow_planner.dpo.eval_multidim_utils import load_closed_loop_gate_model
                    self.closed_loop_gate = load_closed_loop_gate_model(
                        self._planner, self.closed_loop_gate_ckpt, device=self._device
                    )
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        data = NuPlanDataSample(
            batched=(model_inputs['ego_current_state'].dim() > 1),
            ego_past=model_inputs['ego_agent_past'],
            ego_current=model_inputs['ego_current_state'],
            neighbor_past=model_inputs['neighbor_agents_past'],
            lanes=model_inputs['lanes'],
            lanes_speedlimit=model_inputs['lanes_speed_limit'],
            lanes_has_speedlimit=model_inputs['lanes_has_speed_limit'],
            routes=model_inputs['route_lanes'],
            routes_speedlimit=model_inputs['route_lanes_speed_limit'],
            routes_has_speedlimit=model_inputs['route_lanes_has_speed_limit'],
            map_objects=model_inputs['static_objects']
        )

        return data

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        predictions = outputs[0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states

    def _numpy_traj_to_planner_output(self, trajectory: np.ndarray) -> torch.Tensor:
        traj = torch.from_numpy(np.asarray(trajectory)).float().to(self._device)
        if traj.dim() == 2:
            traj = traj.unsqueeze(0).unsqueeze(0)
        elif traj.dim() == 3:
            traj = traj.unsqueeze(0)
        return traj

    def _baseline_planner_output(self, inputs: NuPlanDataSample) -> torch.Tensor:
        return self.core.inference(
            self._planner,
            inputs,
            use_cfg=self.use_cfg,
            cfg_weight=self.cfg_weight,
            num_candidates=self.num_candidates,
            bon_seed=self.bon_seed,
        )

    def _compute_anchor_planner_output(
        self,
        inputs: NuPlanDataSample,
        trace_context: Optional[Dict[str, object]] = None,
        forced_candidate: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.anchor_predictor is None:
            raise RuntimeError("anchor planner was not initialized; call initialize() first")

        from flow_planner.dpo.eval_multidim_utils import (
            infer_candidate_selector_trajectory,
            infer_reranked_anchor_trajectory,
            infer_single_trajectory,
        )

        if self.anchor_mode == "predicted_anchor":
            prediction = self.anchor_predictor.predict_topk(inputs, top_k=1)
            anchor_traj = prediction["anchor_trajs"][:, 0, :, :]
            trajectory = infer_single_trajectory(
                self._planner,
                inputs,
                use_cfg=self.use_cfg,
                cfg_weight=self.cfg_weight,
                bon_seed=self.bon_seed,
                anchor_traj=anchor_traj,
            )
        elif self.anchor_mode == "predicted_anchor_rerank":
            trajectory = infer_reranked_anchor_trajectory(
                self._planner,
                inputs,
                anchor_predictor=self.anchor_predictor,
                top_k=self.anchor_top_k,
                use_cfg=self.use_cfg,
                cfg_weight=self.cfg_weight,
                bon_seed=self.bon_seed,
            )
        elif self.anchor_mode in {
            "predicted_anchor_candidate_selector",
            "predicted_anchor_candidate_selector_hybrid",
            "predicted_anchor_candidate_selector_hybrid_gate",
            "predicted_anchor_candidate_selector_strict_gate",
            "predicted_anchor_candidate_selector_closed_loop_gate",
            "predicted_anchor_candidate_selector_intervention",
        }:
            if self.candidate_selector is None:
                raise RuntimeError("candidate selector was not initialized")
            is_intervention = self.anchor_mode == "predicted_anchor_candidate_selector_intervention"
            trajectory = infer_candidate_selector_trajectory(
                self._planner,
                inputs,
                anchor_predictor=self.anchor_predictor,
                candidate_selector=self.candidate_selector,
                closed_loop_gate=self.closed_loop_gate,
                closed_loop_gate_threshold=self.closed_loop_gate_threshold,
                top_k=self.anchor_top_k,
                samples_per_anchor=self.candidate_samples_per_anchor,
                sample_counts_per_anchor=self.candidate_samples_per_anchor_list,
                use_cfg=self.use_cfg,
                cfg_weight=self.cfg_weight,
                bon_seed=self.bon_seed,
                include_unconditioned_candidate=(
                    is_intervention
                    or self.anchor_mode
                    in {
                        "predicted_anchor_candidate_selector_hybrid",
                        "predicted_anchor_candidate_selector_hybrid_gate",
                        "predicted_anchor_candidate_selector_strict_gate",
                        "predicted_anchor_candidate_selector_closed_loop_gate",
                    }
                ),
                fallback_progress_guard=(
                    self.anchor_mode
                    in {
                        "predicted_anchor_candidate_selector_hybrid_gate",
                        "predicted_anchor_candidate_selector_strict_gate",
                    }
                ),
                fallback_strict_safety_guard=(
                    self.anchor_mode == "predicted_anchor_candidate_selector_strict_gate"
                ),
                forced_candidate=forced_candidate,
                trace_context=trace_context,
            )
        else:
            raise ValueError("unsupported anchor_mode %r" % (self.anchor_mode,))
        return self._numpy_traj_to_planner_output(trajectory)
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        iteration_time_us = int(getattr(current_input.iteration, "time_us", -1))
        forced_candidate = None
        if self.anchor_mode == "predicted_anchor_candidate_selector_intervention":
            forced_candidate = self._candidate_interventions.get(iteration_time_us)

        trace_context = None
        if self.candidate_trace_path and self.anchor_mode != "none":
            trace_context = {
                "path": self.candidate_trace_path,
                "planner_instance_id": self._planner_instance_id,
                "anchor_mode": self.anchor_mode,
                "iteration_index": int(getattr(current_input.iteration, "index", -1)),
                "iteration_time_us": iteration_time_us,
                "write_training_payload": self.candidate_trace_training_payload,
            }
            if forced_candidate is not None:
                trace_context["intervention"] = forced_candidate
            try:
                ego_state = current_input.history.ego_states[-1]
                trace_context.update(
                    {
                        "ego_x": float(ego_state.center.x),
                        "ego_y": float(ego_state.center.y),
                        "ego_heading": float(ego_state.center.heading),
                    }
                )
            except Exception:
                pass

        if self.anchor_mode == "none":
            outputs = self._baseline_planner_output(inputs)
        elif self.anchor_mode == "predicted_anchor_candidate_selector_intervention" and forced_candidate is None:
            outputs = self._baseline_planner_output(inputs)
        else:
            outputs = self._compute_anchor_planner_output(
                inputs,
                trace_context=trace_context,
                forced_candidate=forced_candidate,
            )

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    

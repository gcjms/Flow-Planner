from __future__ import annotations

import glob
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.dpo.config_utils import load_composed_config
from flow_planner.goal.anchor_predictor import AnchorPredictor
from flow_planner.goal.candidate_selector import CandidateSelector
from flow_planner.goal.anchor_utils import (
    find_nearest_anchor,
    find_topk_nearest_anchors,
    load_anchor_vocab,
    select_anchor_from_route,
)
from flow_planner.goal.goal_predictor import GoalPredictor
from flow_planner.goal.goal_utils import (
    find_nearest_goal,
    load_goal_vocab,
    select_goal_from_route,
)

logger = logging.getLogger(__name__)
_CANDIDATE_TRACE_LOCK = threading.Lock()


def _tensor_to_float_list(values: Optional[torch.Tensor]) -> Optional[List[float]]:
    if values is None:
        return None
    try:
        return [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
    except Exception:
        return None


def _write_candidate_selector_trace(
    trace_context: Optional[Dict[str, Any]],
    *,
    candidate_meta: List[Dict[str, Any]],
    logits: torch.Tensor,
    raw_best_idx: int,
    final_idx: int,
    gate_reasons: List[str],
    rule_scores: Optional[torch.Tensor] = None,
    collision_scores: Optional[torch.Tensor] = None,
    ttc_scores: Optional[torch.Tensor] = None,
    forced_candidate: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one selector/gate/intervention decision row for analysis."""
    if not trace_context:
        return
    trace_path = trace_context.get("path")
    if not trace_path:
        return
    try:
        path = Path(str(trace_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_meta = candidate_meta[raw_best_idx] if 0 <= raw_best_idx < len(candidate_meta) else {}
        final_meta = candidate_meta[final_idx] if 0 <= final_idx < len(candidate_meta) else {}
        logits_list = _tensor_to_float_list(logits)
        rule_list = _tensor_to_float_list(rule_scores)
        collision_list = _tensor_to_float_list(collision_scores)
        ttc_list = _tensor_to_float_list(ttc_scores)
        row: Dict[str, Any] = {
            key: value
            for key, value in trace_context.items()
            if key != "path"
        }
        row.update(
            {
                "pid": os.getpid(),
                "wall_time": time.time(),
                "num_candidates": len(candidate_meta),
                "raw_best_idx": int(raw_best_idx),
                "raw_best_type": raw_meta.get("type"),
                "raw_best_anchor_rank": raw_meta.get("anchor_rank"),
                "raw_best_sample_i": raw_meta.get("sample_i"),
                "final_idx": int(final_idx),
                "final_type": final_meta.get("type"),
                "final_anchor_rank": final_meta.get("anchor_rank"),
                "final_sample_i": final_meta.get("sample_i"),
                "fallback_triggered": bool(raw_best_idx != final_idx),
                "gate_reasons": gate_reasons,
                "forced_candidate": forced_candidate,
                "logits": logits_list,
                "raw_best_logit": logits_list[raw_best_idx] if logits_list and 0 <= raw_best_idx < len(logits_list) else None,
                "unconditioned_logit": logits_list[0] if logits_list else None,
                "raw_best_rule_score": rule_list[raw_best_idx] if rule_list and 0 <= raw_best_idx < len(rule_list) else None,
                "unconditioned_rule_score": rule_list[0] if rule_list else None,
                "raw_best_collision_score": collision_list[raw_best_idx] if collision_list and 0 <= raw_best_idx < len(collision_list) else None,
                "unconditioned_collision_score": collision_list[0] if collision_list else None,
                "raw_best_ttc_score": ttc_list[raw_best_idx] if ttc_list and 0 <= raw_best_idx < len(ttc_list) else None,
                "unconditioned_ttc_score": ttc_list[0] if ttc_list else None,
                "candidate_meta": candidate_meta,
            }
        )
        with _CANDIDATE_TRACE_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:  # pragma: no cover - trace must never break planning
        logger.warning("failed to write candidate selector trace: %s", exc)


def _resolve_forced_candidate_idx(
    candidate_meta: List[Dict[str, Any]],
    forced_candidate: Optional[Dict[str, Any]],
    logits: Optional[torch.Tensor] = None,
    raw_best_idx: Optional[int] = None,
) -> Optional[int]:
    if not forced_candidate:
        return None

    candidate_type = str(forced_candidate.get("type", "anchor"))
    if candidate_type == "baseline":
        candidate_type = "unconditioned"
    if candidate_type in {"raw_best", "raw_best_candidate"}:
        return raw_best_idx
    if candidate_type == "raw_best_anchor":
        if logits is None:
            return raw_best_idx
        flat_logits = logits.detach().cpu().reshape(-1)
        best_idx: Optional[int] = None
        best_logit: Optional[float] = None
        for idx, meta in enumerate(candidate_meta):
            if meta.get("type") != "anchor":
                continue
            value = float(flat_logits[idx].item())
            if best_logit is None or value > best_logit:
                best_idx = idx
                best_logit = value
        return best_idx
    anchor_rank = forced_candidate.get("anchor_rank")
    sample_i = forced_candidate.get("sample_i")
    for idx, meta in enumerate(candidate_meta):
        if meta.get("type") != candidate_type:
            continue
        if candidate_type == "unconditioned":
            return idx
        if anchor_rank is not None and int(meta.get("anchor_rank", -1)) != int(anchor_rank):
            continue
        if sample_i is not None and int(meta.get("sample_i", -1)) != int(sample_i):
            continue
        return idx
    if "candidate_idx" in forced_candidate and forced_candidate["candidate_idx"] is not None:
        return int(forced_candidate["candidate_idx"])
    return None


def _read_scene_manifest(scene_manifest: str) -> List[str]:
    manifest_path = Path(scene_manifest)
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Scene manifest JSON must be a list, got {type(payload)!r}")
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _write_scene_manifest(scene_manifest_out: str, scene_files: List[str], scene_dir: str) -> None:
    manifest_path = Path(scene_manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rel_paths = [os.path.relpath(scene_file, scene_dir) for scene_file in scene_files]
    if manifest_path.suffix.lower() == ".json":
        manifest_path.write_text(
            json.dumps(rel_paths, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        manifest_path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
    logger.info("Saved scene manifest to %s", manifest_path)


def resolve_scene_files(
    scene_dir: str,
    max_scenes: int = 500,
    scene_manifest: Optional[str] = None,
    manifest_seed: Optional[int] = None,
    scene_manifest_out: Optional[str] = None,
) -> List[str]:
    all_scene_files = sorted(glob.glob(os.path.join(scene_dir, "*.npz")))
    if scene_manifest:
        manifest_entries = _read_scene_manifest(scene_manifest)
        scene_files: List[str] = []
        missing_entries: List[str] = []
        for entry in manifest_entries:
            scene_file = entry if os.path.isabs(entry) else os.path.join(scene_dir, entry)
            if os.path.exists(scene_file):
                scene_files.append(scene_file)
            else:
                missing_entries.append(entry)
        if missing_entries:
            raise FileNotFoundError(
                f"{len(missing_entries)} scene manifest entries are missing. "
                f"First few: {missing_entries[:5]}"
            )
        if max_scenes:
            scene_files = scene_files[:max_scenes]
        logger.info("Loaded %d scenes from manifest %s", len(scene_files), scene_manifest)
        return scene_files

    scene_files = all_scene_files
    if max_scenes and len(scene_files) > max_scenes:
        if manifest_seed is None:
            scene_files = scene_files[:max_scenes]
        else:
            rng = np.random.default_rng(manifest_seed)
            selected_idx = np.sort(rng.choice(len(scene_files), size=max_scenes, replace=False))
            scene_files = [scene_files[int(idx)] for idx in selected_idx]

    if scene_manifest_out:
        _write_scene_manifest(scene_manifest_out, scene_files, scene_dir)

    return scene_files


def _ensure_device_available(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}' but CUDA is not available")


def _unwrap_state_dict(ckpt: Any) -> Dict[str, Any]:
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in (
            "ema_state_dict",
            "state_dict",
            "model_state_dict",
            "anchor_predictor_state_dict",
            "goal_predictor_state_dict",
            "model",
        ):
            value = ckpt.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state_dict)!r}")

    for prefix in ("module.", "model.", "goal_predictor.", "anchor_predictor."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def _extract_predictor_head_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize predictor checkpoints to a clean ``head.*`` state dict.

    ``train_goal_predictor.py`` / ``train_anchor_predictor.py`` save
    ``payload["model"] = predictor.state_dict()`` which contains both
    ``backbone.*`` and ``head.*`` keys. At eval time the planner backbone is
    supplied separately, so we only want the predictor head weights.
    """
    if any(k.startswith("head.") for k in state_dict):
        return {
            k[len("head."):]: v
            for k, v in state_dict.items()
            if k.startswith("head.")
        }
    return state_dict


def _extract_candidate_selector_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("scorer.") for k in state_dict):
        return {
            k[len("scorer."):]: v
            for k, v in state_dict.items()
            if k.startswith("scorer.")
        }
    return state_dict


def load_planner_model(
    config_path: str,
    ckpt_path: str,
    device: str = "cuda",
    goal_vocab_path: Optional[str] = None,
    anchor_vocab_path: Optional[str] = None,
):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _ensure_device_available(device)

    logger.info("Loading config from %s", config_path)
    cfg = load_composed_config(config_path)
    OmegaConf.update(
        cfg,
        "data.dataset.train.future_downsampling_method",
        "uniform",
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "data.dataset.train.predicted_neighbor_num",
        0,
        force_add=True,
    )
    if cfg.get("normalization_stats") is not None:
        OmegaConf.update(
            cfg,
            "normalization_stats",
            cfg.get("normalization_stats"),
            force_add=True,
        )
    if goal_vocab_path is not None and anchor_vocab_path is not None:
        raise ValueError(
            "goal_vocab_path and anchor_vocab_path are mutually exclusive."
        )
    if goal_vocab_path is not None:
        OmegaConf.update(cfg, "model.goal_vocab_path", goal_vocab_path, force_add=True)
    if anchor_vocab_path is not None:
        OmegaConf.update(cfg, "model.anchor_vocab_path", anchor_vocab_path, force_add=True)
        # Patch the decoder so the anchor_encoder / anchor_cross_attn modules
        # are actually instantiated. Otherwise decoder.anchor_state_dim defaults
        # to 0 and the anchor input is silently ignored at inference, which
        # would silently sabotage the oracle_anchor gate.
        future_len = OmegaConf.select(cfg, "model.future_len")
        if future_len is None:
            raise ValueError(
                "model.future_len missing from config; cannot derive anchor_len."
            )
        dec_anchor_state = OmegaConf.select(cfg, "model.model_decoder.anchor_state_dim")
        if dec_anchor_state in (None, 0):
            OmegaConf.update(cfg, "model.model_decoder.goal_dim", 0, force_add=True)
            OmegaConf.update(cfg, "model.model_decoder.anchor_state_dim", 3, force_add=True)
            OmegaConf.update(cfg, "model.model_decoder.anchor_len", int(future_len), force_add=True)
            if OmegaConf.select(cfg, "model.model_decoder.anchor_token_num") is None:
                OmegaConf.update(cfg, "model.model_decoder.anchor_token_num", 4, force_add=True)
            if OmegaConf.select(cfg, "model.model_decoder.anchor_attn_heads") is None:
                OmegaConf.update(cfg, "model.model_decoder.anchor_attn_heads", 8, force_add=True)

    model = instantiate(cfg.model)

    logger.info("Loading checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Planner missing %d keys: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Planner unexpected %d keys: %s", len(unexpected), unexpected[:5])

    model = model.to(device).eval()
    model.device = device
    return model


def load_goal_predictor_model(
    planner_model,
    ckpt_path: str,
    device: str = "cuda",
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> GoalPredictor:
    _ensure_device_available(device)

    predictor = GoalPredictor(
        planner_backbone=planner_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_predictor_head_state_dict(_unwrap_state_dict(ckpt))
    missing, unexpected = predictor.head.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning("Goal predictor missing %d keys: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning(
            "Goal predictor unexpected %d keys: %s", len(unexpected), unexpected[:5]
        )

    predictor = predictor.to(device).eval()
    predictor.backbone.device = device
    return predictor


def resolve_goal_vocab(model, goal_vocab_path: Optional[str] = None) -> np.ndarray:
    if goal_vocab_path is not None:
        return load_goal_vocab(goal_vocab_path)
    if getattr(model, "_goal_vocab", None) is not None:
        return np.asarray(model._goal_vocab, dtype=np.float32)
    raise ValueError(
        "Goal vocabulary is required for route_goal/predicted_goal. "
        "Pass --goal_vocab_path or use a config with model.goal_vocab_path."
    )


def load_anchor_predictor_model(
    planner_model,
    ckpt_path: str,
    device: str = "cuda",
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> AnchorPredictor:
    """Instantiate an AnchorPredictor on top of ``planner_model`` and load ``ckpt_path``."""
    _ensure_device_available(device)

    predictor = AnchorPredictor(
        planner_backbone=planner_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_predictor_head_state_dict(_unwrap_state_dict(ckpt))
    missing, unexpected = predictor.head.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning("Anchor predictor missing %d keys: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning(
            "Anchor predictor unexpected %d keys: %s", len(unexpected), unexpected[:5]
        )

    predictor = predictor.to(device).eval()
    predictor.backbone.device = device
    return predictor


def load_candidate_selector_model(
    planner_model,
    ckpt_path: str,
    device: str = "cuda",
) -> CandidateSelector:
    """Instantiate a CandidateSelector and load its scorer weights."""
    _ensure_device_available(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = int(args.get("hidden_dim", 256))
    dropout = float(args.get("dropout", 0.1))

    selector = CandidateSelector(
        planner_backbone=planner_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=True,
    )
    state_dict = _extract_candidate_selector_state_dict(_unwrap_state_dict(ckpt))
    missing, unexpected = selector.scorer.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning("Candidate selector missing %d keys: %s", len(missing), missing[:5])
    if unexpected:
        logger.warning(
            "Candidate selector unexpected %d keys: %s", len(unexpected), unexpected[:5]
        )

    selector = selector.to(device).eval()
    selector.backbone.device = device
    return selector


def resolve_anchor_vocab(model, anchor_vocab_path: Optional[str] = None) -> np.ndarray:
    if anchor_vocab_path is not None:
        return load_anchor_vocab(anchor_vocab_path)
    if getattr(model, "_anchor_vocab", None) is not None:
        return np.asarray(model._anchor_vocab, dtype=np.float32)
    raise ValueError(
        "Anchor vocabulary is required for route_anchor/predicted_anchor/"
        "predicted_anchor_rerank/oracle_anchor/oracle_anchor_rerank. "
        "Pass --anchor_vocab_path or use a config with model.anchor_vocab_path."
    )


def _require_scene_key(scene_data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in scene_data:
        raise KeyError(f"Scene NPZ is missing required key '{key}'")
    return np.asarray(scene_data[key])


def _optional_scene_key(
    scene_data: Dict[str, np.ndarray],
    key: str,
    default_shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if key in scene_data:
        return np.asarray(scene_data[key])
    return np.zeros(default_shape, dtype=dtype)


def scene_to_datasample(
    scene_data: Dict[str, np.ndarray],
    device: str = "cuda",
    neighbor_limit: Optional[int] = None,
) -> NuPlanDataSample:
    ego_past = _require_scene_key(scene_data, "ego_agent_past")
    ego_current = _require_scene_key(scene_data, "ego_current_state")
    ego_future = np.asarray(scene_data.get("ego_agent_future", np.zeros((0, 3), dtype=np.float32)))

    neighbor_past = _require_scene_key(scene_data, "neighbor_agents_past")
    neighbor_future_obs = np.asarray(
        scene_data.get("neighbor_agents_future", np.zeros((0, 0, 3), dtype=np.float32))
    )
    if neighbor_limit is not None:
        neighbor_past = neighbor_past[:neighbor_limit]
        neighbor_future_obs = neighbor_future_obs[:neighbor_limit]

    lanes = _require_scene_key(scene_data, "lanes")
    routes = _require_scene_key(scene_data, "route_lanes")
    map_objects = _require_scene_key(scene_data, "static_objects")

    lanes_speed = _optional_scene_key(
        scene_data,
        "lanes_speed_limit",
        default_shape=(lanes.shape[0], 1),
        dtype=np.float32,
    )
    lanes_has_speed = _optional_scene_key(
        scene_data,
        "lanes_has_speed_limit",
        default_shape=(lanes.shape[0], 1),
        dtype=bool,
    )
    routes_speed = _optional_scene_key(
        scene_data,
        "route_lanes_speed_limit",
        default_shape=(routes.shape[0], 1),
        dtype=np.float32,
    )
    routes_has_speed = _optional_scene_key(
        scene_data,
        "route_lanes_has_speed_limit",
        default_shape=(routes.shape[0], 1),
        dtype=bool,
    )

    future_steps = 0
    if neighbor_future_obs.ndim >= 2:
        future_steps = int(neighbor_future_obs.shape[1])
    elif ego_future.ndim >= 1:
        future_steps = int(ego_future.shape[0])

    empty_neighbor_future = torch.zeros(
        1,
        0,
        max(future_steps, 1),
        3,
        device=device,
        dtype=torch.float32,
    )

    return NuPlanDataSample(
        batched=True,
        ego_past=torch.from_numpy(ego_past).float().unsqueeze(0).to(device),
        ego_current=torch.from_numpy(ego_current).float().unsqueeze(0).to(device),
        ego_future=torch.from_numpy(ego_future).float().unsqueeze(0).to(device),
        neighbor_past=torch.from_numpy(neighbor_past).float().unsqueeze(0).to(device),
        neighbor_future=empty_neighbor_future,
        neighbor_future_observed=torch.from_numpy(neighbor_future_obs)
        .float()
        .unsqueeze(0)
        .to(device),
        lanes=torch.from_numpy(lanes).float().unsqueeze(0).to(device),
        lanes_speedlimit=torch.from_numpy(lanes_speed).float().unsqueeze(0).to(device),
        lanes_has_speedlimit=torch.from_numpy(lanes_has_speed)
        .bool()
        .unsqueeze(0)
        .to(device),
        routes=torch.from_numpy(routes).float().unsqueeze(0).to(device),
        routes_speedlimit=torch.from_numpy(routes_speed).float().unsqueeze(0).to(device),
        routes_has_speedlimit=torch.from_numpy(routes_has_speed)
        .bool()
        .unsqueeze(0)
        .to(device),
        map_objects=torch.from_numpy(map_objects).float().unsqueeze(0).to(device),
    )


def choose_goal_point(
    goal_mode: str,
    scene_data: Dict[str, np.ndarray],
    data: NuPlanDataSample,
    device: str,
    goal_vocab: Optional[np.ndarray] = None,
    goal_predictor: Optional[GoalPredictor] = None,
) -> Optional[torch.Tensor]:
    if goal_mode == "none":
        return None

    if goal_mode == "route_goal":
        if goal_vocab is None:
            raise ValueError("route_goal requires a goal vocabulary")
        goal = select_goal_from_route(scene_data["route_lanes"], goal_vocab)
        return torch.from_numpy(np.asarray(goal, dtype=np.float32)).unsqueeze(0).to(device)

    if goal_mode == "predicted_goal":
        if goal_predictor is None:
            raise ValueError("predicted_goal requires a goal predictor checkpoint")
        prediction = goal_predictor.predict_topk(data, top_k=1)
        goal_points = prediction["goal_points"]
        if goal_points.ndim != 3 or goal_points.shape[1] == 0:
            raise RuntimeError("Goal predictor did not return a usable top-1 goal")
        return goal_points[:, 0, :].to(device)

    if goal_mode == "oracle_goal":
        # Oracle (cheating) mode: snap the GT future endpoint to its nearest
        # goal-vocabulary cluster. This reproduces the `gt_nearest` setting in
        # docs/dpo_自动驾驶综述.md Ch 4.3 and provides the upper bound on
        # decoder performance when the "right" goal is supplied. Not deployable.
        if goal_vocab is None:
            raise ValueError("oracle_goal requires a goal vocabulary")
        ego_future = scene_data.get("ego_agent_future")
        if ego_future is None:
            raise RuntimeError(
                "oracle_goal requires scene NPZ to contain 'ego_agent_future'"
            )
        ego_future_arr = np.asarray(ego_future, dtype=np.float32)
        if ego_future_arr.ndim < 2 or ego_future_arr.shape[0] == 0:
            raise RuntimeError(
                f"oracle_goal: 'ego_agent_future' has invalid shape {ego_future_arr.shape}"
            )
        gt_endpoint = ego_future_arr[-1, :2]
        vocab_arr = np.asarray(goal_vocab, dtype=np.float32)
        nearest_idx = int(find_nearest_goal(gt_endpoint[None, :], vocab_arr)[0])
        nearest_goal = vocab_arr[nearest_idx]
        return torch.from_numpy(nearest_goal).unsqueeze(0).to(device)

    raise ValueError(f"Unsupported goal_mode '{goal_mode}'")


def choose_anchor(
    anchor_mode: str,
    scene_data: Dict[str, np.ndarray],
    data: NuPlanDataSample,
    device: str,
    anchor_vocab: Optional[np.ndarray] = None,
    anchor_predictor: Optional[AnchorPredictor] = None,
) -> Optional[torch.Tensor]:
    """Select a trajectory anchor (B=1, T, 3) to feed the decoder.

    Modes mirror ``choose_goal_point`` but return full anchor templates:
      - "none"              → no conditioning
      - "route_anchor"      → pure geometry, pick anchor best matching the route polyline
      - "predicted_anchor"  → use trained AnchorPredictor's top-1 output
      - "oracle_anchor"     → cheat: snap GT future to its nearest vocab anchor
    """
    if anchor_mode == "none":
        return None

    if anchor_mode == "route_anchor":
        if anchor_vocab is None:
            raise ValueError("route_anchor requires an anchor vocabulary")
        anchor = select_anchor_from_route(scene_data["route_lanes"], anchor_vocab)
        return torch.from_numpy(np.asarray(anchor, dtype=np.float32)).unsqueeze(0).to(device)

    if anchor_mode == "predicted_anchor":
        if anchor_predictor is None:
            raise ValueError("predicted_anchor requires an anchor predictor checkpoint")
        prediction = anchor_predictor.predict_topk(data, top_k=1)
        anchor_trajs = prediction["anchor_trajs"]  # (B, k, T, 3)
        if anchor_trajs.ndim != 4 or anchor_trajs.shape[1] == 0:
            raise RuntimeError("Anchor predictor did not return a usable top-1 anchor")
        return anchor_trajs[:, 0, :, :].to(device)  # (B, T, 3)

    if anchor_mode == "oracle_anchor":
        # Oracle (cheating): snap the GT future trajectory to its nearest vocab anchor.
        # Provides the upper bound on decoder performance when the "right" anchor is
        # supplied. Not deployable — requires access to ego_agent_future.
        if anchor_vocab is None:
            raise ValueError("oracle_anchor requires an anchor vocabulary")
        ego_future = scene_data.get("ego_agent_future")
        if ego_future is None:
            raise RuntimeError(
                "oracle_anchor requires scene NPZ to contain 'ego_agent_future'"
            )
        ego_future_arr = np.asarray(ego_future, dtype=np.float32)
        T = anchor_vocab.shape[1]
        if ego_future_arr.ndim < 2 or ego_future_arr.shape[0] < T or ego_future_arr.shape[-1] < 3:
            raise RuntimeError(
                f"oracle_anchor: ego_agent_future shape {ego_future_arr.shape} "
                f"incompatible with anchor vocab horizon T={T} (needs (>=T, >=3))"
            )
        gt_traj = ego_future_arr[-T:, :3]
        nearest_idx = int(find_nearest_anchor(gt_traj, anchor_vocab))
        nearest_anchor = anchor_vocab[nearest_idx]   # (T, 3)
        return torch.from_numpy(nearest_anchor).unsqueeze(0).to(device)

    raise ValueError(f"Unsupported anchor_mode '{anchor_mode}'")


def normalize_prediction(prediction: torch.Tensor) -> np.ndarray:
    result = prediction.detach().cpu().numpy()
    if result.ndim >= 1 and result.shape[0] == 1:
        result = result[0]
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    return result


def infer_single_trajectory(
    model,
    data: NuPlanDataSample,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    goal_point: Optional[torch.Tensor] = None,
    anchor_traj: Optional[torch.Tensor] = None,
) -> np.ndarray:
    with torch.no_grad():
        prediction = model(
            data,
            mode="inference",
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
            num_candidates=1,
            return_all_candidates=False,
            bon_seed=bon_seed,
            goal_point=goal_point,
            anchor_traj=anchor_traj,
        )
    return normalize_prediction(prediction)


def _build_rerank_context(
    data: NuPlanDataSample,
    future_steps: int,
):
    """Build lightweight online-available context for trajectory reranking.

    We intentionally avoid GT future here: use neighbor past -> constant-velocity
    extrapolation plus route polyline, matching the deployment-style heuristic used
    by ``FlowPlanner.forward_inference(num_candidates>1)``.
    """
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    neighbors_future = None
    if hasattr(data, "neighbor_past") and data.neighbor_past is not None and data.neighbor_past.numel() > 0:
        neighbor_past = data.neighbor_past[0] if data.neighbor_past.dim() == 4 else data.neighbor_past
        neighbors_future = TrajectoryScorer.extrapolate_neighbor_future(
            neighbor_past,
            future_steps=future_steps,
            dt=0.1,
        )
        if neighbors_future is not None:
            neighbors_future = neighbors_future.cpu()

    route = None
    if hasattr(data, "routes") and data.routes is not None and data.routes.numel() > 0:
        route_data = data.routes[0] if data.routes.dim() == 4 else data.routes
        route = route_data[:, :, :2].reshape(-1, 2).cpu()

    return neighbors_future, route


def _get_oracle_topk_anchor_trajs(
    scene_data: Dict[str, np.ndarray],
    anchor_vocab: np.ndarray,
    top_k: int,
    device: str,
) -> torch.Tensor:
    """Return GT-nearest top-k anchor templates as ``(1, k, T, 3)``."""
    ego_future = scene_data.get("ego_agent_future")
    if ego_future is None:
        raise RuntimeError(
            "oracle_anchor_rerank requires scene NPZ to contain 'ego_agent_future'"
        )

    ego_future_arr = np.asarray(ego_future, dtype=np.float32)
    T = anchor_vocab.shape[1]
    if ego_future_arr.ndim < 2 or ego_future_arr.shape[0] < T or ego_future_arr.shape[-1] < 3:
        raise RuntimeError(
            f"oracle_anchor_rerank: ego_agent_future shape {ego_future_arr.shape} "
            f"incompatible with anchor vocab horizon T={T} (needs (>=T, >=3))"
        )

    gt_traj = ego_future_arr[-T:, :3]
    topk_idx = np.asarray(
        find_topk_nearest_anchors(gt_traj, anchor_vocab, top_k=top_k),
        dtype=np.int64,
    )
    return torch.from_numpy(anchor_vocab[topk_idx]).unsqueeze(0).to(device)


def _infer_reranked_anchor_trajectory_from_candidates(
    model,
    data: NuPlanDataSample,
    anchor_trajs: torch.Tensor,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    collision_weight: float = 40.0,
    ttc_weight: float = 20.0,
    route_weight: float = 25.0,
    comfort_weight: float = 10.0,
    progress_weight: float = 0.0,
    collision_dist: float = 2.0,
) -> np.ndarray:
    """Generate one trajectory per candidate anchor, then rerank them."""
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    if anchor_trajs.ndim != 4 or anchor_trajs.shape[0] == 0 or anchor_trajs.shape[1] == 0:
        raise RuntimeError("Anchor rerank requires a non-empty candidate set shaped (B, k, T, 3)")

    candidate_trajs = []
    anchor_device = getattr(model, "device", data.ego_current.device)
    for k_idx in range(anchor_trajs.shape[1]):
        candidate_anchor = anchor_trajs[:, k_idx, :, :].to(anchor_device)
        candidate_traj = infer_single_trajectory(
            model,
            data,
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
            bon_seed=bon_seed,
            anchor_traj=candidate_anchor,
        )
        candidate_trajs.append(candidate_traj)

    traj_tensor = torch.from_numpy(np.stack(candidate_trajs, axis=0)).float()
    neighbors_future, route = _build_rerank_context(
        data,
        future_steps=traj_tensor.shape[1],
    )
    scorer = TrajectoryScorer(
        collision_weight=collision_weight,
        ttc_weight=ttc_weight,
        route_weight=route_weight,
        comfort_weight=comfort_weight,
        progress_weight=progress_weight,
        collision_threshold=collision_dist,
        ttc_threshold=3.0,
        dt=0.1,
        verbose=False,
    )
    scores = scorer.score_trajectories(
        traj_tensor,
        neighbors=neighbors_future,
        route=route,
    )
    best_idx = int(scores.argmax().item())
    return candidate_trajs[best_idx]


def _infer_candidate_selected_trajectory_from_candidates(
    model,
    data: NuPlanDataSample,
    anchor_trajs: torch.Tensor,
    candidate_selector: CandidateSelector,
    samples_per_anchor: int = 3,
    sample_counts_per_anchor: Optional[List[int]] = None,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    include_unconditioned_candidate: bool = False,
    fallback_progress_guard: bool = False,
    fallback_strict_safety_guard: bool = False,
    forced_candidate: Optional[Dict[str, Any]] = None,
    trace_context: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Generate trajectories and let a learned selector or intervention pick one."""
    if anchor_trajs.ndim != 4 or anchor_trajs.shape[0] == 0 or anchor_trajs.shape[1] == 0:
        raise RuntimeError(
            "Candidate selector requires a non-empty anchor candidate set shaped (B, k, T, 3)"
        )
    num_anchor_candidates = int(anchor_trajs.shape[1])
    if sample_counts_per_anchor is None:
        if samples_per_anchor <= 0:
            raise ValueError("samples_per_anchor must be positive")
        samples_by_anchor = [int(samples_per_anchor)] * num_anchor_candidates
    else:
        samples_by_anchor = [int(value) for value in sample_counts_per_anchor]
        if len(samples_by_anchor) != num_anchor_candidates:
            raise ValueError(
                "sample_counts_per_anchor length must match anchor candidate count"
            )
        if any(value <= 0 for value in samples_by_anchor):
            raise ValueError("sample_counts_per_anchor values must be positive")

    candidate_trajs: List[np.ndarray] = []
    repeated_anchors: List[np.ndarray] = []
    candidate_meta: List[Dict[str, Any]] = []
    anchor_device = getattr(model, "device", data.ego_current.device)
    if include_unconditioned_candidate:
        fallback_traj = infer_single_trajectory(
            model,
            data,
            use_cfg=use_cfg,
            cfg_weight=cfg_weight,
            bon_seed=bon_seed,
            anchor_traj=None,
        )
        candidate_trajs.append(fallback_traj.astype(np.float32))
        repeated_anchors.append(
            np.zeros_like(anchor_trajs[0, 0].detach().cpu().numpy(), dtype=np.float32)
        )
        candidate_meta.append(
            {
                "candidate_idx": len(candidate_meta),
                "type": "unconditioned",
                "anchor_rank": None,
                "sample_i": None,
            }
        )
    for anchor_rank in range(anchor_trajs.shape[1]):
        candidate_anchor = anchor_trajs[:, anchor_rank, :, :].to(anchor_device)
        for sample_i in range(samples_by_anchor[anchor_rank]):
            sample_seed = bon_seed
            if bon_seed >= 0:
                sample_seed = bon_seed + anchor_rank * 100 + sample_i
            candidate_traj = infer_single_trajectory(
                model,
                data,
                use_cfg=use_cfg,
                cfg_weight=cfg_weight,
                bon_seed=sample_seed,
                anchor_traj=candidate_anchor,
            )
            candidate_trajs.append(candidate_traj.astype(np.float32))
            repeated_anchors.append(
                candidate_anchor[0].detach().cpu().numpy().astype(np.float32)
            )
            candidate_meta.append(
                {
                    "candidate_idx": len(candidate_meta),
                    "type": "anchor",
                    "anchor_rank": int(anchor_rank),
                    "anchor_rank_1based": int(anchor_rank) + 1,
                    "sample_i": int(sample_i),
                }
            )

    candidate_tensor = torch.from_numpy(np.stack(candidate_trajs, axis=0)).unsqueeze(0).float().to(anchor_device)
    anchor_tensor = torch.from_numpy(np.stack(repeated_anchors, axis=0)).unsqueeze(0).float().to(anchor_device)
    candidate_mask = torch.ones(
        (1, candidate_tensor.shape[1]),
        dtype=torch.bool,
        device=anchor_device,
    )
    with torch.no_grad():
        logits = candidate_selector(
            data,
            candidate_tensor,
            anchor_tensor,
            candidate_mask=candidate_mask,
        )
    raw_best_idx = int(logits[0].argmax().item())
    best_idx = raw_best_idx
    gate_reasons: List[str] = []
    rule_scores_trace: Optional[torch.Tensor] = None
    collision_scores_trace: Optional[torch.Tensor] = None
    ttc_scores_trace: Optional[torch.Tensor] = None

    forced_idx = _resolve_forced_candidate_idx(
        candidate_meta,
        forced_candidate,
        logits=logits,
        raw_best_idx=raw_best_idx,
    )
    if forced_candidate is not None:
        if forced_idx is None or not (0 <= forced_idx < len(candidate_trajs)):
            raise ValueError(f"forced candidate not found: {forced_candidate}")
        best_idx = int(forced_idx)
        gate_reasons.append("forced_candidate")
    elif include_unconditioned_candidate and fallback_progress_guard and best_idx != 0:
        fallback = candidate_trajs[0]
        selected = candidate_trajs[best_idx]
        fallback_final_x = float(fallback[-1, 0])
        selected_final_x = float(selected[-1, 0])
        fallback_path = float(np.linalg.norm(np.diff(fallback[:, :2], axis=0), axis=1).sum())
        selected_path = float(np.linalg.norm(np.diff(selected[:, :2], axis=0), axis=1).sum())
        lateral_delta = float(abs(selected[-1, 1] - fallback[-1, 1]))
        should_fallback = False
        if selected_final_x < fallback_final_x - 2.0:
            should_fallback = True
            gate_reasons.append("selected_final_x_lt_fallback_minus_2m")
        if selected_path < 0.75 * max(fallback_path, 1e-3):
            should_fallback = True
            gate_reasons.append("selected_path_lt_0p75_fallback")
        if lateral_delta > 3.0 and selected_final_x <= fallback_final_x + 1.0:
            should_fallback = True
            gate_reasons.append("large_lateral_delta_without_progress")
        if not should_fallback:
            from flow_planner.risk.trajectory_scorer import TrajectoryScorer

            neighbors_future, route = _build_rerank_context(
                data,
                future_steps=candidate_tensor.shape[2],
            )
            scorer = TrajectoryScorer(
                collision_weight=40.0,
                ttc_weight=20.0,
                route_weight=25.0,
                comfort_weight=10.0,
                progress_weight=5.0,
                collision_threshold=2.0,
                ttc_threshold=3.0,
                dt=0.1,
                verbose=False,
            )
            candidate_cpu = candidate_tensor[0].detach().cpu()
            rule_scores = scorer.score_trajectories(
                candidate_cpu,
                neighbors=neighbors_future,
                route=route,
            )
            rule_scores_trace = rule_scores.detach().cpu()
            if float(rule_scores[best_idx].item()) < float(rule_scores[0].item()):
                should_fallback = True
                gate_reasons.append("rule_score_lt_unconditioned")
            if fallback_strict_safety_guard:
                collision_scores = scorer._collision_score(candidate_cpu, neighbors_future)
                ttc_scores = scorer._ttc_score(candidate_cpu, neighbors_future)
                collision_scores_trace = collision_scores.detach().cpu()
                ttc_scores_trace = ttc_scores.detach().cpu()
                if float(collision_scores[best_idx].item()) < float(collision_scores[0].item()):
                    should_fallback = True
                    gate_reasons.append("collision_score_lt_unconditioned")
                if float(ttc_scores[best_idx].item()) < float(ttc_scores[0].item()):
                    should_fallback = True
                    gate_reasons.append("ttc_score_lt_unconditioned")
                if float(rule_scores[best_idx].item()) < float(rule_scores[0].item()) + 1.0:
                    should_fallback = True
                    gate_reasons.append("rule_score_margin_lt_1p0")
        if should_fallback:
            best_idx = 0

    _write_candidate_selector_trace(
        trace_context,
        candidate_meta=candidate_meta,
        logits=logits.detach().cpu(),
        raw_best_idx=raw_best_idx,
        final_idx=best_idx,
        gate_reasons=gate_reasons,
        rule_scores=rule_scores_trace,
        collision_scores=collision_scores_trace,
        ttc_scores=ttc_scores_trace,
        forced_candidate=forced_candidate,
    )
    return candidate_trajs[best_idx]


def infer_reranked_anchor_trajectory(
    model,
    data: NuPlanDataSample,
    anchor_predictor: AnchorPredictor,
    top_k: int = 3,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    collision_weight: float = 40.0,
    ttc_weight: float = 20.0,
    route_weight: float = 25.0,
    comfort_weight: float = 10.0,
    progress_weight: float = 0.0,
    collision_dist: float = 2.0,
) -> np.ndarray:
    """Top-k anchor ablation for Phase 1:

    1. Predictor proposes k anchor templates.
    2. Planner generates one trajectory for each anchor.
    3. TrajectoryScorer reranks those trajectories using route/safety heuristics.

    This directly tests whether the current failure is mostly "candidate ranking"
    rather than "candidate generation".
    """
    prediction = anchor_predictor.predict_topk(data, top_k=top_k)
    anchor_trajs = prediction["anchor_trajs"]  # (B, k, T, 3)
    if anchor_trajs.ndim != 4 or anchor_trajs.shape[0] == 0 or anchor_trajs.shape[1] == 0:
        raise RuntimeError("Anchor predictor did not return usable top-k anchors")
    return _infer_reranked_anchor_trajectory_from_candidates(
        model,
        data,
        anchor_trajs=anchor_trajs,
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
        bon_seed=bon_seed,
        collision_weight=collision_weight,
        ttc_weight=ttc_weight,
        route_weight=route_weight,
        comfort_weight=comfort_weight,
        progress_weight=progress_weight,
        collision_dist=collision_dist,
    )


def infer_candidate_selector_trajectory(
    model,
    data: NuPlanDataSample,
    anchor_predictor: AnchorPredictor,
    candidate_selector: CandidateSelector,
    top_k: int = 3,
    samples_per_anchor: int = 3,
    sample_counts_per_anchor: Optional[List[int]] = None,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    include_unconditioned_candidate: bool = False,
    fallback_progress_guard: bool = False,
    fallback_strict_safety_guard: bool = False,
    forced_candidate: Optional[Dict[str, Any]] = None,
    trace_context: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Use learned scene+anchor+trajectory scoring to pick from top-k candidates."""
    prediction = anchor_predictor.predict_topk(data, top_k=top_k)
    anchor_trajs = prediction["anchor_trajs"]
    if anchor_trajs.ndim != 4 or anchor_trajs.shape[0] == 0 or anchor_trajs.shape[1] == 0:
        raise RuntimeError("Anchor predictor did not return usable top-k anchors")
    return _infer_candidate_selected_trajectory_from_candidates(
        model,
        data,
        anchor_trajs=anchor_trajs,
        candidate_selector=candidate_selector,
        samples_per_anchor=samples_per_anchor,
        sample_counts_per_anchor=sample_counts_per_anchor,
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
        bon_seed=bon_seed,
        include_unconditioned_candidate=include_unconditioned_candidate,
        fallback_progress_guard=fallback_progress_guard,
        fallback_strict_safety_guard=fallback_strict_safety_guard,
        forced_candidate=forced_candidate,
        trace_context=trace_context,
    )


def infer_oracle_reranked_anchor_trajectory(
    model,
    data: NuPlanDataSample,
    scene_data: Dict[str, np.ndarray],
    anchor_vocab: np.ndarray,
    top_k: int = 3,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    collision_weight: float = 40.0,
    ttc_weight: float = 20.0,
    route_weight: float = 25.0,
    comfort_weight: float = 10.0,
    progress_weight: float = 0.0,
    collision_dist: float = 2.0,
) -> np.ndarray:
    """Oracle rerank ablation: use GT-nearest top-k anchors as the candidate pool."""
    if anchor_vocab is None:
        raise ValueError("oracle_anchor_rerank requires an anchor vocabulary")

    anchor_trajs = _get_oracle_topk_anchor_trajs(
        scene_data=scene_data,
        anchor_vocab=anchor_vocab,
        top_k=top_k,
        device=getattr(model, "device", data.ego_current.device),
    )
    return _infer_reranked_anchor_trajectory_from_candidates(
        model,
        data,
        anchor_trajs=anchor_trajs,
        use_cfg=use_cfg,
        cfg_weight=cfg_weight,
        bon_seed=bon_seed,
        collision_weight=collision_weight,
        ttc_weight=ttc_weight,
        route_weight=route_weight,
        comfort_weight=comfort_weight,
        progress_weight=progress_weight,
        collision_dist=collision_dist,
    )


def evaluate_trajectory(
    traj: np.ndarray,
    neighbor_future_gt: np.ndarray,
    route_lanes: Optional[np.ndarray],
    collision_dist: float = 2.0,
) -> Dict[str, float]:
    from flow_planner.risk.trajectory_scorer import TrajectoryScorer

    traj_tensor = torch.from_numpy(np.asarray(traj)).float().unsqueeze(0)
    nb_future = torch.from_numpy(np.asarray(neighbor_future_gt)[:, :, :2]).float()

    scorer = TrajectoryScorer(
        collision_threshold=collision_dist,
        ttc_threshold=3.0,
        dt=0.1,
    )

    collision_score = scorer._collision_score(traj_tensor, nb_future).item()
    ttc_score = scorer._ttc_score(traj_tensor, nb_future).item()
    comfort_score = scorer._comfort_score(traj_tensor).item()
    progress_score = scorer._progress_score(traj_tensor).item()

    route_tensor = None
    if route_lanes is not None and np.asarray(route_lanes).size > 0:
        rl = np.asarray(route_lanes)
        if rl.ndim == 3:
            valid_mask = np.abs(rl).sum(axis=-1).sum(axis=-1) > 1e-6
            if valid_mask.any():
                rl_flat = rl[valid_mask].reshape(-1, rl.shape[-1])
                route_tensor = torch.from_numpy(rl_flat[:, :2]).float()
        elif rl.ndim == 2:
            route_tensor = torch.from_numpy(rl[:, :2]).float()
    route_score = scorer._route_score(traj_tensor, route_tensor).item()

    collided = False
    traj = np.asarray(traj)
    neighbor_future_gt = np.asarray(neighbor_future_gt)
    max_steps = min(traj.shape[0], neighbor_future_gt.shape[1])
    for t in range(max_steps):
        ex, ey = float(traj[t, 0]), float(traj[t, 1])
        for m in range(neighbor_future_gt.shape[0]):
            nx, ny = float(neighbor_future_gt[m, t, 0]), float(neighbor_future_gt[m, t, 1])
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                continue
            if ((ex - nx) ** 2 + (ey - ny) ** 2) ** 0.5 < collision_dist:
                collided = True
                break
        if collided:
            break

    return {
        "collided": 1.0 if collided else 0.0,
        "collision_score": collision_score,
        "ttc_score": ttc_score,
        "comfort_score": comfort_score,
        "progress_score": progress_score,
        "route_score": route_score,
    }


def run_multidim_evaluation(
    model,
    scene_dir: str,
    device: str = "cuda",
    max_scenes: int = 500,
    collision_dist: float = 2.0,
    use_cfg: bool = True,
    cfg_weight: float = 1.8,
    bon_seed: int = -1,
    goal_mode: str = "none",
    goal_vocab: Optional[np.ndarray] = None,
    goal_predictor: Optional[GoalPredictor] = None,
    anchor_mode: str = "none",
    anchor_vocab: Optional[np.ndarray] = None,
    anchor_predictor: Optional[AnchorPredictor] = None,
    candidate_selector: Optional[CandidateSelector] = None,
    predicted_anchor_top_k: int = 3,
    candidate_samples_per_anchor: int = 3,
    candidate_samples_per_anchor_list: Optional[List[int]] = None,
    rerank_collision_weight: float = 40.0,
    rerank_ttc_weight: float = 20.0,
    rerank_route_weight: float = 25.0,
    rerank_comfort_weight: float = 10.0,
    rerank_progress_weight: float = 0.0,
    scene_manifest: Optional[str] = None,
    manifest_seed: Optional[int] = None,
    scene_manifest_out: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    if goal_mode != "none" and anchor_mode != "none":
        raise ValueError(
            f"goal_mode={goal_mode!r} and anchor_mode={anchor_mode!r} are mutually exclusive."
        )
    scene_files = resolve_scene_files(
        scene_dir=scene_dir,
        max_scenes=max_scenes,
        scene_manifest=scene_manifest,
        manifest_seed=manifest_seed,
        scene_manifest_out=scene_manifest_out,
    )
    logger.info("Evaluating %d scenes from %s", len(scene_files), scene_dir)

    metrics = {
        "collided": [],
        "collision_score": [],
        "ttc_score": [],
        "comfort_score": [],
        "progress_score": [],
        "route_score": [],
    }
    failures: List[Dict[str, str]] = []

    start_time = time.time()
    neighbor_limit = int(model.planner_params.get("neighbor_num", 32))

    for i, scene_file in enumerate(scene_files):
        try:
            with np.load(scene_file, allow_pickle=True) as raw:
                scene_data = {key: raw[key] for key in raw.files}

            data = scene_to_datasample(
                scene_data,
                device=device,
                neighbor_limit=neighbor_limit,
            )
            goal_point = choose_goal_point(
                goal_mode=goal_mode,
                scene_data=scene_data,
                data=data,
                device=device,
                goal_vocab=goal_vocab,
                goal_predictor=goal_predictor,
            )
            if anchor_mode in (
                "predicted_anchor_rerank",
                "oracle_anchor_rerank",
                "predicted_anchor_candidate_selector",
            ):
                if goal_point is not None:
                    raise ValueError(
                        f"{anchor_mode} is anchor-only; goal_mode must be 'none'"
                    )
                if anchor_mode == "predicted_anchor_rerank":
                    if anchor_predictor is None:
                        raise ValueError(
                            "predicted_anchor_rerank requires an anchor predictor checkpoint"
                        )
                    pred_traj = infer_reranked_anchor_trajectory(
                        model,
                        data,
                        anchor_predictor=anchor_predictor,
                        top_k=predicted_anchor_top_k,
                        use_cfg=use_cfg,
                        cfg_weight=cfg_weight,
                        bon_seed=bon_seed,
                        collision_weight=rerank_collision_weight,
                        ttc_weight=rerank_ttc_weight,
                        route_weight=rerank_route_weight,
                        comfort_weight=rerank_comfort_weight,
                        progress_weight=rerank_progress_weight,
                        collision_dist=collision_dist,
                    )
                elif anchor_mode == "predicted_anchor_candidate_selector":
                    if anchor_predictor is None:
                        raise ValueError(
                            "predicted_anchor_candidate_selector requires an anchor predictor checkpoint"
                        )
                    if candidate_selector is None:
                        raise ValueError(
                            "predicted_anchor_candidate_selector requires a candidate selector checkpoint"
                        )
                    pred_traj = infer_candidate_selector_trajectory(
                        model,
                        data,
                        anchor_predictor=anchor_predictor,
                        candidate_selector=candidate_selector,
                        top_k=predicted_anchor_top_k,
                        samples_per_anchor=candidate_samples_per_anchor,
                        sample_counts_per_anchor=candidate_samples_per_anchor_list,
                        use_cfg=use_cfg,
                        cfg_weight=cfg_weight,
                        bon_seed=bon_seed,
                    )
                else:
                    pred_traj = infer_oracle_reranked_anchor_trajectory(
                        model,
                        data,
                        scene_data=scene_data,
                        anchor_vocab=anchor_vocab,
                        top_k=predicted_anchor_top_k,
                        use_cfg=use_cfg,
                        cfg_weight=cfg_weight,
                        bon_seed=bon_seed,
                        collision_weight=rerank_collision_weight,
                        ttc_weight=rerank_ttc_weight,
                        route_weight=rerank_route_weight,
                        comfort_weight=rerank_comfort_weight,
                        progress_weight=rerank_progress_weight,
                        collision_dist=collision_dist,
                    )
            else:
                anchor_traj = choose_anchor(
                    anchor_mode=anchor_mode,
                    scene_data=scene_data,
                    data=data,
                    device=device,
                    anchor_vocab=anchor_vocab,
                    anchor_predictor=anchor_predictor,
                )
                pred_traj = infer_single_trajectory(
                    model,
                    data,
                    use_cfg=use_cfg,
                    cfg_weight=cfg_weight,
                    bon_seed=bon_seed,
                    goal_point=goal_point,
                    anchor_traj=anchor_traj,
                )
            scene_metrics = evaluate_trajectory(
                pred_traj,
                neighbor_future_gt=_require_scene_key(scene_data, "neighbor_agents_future"),
                route_lanes=scene_data.get("route_lanes"),
                collision_dist=collision_dist,
            )
            for key, value in scene_metrics.items():
                metrics[key].append(value)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1e-6)
                collision_rate = np.mean(metrics["collided"]) * 100 if metrics["collided"] else 0.0
                logger.info(
                    "[%d/%d] collision_rate=%.1f%% | %.1f scenes/s | ETA: %.1fmin",
                    i + 1,
                    len(scene_files),
                    collision_rate,
                    rate,
                    (len(scene_files) - i - 1) / max(rate, 1e-6) / 60.0,
                )
        except Exception as exc:  # pragma: no cover - deployment-oriented fallback
            failures.append(
                {
                    "scene": Path(scene_file).name,
                    "error": str(exc),
                }
            )
            logger.warning("Scene %s failed: %s", scene_file, exc)

    elapsed = time.time() - start_time
    evaluated = len(metrics["collided"])
    if evaluated == 0:
        first_error = failures[0]["error"] if failures else "unknown error"
        raise RuntimeError(
            f"No scenes were successfully evaluated from {scene_dir}. "
            f"First failure: {first_error}"
        )

    conditioning_family = (
        "anchor" if anchor_mode != "none"
        else ("goal" if goal_mode != "none" else "none")
    )
    conditioning_mode = (
        anchor_mode if anchor_mode != "none"
        else (goal_mode if goal_mode != "none" else "none")
    )

    summary = {
        "conditioning_family": conditioning_family,
        "conditioning_mode": conditioning_mode,
        "goal_mode": goal_mode,
        "anchor_mode": anchor_mode,
        "use_cfg": bool(use_cfg),
        "cfg_weight": float(cfg_weight),
        "bon_seed": int(bon_seed),
        "scenes_requested": int(len(scene_files)),
        "scenes_evaluated": int(evaluated),
        "scenes_failed": int(len(failures)),
        "collision_rate": float(np.mean(metrics["collided"]) * 100.0),
        "avg_collision_score": float(np.mean(metrics["collision_score"])),
        "avg_ttc": float(np.mean(metrics["ttc_score"])),
        "avg_comfort": float(np.mean(metrics["comfort_score"])),
        "avg_progress": float(np.mean(metrics["progress_score"])),
        "avg_route": float(np.mean(metrics["route_score"])),
        "elapsed_seconds": float(elapsed),
        "elapsed_minutes": float(elapsed / 60.0),
    }
    return summary, failures


def log_summary(summary: Dict[str, float], ckpt_path: str) -> None:
    logger.info("=" * 60)
    logger.info("SUMMARY - Multi-Dimensional Open-Loop Evaluation")
    logger.info("  Checkpoint: %s", ckpt_path)
    logger.info(
        "  conditioning: %s / %s",
        summary.get("conditioning_family", "none"),
        summary.get("conditioning_mode", "none"),
    )
    logger.info("  use_cfg: %s", summary["use_cfg"])
    logger.info("  cfg_weight: %.3f", summary["cfg_weight"])
    logger.info("  Scenes requested: %d", summary["scenes_requested"])
    logger.info("  Scenes evaluated: %d", summary["scenes_evaluated"])
    logger.info("  Scenes failed: %d", summary["scenes_failed"])
    logger.info("  collision_rate: %.1f%%", summary["collision_rate"])
    logger.info("  avg_collision_score: %.4f", summary["avg_collision_score"])
    logger.info("  avg_ttc: %.4f", summary["avg_ttc"])
    logger.info("  avg_comfort: %.4f", summary["avg_comfort"])
    logger.info("  avg_progress: %.4f", summary["avg_progress"])
    logger.info("  avg_route: %.4f", summary["avg_route"])
    logger.info("  Time: %.1f min", summary["elapsed_minutes"])
    logger.info("=" * 60)


def save_summary_json(
    output_json: str,
    summary: Dict[str, float],
    failures: List[Dict[str, str]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "summary": summary,
        "failures": failures,
    }
    if extra:
        payload["extra"] = extra

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved summary JSON to %s", output_path)

from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.dpo.config_utils import load_composed_config
from flow_planner.goal.goal_predictor import GoalPredictor
from flow_planner.goal.goal_utils import (
    find_nearest_goal,
    load_goal_vocab,
    select_goal_from_route,
)

logger = logging.getLogger(__name__)


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
            "goal_predictor_state_dict",
        ):
            value = ckpt.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state_dict)!r}")

    for prefix in ("module.", "model.", "goal_predictor."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_planner_model(
    config_path: str,
    ckpt_path: str,
    device: str = "cuda",
    goal_vocab_path: Optional[str] = None,
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
    if goal_vocab_path is not None:
        OmegaConf.update(cfg, "model.goal_vocab_path", goal_vocab_path, force_add=True)

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
    state_dict = _unwrap_state_dict(ckpt)

    if any(k.startswith(("backbone.", "head.")) for k in state_dict):
        predictor_state = {
            k: v for k, v in state_dict.items() if not k.startswith("backbone.")
        }
        missing, unexpected = predictor.load_state_dict(predictor_state, strict=False)
    else:
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
        )
    return normalize_prediction(prediction)


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
    scene_manifest: Optional[str] = None,
    manifest_seed: Optional[int] = None,
    scene_manifest_out: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
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
            pred_traj = infer_single_trajectory(
                model,
                data,
                use_cfg=use_cfg,
                cfg_weight=cfg_weight,
                bon_seed=bon_seed,
                goal_point=goal_point,
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

    summary = {
        "goal_mode": goal_mode,
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
    logger.info("  goal_mode: %s", summary["goal_mode"])
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

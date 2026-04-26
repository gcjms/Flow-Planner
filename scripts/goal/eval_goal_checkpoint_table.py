#!/usr/bin/env python3
"""Compare base and goal-finetune checkpoints on a shared scene subset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flow_planner.data.data_process.utils import convert_to_model_inputs
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.goal.goal_utils import find_nearest_goal, select_goal_from_route
from flow_planner.risk.trajectory_scorer import TrajectoryScorer


@dataclass
class CheckpointSpec:
    name: str
    ckpt_path: str
    config_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Goal finetune run directory that contains .hydra/config.yaml and epoch checkpoints.",
    )
    parser.add_argument(
        "--base-ckpt",
        default="/root/autodl-tmp/Flow-Planner/checkpoints/model.pth",
        help="Base Flow-Planner checkpoint to use as the reference row.",
    )
    parser.add_argument(
        "--scene-dir",
        default="/root/autodl-tmp/hard_scenarios_v2",
        help="Directory containing evaluation npz scenes.",
    )
    parser.add_argument(
        "--data-list",
        default="/root/autodl-tmp/hard_scenarios_v2/train_list.json",
        help="JSON list of npz filenames to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of scenes to evaluate from the list.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[50, 60, 70],
        help="Goal-finetune epochs to compare if matching checkpoint files exist.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use for evaluation.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the table as JSON.",
    )
    parser.add_argument(
        "--goal-mode",
        choices=["none", "gt_nearest", "route_nearest", "self_nearest"],
        default="gt_nearest",
        help=(
            "How to provide goal conditioning for goal-finetuned checkpoints. "
            "'gt_nearest' uses the nearest goal cluster to the GT endpoint; "
            "'self_nearest' bootstraps from the model's own no-goal endpoint; "
            "'route_nearest' retrieves a goal from route geometry only; "
            "'none' preserves the old no-goal evaluation path."
        ),
    )
    return parser.parse_args()


def load_cfg(config_kind: str, run_dir: str):
    if config_kind == "base":
        with initialize_config_module(config_module="flow_planner.script", version_base=None):
            cfg = compose(config_name="flow_planner_standard")
    elif config_kind == "goal":
        cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    else:
        cfg = OmegaConf.load(config_kind)

    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)
    return cfg


def load_model(cfg, ckpt_path: str, device: str):
    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    return model.to(device).eval()


def npz_to_data_sample(npz, device: str) -> NuPlanDataSample:
    raw = {
        "ego_agent_past": npz["ego_agent_past"],
        "ego_current_state": npz["ego_current_state"],
        "neighbor_agents_past": npz["neighbor_agents_past"],
        "static_objects": npz["static_objects"],
        "lanes": npz["lanes"],
        "lanes_speed_limit": npz["lanes_speed_limit"],
        "lanes_has_speed_limit": npz["lanes_has_speed_limit"],
        "route_lanes": npz["route_lanes"],
        "route_lanes_speed_limit": npz["route_lanes_speed_limit"],
        "route_lanes_has_speed_limit": npz["route_lanes_has_speed_limit"],
    }
    model_inputs = convert_to_model_inputs(raw, device)
    return NuPlanDataSample(
        batched=(model_inputs["ego_current_state"].dim() > 1),
        ego_past=model_inputs["ego_agent_past"],
        ego_current=model_inputs["ego_current_state"],
        ego_future=torch.from_numpy(npz["ego_agent_future"]).unsqueeze(0).to(device),
        neighbor_past=model_inputs["neighbor_agents_past"],
        lanes=model_inputs["lanes"],
        lanes_speedlimit=model_inputs["lanes_speed_limit"],
        lanes_has_speedlimit=model_inputs["lanes_has_speed_limit"],
        routes=model_inputs["route_lanes"],
        routes_speedlimit=model_inputs["route_lanes_speed_limit"],
        routes_has_speedlimit=model_inputs["route_lanes_has_speed_limit"],
        map_objects=model_inputs["static_objects"],
    )


@torch.no_grad()
def infer(model, npz, device: str, goal_mode: str):
    data = npz_to_data_sample(npz, device)
    infer_kwargs = dict(mode="inference", use_cfg=True, cfg_weight=1.8)
    if goal_mode == "gt_nearest":
        infer_kwargs["goal_point"] = model._get_goal_for_gt(data)
    elif goal_mode == "route_nearest":
        route_goal = select_goal_from_route(npz["route_lanes"], model._goal_vocab)
        infer_kwargs["goal_point"] = torch.from_numpy(route_goal).float().unsqueeze(0).to(device)
    elif goal_mode == "self_nearest":
        bootstrap_outputs = model(data, mode="inference", use_cfg=True, cfg_weight=1.8)
        endpoint = bootstrap_outputs[0, 0, -1, :2].detach().cpu().numpy()[None, :]
        goal_idx = find_nearest_goal(endpoint, model._goal_vocab)[0]
        infer_kwargs["goal_point"] = torch.from_numpy(model._goal_vocab[goal_idx]).float().unsqueeze(0).to(device)
    outputs = model(data, **infer_kwargs)
    pred = outputs[0, 0].cpu().numpy()
    pred_xy = pred[:, :2]
    pred_heading = np.arctan2(pred[:, 3], pred[:, 2])
    return pred, pred_xy, pred_heading


def eval_multidim_metrics(traj: np.ndarray, scene_data: dict, collision_dist: float = 2.0):
    nb_future = torch.from_numpy(scene_data["neighbor_agents_future"][:, :, :2]).float()
    traj_tensor = torch.from_numpy(traj).float().unsqueeze(0)
    scorer = TrajectoryScorer(collision_threshold=collision_dist, ttc_threshold=3.0, dt=0.1)
    collision_score = float(scorer._collision_score(traj_tensor, nb_future).item())
    progress_score = float(scorer._progress_score(traj_tensor).item())

    collided = False
    num_neighbors = scene_data["neighbor_agents_future"].shape[0]
    future_steps = min(traj.shape[0], scene_data["neighbor_agents_future"].shape[1])
    for t in range(future_steps):
        ex, ey = float(traj[t, 0]), float(traj[t, 1])
        for m in range(num_neighbors):
            nx, ny = float(scene_data["neighbor_agents_future"][m, t, 0]), float(scene_data["neighbor_agents_future"][m, t, 1])
            if abs(nx) < 1e-6 and abs(ny) < 1e-6:
                continue
            if ((ex - nx) ** 2 + (ey - ny) ** 2) ** 0.5 < collision_dist:
                collided = True
                break
        if collided:
            break

    return float(collided), collision_score, progress_score


def find_goal_checkpoint(run_dir: str, epoch: int) -> str | None:
    prefix = f"model_epoch_{epoch}_trainloss_"
    for name in sorted(os.listdir(run_dir)):
        if name.startswith(prefix) and name.endswith(".pth"):
            return os.path.join(run_dir, name)
    return None


def build_checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs = [CheckpointSpec(name="base", ckpt_path=args.base_ckpt, config_kind="base")]
    for epoch in args.epochs:
        ckpt_path = find_goal_checkpoint(args.run_dir, epoch)
        if ckpt_path:
            specs.append(CheckpointSpec(name=f"e{epoch}", ckpt_path=ckpt_path, config_kind="goal"))
    return specs


def evaluate_checkpoint(spec: CheckpointSpec, args: argparse.Namespace) -> dict[str, float | str]:
    cfg = load_cfg(spec.config_kind, args.run_dir)
    model = load_model(cfg, spec.ckpt_path, args.device)
    goal_mode = args.goal_mode if spec.config_kind == "goal" else "none"

    with open(args.data_list) as f:
        files = json.load(f)[: args.max_samples]

    metrics = {
        "ADE": [],
        "FDE": [],
        "ADE1": [],
        "ADE3": [],
        "HeadDeg": [],
        "CollRate": [],
        "CollScore": [],
        "Progress": [],
    }

    for file_name in files:
        scene_path = os.path.join(args.scene_dir, file_name)
        npz = np.load(scene_path, allow_pickle=True)
        pred, pred_xy, pred_heading = infer(model, npz, args.device, goal_mode)

        gt = npz["ego_agent_future"]
        gt_xy = gt[:, :2]
        gt_heading = gt[:, 2]
        errors = np.linalg.norm(pred_xy - gt_xy, axis=-1)

        metrics["ADE"].append(float(errors.mean()))
        metrics["FDE"].append(float(errors[-1]))
        metrics["ADE1"].append(float(errors[:10].mean()))
        metrics["ADE3"].append(float(errors[:30].mean()))

        heading_error = np.abs(pred_heading - gt_heading)
        heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)
        metrics["HeadDeg"].append(float(np.degrees(heading_error.mean())))

        collided, collision_score, progress_score = eval_multidim_metrics(pred, dict(npz))
        metrics["CollRate"].append(collided)
        metrics["CollScore"].append(collision_score)
        metrics["Progress"].append(progress_score)

    return {
        "name": spec.name,
        "samples": len(files),
        "ADE": float(np.mean(metrics["ADE"])),
        "FDE": float(np.mean(metrics["FDE"])),
        "ADE1": float(np.mean(metrics["ADE1"])),
        "ADE3": float(np.mean(metrics["ADE3"])),
        "HeadDeg": float(np.mean(metrics["HeadDeg"])),
        "CollRate": float(np.mean(metrics["CollRate"]) * 100.0),
        "CollScore": float(np.mean(metrics["CollScore"])),
        "Progress": float(np.mean(metrics["Progress"])),
        "ckpt_path": spec.ckpt_path,
        "goal_mode": goal_mode,
    }


def print_table(rows: Iterable[dict[str, float | str]]) -> None:
    headers = ["name", "samples", "ADE", "FDE", "ADE1", "ADE3", "HeadDeg", "CollRate", "CollScore", "Progress"]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["name"]),
                    str(row["samples"]),
                    f"{row['ADE']:.4f}",
                    f"{row['FDE']:.4f}",
                    f"{row['ADE1']:.4f}",
                    f"{row['ADE3']:.4f}",
                    f"{row['HeadDeg']:.4f}",
                    f"{row['CollRate']:.2f}",
                    f"{row['CollScore']:.4f}",
                    f"{row['Progress']:.4f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    rows = [evaluate_checkpoint(spec, args) for spec in build_checkpoint_specs(args)]
    print_table(rows)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()

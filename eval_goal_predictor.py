#!/usr/bin/env python3
"""
Evaluate a trained goal predictor on top-k goal-cluster accuracy.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_planner.data.utils.collect import collect_batch
from flow_planner.goal.goal_predictor import GoalPredictor
from train_goal_predictor import build_dataset, load_planner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--planner-ckpt", required=True)
    parser.add_argument("--goal-vocab-path", default=None)
    parser.add_argument("--predictor-ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def topk_hits(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5)):
    max_k = min(max(ks), logits.shape[-1])
    topk = torch.topk(logits, k=max_k, dim=-1).indices
    out = {}
    for k in ks:
        kk = min(k, max_k)
        out[k] = (topk[:, :kk] == labels.unsqueeze(1)).any(dim=1).float()
    return out


def main() -> None:
    args = parse_args()
    planner_config_path = args.planner_config
    if args.goal_vocab_path:
        from omegaconf import OmegaConf
        from flow_planner.dpo.config_utils import load_composed_config

        cfg_for_vocab = load_composed_config(args.planner_config)
        OmegaConf.update(cfg_for_vocab, "model.goal_vocab_path", args.goal_vocab_path, force_add=True)
        planner_config_path = args.planner_config + ".goal_predictor_eval.resolved.yaml"
        OmegaConf.save(cfg_for_vocab, planner_config_path)

    planner, cfg = load_planner(planner_config_path, args.planner_ckpt, args.device)
    checkpoint = torch.load(args.predictor_ckpt, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    predictor = GoalPredictor(
        planner,
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
        freeze_backbone=True,
    ).to(args.device)
    predictor.load_state_dict(checkpoint["model"], strict=False)
    predictor.eval()

    dataset = build_dataset(cfg, args.data_dir, args.data_list, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collect_batch,
    )

    top1, top3, top5 = [], [], []
    for batch in tqdm(loader, desc="Eval"):
        batch = batch.to(args.device)
        with torch.no_grad():
            logits = predictor(batch)
            labels = predictor.get_goal_labels(batch)
        hits = topk_hits(logits, labels, ks=(1, 3, 5))
        top1.append(hits[1].cpu().numpy())
        top3.append(hits[3].cpu().numpy())
        top5.append(hits[5].cpu().numpy())

    metrics = {
        "top1": float(np.concatenate(top1).mean()),
        "top3": float(np.concatenate(top3).mean()),
        "top5": float(np.concatenate(top5).mean()),
        "num_samples": int(sum(len(x) for x in top1)),
    }
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

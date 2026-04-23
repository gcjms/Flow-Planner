#!/usr/bin/env python3
"""
Train a scene-conditioned trajectory-anchor classifier (Phase 1).

This is the anchor-world twin of ``train_goal_predictor.py``. It loads a
pretrained FlowPlanner backbone (frozen by default), points it at an
anchor vocabulary (K, T, 3), and trains a small MLP head to pick the
GT-nearest anchor id from scene features.

Usage (AutoDL):
  python train_anchor_predictor.py \
      --planner-config conf/planner.yaml \
      --planner-ckpt    /root/autodl-tmp/ckpts/flowplanner.pth \
      --anchor-vocab-path /root/Flow-Planner/anchor_vocab.npy \
      --train-data-dir  /root/autodl-tmp/nuplan_npz \
      --train-data-list /root/autodl-tmp/nuplan_npz/train_list.json \
      --val-data-dir    /root/autodl-tmp/nuplan_npz \
      --val-data-list   /root/autodl-tmp/nuplan_npz/val_list.json \
      --save-dir /root/Flow-Planner/checkpoints/anchor_predictor_run1 \
      --epochs 10 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_planner.data.utils.collect import collect_batch
from flow_planner.dpo.config_utils import load_composed_config
from flow_planner.goal.anchor_predictor import AnchorPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner-config", required=True, help="Path to planner config yaml.")
    parser.add_argument("--planner-ckpt", required=True, help="Path to pretrained planner checkpoint.")
    parser.add_argument("--anchor-vocab-path", required=True,
                        help="Path to anchor_vocab.npy produced by cluster_trajectories.py.")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--train-data-list", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--val-data-list", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_planner(cfg_path: str, ckpt_path: str, anchor_vocab_path: str, device: str):
    cfg = load_composed_config(cfg_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    # Attach anchor vocab to the backbone so predictor can read it.
    OmegaConf.update(cfg, "model.anchor_vocab_path", anchor_vocab_path, force_add=True)
    # And make sure legacy goal_vocab is NOT set (mutually exclusive guard in __init__).
    if OmegaConf.select(cfg, "model.goal_vocab_path") is not None:
        OmegaConf.update(cfg, "model.goal_vocab_path", None, force_add=True)
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
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    # When loading a no-anchor planner ckpt into an anchor-enabled config,
    # the new anchor_encoder weights will show up as "missing". That's expected
    # at Phase 1 bootstrap (we didn't train them yet).
    if missing:
        print(f"[warn] planner missing {len(missing)} keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] planner unexpected {len(unexpected)} keys (first 5): {unexpected[:5]}")
    model = model.to(device).eval()
    model.device = device
    return model, cfg


def build_dataset(cfg, data_dir: str, data_list: str, max_samples: int | None):
    os.environ["TRAINING_DATA"] = data_dir
    os.environ["TRAINING_JSON"] = data_list
    ds_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.dataset.train, resolve=True))
    ds_cfg.data_dir = data_dir
    ds_cfg.data_list = data_list
    if max_samples is not None:
        ds_cfg.max_num = max_samples
    return instantiate(ds_cfg)


def topk_hits(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5)):
    max_k = min(max(ks), logits.shape[-1])
    topk = torch.topk(logits, k=max_k, dim=-1).indices
    out = {}
    for k in ks:
        kk = min(k, max_k)
        hits = (topk[:, :kk] == labels.unsqueeze(1)).any(dim=1).float().mean()
        out[k] = float(hits.item())
    return out


@torch.no_grad()
def evaluate(model: AnchorPredictor, loader: DataLoader, device: str):
    model.eval()
    losses = []
    top1, top3, top5 = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = batch.to(device)
        logits = model(batch)
        labels = model.get_anchor_labels(batch)
        loss = criterion(logits, labels)
        hits = topk_hits(logits, labels, ks=(1, 3, 5))
        losses.append(loss.item())
        top1.append(hits[1])
        top3.append(hits[3])
        top5.append(hits[5])
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "top3": float(np.mean(top3)) if top3 else 0.0,
        "top5": float(np.mean(top5)) if top5 else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    planner, cfg = load_planner(
        args.planner_config,
        args.planner_ckpt,
        args.anchor_vocab_path,
        args.device,
    )
    model = AnchorPredictor(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)

    train_set = build_dataset(cfg, args.train_data_dir, args.train_data_list, args.max_train_samples)
    val_data_dir = args.val_data_dir or args.train_data_dir
    val_data_list = args.val_data_list or args.train_data_list
    val_set = build_dataset(cfg, val_data_dir, val_data_list, args.max_val_samples)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collect_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collect_batch,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_top1 = -1.0
    best_path = Path(args.save_dir) / "anchor_predictor_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_top1 = []
        epoch_top3 = []
        epoch_top5 = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(args.device)
            logits = model(batch)
            labels = model.get_anchor_labels(batch)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            hits = topk_hits(logits.detach(), labels, ks=(1, 3, 5))
            epoch_losses.append(loss.item())
            epoch_top1.append(hits[1])
            epoch_top3.append(hits[3])
            epoch_top5.append(hits[5])
            progress.set_postfix(loss=f"{loss.item():.3f}", top1=f"{hits[1]*100:.1f}%")

        scheduler.step()

        train_metrics = {
            "loss": float(np.mean(epoch_losses)),
            "top1": float(np.mean(epoch_top1)),
            "top3": float(np.mean(epoch_top3)),
            "top5": float(np.mean(epoch_top5)),
        }
        val_metrics = evaluate(model, val_loader, args.device)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        print(
            f"[epoch {epoch}] "
            f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.3f} "
            f"val loss={val_metrics['loss']:.4f} top1={val_metrics['top1']:.3f} "
            f"top3={val_metrics['top3']:.3f} top5={val_metrics['top5']:.3f}"
        )

        latest_path = Path(args.save_dir) / "anchor_predictor_latest.pth"
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, latest_path)

        if val_metrics["top1"] >= best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(payload, best_path)

        with open(Path(args.save_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

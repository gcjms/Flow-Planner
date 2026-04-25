#!/usr/bin/env python3
"""
Finetune a FlowPlanner to actually use trajectory anchors (Phase 1 hard gate).

Why this script exists
----------------------
When you load a pre-anchor planner checkpoint into an anchor-enabled config
(see flow_planner_anchor.yaml), the new modules

    decoder.anchor_encoder      # AnchorTokenEncoder
    decoder.anchor_cross_attn   # AnchorCrossAttention

are absent from the ckpt. They are freshly initialized, and in particular
``anchor_cross_attn.out_proj`` is **zero-initialized by design**, which means
the anchor contribution starts at exactly 0. At that point the model is
architecturally capable of consuming an anchor but ignores it completely:

    oracle_anchor metrics == none metrics

This script takes a short continued-training pass that teacher-forces the
oracle (GT-nearest) anchor on every batch by default. It can optionally enable
scheduled anchor sampling, where a linearly increasing fraction of samples use
the trained AnchorPredictor's top-1 anchor instead of the oracle anchor. During
this window:

    - anchor_encoder learns to produce useful anchor-summary tokens
    - anchor_cross_attn.out_proj grows away from zero
    - the DiT main stack learns to route those tokens into the final
      trajectory prediction

Only after this step can Phase 1 Exit Criteria be evaluated meaningfully
(see ANCHOR_DEPLOYMENT_AND_VERIFICATION.md §2.6).

Default parameter-group policy
------------------------------
- ``decoder.anchor_encoder``, ``decoder.anchor_cross_attn``: **full lr**
  (new modules, need to train from scratch).
- rest of ``decoder``: ``base_lr * decoder_lr_mult`` (default 0.1x) so the
  pretrained DiT stack only gently re-adapts to the new cross-attn residual.
- ``model_encoder``: **frozen by default**. The AnchorPredictor you train on
  top of the same backbone (train_anchor_predictor.py) relies on these
  features staying roughly constant; unfreezing here makes the two models
  drift apart.

Usage (AutoDL)
--------------
    python finetune_anchor_planner.py \
        --planner-config flow_planner/script/anchor_finetune.yaml \
        --planner-ckpt   /root/autodl-tmp/ckpts/flowplanner_no_goal.pth \
        --anchor-vocab-path /root/autodl-tmp/anchor_runs/anchor_vocab.npy \
        --train-data-dir  /root/autodl-tmp/nuplan_npz \
        --train-data-list /root/autodl-tmp/nuplan_npz/train_list.json \
        --val-data-dir    /root/autodl-tmp/nuplan_npz \
        --val-data-list   /root/autodl-tmp/nuplan_npz/val_list.json \
        --save-dir        /root/autodl-tmp/anchor_runs/planner_ft_run1 \
        --epochs 10 --batch-size 32 --lr 2e-5 --decoder-lr-mult 0.1 \
        --max-train-samples 80000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_planner.data.utils.collect import collect_batch
from flow_planner.dpo.eval_multidim_utils import (
    _extract_predictor_head_state_dict,
    _unwrap_state_dict,
)
from flow_planner.goal.anchor_predictor import AnchorPredictor

# Re-use the patching + ckpt-loading helper that already handles decoder
# anchor-field injection (goal_dim=0 / anchor_state_dim=3 / anchor_len / ...).
from train_anchor_predictor import build_dataset, load_planner, set_seed


DEFAULT_LOSS_WEIGHTS = {
    "ego_planning_loss": 1.0,
    "consistency_loss": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--planner-config", required=True,
                        help="Path to a top-level planner yaml with anchor model "
                             "defaults (e.g. flow_planner/script/anchor_finetune.yaml).")
    parser.add_argument("--planner-ckpt", required=True,
                        help="Pretrained planner ckpt (no-goal or goal era). "
                             "Loaded strict=False; new anchor modules start fresh.")
    parser.add_argument("--anchor-vocab-path", required=True,
                        help="(K, T, 3) numpy produced by cluster_trajectories.py.")
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--train-data-list", required=True)
    parser.add_argument("--val-data-dir", default=None)
    parser.add_argument("--val-data-list", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # LR groups
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Base lr for NEW anchor modules "
                             "(anchor_encoder + anchor_cross_attn).")
    parser.add_argument("--decoder-lr-mult", type=float, default=0.1,
                        help="Multiplier for the rest of the decoder. Default 0.1x "
                             "keeps the pretrained DiT from drifting too much.")
    parser.add_argument("--encoder-lr-mult", type=float, default=0.0,
                        help="Multiplier for model_encoder. 0.0 = frozen (default). "
                             "Set to e.g. 0.05 if you intentionally want the encoder "
                             "to co-adapt; note this will invalidate a separately "
                             "trained AnchorPredictor on the old encoder.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Linear warmup of all lr groups during the first N epochs.")

    # Loss weights (match core/flow_matching.yaml defaults)
    parser.add_argument("--ego-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)

    # Anchor config patching (fed straight into load_planner)
    parser.add_argument("--anchor-state-dim", type=int, default=3)
    parser.add_argument("--anchor-token-num", type=int, default=4)
    parser.add_argument("--anchor-attn-heads", type=int, default=8)

    # Scheduled anchor sampling (Phase 2 robustness probe)
    parser.add_argument("--anchor-predictor-ckpt", default=None,
                        help="AnchorPredictor checkpoint used when scheduled "
                             "sampling replaces oracle anchors with predictor top-1.")
    parser.add_argument("--scheduled-sampling-p-max", type=float, default=0.0,
                        help="Maximum probability of replacing an oracle anchor "
                             "with predictor top-1 during finetuning. 0 disables it.")
    parser.add_argument("--scheduled-sampling-ramp-epochs", type=float, default=None,
                        help="Linearly ramp replacement probability to p_max over "
                             "this many epochs. Defaults to --epochs.")
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--predictor-dropout", type=float, default=0.1)

    return parser.parse_args()


def build_param_groups(
    planner: nn.Module,
    base_lr: float,
    decoder_lr_mult: float,
    encoder_lr_mult: float,
) -> list[dict]:
    """Split params into {anchor_new, decoder_other, encoder} and freeze
    whatever lr mult is 0."""
    anchor_params: list[nn.Parameter] = []
    decoder_other_params: list[nn.Parameter] = []
    encoder_params: list[nn.Parameter] = []

    decoder = planner.model_decoder
    for name, p in decoder.named_parameters():
        if name.startswith("anchor_encoder.") or name.startswith("anchor_cross_attn."):
            anchor_params.append(p)
        else:
            decoder_other_params.append(p)

    for _, p in planner.model_encoder.named_parameters():
        encoder_params.append(p)

    # Hard-freeze encoder when mult is 0 (default).
    if encoder_lr_mult == 0.0:
        for p in encoder_params:
            p.requires_grad = False
    # Same treatment for decoder-other if someone explicitly sets mult=0
    # (e.g. to sanity check that anchor path alone can already move L2).
    if decoder_lr_mult == 0.0:
        for p in decoder_other_params:
            p.requires_grad = False

    groups: list[dict] = []
    if len(anchor_params) > 0:
        groups.append({
            "params": [p for p in anchor_params if p.requires_grad],
            "lr": base_lr,
            "initial_lr": base_lr,
            "name": "anchor_new",
        })
    if decoder_lr_mult > 0.0:
        lr = base_lr * decoder_lr_mult
        groups.append({
            "params": [p for p in decoder_other_params if p.requires_grad],
            "lr": lr,
            "initial_lr": lr,
            "name": "decoder_other",
        })
    if encoder_lr_mult > 0.0:
        lr = base_lr * encoder_lr_mult
        groups.append({
            "params": [p for p in encoder_params if p.requires_grad],
            "lr": lr,
            "initial_lr": lr,
            "name": "encoder",
        })

    # Drop groups that ended up empty.
    groups = [g for g in groups if len(g["params"]) > 0]
    return groups


def compute_total_loss(loss_dict: dict, weights: dict) -> torch.Tensor:
    total = None
    for k, w in weights.items():
        if k not in loss_dict:
            continue
        term = loss_dict[k] * w
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError(
            f"forward_train returned no weightable losses. "
            f"keys={list(loss_dict.keys())} weighted={list(weights.keys())}"
        )
    return total


def load_anchor_predictor_for_sampling(
    planner: nn.Module,
    ckpt_path: str,
    device: str,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> AnchorPredictor:
    """Load a predictor head without freezing the planner being finetuned."""
    predictor = AnchorPredictor(
        planner_backbone=planner,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_predictor_head_state_dict(_unwrap_state_dict(ckpt))
    missing, unexpected = predictor.head.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] scheduled predictor missing {len(missing)} keys: {missing[:5]}")
    if unexpected:
        print(f"[warn] scheduled predictor unexpected {len(unexpected)} keys: {unexpected[:5]}")

    predictor = predictor.to(device)
    for p in predictor.head.parameters():
        p.requires_grad_(False)
    predictor.head.eval()
    return predictor


def scheduled_sampling_probability(
    global_step: int,
    total_steps: int,
    p_max: float,
    ramp_steps: int,
) -> float:
    if p_max <= 0.0 or total_steps <= 0:
        return 0.0
    if ramp_steps <= 0:
        return float(p_max)
    progress = min(1.0, max(0.0, global_step / float(ramp_steps)))
    return float(p_max) * progress


@torch.no_grad()
def predict_top1_anchor_traj(
    predictor: AnchorPredictor,
    batch,
) -> torch.Tensor:
    """Predict top-1 anchors while temporarily putting the shared backbone in eval mode."""
    backbone = predictor.backbone
    was_training = backbone.training
    backbone.eval()
    predictor.head.eval()
    try:
        prediction = predictor.predict_topk(batch, top_k=1)
    finally:
        if was_training:
            backbone.train()
    return prediction["anchor_trajs"][:, 0, :, :].detach()


def apply_scheduled_anchor_override(
    planner: nn.Module,
    batch,
    predictor: AnchorPredictor | None,
    probability: float,
) -> dict:
    """Attach ``batch.anchor_traj_override`` with oracle/predicted mixed anchors."""
    B = batch.ego_current.shape[0]
    stats = {
        "scheduled_probability": float(probability),
        "scheduled_samples": 0,
        "scheduled_total": int(B),
    }

    if probability <= 0.0:
        if hasattr(batch, "anchor_traj_override"):
            delattr(batch, "anchor_traj_override")
        return stats
    if predictor is None:
        raise RuntimeError(
            "--scheduled-sampling-p-max > 0 requires --anchor-predictor-ckpt"
        )

    replace_mask = torch.rand((B,), device=batch.ego_current.device) < probability
    stats["scheduled_samples"] = int(replace_mask.sum().item())
    if stats["scheduled_samples"] == 0:
        if hasattr(batch, "anchor_traj_override"):
            delattr(batch, "anchor_traj_override")
        return stats

    oracle_anchor = planner._get_anchor_for_gt(batch).to(batch.ego_current.device).float()
    predicted_anchor = predict_top1_anchor_traj(predictor, batch).to(oracle_anchor.device).float()
    mixed_anchor = oracle_anchor.clone()
    mixed_anchor[replace_mask] = predicted_anchor[replace_mask]
    batch.anchor_traj_override = mixed_anchor
    return stats


def format_metrics(metrics: dict) -> str:
    parts = []
    for k in ("total_loss", "ego_planning_loss", "consistency_loss"):
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.4f}")
    return " ".join(parts)


@torch.no_grad()
def evaluate(planner: nn.Module, loader: DataLoader, device: str,
             weights: dict) -> dict:
    planner.eval()
    totals = {"total_loss": [], "ego_planning_loss": [], "consistency_loss": []}
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = batch.to(device)
        _, loss_dict = planner(batch, mode="train")
        total = compute_total_loss(loss_dict, weights)
        totals["total_loss"].append(float(total.item()))
        for k in ("ego_planning_loss", "consistency_loss"):
            if k in loss_dict:
                totals[k].append(float(loss_dict[k].item()))
    out = {}
    for k, vs in totals.items():
        out[k] = float(np.mean(vs)) if vs else 0.0
    return out


def warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """Epoch-level linear warmup. Returns a multiplier in (0, 1]."""
    if warmup_epochs <= 0:
        return 1.0
    # +1 so epoch 0 starts at 1/(warmup_epochs+1) instead of 0
    if epoch >= warmup_epochs:
        return 1.0
    return (epoch + 1) / (warmup_epochs + 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Model ----
    planner, cfg = load_planner(
        args.planner_config,
        args.planner_ckpt,
        args.anchor_vocab_path,
        args.device,
        anchor_state_dim=args.anchor_state_dim,
        anchor_token_num=args.anchor_token_num,
        anchor_attn_heads=args.anchor_attn_heads,
    )
    # load_planner leaves the model in .eval(); flip back for training.
    planner.train()

    # Sanity: anchor modules must actually exist after config patching.
    if not hasattr(planner.model_decoder, "anchor_encoder") or \
            not hasattr(planner.model_decoder, "anchor_cross_attn"):
        raise RuntimeError(
            "Planner decoder has no anchor_encoder / anchor_cross_attn. "
            "Check that --planner-config resolves anchor_state_dim>0 at "
            "instantiation time (see flow_planner_anchor.yaml)."
        )
    if planner._anchor_vocab_tensor is None:
        raise RuntimeError(
            "planner._anchor_vocab_tensor is None; anchor_vocab_path did not "
            "propagate into the model. Check load_planner patching."
        )
    if args.scheduled_sampling_p_max < 0.0 or args.scheduled_sampling_p_max > 1.0:
        raise ValueError("--scheduled-sampling-p-max must be in [0, 1]")

    # ---- Data ----
    train_set = build_dataset(cfg, args.train_data_dir, args.train_data_list,
                              args.max_train_samples)
    val_data_dir = args.val_data_dir or args.train_data_dir
    val_data_list = args.val_data_list or args.train_data_list
    val_set = build_dataset(cfg, val_data_dir, val_data_list,
                            args.max_val_samples)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collect_batch,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collect_batch,
    )

    # ---- Param groups / optimizer ----
    param_groups = build_param_groups(
        planner,
        base_lr=args.lr,
        decoder_lr_mult=args.decoder_lr_mult,
        encoder_lr_mult=args.encoder_lr_mult,
    )
    for g in param_groups:
        print(f"[param-group] name={g['name']:15s} "
              f"lr={g['lr']:.3e} n_params={sum(p.numel() for p in g['params'])}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01,
    )

    weights = {
        "ego_planning_loss": args.ego_loss_weight,
        "consistency_loss": args.consistency_loss_weight,
    }

    scheduled_predictor = None
    if args.scheduled_sampling_p_max > 0.0:
        if args.anchor_predictor_ckpt is None:
            raise ValueError(
                "--scheduled-sampling-p-max > 0 requires --anchor-predictor-ckpt"
            )
        scheduled_predictor = load_anchor_predictor_for_sampling(
            planner,
            args.anchor_predictor_ckpt,
            args.device,
            hidden_dim=args.predictor_hidden_dim,
            dropout=args.predictor_dropout,
        )
        print(
            "[scheduled-sampling] enabled "
            f"p_max={args.scheduled_sampling_p_max:.3f} "
            f"predictor_ckpt={args.anchor_predictor_ckpt}"
        )

    # ---- Train loop ----
    history = []
    best_val = float("inf")
    best_path = Path(args.save_dir) / "planner_anchor_best.pth"
    global_step = 0
    total_steps = max(1, args.epochs * len(train_loader))
    ramp_epochs = args.scheduled_sampling_ramp_epochs
    if ramp_epochs is None:
        ramp_epochs = float(args.epochs)
    ramp_steps = int(max(0, ramp_epochs) * len(train_loader))

    for epoch in range(1, args.epochs + 1):
        planner.train()

        # Warmup (epoch-level linear): scale each group's lr for this epoch.
        warm = warmup_scale(epoch - 1, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * warm

        epoch_totals = {"total_loss": [], "ego_planning_loss": [],
                        "consistency_loss": []}
        out_proj_norms = []
        scheduled_probs = []
        scheduled_samples = 0
        scheduled_total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            batch = batch.to(args.device)
            sampling_p = scheduled_sampling_probability(
                global_step=global_step,
                total_steps=total_steps,
                p_max=args.scheduled_sampling_p_max,
                ramp_steps=ramp_steps,
            )
            sched_stats = apply_scheduled_anchor_override(
                planner,
                batch,
                scheduled_predictor,
                sampling_p,
            )
            scheduled_probs.append(sched_stats["scheduled_probability"])
            scheduled_samples += sched_stats["scheduled_samples"]
            scheduled_total += sched_stats["scheduled_total"]
            _, loss_dict = planner(batch, mode="train")
            total = compute_total_loss(loss_dict, weights)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                args.grad_clip,
            )
            optimizer.step()

            epoch_totals["total_loss"].append(float(total.item()))
            for k in ("ego_planning_loss", "consistency_loss"):
                if k in loss_dict:
                    epoch_totals[k].append(float(loss_dict[k].item()))

            # Diagnostic: monitor anchor_cross_attn.out_proj.weight norm.
            # Starts at 0 (zero-init) and should grow as the model learns to
            # use the anchor. If this stays at 0, finetuning isn't routing
            # gradient into the anchor path.
            with torch.no_grad():
                out_proj = planner.model_decoder.anchor_cross_attn.out_proj.weight
                out_proj_norms.append(float(out_proj.abs().mean().item()))

            progress.set_postfix(
                loss=f"{total.item():.3f}",
                out_proj=f"{out_proj_norms[-1]:.2e}",
                sched=f"{sampling_p:.2f}",
            )
            global_step += 1

        scheduler.step()

        # Restore cosine-scheduled lr after our manual warmup override.
        # (CosineAnnealingLR was stepped against "initial_lr" via last_epoch;
        # it already put the right value in ``g['lr']`` for the NEXT epoch.
        # Our warmup override above only applied to THIS epoch.)

        train_metrics = {k: float(np.mean(v)) if v else 0.0
                         for k, v in epoch_totals.items()}
        val_metrics = evaluate(planner, val_loader, args.device, weights)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": {g["name"]: float(g["lr"]) for g in optimizer.param_groups
                   if "name" in g},
            "anchor_out_proj_abs_mean_end": out_proj_norms[-1]
            if out_proj_norms else 0.0,
            "scheduled_sampling": {
                "p_max": float(args.scheduled_sampling_p_max),
                "p_mean": float(np.mean(scheduled_probs)) if scheduled_probs else 0.0,
                "sample_fraction": float(scheduled_samples / max(scheduled_total, 1)),
                "samples": int(scheduled_samples),
                "total": int(scheduled_total),
            },
        }
        history.append(record)

        print(
            f"[epoch {epoch}] train {format_metrics(train_metrics)} | "
            f"val {format_metrics(val_metrics)} | "
            f"out_proj={record['anchor_out_proj_abs_mean_end']:.2e} | "
            f"sched_p_mean={record['scheduled_sampling']['p_mean']:.3f} "
            f"sched_frac={record['scheduled_sampling']['sample_fraction']:.3f}"
        )

        # Save latest + best.
        payload = {
            "epoch": epoch,
            "state_dict": planner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "args": vars(args),
            "planner_config": args.planner_config,
            "source_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, Path(args.save_dir) / "planner_anchor_latest.pth")
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            torch.save(payload, best_path)

        with open(Path(args.save_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"best checkpoint: {best_path} (val total_loss={best_val:.4f})")


if __name__ == "__main__":
    main()

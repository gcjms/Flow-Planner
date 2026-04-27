#!/usr/bin/env python3
"""Train an anchor-level soft preference selector.

This is deliberately different from planner-DPO: the planner generates
candidate trajectories for top-k anchors, but the learned object here is the
discrete AnchorPredictor head.  The target is a soft distribution over anchor
ids aggregated from candidate scores in ``generate_anchor_softpref_candidates``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flow_planner.data.utils.collect import collect_batch
from flow_planner.dpo.anchor_candidate_scorer import aggregate_anchor_scores
from flow_planner.goal.anchor_predictor import AnchorPredictor
from train_anchor_predictor import build_dataset, load_planner, set_seed


@dataclass
class SelectorRecord:
    scenario_id: str
    scene_relpath: str
    target: np.ndarray
    best_anchor: int
    anchor_indices: List[int]
    anchor_scores: List[float]
    score_std: float
    score_gap: float
    top_prob: float
    has_collision_candidate: bool


class AnchorSelectorDataset(Dataset):
    def __init__(self, base_dataset: Dataset, records: Sequence[SelectorRecord]):
        if len(base_dataset) != len(records):
            raise ValueError(f"dataset/record length mismatch: {len(base_dataset)} vs {len(records)}")
        self.base_dataset = base_dataset
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        return (
            self.base_dataset[idx],
            torch.from_numpy(record.target).float(),
            torch.tensor(record.best_anchor, dtype=torch.long),
        )


def collate_selector(batch):
    samples, targets, best_anchors = zip(*batch)
    return collect_batch(list(samples)), torch.stack(list(targets), dim=0), torch.stack(list(best_anchors), dim=0)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = values.astype(np.float64) / temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    return (probs / probs.sum()).astype(np.float32)


def _has_collision_candidate(candidates: Sequence[Dict[str, object]]) -> bool:
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        if float(metrics.get("collided", 0.0)) > 0.5:
            return True
    return False


def load_selector_records(
    scored_dir: str,
    data_dir: str,
    num_anchors: int,
    target_temp: float,
    score_agg: str,
    max_scenes: int | None,
    min_score_std: float,
    min_score_gap: float,
    min_top_prob: float,
) -> Tuple[List[SelectorRecord], Dict[str, object]]:
    json_paths = sorted(Path(scored_dir).glob("*.json"))
    if max_scenes is not None:
        json_paths = json_paths[:max_scenes]

    records: List[SelectorRecord] = []
    skipped = {
        "no_candidates": 0,
        "bad_scene_path": 0,
        "low_score_std": 0,
        "low_score_gap": 0,
        "low_top_prob": 0,
    }
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", [])
        if not candidates:
            skipped["no_candidates"] += 1
            continue

        anchor_indices, anchor_scores = aggregate_anchor_scores(candidates, score_agg)
        score_arr = np.asarray(anchor_scores, dtype=np.float32)
        order = np.argsort(score_arr)[::-1]
        score_std = float(score_arr.std())
        score_gap = float(score_arr[order[0]] - score_arr[order[1]]) if len(order) > 1 else 0.0
        probs = _softmax(score_arr, target_temp)
        top_prob = float(probs.max())

        if score_std < min_score_std:
            skipped["low_score_std"] += 1
            continue
        if score_gap < min_score_gap:
            skipped["low_score_gap"] += 1
            continue
        if top_prob < min_top_prob:
            skipped["low_top_prob"] += 1
            continue

        scene_file = str(payload["scene_file"])
        try:
            scene_relpath = os.path.relpath(scene_file, data_dir)
        except ValueError:
            skipped["bad_scene_path"] += 1
            continue
        if scene_relpath.startswith(".."):
            skipped["bad_scene_path"] += 1
            continue

        target = np.zeros((num_anchors,), dtype=np.float32)
        for anchor_idx, prob in zip(anchor_indices, probs):
            target[int(anchor_idx)] = float(prob)

        best_anchor = int(anchor_indices[int(order[0])])
        records.append(
            SelectorRecord(
                scenario_id=str(payload.get("scenario_id", path.stem)),
                scene_relpath=scene_relpath,
                target=target,
                best_anchor=best_anchor,
                anchor_indices=[int(x) for x in anchor_indices],
                anchor_scores=[float(x) for x in anchor_scores],
                score_std=score_std,
                score_gap=score_gap,
                top_prob=top_prob,
                has_collision_candidate=_has_collision_candidate(candidates),
            )
        )

    stats = {
        "scored_dir": scored_dir,
        "data_dir": data_dir,
        "num_input_json": len(json_paths),
        "num_records": len(records),
        "skipped": skipped,
        "target_temp": target_temp,
        "score_agg": score_agg,
        "score_std_mean": float(np.mean([r.score_std for r in records])) if records else 0.0,
        "score_gap_mean": float(np.mean([r.score_gap for r in records])) if records else 0.0,
        "top_prob_mean": float(np.mean([r.top_prob for r in records])) if records else 0.0,
        "collision_scene_count": int(sum(1 for r in records if r.has_collision_candidate)),
    }
    return records, stats


def split_records(records: Sequence[SelectorRecord], val_fraction: float, seed: int):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    if val_fraction <= 0 or len(records) < 5:
        return records, []
    val_size = max(1, int(round(len(records) * val_fraction)))
    return records[val_size:], records[:val_size]


def build_loader(
    cfg,
    records: Sequence[SelectorRecord],
    data_dir: str,
    manifest_path: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    manifest_path.write_text(
        json.dumps([r.scene_relpath for r in records], indent=2) + "\n",
        encoding="utf-8",
    )
    base_dataset = build_dataset(cfg, data_dir, str(manifest_path), max_samples=None)
    dataset = AnchorSelectorDataset(base_dataset, records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_selector,
    )


def load_predictor_head(model: AnchorPredictor, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("head.") for k in state_dict):
        state_dict = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
    missing, unexpected = model.head.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] predictor head missing {len(missing)} keys: {missing[:5]}")
    if unexpected:
        print(f"[warn] predictor head unexpected {len(unexpected)} keys: {unexpected[:5]}")


def run_epoch(
    model: AnchorPredictor,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    gt_ce_weight: float,
):
    train = optimizer is not None
    model.train(train)
    pref_losses: List[float] = []
    gt_losses: List[float] = []
    total_losses: List[float] = []
    top1_matches: List[float] = []
    gt_top1_hits: List[float] = []
    target_probs_on_pred: List[float] = []

    desc = "Train" if train else "Eval"
    for batch, targets, best_anchors in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        targets = targets.to(device)
        best_anchors = best_anchors.to(device)

        logits = model(batch)
        log_probs = F.log_softmax(logits, dim=-1)
        pref_loss = -(targets * log_probs).sum(dim=-1).mean()

        gt_loss = torch.zeros((), device=device)
        if gt_ce_weight > 0:
            labels = model.get_anchor_labels(batch)
            gt_loss = F.cross_entropy(logits, labels)
        loss = pref_loss + gt_ce_weight * gt_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        pred = logits.argmax(dim=-1)
        top1_matches.append((pred == best_anchors).float().mean().item())
        target_probs_on_pred.append(targets.gather(1, pred.unsqueeze(1)).mean().item())
        if gt_ce_weight > 0:
            labels = model.get_anchor_labels(batch)
            gt_top1_hits.append((pred == labels).float().mean().item())
        pref_losses.append(pref_loss.item())
        gt_losses.append(gt_loss.item())
        total_losses.append(loss.item())

    return {
        "loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "pref_loss": float(np.mean(pref_losses)) if pref_losses else 0.0,
        "gt_loss": float(np.mean(gt_losses)) if gt_losses else 0.0,
        "top1_match": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "gt_top1": float(np.mean(gt_top1_hits)) if gt_top1_hits else 0.0,
        "target_prob_on_pred": float(np.mean(target_probs_on_pred)) if target_probs_on_pred else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--planner-ckpt", required=True)
    parser.add_argument("--anchor-vocab-path", required=True)
    parser.add_argument("--init-predictor-ckpt", required=True)
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-temp", type=float, default=0.5)
    parser.add_argument("--score-agg", choices=("mean", "max", "logmeanexp"), default="mean")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-score-std", type=float, default=0.0)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--min-top-prob", type=float, default=0.0)
    parser.add_argument("--gt-ce-weight", type=float, default=0.05)
    parser.add_argument("--anchor-state-dim", type=int, default=3)
    parser.add_argument("--anchor-token-num", type=int, default=4)
    parser.add_argument("--anchor-attn-heads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    planner, cfg = load_planner(
        args.planner_config,
        args.planner_ckpt,
        args.anchor_vocab_path,
        args.device,
        anchor_state_dim=args.anchor_state_dim,
        anchor_token_num=args.anchor_token_num,
        anchor_attn_heads=args.anchor_attn_heads,
    )
    model = AnchorPredictor(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)
    load_predictor_head(model, args.init_predictor_ckpt)

    records, target_stats = load_selector_records(
        scored_dir=args.scored_dir,
        data_dir=args.data_dir,
        num_anchors=model.num_anchors,
        target_temp=args.target_temp,
        score_agg=args.score_agg,
        max_scenes=args.max_scenes,
        min_score_std=args.min_score_std,
        min_score_gap=args.min_score_gap,
        min_top_prob=args.min_top_prob,
    )
    if not records:
        raise RuntimeError(f"no usable selector records loaded from {args.scored_dir}")
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    target_stats["num_train_records"] = len(train_records)
    target_stats["num_val_records"] = len(val_records)
    (save_dir / "target_stats.json").write_text(
        json.dumps(target_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(target_stats, indent=2, ensure_ascii=False))

    train_loader = build_loader(
        cfg,
        train_records,
        args.data_dir,
        save_dir / "selector_train_list.json",
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    val_loader = None
    if val_records:
        val_loader = build_loader(
            cfg,
            val_records,
            args.data_dir,
            save_dir / "selector_val_list.json",
            args.batch_size,
            args.num_workers,
            shuffle=False,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_metric = -1.0
    best_path = save_dir / "anchor_selector_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args.device, optimizer, args.gt_ce_weight)
        scheduler.step()
        val_metrics = (
            run_epoch(model, val_loader, args.device, None, args.gt_ce_weight)
            if val_loader is not None
            else {}
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        metric = val_metrics.get("top1_match", train_metrics["top1_match"])
        print(
            f"[epoch {epoch}] "
            f"train pref={train_metrics['pref_loss']:.4f} top1={train_metrics['top1_match']:.3f} "
            f"val top1={val_metrics.get('top1_match', 0.0):.3f} "
            f"val prob={val_metrics.get('target_prob_on_pred', 0.0):.3f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "target_stats": target_stats,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, save_dir / "anchor_selector_latest.pth")
        if metric >= best_metric:
            best_metric = metric
            torch.save(payload, best_path)
        (save_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

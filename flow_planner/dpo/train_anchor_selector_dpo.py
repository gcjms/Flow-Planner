#!/usr/bin/env python3
"""Train an anchor selector with DPO on discrete anchor scores.

This script is the DPO counterpart of ``train_anchor_selector_softpref.py``.
It avoids planner-level continuous flow-matching log-prob and instead applies
DPO directly to ``AnchorPredictor`` logits:

    s_theta(scene, chosen_anchor) > s_theta(scene, rejected_anchor)

The preference pairs are mined from scored anchor-conditioned candidate sets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flow_planner.data.utils.collect import collect_batch
from flow_planner.dpo.anchor_candidate_scorer import pair_label, summarize_anchor_groups
from flow_planner.goal.anchor_predictor import AnchorPredictor
from train_anchor_predictor import build_dataset, load_planner, set_seed


@dataclass
class AnchorPairRecord:
    scenario_id: str
    scene_relpath: str
    chosen_anchor: int
    rejected_anchor: int
    chosen_score: float
    rejected_score: float
    score_gap: float
    label: str


class AnchorSelectorDPODataset(Dataset):
    def __init__(self, base_dataset: Dataset, records: Sequence[AnchorPairRecord]):
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
            torch.tensor(record.chosen_anchor, dtype=torch.long),
            torch.tensor(record.rejected_anchor, dtype=torch.long),
            torch.tensor(record.score_gap, dtype=torch.float32),
        )


def collate_dpo(batch):
    samples, chosen, rejected, gaps = zip(*batch)
    return (
        collect_batch(list(samples)),
        torch.stack(list(chosen), dim=0),
        torch.stack(list(rejected), dim=0),
        torch.stack(list(gaps), dim=0),
    )


def load_pair_records(
    scored_dir: str,
    data_dir: str,
    score_agg: str,
    pair_mode: str,
    max_scenes: int | None,
    min_score_gap: float,
    min_score_std: float,
    require_collision_pair: bool,
) -> Tuple[List[AnchorPairRecord], Dict[str, object]]:
    json_paths = sorted(Path(scored_dir).glob("*.json"))
    if max_scenes is not None:
        json_paths = json_paths[:max_scenes]

    records: List[AnchorPairRecord] = []
    skipped = {
        "no_candidates": 0,
        "too_few_anchors": 0,
        "bad_scene_path": 0,
        "low_score_std": 0,
        "low_score_gap": 0,
        "require_collision_pair": 0,
    }
    label_counts: Dict[str, int] = {}
    scene_count_with_pairs = 0

    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates", [])
        if not candidates:
            skipped["no_candidates"] += 1
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

        summaries = summarize_anchor_groups(candidates, method=score_agg)
        if len(summaries) < 2:
            skipped["too_few_anchors"] += 1
            continue

        anchors = sorted(summaries.keys(), key=lambda idx: summaries[idx]["score"], reverse=True)
        scores = np.asarray([summaries[idx]["score"] for idx in anchors], dtype=np.float32)
        if float(scores.std()) < min_score_std:
            skipped["low_score_std"] += 1
            continue

        candidate_pairs: List[Tuple[int, int]] = []
        if pair_mode == "best_worst":
            candidate_pairs.append((anchors[0], anchors[-1]))
        elif pair_mode == "all":
            for i in range(len(anchors)):
                for j in range(i + 1, len(anchors)):
                    candidate_pairs.append((anchors[i], anchors[j]))
        else:
            raise ValueError(f"unknown pair_mode: {pair_mode}")

        scene_added = 0
        for chosen_anchor, rejected_anchor in candidate_pairs:
            chosen_summary = summaries[chosen_anchor]
            rejected_summary = summaries[rejected_anchor]
            gap = float(chosen_summary["score"] - rejected_summary["score"])
            if gap < min_score_gap:
                skipped["low_score_gap"] += 1
                continue
            label = pair_label(chosen_summary, rejected_summary)
            if require_collision_pair and not label.startswith("anchor_collision"):
                skipped["require_collision_pair"] += 1
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
            records.append(
                AnchorPairRecord(
                    scenario_id=str(payload.get("scenario_id", path.stem)),
                    scene_relpath=scene_relpath,
                    chosen_anchor=int(chosen_anchor),
                    rejected_anchor=int(rejected_anchor),
                    chosen_score=float(chosen_summary["score"]),
                    rejected_score=float(rejected_summary["score"]),
                    score_gap=gap,
                    label=label,
                )
            )
            scene_added += 1
        if scene_added:
            scene_count_with_pairs += 1

    stats = {
        "scored_dir": scored_dir,
        "data_dir": data_dir,
        "num_input_json": len(json_paths),
        "num_pairs": len(records),
        "num_scenes_with_pairs": scene_count_with_pairs,
        "skipped": skipped,
        "label_counts": label_counts,
        "score_agg": score_agg,
        "pair_mode": pair_mode,
        "min_score_gap": min_score_gap,
        "min_score_std": min_score_std,
        "require_collision_pair": require_collision_pair,
        "mean_gap": float(np.mean([r.score_gap for r in records])) if records else 0.0,
        "median_gap": float(np.median([r.score_gap for r in records])) if records else 0.0,
    }
    return records, stats


def split_records(records: Sequence[AnchorPairRecord], val_fraction: float, seed: int):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    if val_fraction <= 0 or len(records) < 5:
        return records, []
    val_size = max(1, int(round(len(records) * val_fraction)))
    return records[val_size:], records[:val_size]


def build_loader(
    cfg,
    records: Sequence[AnchorPairRecord],
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
    dataset = AnchorSelectorDPODataset(base_dataset, records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_dpo,
    )


def _extract_head_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("head.") for k in state_dict):
        return {
            k[len("head."):]: v
            for k, v in state_dict.items()
            if k.startswith("head.")
        }
    return state_dict


def load_predictor_head(model: AnchorPredictor, ckpt_path: str, name: str) -> None:
    state_dict = _extract_head_state_dict(ckpt_path)
    missing, unexpected = model.head.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {name} head missing {len(missing)} keys: {missing[:5]}")
    if unexpected:
        print(f"[warn] {name} head unexpected {len(unexpected)} keys: {unexpected[:5]}")


def run_epoch(
    policy: AnchorPredictor,
    reference: AnchorPredictor,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    sft_weight: float,
    gap_weight: float,
):
    train = optimizer is not None
    policy.train(train)
    reference.eval()

    losses: List[float] = []
    dpo_losses: List[float] = []
    sft_losses: List[float] = []
    accuracies: List[float] = []
    margins: List[float] = []
    ref_margins: List[float] = []
    chosen_probs: List[float] = []

    desc = "Train" if train else "Eval"
    for batch, chosen, rejected, gaps in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        chosen = chosen.to(device)
        rejected = rejected.to(device)
        gaps = gaps.to(device)

        logits = policy(batch)
        with torch.no_grad():
            ref_logits = reference(batch)

        chosen_score = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
        rejected_score = logits.gather(1, rejected.unsqueeze(1)).squeeze(1)
        ref_chosen = ref_logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
        ref_rejected = ref_logits.gather(1, rejected.unsqueeze(1)).squeeze(1)

        margin = chosen_score - rejected_score
        ref_margin = ref_chosen - ref_rejected
        dpo_logits = beta * (margin - ref_margin)
        per_item_dpo = -F.logsigmoid(dpo_logits)
        if gap_weight > 0:
            weights = (1.0 + gap_weight * gaps).clamp(max=5.0)
            dpo_loss = (per_item_dpo * weights).mean()
        else:
            dpo_loss = per_item_dpo.mean()

        sft_loss = torch.zeros((), device=device)
        if sft_weight > 0:
            sft_loss = F.cross_entropy(logits, chosen)
        loss = dpo_loss + sft_weight * sft_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()

        probs = torch.softmax(logits, dim=-1)
        losses.append(loss.item())
        dpo_losses.append(dpo_loss.item())
        sft_losses.append(sft_loss.item())
        accuracies.append((margin > 0).float().mean().item())
        margins.append(margin.mean().item())
        ref_margins.append(ref_margin.mean().item())
        chosen_probs.append(probs.gather(1, chosen.unsqueeze(1)).mean().item())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "dpo_loss": float(np.mean(dpo_losses)) if dpo_losses else 0.0,
        "sft_loss": float(np.mean(sft_losses)) if sft_losses else 0.0,
        "pair_acc": float(np.mean(accuracies)) if accuracies else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "ref_margin": float(np.mean(ref_margins)) if ref_margins else 0.0,
        "chosen_prob": float(np.mean(chosen_probs)) if chosen_probs else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--planner-ckpt", required=True)
    parser.add_argument("--anchor-vocab-path", required=True)
    parser.add_argument("--init-predictor-ckpt", required=True)
    parser.add_argument("--ref-predictor-ckpt", default=None)
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--score-agg", choices=("mean", "max", "logmeanexp"), default="mean")
    parser.add_argument("--pair-mode", choices=("all", "best_worst"), default="all")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-score-gap", type=float, default=0.05)
    parser.add_argument("--min-score-std", type=float, default=0.0)
    parser.add_argument("--require-collision-pair", action="store_true")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--sft-weight", type=float, default=0.05)
    parser.add_argument("--gap-weight", type=float, default=0.0)
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
    policy = AnchorPredictor(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)
    load_predictor_head(policy, args.init_predictor_ckpt, name="policy")

    reference = AnchorPredictor(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)
    load_predictor_head(reference, args.ref_predictor_ckpt or args.init_predictor_ckpt, name="reference")
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    records, pair_stats = load_pair_records(
        scored_dir=args.scored_dir,
        data_dir=args.data_dir,
        score_agg=args.score_agg,
        pair_mode=args.pair_mode,
        max_scenes=args.max_scenes,
        min_score_gap=args.min_score_gap,
        min_score_std=args.min_score_std,
        require_collision_pair=args.require_collision_pair,
    )
    if not records:
        raise RuntimeError(f"no usable DPO pairs loaded from {args.scored_dir}")
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    pair_stats["num_train_pairs"] = len(train_records)
    pair_stats["num_val_pairs"] = len(val_records)
    (save_dir / "pair_stats.json").write_text(
        json.dumps(pair_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(pair_stats, indent=2, ensure_ascii=False))

    train_loader = build_loader(
        cfg,
        train_records,
        args.data_dir,
        save_dir / "selector_dpo_train_list.json",
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
            save_dir / "selector_dpo_val_list.json",
            args.batch_size,
            args.num_workers,
            shuffle=False,
        )

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_metric = -1.0
    best_path = save_dir / "anchor_selector_dpo_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            policy,
            reference,
            train_loader,
            args.device,
            optimizer,
            beta=args.beta,
            sft_weight=args.sft_weight,
            gap_weight=args.gap_weight,
        )
        scheduler.step()
        val_metrics = (
            run_epoch(
                policy,
                reference,
                val_loader,
                args.device,
                None,
                beta=args.beta,
                sft_weight=args.sft_weight,
                gap_weight=args.gap_weight,
            )
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
        metric = val_metrics.get("pair_acc", train_metrics["pair_acc"])
        print(
            f"[epoch {epoch}] "
            f"train dpo={train_metrics['dpo_loss']:.4f} acc={train_metrics['pair_acc']:.3f} "
            f"margin={train_metrics['margin']:.4f} "
            f"val acc={val_metrics.get('pair_acc', 0.0):.3f} "
            f"val margin={val_metrics.get('margin', 0.0):.4f}"
        )

        payload = {
            "epoch": epoch,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "pair_stats": pair_stats,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, save_dir / "anchor_selector_dpo_latest.pth")
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

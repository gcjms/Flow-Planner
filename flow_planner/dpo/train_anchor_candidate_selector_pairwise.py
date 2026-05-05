#!/usr/bin/env python3
"""Train a candidate-level selector with hard safety pairs.

This script keeps the candidate-level scorer from
``train_anchor_candidate_selector_softpref.py`` but switches the supervision
from soft teacher distributions to explicit pairwise preferences:

    safe candidate > collided candidate

The goal is to test whether candidate-level ranking works better with clean
binary supervision than with soft CE over noisy teacher scores.
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
from flow_planner.goal.candidate_selector import CandidateSelector
from train_anchor_predictor import build_dataset, load_planner, set_seed


@dataclass
class CandidatePairRecord:
    scenario_id: str
    scene_relpath: str
    candidate_npz_path: str
    chosen_idx: int
    rejected_idx: int
    chosen_anchor: int
    rejected_anchor: int
    score_gap: float
    same_anchor: bool


class CandidatePairDataset(Dataset):
    def __init__(self, base_dataset: Dataset, records: Sequence[CandidatePairRecord]):
        if len(base_dataset) != len(records):
            raise ValueError(f"dataset/record length mismatch: {len(base_dataset)} vs {len(records)}")
        self.base_dataset = base_dataset
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        with np.load(record.candidate_npz_path, allow_pickle=True) as raw:
            candidate_trajs = np.asarray(raw["candidates"], dtype=np.float32)
            anchor_trajs = np.asarray(raw["anchor_trajs"], dtype=np.float32)

        return (
            self.base_dataset[idx],
            torch.from_numpy(candidate_trajs).float(),
            torch.from_numpy(anchor_trajs).float(),
            torch.tensor(record.chosen_idx, dtype=torch.long),
            torch.tensor(record.rejected_idx, dtype=torch.long),
            torch.tensor(record.score_gap, dtype=torch.float32),
        )


def collate_candidate_pairs(batch):
    samples, candidate_trajs, anchor_trajs, chosen, rejected, gaps = zip(*batch)
    batch_size = len(batch)
    max_candidates = max(item.shape[0] for item in candidate_trajs)
    future_steps = candidate_trajs[0].shape[1]
    candidate_dim = candidate_trajs[0].shape[2]
    anchor_dim = anchor_trajs[0].shape[2]

    padded_candidates = torch.zeros(batch_size, max_candidates, future_steps, candidate_dim, dtype=torch.float32)
    padded_anchors = torch.zeros(batch_size, max_candidates, future_steps, anchor_dim, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    for row_idx, (cand, anchor) in enumerate(zip(candidate_trajs, anchor_trajs)):
        count = cand.shape[0]
        padded_candidates[row_idx, :count] = cand
        padded_anchors[row_idx, :count] = anchor
        candidate_mask[row_idx, :count] = True

    return (
        collect_batch(list(samples)),
        padded_candidates,
        padded_anchors,
        candidate_mask,
        torch.stack(list(chosen), dim=0),
        torch.stack(list(rejected), dim=0),
        torch.stack(list(gaps), dim=0),
    )


def _pair_scope_match(scope: str, chosen_anchor: int, rejected_anchor: int) -> bool:
    if scope == "any":
        return True
    if scope == "same_anchor":
        return chosen_anchor == rejected_anchor
    if scope == "cross_anchor":
        return chosen_anchor != rejected_anchor
    raise ValueError(f"unknown pair_scope: {scope}")


def load_candidate_pair_records(
    scored_dir: str,
    data_dir: str,
    max_scenes: int | None,
    pair_scope: str,
    pair_reduce: str,
    min_score_gap: float,
) -> Tuple[List[CandidatePairRecord], Dict[str, object]]:
    json_paths = sorted(Path(scored_dir).glob("*.json"))
    if max_scenes is not None:
        json_paths = json_paths[:max_scenes]

    records: List[CandidatePairRecord] = []
    skipped = {
        "no_candidates": 0,
        "bad_scene_path": 0,
        "missing_npz": 0,
        "no_mixed_collision": 0,
        "scope_filtered": 0,
        "low_gap": 0,
    }
    scene_counts = {
        "scenes": 0,
        "scenes_with_pairs": 0,
        "mixed_collision_scenes": 0,
    }

    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_infos = payload.get("candidates", [])
        if not candidate_infos:
            skipped["no_candidates"] += 1
            continue

        scene_counts["scenes"] += 1
        scene_file = str(payload["scene_file"])
        try:
            scene_relpath = os.path.relpath(scene_file, data_dir)
        except ValueError:
            skipped["bad_scene_path"] += 1
            continue
        if scene_relpath.startswith(".."):
            skipped["bad_scene_path"] += 1
            continue

        candidate_npz_path = str(payload.get("source_npz", ""))
        if not candidate_npz_path or not os.path.exists(candidate_npz_path):
            skipped["missing_npz"] += 1
            continue

        safe: List[Tuple[int, int, float]] = []
        collided: List[Tuple[int, int, float]] = []
        by_anchor: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}
        for item in candidate_infos:
            idx = int(item["candidate_idx"])
            anchor = int(item["anchor_index"])
            score = float(item["total_score"])
            is_safe = float(item.get("metrics", {}).get("collided", 0.0)) < 0.5
            by_anchor.setdefault(anchor, {"safe": [], "collided": []})
            if is_safe:
                safe.append((idx, anchor, score))
                by_anchor[anchor]["safe"].append((idx, score))
            else:
                collided.append((idx, anchor, score))
                by_anchor[anchor]["collided"].append((idx, score))

        if not safe or not collided:
            skipped["no_mixed_collision"] += 1
            continue
        scene_counts["mixed_collision_scenes"] += 1

        scene_added = 0
        if pair_reduce == "all":
            for chosen_idx, chosen_anchor, chosen_score in safe:
                for rejected_idx, rejected_anchor, rejected_score in collided:
                    if not _pair_scope_match(pair_scope, chosen_anchor, rejected_anchor):
                        skipped["scope_filtered"] += 1
                        continue
                    gap = chosen_score - rejected_score
                    if gap < min_score_gap:
                        skipped["low_gap"] += 1
                        continue
                    records.append(
                        CandidatePairRecord(
                            scenario_id=str(payload.get("scenario_id", path.stem)),
                            scene_relpath=scene_relpath,
                            candidate_npz_path=candidate_npz_path,
                            chosen_idx=chosen_idx,
                            rejected_idx=rejected_idx,
                            chosen_anchor=chosen_anchor,
                            rejected_anchor=rejected_anchor,
                            score_gap=gap,
                            same_anchor=(chosen_anchor == rejected_anchor),
                        )
                    )
                    scene_added += 1
        elif pair_reduce == "scene_best":
            chosen_idx, chosen_anchor, chosen_score = max(safe, key=lambda x: x[2])
            rejected_idx, rejected_anchor, rejected_score = min(collided, key=lambda x: x[2])
            if _pair_scope_match(pair_scope, chosen_anchor, rejected_anchor):
                gap = chosen_score - rejected_score
                if gap >= min_score_gap:
                    records.append(
                        CandidatePairRecord(
                            scenario_id=str(payload.get("scenario_id", path.stem)),
                            scene_relpath=scene_relpath,
                            candidate_npz_path=candidate_npz_path,
                            chosen_idx=chosen_idx,
                            rejected_idx=rejected_idx,
                            chosen_anchor=chosen_anchor,
                            rejected_anchor=rejected_anchor,
                            score_gap=gap,
                            same_anchor=(chosen_anchor == rejected_anchor),
                        )
                    )
                    scene_added += 1
                else:
                    skipped["low_gap"] += 1
            else:
                skipped["scope_filtered"] += 1
        elif pair_reduce == "anchor_best":
            for anchor, groups in by_anchor.items():
                safe_rows = groups["safe"]
                collided_rows = groups["collided"]
                if not safe_rows or not collided_rows:
                    continue
                chosen_idx, chosen_score = max(safe_rows, key=lambda x: x[1])
                rejected_idx, rejected_score = min(collided_rows, key=lambda x: x[1])
                gap = chosen_score - rejected_score
                if gap < min_score_gap:
                    skipped["low_gap"] += 1
                    continue
                records.append(
                    CandidatePairRecord(
                        scenario_id=str(payload.get("scenario_id", path.stem)),
                        scene_relpath=scene_relpath,
                        candidate_npz_path=candidate_npz_path,
                        chosen_idx=chosen_idx,
                        rejected_idx=rejected_idx,
                        chosen_anchor=anchor,
                        rejected_anchor=anchor,
                        score_gap=gap,
                        same_anchor=True,
                    )
                )
                scene_added += 1
        else:
            raise ValueError(f"unknown pair_reduce: {pair_reduce}")

        if scene_added:
            scene_counts["scenes_with_pairs"] += 1

    gap_values = [r.score_gap for r in records]
    stats = {
        "scored_dir": scored_dir,
        "data_dir": data_dir,
        "num_input_json": len(json_paths),
        "num_pairs": len(records),
        "pair_scope": pair_scope,
        "pair_reduce": pair_reduce,
        "min_score_gap": min_score_gap,
        "scene_counts": scene_counts,
        "skipped": skipped,
        "same_anchor_pairs": int(sum(1 for r in records if r.same_anchor)),
        "cross_anchor_pairs": int(sum(1 for r in records if not r.same_anchor)),
        "gap_mean": float(np.mean(gap_values)) if gap_values else 0.0,
        "gap_p50": float(np.percentile(gap_values, 50)) if gap_values else 0.0,
    }
    return records, stats


def split_records(records: Sequence[CandidatePairRecord], val_fraction: float, seed: int):
    records = list(records)
    rng = random.Random(seed)
    scene_ids = sorted({record.scenario_id for record in records})
    rng.shuffle(scene_ids)
    if val_fraction <= 0 or len(scene_ids) < 2:
        return records, []
    val_scene_count = max(1, int(round(len(scene_ids) * val_fraction)))
    val_scene_count = min(val_scene_count, len(scene_ids) - 1)
    val_scenes = set(scene_ids[:val_scene_count])
    train_records = [record for record in records if record.scenario_id not in val_scenes]
    val_records = [record for record in records if record.scenario_id in val_scenes]
    return train_records, val_records


def split_metadata(train_records: Sequence[CandidatePairRecord], val_records: Sequence[CandidatePairRecord]) -> Dict[str, object]:
    train_scenes = {record.scenario_id for record in train_records}
    val_scenes = {record.scenario_id for record in val_records}
    return {
        "split_strategy": "scene_grouped",
        "num_train_scenes": len(train_scenes),
        "num_val_scenes": len(val_scenes),
        "train_val_scene_overlap": len(train_scenes & val_scenes),
    }


def build_loader(
    cfg,
    records: Sequence[CandidatePairRecord],
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
    dataset = CandidatePairDataset(base_dataset, records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_candidate_pairs,
    )


def run_epoch(
    model: CandidateSelector,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    gap_weight: float,
    chosen_ce_weight: float,
):
    train = optimizer is not None
    model.train(train)

    losses: List[float] = []
    pair_losses: List[float] = []
    ce_losses: List[float] = []
    pair_accs: List[float] = []
    margins: List[float] = []
    chosen_probs: List[float] = []

    desc = "Train" if train else "Eval"
    for batch, candidate_trajs, anchor_trajs, candidate_mask, chosen, rejected, gaps in tqdm(
        loader,
        desc=desc,
        leave=False,
    ):
        batch = batch.to(device)
        candidate_trajs = candidate_trajs.to(device)
        anchor_trajs = anchor_trajs.to(device)
        candidate_mask = candidate_mask.to(device)
        chosen = chosen.to(device)
        rejected = rejected.to(device)
        gaps = gaps.to(device)

        logits = model(batch, candidate_trajs, anchor_trajs, candidate_mask=candidate_mask)
        masked_logits = logits.masked_fill(~candidate_mask, -1e9)
        chosen_score = masked_logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
        rejected_score = masked_logits.gather(1, rejected.unsqueeze(1)).squeeze(1)
        margin = chosen_score - rejected_score
        per_item_pair = -F.logsigmoid(margin)
        if gap_weight > 0:
            weights = (1.0 + gap_weight * gaps).clamp(max=5.0)
            pair_loss = (per_item_pair * weights).mean()
        else:
            pair_loss = per_item_pair.mean()

        ce_loss = torch.zeros((), device=device)
        if chosen_ce_weight > 0:
            ce_loss = F.cross_entropy(masked_logits, chosen)
        loss = pair_loss + chosen_ce_weight * ce_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        probs = torch.softmax(masked_logits, dim=-1)
        losses.append(loss.item())
        pair_losses.append(pair_loss.item())
        ce_losses.append(ce_loss.item())
        pair_accs.append((margin > 0).float().mean().item())
        margins.append(margin.mean().item())
        chosen_probs.append(probs.gather(1, chosen.unsqueeze(1)).mean().item())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pair_loss": float(np.mean(pair_losses)) if pair_losses else 0.0,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "pair_acc": float(np.mean(pair_accs)) if pair_accs else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "chosen_prob": float(np.mean(chosen_probs)) if chosen_probs else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--planner-ckpt", required=True)
    parser.add_argument("--anchor-vocab-path", required=True)
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
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--pair-scope", choices=("any", "same_anchor", "cross_anchor"), default="same_anchor")
    parser.add_argument("--pair-reduce", choices=("all", "scene_best", "anchor_best"), default="anchor_best")
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--gap-weight", type=float, default=0.0)
    parser.add_argument("--chosen-ce-weight", type=float, default=0.0)
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
    model = CandidateSelector(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)

    records, pair_stats = load_candidate_pair_records(
        scored_dir=args.scored_dir,
        data_dir=args.data_dir,
        max_scenes=args.max_scenes,
        pair_scope=args.pair_scope,
        pair_reduce=args.pair_reduce,
        min_score_gap=args.min_score_gap,
    )
    if not records:
        raise RuntimeError(f"no usable candidate pairs loaded from {args.scored_dir}")

    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    pair_stats["num_train_pairs"] = len(train_records)
    pair_stats["num_val_pairs"] = len(val_records)
    pair_stats.update(split_metadata(train_records, val_records))
    (save_dir / "pair_stats.json").write_text(
        json.dumps(pair_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(pair_stats, indent=2, ensure_ascii=False))

    train_loader = build_loader(
        cfg,
        train_records,
        args.data_dir,
        save_dir / "candidate_pair_train_list.json",
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
            save_dir / "candidate_pair_val_list.json",
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
    best_path = save_dir / "anchor_candidate_selector_pairwise_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer,
            gap_weight=args.gap_weight,
            chosen_ce_weight=args.chosen_ce_weight,
        )
        scheduler.step()
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                args.device,
                None,
                gap_weight=args.gap_weight,
                chosen_ce_weight=args.chosen_ce_weight,
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
            f"train pair={train_metrics['pair_loss']:.4f} acc={train_metrics['pair_acc']:.3f} "
            f"val acc={val_metrics.get('pair_acc', 0.0):.3f} "
            f"val chosen_prob={val_metrics.get('chosen_prob', 0.0):.3f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "pair_stats": pair_stats,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, save_dir / "anchor_candidate_selector_pairwise_latest.pth")
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

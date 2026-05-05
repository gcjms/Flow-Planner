#!/usr/bin/env python3
"""Train a candidate-level selector from scored anchor-conditioned candidates.

Unlike anchor-level soft preference, this model scores each generated candidate
trajectory directly:

    scene + anchor_traj + candidate_traj -> score

Targets are within-scene soft distributions derived from teacher structured
scores written by ``generate_anchor_softpref_candidates.py``.
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
class CandidateSceneRecord:
    scenario_id: str
    scene_relpath: str
    candidate_npz_path: str
    target: np.ndarray
    best_candidate: int
    num_candidates: int
    score_std: float
    score_gap: float
    top_prob: float


class CandidateSelectorDataset(Dataset):
    def __init__(self, base_dataset: Dataset, records: Sequence[CandidateSceneRecord]):
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
            torch.from_numpy(record.target).float(),
            torch.tensor(record.best_candidate, dtype=torch.long),
        )


def collate_candidate_selector(batch):
    samples, candidate_trajs, anchor_trajs, targets, best_indices = zip(*batch)
    batch_size = len(batch)
    max_candidates = max(item.shape[0] for item in candidate_trajs)
    future_steps = candidate_trajs[0].shape[1]
    candidate_dim = candidate_trajs[0].shape[2]
    anchor_dim = anchor_trajs[0].shape[2]

    padded_candidates = torch.zeros(batch_size, max_candidates, future_steps, candidate_dim, dtype=torch.float32)
    padded_anchors = torch.zeros(batch_size, max_candidates, future_steps, anchor_dim, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_candidates, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    for row_idx, (cand, anchor, target) in enumerate(zip(candidate_trajs, anchor_trajs, targets)):
        count = cand.shape[0]
        padded_candidates[row_idx, :count] = cand
        padded_anchors[row_idx, :count] = anchor
        padded_targets[row_idx, :count] = target
        candidate_mask[row_idx, :count] = True

    return (
        collect_batch(list(samples)),
        padded_candidates,
        padded_anchors,
        padded_targets,
        candidate_mask,
        torch.stack(list(best_indices), dim=0),
    )


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = values.astype(np.float64) / temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    return (probs / probs.sum()).astype(np.float32)


def load_candidate_records(
    scored_dir: str,
    data_dir: str,
    target_temp: float,
    max_scenes: int | None,
    min_score_std: float,
    min_score_gap: float,
    min_top_prob: float,
) -> Tuple[List[CandidateSceneRecord], Dict[str, object]]:
    json_paths = sorted(Path(scored_dir).glob("*.json"))
    if max_scenes is not None:
        json_paths = json_paths[:max_scenes]

    records: List[CandidateSceneRecord] = []
    skipped = {
        "no_candidates": 0,
        "bad_scene_path": 0,
        "missing_npz": 0,
        "missing_anchor_trajs": 0,
        "shape_mismatch": 0,
        "low_score_std": 0,
        "low_score_gap": 0,
        "low_top_prob": 0,
    }

    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_infos = payload.get("candidates", [])
        if not candidate_infos:
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

        candidate_npz_path = str(payload.get("source_npz", ""))
        if not candidate_npz_path or not os.path.exists(candidate_npz_path):
            skipped["missing_npz"] += 1
            continue

        try:
            with np.load(candidate_npz_path, allow_pickle=True) as raw:
                candidate_count = int(np.asarray(raw["candidates"]).shape[0])
                if "anchor_trajs" not in raw.files:
                    skipped["missing_anchor_trajs"] += 1
                    continue
        except Exception:
            skipped["missing_npz"] += 1
            continue

        score_arr = np.asarray(
            [float(item["total_score"]) for item in candidate_infos],
            dtype=np.float32,
        )
        if candidate_count != int(score_arr.shape[0]):
            skipped["shape_mismatch"] += 1
            continue

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

        records.append(
            CandidateSceneRecord(
                scenario_id=str(payload.get("scenario_id", path.stem)),
                scene_relpath=scene_relpath,
                candidate_npz_path=candidate_npz_path,
                target=probs,
                best_candidate=int(order[0]),
                num_candidates=int(candidate_count),
                score_std=score_std,
                score_gap=score_gap,
                top_prob=top_prob,
            )
        )

    stats = {
        "scored_dir": scored_dir,
        "data_dir": data_dir,
        "num_input_json": len(json_paths),
        "num_records": len(records),
        "skipped": skipped,
        "target_temp": target_temp,
        "score_std_mean": float(np.mean([r.score_std for r in records])) if records else 0.0,
        "score_gap_mean": float(np.mean([r.score_gap for r in records])) if records else 0.0,
        "top_prob_mean": float(np.mean([r.top_prob for r in records])) if records else 0.0,
        "num_candidates_mean": float(np.mean([r.num_candidates for r in records])) if records else 0.0,
    }
    return records, stats


def split_records(records: Sequence[CandidateSceneRecord], val_fraction: float, seed: int):
    records = list(records)
    rng = random.Random(seed)
    scene_ids = sorted({record.scenario_id for record in records})
    rng.shuffle(scene_ids)
    if val_fraction <= 0 or len(scene_ids) < 2:
        return records, []
    val_scene_count = max(1, int(round(len(scene_ids) * val_fraction)))
    val_scenes = set(scene_ids[:val_scene_count])
    train_records = [record for record in records if record.scenario_id not in val_scenes]
    val_records = [record for record in records if record.scenario_id in val_scenes]
    return train_records, val_records


def split_metadata(train_records: Sequence[CandidateSceneRecord], val_records: Sequence[CandidateSceneRecord]) -> Dict[str, object]:
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
    records: Sequence[CandidateSceneRecord],
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
    dataset = CandidateSelectorDataset(base_dataset, records)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_candidate_selector,
    )


def run_epoch(
    model: CandidateSelector,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
):
    train = optimizer is not None
    model.train(train)

    pref_losses: List[float] = []
    top1_matches: List[float] = []
    target_probs_on_pred: List[float] = []
    mean_best_logit: List[float] = []

    desc = "Train" if train else "Eval"
    for batch, candidate_trajs, anchor_trajs, targets, candidate_mask, best_indices in tqdm(
        loader,
        desc=desc,
        leave=False,
    ):
        batch = batch.to(device)
        candidate_trajs = candidate_trajs.to(device)
        anchor_trajs = anchor_trajs.to(device)
        targets = targets.to(device)
        candidate_mask = candidate_mask.to(device)
        best_indices = best_indices.to(device)

        logits = model(batch, candidate_trajs, anchor_trajs, candidate_mask=candidate_mask)
        masked_logits = logits.masked_fill(~candidate_mask, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        pref_loss = -(targets * log_probs).sum(dim=-1).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            pref_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        pred = masked_logits.argmax(dim=-1)
        top1_matches.append((pred == best_indices).float().mean().item())
        target_probs_on_pred.append(targets.gather(1, pred.unsqueeze(1)).mean().item())
        mean_best_logit.append(masked_logits.gather(1, best_indices.unsqueeze(1)).mean().item())
        pref_losses.append(pref_loss.item())

    return {
        "loss": float(np.mean(pref_losses)) if pref_losses else 0.0,
        "top1_match": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "target_prob_on_pred": float(np.mean(target_probs_on_pred)) if target_probs_on_pred else 0.0,
        "best_logit": float(np.mean(mean_best_logit)) if mean_best_logit else 0.0,
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-temp", type=float, default=0.5)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-score-std", type=float, default=0.0)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--min-top-prob", type=float, default=0.0)
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

    records, candidate_stats = load_candidate_records(
        scored_dir=args.scored_dir,
        data_dir=args.data_dir,
        target_temp=args.target_temp,
        max_scenes=args.max_scenes,
        min_score_std=args.min_score_std,
        min_score_gap=args.min_score_gap,
        min_top_prob=args.min_top_prob,
    )
    if not records:
        raise RuntimeError(f"no usable candidate selector records loaded from {args.scored_dir}")

    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    candidate_stats["num_train_records"] = len(train_records)
    candidate_stats["num_val_records"] = len(val_records)
    candidate_stats.update(split_metadata(train_records, val_records))
    (save_dir / "candidate_stats.json").write_text(
        json.dumps(candidate_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(candidate_stats, indent=2, ensure_ascii=False))

    train_loader = build_loader(
        cfg,
        train_records,
        args.data_dir,
        save_dir / "candidate_selector_train_list.json",
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
            save_dir / "candidate_selector_val_list.json",
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
    best_path = save_dir / "anchor_candidate_selector_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args.device, optimizer)
        scheduler.step()
        val_metrics = run_epoch(model, val_loader, args.device, None) if val_loader is not None else {}

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
            f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1_match']:.3f} "
            f"val top1={val_metrics.get('top1_match', 0.0):.3f} "
            f"val prob={val_metrics.get('target_prob_on_pred', 0.0):.3f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "candidate_stats": candidate_stats,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, save_dir / "anchor_candidate_selector_latest.pth")
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

#!/usr/bin/env python3
"""Train an accept/reject gate from closed-loop intervention labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_planner.goal.closed_loop_gate import ClosedLoopGate
from train_anchor_predictor import load_planner, set_seed


@dataclass
class ClosedLoopGateRecord:
    experiment_name: str
    scenario_name: str
    label: int
    reason: str
    trace_path: str


class ClosedLoopGateDataset(Dataset):
    def __init__(self, records: Sequence[ClosedLoopGateRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        trace = _load_single_trace(Path(record.trace_path))
        required = ("scene_features", "selected_candidate_traj", "selected_anchor_traj")
        missing = [key for key in required if key not in trace]
        if missing:
            raise KeyError(
                f"trace missing training payload fields {missing}: {record.trace_path}"
            )
        return {
            "scene_features": torch.tensor(trace["scene_features"], dtype=torch.float32),
            "candidate_traj": torch.tensor(trace["selected_candidate_traj"], dtype=torch.float32),
            "anchor_traj": torch.tensor(trace["selected_anchor_traj"], dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "experiment_name": record.experiment_name,
            "scenario_name": record.scenario_name,
            "reason": record.reason,
        }


def _load_single_trace(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if row.get("gate_reasons") == ["forced_candidate"]:
                return row
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(f"expected one trace row or forced_candidate row in {path}, got {len(rows)}")
    return rows[0]


def _collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scene_features": torch.stack([item["scene_features"] for item in batch], dim=0),
        "candidate_trajs": torch.stack([item["candidate_traj"] for item in batch], dim=0).unsqueeze(1),
        "anchor_trajs": torch.stack([item["anchor_traj"] for item in batch], dim=0).unsqueeze(1),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
        "experiment_names": [str(item["experiment_name"]) for item in batch],
        "scenario_names": [str(item["scenario_name"]) for item in batch],
        "reasons": [str(item["reason"]) for item in batch],
    }


def _trace_path_for_row(row: dict[str, Any], run_root: Path) -> Path:
    return run_root / f"{row['experiment_name']}_trace.jsonl"


def load_records(labels_csv: Path, run_root: Path) -> list[ClosedLoopGateRecord]:
    table = pd.read_csv(labels_csv)
    records: list[ClosedLoopGateRecord] = []
    for row in table.to_dict("records"):
        label_name = str(row.get("closed_loop_label", ""))
        if label_name not in {"accept", "reject"}:
            continue
        trace_path = _trace_path_for_row(row, run_root)
        records.append(
            ClosedLoopGateRecord(
                experiment_name=str(row["experiment_name"]),
                scenario_name=str(row["scenario_name"]),
                label=1 if label_name == "accept" else 0,
                reason=str(row.get("label_reason", "")),
                trace_path=str(trace_path),
            )
        )
    return records


def split_records(
    records: Sequence[ClosedLoopGateRecord],
    val_fraction: float,
    seed: int,
) -> tuple[list[ClosedLoopGateRecord], list[ClosedLoopGateRecord], dict[str, Any]]:
    rng = random.Random(seed)
    scene_ids = sorted({record.scenario_name for record in records})
    rng.shuffle(scene_ids)
    if val_fraction <= 0 or len(scene_ids) < 2:
        train_records = list(records)
        val_records: list[ClosedLoopGateRecord] = []
    else:
        val_count = max(1, int(round(len(scene_ids) * val_fraction)))
        val_count = min(val_count, len(scene_ids) - 1)
        val_scenes = set(scene_ids[:val_count])
        train_records = [record for record in records if record.scenario_name not in val_scenes]
        val_records = [record for record in records if record.scenario_name in val_scenes]
    meta = {
        "split_strategy": "scene_grouped",
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "num_scenes": len(scene_ids),
        "num_train_scenes": len({record.scenario_name for record in train_records}),
        "num_val_scenes": len({record.scenario_name for record in val_records}),
        "train_val_scene_overlap": len(
            {record.scenario_name for record in train_records}
            & {record.scenario_name for record in val_records}
        ),
        "num_accept": int(sum(record.label == 1 for record in records)),
        "num_reject": int(sum(record.label == 0 for record in records)),
    }
    return train_records, val_records, meta


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_epoch(
    model: ClosedLoopGate,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    accs: list[float] = []
    recalls_pos: list[float] = []
    recalls_neg: list[float] = []
    probs_pos: list[float] = []

    desc = "Train" if train else "Eval"
    for batch in tqdm(loader, desc=desc, leave=False):
        scene_features = batch["scene_features"].to(device)
        candidate_trajs = batch["candidate_trajs"].to(device)
        anchor_trajs = batch["anchor_trajs"].to(device)
        labels = batch["labels"].to(device)

        logits = model.score_features(scene_features, candidate_trajs, anchor_trajs)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        probs = torch.sigmoid(logits)
        preds = probs >= 0.5
        labels_bool = labels >= 0.5
        accs.append((preds == labels_bool).float().mean().item())
        if labels_bool.any():
            recalls_pos.append((preds[labels_bool] == labels_bool[labels_bool]).float().mean().item())
            probs_pos.append(probs[labels_bool].mean().item())
        if (~labels_bool).any():
            recalls_neg.append((preds[~labels_bool] == labels_bool[~labels_bool]).float().mean().item())
        losses.append(loss.item())

    return {
        "loss": _safe_mean(losses),
        "acc": _safe_mean(accs),
        "accept_recall": _safe_mean(recalls_pos),
        "reject_recall": _safe_mean(recalls_neg),
        "accept_prob": _safe_mean(probs_pos),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--planner-ckpt", required=True)
    parser.add_argument("--anchor-vocab-path", required=True)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--anchor-state-dim", type=int, default=3)
    parser.add_argument("--anchor-token-num", type=int, default=4)
    parser.add_argument("--anchor-attn-heads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.labels_csv, args.run_root)
    if not records:
        raise RuntimeError(f"no accept/reject labels loaded from {args.labels_csv}")

    train_records, val_records, split_meta = split_records(records, args.val_fraction, args.seed)
    if not train_records:
        raise RuntimeError("scene split produced no training records")
    (args.save_dir / "split_meta.json").write_text(
        json.dumps(split_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_meta, indent=2, ensure_ascii=False))

    planner, _ = load_planner(
        args.planner_config,
        args.planner_ckpt,
        args.anchor_vocab_path,
        args.device,
        anchor_state_dim=args.anchor_state_dim,
        anchor_token_num=args.anchor_token_num,
        anchor_attn_heads=args.anchor_attn_heads,
    )
    model = ClosedLoopGate(
        planner_backbone=planner,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)

    train_loader = DataLoader(
        ClosedLoopGateDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = None
    if val_records:
        val_loader = DataLoader(
            ClosedLoopGateDataset(val_records),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_collate,
        )

    if args.pos_weight is None:
        num_pos = max(1, sum(record.label == 1 for record in train_records))
        num_neg = max(1, sum(record.label == 0 for record in train_records))
        pos_weight_value = num_neg / num_pos
    else:
        pos_weight_value = float(args.pos_weight)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=args.device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_path = args.save_dir / "closed_loop_gate_best.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args.device, optimizer, pos_weight)
        scheduler.step()
        val_metrics = (
            run_epoch(model, val_loader, args.device, None, pos_weight)
            if val_loader is not None
            else {}
        )
        metric = val_metrics.get("acc", train_metrics["acc"])
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "pos_weight": float(pos_weight_value),
        }
        history.append(record)
        print(
            f"[epoch {epoch}] "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} "
            f"val acc={val_metrics.get('acc', 0.0):.3f} "
            f"val reject_recall={val_metrics.get('reject_recall', 0.0):.3f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "split_meta": split_meta,
            "args": vars(args),
            "planner_config": args.planner_config,
            "planner_ckpt": args.planner_ckpt,
            "anchor_vocab_path": args.anchor_vocab_path,
        }
        torch.save(payload, args.save_dir / "closed_loop_gate_latest.pth")
        if metric >= best_metric:
            best_metric = metric
            torch.save(payload, best_path)
        (args.save_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

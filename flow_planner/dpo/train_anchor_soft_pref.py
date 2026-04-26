"""
Anchor soft preference distillation for Flow-Planner.

This is the anchor-conditioned analogue of ``train_soft_pref.py`` from the
goal branch. It learns a scene-level soft ranking over all generated anchor
candidates instead of forcing ambiguous safe trajectories into hard
chosen/rejected labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flow_planner.dpo.analyze_candidate_modes import ensure_candidates_shape
from flow_planner.dpo.dpo_loss import FlowMatchingDPOLoss
from flow_planner.dpo.lora import (
    get_lora_params,
    inject_lora,
    merge_lora,
    print_lora_summary,
    save_lora,
)
from flow_planner.dpo.train_dpo import (
    attach_anchor_to_decoder_inputs,
    load_flow_planner,
    prepare_encoder_outputs,
)

logger = logging.getLogger(__name__)


def _scene_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return values - mean
    return (values - mean) / std


def _softmax_np(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32) / max(float(temperature), 1e-6)
    scaled = scaled - float(np.max(scaled))
    probs = np.exp(scaled)
    denom = float(np.sum(probs))
    if denom <= 0.0:
        return np.full_like(probs, 1.0 / max(len(probs), 1), dtype=np.float32)
    return (probs / denom).astype(np.float32)


def _resolve_candidate_npz(
    scenario_id: str,
    scene_payload: Dict[str, object],
    candidates_dir: Optional[str],
) -> Optional[str]:
    source_npz = scene_payload.get("source_npz")
    if isinstance(source_npz, str) and os.path.exists(source_npz):
        return source_npz
    if candidates_dir:
        fallback = os.path.join(candidates_dir, f"{scenario_id}_candidates.npz")
        if os.path.exists(fallback):
            return fallback
    return None


def _load_scene_condition(scene_path: str) -> Dict[str, torch.Tensor]:
    scene = np.load(scene_path, allow_pickle=True)
    return {
        "neighbor_past": torch.from_numpy(scene["neighbor_agents_past"]).float(),
        "lanes": torch.from_numpy(scene["lanes"]).float(),
        "lanes_speedlimit": torch.from_numpy(scene["lanes_speed_limit"]).float(),
        "lanes_has_speedlimit": torch.from_numpy(scene["lanes_has_speed_limit"]).bool(),
        "routes": torch.from_numpy(scene["route_lanes"]).float(),
        "map_objects": torch.from_numpy(scene["static_objects"]).float(),
        "ego_current": torch.from_numpy(scene["ego_current_state"]).float(),
    }


class AnchorSoftPreferenceSceneDataset(Dataset):
    """Load per-scene anchor candidate sets and build soft target distributions."""

    def __init__(
        self,
        scored_dir: str,
        scene_dir: str,
        state_dim: int = 4,
        candidates_dir: Optional[str] = None,
        max_scenes: Optional[int] = None,
        min_candidates: int = 2,
        target_temp: float = 1.0,
        gt_weight: float = 0.2,
        score_weight: float = 1.0,
        gt_ade_weight: float = 0.5,
        gt_fde_weight: float = 1.0,
        require_anchors: bool = True,
        min_score_std: float = 0.0,
        min_top_prob: float = 0.0,
    ):
        self.state_dim = state_dim
        self.scene_dir = scene_dir
        self.target_temp = target_temp
        self.gt_weight = gt_weight
        self.score_weight = score_weight
        self.gt_ade_weight = gt_ade_weight
        self.gt_fde_weight = gt_fde_weight
        self.require_anchors = require_anchors
        self.min_score_std = float(min_score_std)
        self.min_top_prob = float(min_top_prob)
        self.records: List[Dict[str, object]] = []
        self.num_candidates: Optional[int] = None

        scored_files = sorted(Path(scored_dir).glob("*.json"))
        if max_scenes is not None:
            scored_files = scored_files[:max_scenes]

        skipped = {
            "missing_candidate_npz": 0,
            "missing_scene_npz": 0,
            "missing_candidate_infos": 0,
            "missing_anchors": 0,
            "inconsistent_k": 0,
            "too_few_candidates": 0,
            "low_score_std": 0,
            "low_top_prob": 0,
        }

        for scored_path in scored_files:
            with open(scored_path, "r", encoding="utf-8") as fp:
                scene_payload = json.load(fp)

            scenario_id = str(scene_payload.get("scenario_id", scored_path.stem))
            candidate_npz = _resolve_candidate_npz(scenario_id, scene_payload, candidates_dir)
            if candidate_npz is None:
                skipped["missing_candidate_npz"] += 1
                continue

            scene_path = os.path.join(scene_dir, f"{scenario_id}.npz")
            if not os.path.exists(scene_path):
                skipped["missing_scene_npz"] += 1
                continue

            raw = np.load(candidate_npz, allow_pickle=True)
            candidates = ensure_candidates_shape(raw["candidates"]).astype(np.float32)
            if candidates.shape[0] < min_candidates:
                skipped["too_few_candidates"] += 1
                continue

            if "anchor_trajs" not in raw.files:
                skipped["missing_anchors"] += 1
                if require_anchors:
                    continue
                anchor_trajs = None
            else:
                anchor_trajs = np.asarray(raw["anchor_trajs"], dtype=np.float32)

            candidate_infos = scene_payload.get("candidates", [])
            info_by_idx = {
                int(item["candidate_idx"]): item
                for item in candidate_infos
                if isinstance(item, dict) and "candidate_idx" in item
            }
            if any(idx not in info_by_idx for idx in range(candidates.shape[0])):
                skipped["missing_candidate_infos"] += 1
                continue

            if self.num_candidates is None:
                self.num_candidates = int(candidates.shape[0])
            elif int(candidates.shape[0]) != self.num_candidates:
                skipped["inconsistent_k"] += 1
                continue

            gt_similarity_raw: List[float] = []
            score_raw: List[float] = []
            collided: List[float] = []
            for idx in range(candidates.shape[0]):
                info = info_by_idx[idx]
                metrics = info.get("metrics", {})
                ade = float(metrics.get("ade", 0.0))
                fde = float(metrics.get("fde", 0.0))
                gt_similarity_raw.append(
                    -(self.gt_ade_weight * ade + self.gt_fde_weight * fde)
                )
                score_raw.append(float(info.get("total_score", 0.0)))
                collided.append(float(metrics.get("collided", 0.0)))

            gt_similarity = _scene_zscore(np.asarray(gt_similarity_raw, dtype=np.float32))
            score_values = _scene_zscore(np.asarray(score_raw, dtype=np.float32))
            combined = self.gt_weight * gt_similarity + self.score_weight * score_values
            target_probs = _softmax_np(combined, self.target_temp)
            score_std = float(np.asarray(score_raw, dtype=np.float32).std())
            top_prob = float(np.max(target_probs))
            if score_std < self.min_score_std:
                skipped["low_score_std"] += 1
                continue
            if top_prob < self.min_top_prob:
                skipped["low_top_prob"] += 1
                continue

            record = {
                "scenario_id": scenario_id,
                "scene_path": scene_path,
                "candidate_npz": candidate_npz,
                "trajectories": candidates[:, :, : self.state_dim],
                "anchor_trajs": anchor_trajs,
                "target_probs": target_probs,
                "top_idx": int(np.argmax(target_probs)),
                "gt_similarity": gt_similarity.astype(np.float32),
                "score_values": score_values.astype(np.float32),
                "raw_scores": np.asarray(score_raw, dtype=np.float32),
                "collided": np.asarray(collided, dtype=np.float32),
                "score_std": score_std,
                "top_prob": top_prob,
            }
            self.records.append(record)

        if not self.records:
            raise RuntimeError(
                "No valid anchor soft preference scenes were loaded. "
                "Check scored_dir, scene_dir, and candidate paths."
            )

        logger.info(
            "Loaded %d anchor soft-preference scenes (K=%d).",
            len(self.records),
            self.num_candidates or -1,
        )
        logger.info("Skipped scenes: %s", skipped)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.records[idx]
        condition = _load_scene_condition(str(record["scene_path"]))
        return {
            "scenario_id": record["scenario_id"],
            "trajectories": torch.from_numpy(np.asarray(record["trajectories"], dtype=np.float32)).float(),
            "anchor_trajs": torch.from_numpy(np.asarray(record["anchor_trajs"], dtype=np.float32)).float(),
            "target_probs": torch.from_numpy(np.asarray(record["target_probs"], dtype=np.float32)).float(),
            "top_idx": int(record["top_idx"]),
            "gt_similarity": torch.from_numpy(np.asarray(record["gt_similarity"], dtype=np.float32)).float(),
            "score_values": torch.from_numpy(np.asarray(record["score_values"], dtype=np.float32)).float(),
            "raw_scores": torch.from_numpy(np.asarray(record["raw_scores"], dtype=np.float32)).float(),
            "collided": torch.from_numpy(np.asarray(record["collided"], dtype=np.float32)).float(),
            "condition": condition,
        }


def collate_anchor_soft_preferences(batch: List[Dict[str, object]]) -> Dict[str, object]:
    trajectories = torch.stack([item["trajectories"] for item in batch])
    anchor_trajs = torch.stack([item["anchor_trajs"] for item in batch])
    target_probs = torch.stack([item["target_probs"] for item in batch])
    gt_similarity = torch.stack([item["gt_similarity"] for item in batch])
    score_values = torch.stack([item["score_values"] for item in batch])
    raw_scores = torch.stack([item["raw_scores"] for item in batch])
    collided = torch.stack([item["collided"] for item in batch])
    top_idx = torch.tensor([item["top_idx"] for item in batch], dtype=torch.long)
    scenario_ids = [str(item["scenario_id"]) for item in batch]

    conditions: Dict[str, object] = {}
    keys = batch[0]["condition"].keys()
    for key in keys:
        tensors = [item["condition"][key] for item in batch if key in item["condition"]]
        if len(tensors) == len(batch):
            try:
                conditions[key] = torch.stack(tensors)
            except RuntimeError:
                conditions[key] = tensors
        elif len(tensors) > 0:
            conditions[key] = tensors

    return {
        "scenario_ids": scenario_ids,
        "trajectories": trajectories,
        "anchor_trajs": anchor_trajs,
        "target_probs": target_probs,
        "top_idx": top_idx,
        "gt_similarity": gt_similarity,
        "score_values": score_values,
        "raw_scores": raw_scores,
        "collided": collided,
        "condition": conditions,
    }


def compute_anchor_candidate_log_probs(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    anchor_trajs: torch.Tensor,
    encoder_outputs: Dict[str, object],
    logprob_helper: FlowMatchingDPOLoss,
    action_len: int,
    action_overlap: int,
    data_processor,
    device: str,
) -> torch.Tensor:
    batch_size, num_candidates = trajectories.shape[:2]
    log_probs: List[torch.Tensor] = []
    for idx in range(num_candidates):
        candidate = trajectories[:, idx].to(device)
        anchor_traj = anchor_trajs[:, idx].to(device)
        conditioned = attach_anchor_to_decoder_inputs(encoder_outputs, anchor_traj, device)
        candidate_log_prob = logprob_helper.compute_log_prob_multi_t(
            model=model,
            trajectory=candidate,
            encoder_outputs=conditioned,
            action_len=action_len,
            action_overlap=action_overlap,
            data_processor=data_processor,
            num_samples=logprob_helper.num_t_samples,
        )
        if candidate_log_prob.shape[0] != batch_size:
            raise RuntimeError(
                f"Expected batch size {batch_size} for candidate {idx}, "
                f"got {candidate_log_prob.shape[0]}"
            )
        log_probs.append(candidate_log_prob)
    return torch.stack(log_probs, dim=1)


def compute_soft_distill_loss(
    log_probs: torch.Tensor,
    target_probs: torch.Tensor,
    top_idx: torch.Tensor,
    top1_weight: float = 0.0,
    ref_log_probs: Optional[torch.Tensor] = None,
    ref_kl_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    policy_log_dist = F.log_softmax(log_probs, dim=-1)
    distill_loss = -(target_probs * policy_log_dist).sum(dim=-1).mean()

    top1_loss = log_probs.new_zeros(())
    if top1_weight > 0.0:
        top1_loss = -log_probs.gather(1, top_idx.unsqueeze(1)).mean()

    ref_kl = log_probs.new_zeros(())
    if ref_log_probs is not None and ref_kl_weight > 0.0:
        ref_log_dist = F.log_softmax(ref_log_probs, dim=-1)
        ref_dist = ref_log_dist.exp()
        ref_kl = (ref_dist * (ref_log_dist - policy_log_dist)).sum(dim=-1).mean()

    loss = distill_loss + top1_weight * top1_loss + ref_kl_weight * ref_kl

    teacher_top1 = target_probs.argmax(dim=-1)
    policy_top1 = log_probs.argmax(dim=-1)
    metrics = {
        "soft/loss": float(loss.item()),
        "soft/loss_distill": float(distill_loss.item()),
        "soft/loss_top1": float(top1_loss.item()),
        "soft/loss_ref_kl": float(ref_kl.item()),
        "soft/top1_match": float((policy_top1 == teacher_top1).float().mean().item()),
        "soft/target_entropy": float(
            (-(target_probs * torch.log(target_probs.clamp_min(1e-8))).sum(dim=-1).mean()).item()
        ),
        "soft/log_prob_mean": float(log_probs.mean().item()),
        "soft/log_prob_top1": float(log_probs.gather(1, teacher_top1.unsqueeze(1)).mean().item()),
    }
    return loss, metrics


def train_anchor_soft_preferences(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(args.output_dir, "tb_logs")
        tb_writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard: %s", tb_dir)
    except ImportError:
        logger.warning("TensorBoard not available, logging to console only")

    logger.info("=" * 60)
    logger.info("Loading policy model...")
    policy_model = load_flow_planner(
        args.config_path,
        args.ckpt_path,
        device,
        anchor_vocab_path=args.anchor_vocab_path,
    )

    ref_model = None
    if args.ref_kl_weight > 0.0:
        logger.info("Creating frozen reference model for distribution drift control...")
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    logger.info("=" * 60)
    logger.info("Injecting LoRA into decoder...")
    target_modules = args.lora_targets.split(",") if args.lora_targets else None
    inject_lora(
        policy_model.model_decoder,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    policy_model = policy_model.to(device)

    for param in policy_model.parameters():
        param.requires_grad = False
    lora_params = get_lora_params(policy_model.model_decoder)
    for param in lora_params:
        param.requires_grad = True

    print_lora_summary(policy_model.model_decoder)
    trainable_count = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in policy_model.parameters())
    logger.info(
        "Trainable: %s / %s (%.2f%%)",
        f"{trainable_count:,}",
        f"{total_count:,}",
        100.0 * trainable_count / max(total_count, 1),
    )

    logger.info("=" * 60)
    logger.info("Loading anchor scored scenes...")
    dataset = AnchorSoftPreferenceSceneDataset(
        scored_dir=args.scored_dir,
        scene_dir=args.scene_dir,
        state_dim=policy_model.planner_params["state_dim"],
        candidates_dir=args.candidates_dir,
        max_scenes=args.max_scenes,
        min_candidates=args.min_candidates,
        target_temp=args.target_temp,
        gt_weight=args.gt_weight,
        score_weight=args.score_weight,
        gt_ade_weight=args.gt_ade_weight,
        gt_fde_weight=args.gt_fde_weight,
        require_anchors=args.require_anchors,
        min_score_std=args.min_score_std,
        min_top_prob=args.min_top_prob,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_anchor_soft_preferences,
    )

    optimizer = AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    logprob_helper = FlowMatchingDPOLoss(
        beta=0.1,
        sft_weight=0.0,
        num_t_samples=args.num_t_samples,
    )
    action_len = policy_model.planner_params["action_len"]
    action_overlap = policy_model.planner_params["action_overlap"]

    logger.info("=" * 60)
    logger.info("Starting anchor soft preference distillation:")
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Learning rate: %.2e", args.lr)
    logger.info("  Scenes: %d", len(dataset))
    logger.info("  Candidates per scene: %d", dataset.num_candidates or -1)
    logger.info("=" * 60)

    global_step = 0
    best_top1_match = -1.0
    start_time = time.time()

    for epoch in range(args.epochs):
        policy_model.train()
        epoch_losses: List[float] = []
        epoch_top1: List[float] = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in pbar:
            trajectories = batch["trajectories"].to(device)
            anchor_trajs = batch["anchor_trajs"].to(device)
            target_probs = batch["target_probs"].to(device)
            top_idx = batch["top_idx"].to(device)

            with torch.no_grad():
                encoder_outputs = prepare_encoder_outputs(policy_model, batch["condition"], device)

            log_probs = compute_anchor_candidate_log_probs(
                model=policy_model,
                trajectories=trajectories,
                anchor_trajs=anchor_trajs,
                encoder_outputs=encoder_outputs,
                logprob_helper=logprob_helper,
                action_len=action_len,
                action_overlap=action_overlap,
                data_processor=policy_model.data_processor,
                device=device,
            )

            ref_log_probs = None
            if ref_model is not None and args.ref_kl_weight > 0.0:
                with torch.no_grad():
                    ref_log_probs = compute_anchor_candidate_log_probs(
                        model=ref_model,
                        trajectories=trajectories,
                        anchor_trajs=anchor_trajs,
                        encoder_outputs=encoder_outputs,
                        logprob_helper=logprob_helper,
                        action_len=action_len,
                        action_overlap=action_overlap,
                        data_processor=policy_model.data_processor,
                        device=device,
                    )

            loss, metrics = compute_soft_distill_loss(
                log_probs=log_probs,
                target_probs=target_probs,
                top_idx=top_idx,
                top1_weight=args.top1_weight,
                ref_log_probs=ref_log_probs,
                ref_kl_weight=args.ref_kl_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_losses.append(metrics["soft/loss"])
            epoch_top1.append(metrics["soft/top1_match"])
            metrics["soft/grad_norm"] = float(grad_norm.item())

            pbar.set_postfix(
                {
                    "loss": f"{metrics['soft/loss']:.4f}",
                    "match": f"{metrics['soft/top1_match']:.2f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if tb_writer and global_step % args.log_every == 0:
                for key, value in metrics.items():
                    tb_writer.add_scalar(key, value, global_step)
                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        avg_top1 = float(np.mean(epoch_top1)) if epoch_top1 else float("nan")
        elapsed = (time.time() - start_time) / 60.0
        logger.info(
            "Epoch %d/%d | Loss: %.4f | Top1 match: %.2f%% | Time: %.1fmin",
            epoch + 1,
            args.epochs,
            avg_loss,
            100.0 * avg_top1,
            elapsed,
        )

        ckpt_path = os.path.join(args.output_dir, f"lora_epoch_{epoch + 1}.pt")
        save_lora(
            policy_model.model_decoder,
            ckpt_path,
            extra_info={
                "epoch": epoch + 1,
                "loss": avg_loss,
                "top1_match": avg_top1,
                "global_step": global_step,
                "args": vars(args),
            },
        )

        if avg_top1 > best_top1_match:
            best_top1_match = avg_top1
            best_path = os.path.join(args.output_dir, "lora_best.pt")
            save_lora(
                policy_model.model_decoder,
                best_path,
                extra_info={"epoch": epoch + 1, "top1_match": best_top1_match},
            )
            logger.info("  -> New best top1 match: %.2f%%", 100.0 * best_top1_match)

        if tb_writer:
            tb_writer.add_scalar("epoch/loss", avg_loss, epoch + 1)
            tb_writer.add_scalar("epoch/top1_match", avg_top1, epoch + 1)

    total_time = (time.time() - start_time) / 60.0
    logger.info("=" * 60)
    logger.info("Anchor soft preference distillation complete!")
    logger.info("  Total time: %.1f min", total_time)
    logger.info("  Best top1 match: %.2f%%", 100.0 * best_top1_match)
    logger.info("  Output: %s", args.output_dir)
    logger.info("=" * 60)

    if args.save_merged:
        logger.info("Merging LoRA weights and saving full model...")
        merge_lora(policy_model.model_decoder)
        merged_path = os.path.join(args.output_dir, "model_anchor_softpref_merged.pth")
        raw_state_dict = policy_model.state_dict()
        fixed_state_dict = {}
        removed_lora_keys = 0
        for key, value in raw_state_dict.items():
            if ".lora_A" in key or ".lora_B" in key:
                removed_lora_keys += 1
                continue
            fixed_state_dict[key.replace("module.", "")] = value
        torch.save({"state_dict": fixed_state_dict}, merged_path)
        logger.info(
            "Merged model saved to %s (%d keys, removed %d LoRA side keys)",
            merged_path,
            len(fixed_state_dict),
            removed_lora_keys,
        )

    if tb_writer:
        tb_writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anchor soft preference distillation for Flow-Planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--anchor_vocab_path", type=str, required=True)
    parser.add_argument("--scored_dir", type=str, required=True)
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--candidates_dir", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--min_candidates", type=int, default=2)
    parser.add_argument("--require_anchors", action="store_true", default=True)

    parser.add_argument("--output_dir", type=str, default="checkpoints/anchor_soft_pref")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_t_samples", type=int, default=8)

    parser.add_argument("--target_temp", type=float, default=1.0)
    parser.add_argument("--min_score_std", type=float, default=0.0)
    parser.add_argument("--min_top_prob", type=float, default=0.0)
    parser.add_argument("--gt_weight", type=float, default=0.2)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument("--gt_ade_weight", type=float, default=0.5)
    parser.add_argument("--gt_fde_weight", type=float, default=1.0)
    parser.add_argument("--top1_weight", type=float, default=0.1)
    parser.add_argument("--ref_kl_weight", type=float, default=0.0)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_merged", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_anchor_soft_preferences(args)


if __name__ == "__main__":
    main()

"""
DPO 训练脚本
=============
使用 LoRA 对 Flow-Planner Decoder 进行 DPO 微调。

独立脚本，不走 Hydra 系统（与原始 trainer.py 解耦）。

用法：
  python -m flow_planner.dpo.train_dpo \
      --ckpt_path checkpoints/model.pth \
      --config_path checkpoints/model_config.yaml \
      --preference_path dpo_data/preferences.npz \
      --output_dir checkpoints/dpo \
      --epochs 5 \
      --batch_size 8 \
      --lr 1e-5 \
      --lora_rank 4 \
      --lora_alpha 16
"""

import os
import sys
import copy
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from flow_planner.dpo.dpo_loss import FlowMatchingDPOLoss
from flow_planner.dpo.lora import (
    inject_lora, get_lora_params, merge_lora,
    save_lora, load_lora, print_lora_summary,
)

logger = logging.getLogger(__name__)


# ==============================================================
# 偏好对数据集
# ==============================================================

class PreferenceDataset(Dataset):
    """
    加载 generate_preferences 生成的偏好对数据。

    数据格式 (preferences.npz):
      - preferences: array of dicts, 每个 dict 包含:
          - chosen: (T, D) 好轨迹
          - rejected: (T, D) 坏轨迹
          - chosen_score: float
          - rejected_score: float
          - score_gap: float
          - condition: dict of numpy arrays
            - ego_current: (D_ego,)
            - neighbor_past: (M, T_p, D_n)
            - lane: (L, P, D_l)
    """

    def __init__(
        self,
        preference_path: str,
        min_score_gap: float = 2.0,
        max_pairs: Optional[int] = None,
        state_dim: int = 4,
    ):
        """
        Args:
            preference_path: .npz 文件路径
            min_score_gap: 最小分差阈值（过滤噪声偏好对）
            max_pairs: 最多使用多少对（用于调试）
            state_dim: 轨迹的状态维度 (默认 4: x, y, cos_h, sin_h)
        """
        logger.info(f"Loading preferences from {preference_path}")
        data = np.load(preference_path, allow_pickle=True)
        all_prefs = data['preferences']

        # 过滤
        self.pairs = []
        for pref in all_prefs:
            pref_dict = pref if isinstance(pref, dict) else pref.item()
            gap = pref_dict.get('score_gap', float('inf'))
            if gap >= min_score_gap:
                self.pairs.append(pref_dict)

        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

        self.state_dim = state_dim
        logger.info(
            f"Loaded {len(self.pairs)} preference pairs "
            f"(filtered from {len(all_prefs)}, min_gap={min_score_gap})"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pref = self.pairs[idx]

        chosen = torch.from_numpy(pref['chosen']).float()      # (T, D)
        rejected = torch.from_numpy(pref['rejected']).float()  # (T, D)

        # 确保只取前 state_dim 维 (x, y, cos_h, sin_h)
        chosen = chosen[:, :self.state_dim]
        rejected = rejected[:, :self.state_dim]

        # 条件信息
        condition = {}
        if 'condition' in pref:
            cond = pref['condition']
            for key in ['ego_current', 'neighbor_past', 'lane']:
                if key in cond and cond[key] is not None:
                    condition[key] = torch.from_numpy(
                        np.array(cond[key])
                    ).float()

        return {
            'chosen': chosen,
            'rejected': rejected,
            'condition': condition,
            'score_gap': pref.get('score_gap', 0.0),
        }


def collate_preferences(batch: List[dict]) -> dict:
    """
    自定义 collate 函数，处理不同形状的 condition。
    """
    chosen = torch.stack([item['chosen'] for item in batch])       # (B, T, D)
    rejected = torch.stack([item['rejected'] for item in batch])   # (B, T, D)
    score_gaps = torch.tensor([item['score_gap'] for item in batch])

    # 条件：尝试 stack，如果形状不一致则跳过
    conditions = {}
    keys = batch[0]['condition'].keys()
    for key in keys:
        tensors = [item['condition'][key] for item in batch if key in item['condition']]
        if len(tensors) == len(batch):
            try:
                conditions[key] = torch.stack(tensors)
            except RuntimeError:
                # 形状不一致，保持 list
                conditions[key] = tensors
        elif len(tensors) > 0:
            conditions[key] = tensors

    return {
        'chosen': chosen,
        'rejected': rejected,
        'condition': conditions,
        'score_gap': score_gaps,
    }


# ==============================================================
# 模型加载
# ==============================================================

def load_flow_planner(
    config_path: str,
    ckpt_path: str,
    device: str = 'cuda',
) -> nn.Module:
    """
    加载 FlowPlanner 模型。

    Args:
        config_path: 模型配置文件路径
        ckpt_path: 模型权重文件路径
        device: 设备

    Returns:
        model: FlowPlanner 模型
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    logger.info(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)

    logger.info("Instantiating model...")
    model = instantiate(cfg.model)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # 适配不同的 checkpoint 格式
    if 'ema_state_dict' in ckpt:
        state_dict = {
            k.replace('module.', ''): v
            for k, v in ckpt['ema_state_dict'].items()
        }
    elif 'state_dict' in ckpt:
        state_dict = {
            k.replace('module.', ''): v
            for k, v in ckpt['state_dict'].items()
        }
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")

    return model


def prepare_encoder_outputs(
    model: nn.Module,
    condition: dict,
    device: str = 'cuda',
) -> dict:
    """
    运行 encoder 获取 decoder 所需的条件信息。

    在 DPO 训练中，encoder 的输出在 chosen 和 rejected 间共享，
    因此对每个 batch 只需运行一次 encoder。

    Args:
        model: FlowPlanner 模型
        condition: 来自 PreferenceDataset 的条件数据
        device: 设备

    Returns:
        encoder_outputs: decoder 所需的字典
    """
    from flow_planner.data.dataset.nuplan import NuPlanDataSample

    # 构建 NuPlanDataSample（简化版，只包含 encoder 需要的字段）
    B = condition.get('ego_current', condition.get('neighbor_past')).shape[0]

    # 准备 cfg_flags (DPO 训练不需要 CFG, 全部设为 conditioned)
    cfg_flags = torch.ones(B, device=device, dtype=torch.int32)

    # 将条件移到设备
    inputs = {}
    for key, val in condition.items():
        if isinstance(val, torch.Tensor):
            inputs[key] = val.to(device)
        elif isinstance(val, list):
            inputs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in val]

    # 构建 encoder 输入
    encoder_inputs = model.extract_encoder_inputs(inputs)

    # encoder 前向
    with torch.no_grad():
        encoder_outputs = model.encoder(**encoder_inputs)

    # 构建 decoder 所需的 model_extra
    inputs['cfg_flags'] = cfg_flags
    decoder_inputs = model.extract_decoder_inputs(encoder_outputs, inputs)

    return decoder_inputs


# ==============================================================
# 训练循环
# ==============================================================

def train_dpo(args):
    """DPO 微调主训练循环"""

    # ---- 设置 ----
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, 'tb_logs')
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard: {tb_dir}")
    except ImportError:
        logger.warning("TensorBoard not available, logging to console only")

    # ---- 加载模型 ----
    logger.info("=" * 60)
    logger.info("Loading policy model...")
    policy_model = load_flow_planner(args.config_path, args.ckpt_path, device)

    logger.info("Creating reference model (frozen copy)...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ---- 注入 LoRA ----
    logger.info("=" * 60)
    logger.info("Injecting LoRA into policy model decoder...")

    target_modules = args.lora_targets.split(',') if args.lora_targets else None
    lora_info = inject_lora(
        policy_model.model_decoder,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    # 冻结所有参数，只训练 LoRA
    for p in policy_model.parameters():
        p.requires_grad = False
    lora_params = get_lora_params(policy_model.model_decoder)
    for p in lora_params:
        p.requires_grad = True

    print_lora_summary(policy_model.model_decoder)

    trainable_count = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in policy_model.parameters())
    logger.info(f"Trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # ---- 加载数据 ----
    logger.info("=" * 60)
    logger.info("Loading preference dataset...")
    dataset = PreferenceDataset(
        preference_path=args.preference_path,
        min_score_gap=args.min_score_gap,
        max_pairs=args.max_pairs,
        state_dim=policy_model.planner_params['state_dim'],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_preferences,
    )

    # ---- DPO Loss ----
    dpo_loss_fn = FlowMatchingDPOLoss(beta=args.beta)

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(dataloader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    # 获取 action token 参数
    action_len = policy_model.planner_params['action_len']
    action_overlap = policy_model.planner_params['action_overlap']

    # ---- 训练 ----
    logger.info("=" * 60)
    logger.info(f"Starting DPO training:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Beta (DPO temp): {args.beta}")
    logger.info(f"  LoRA rank: {args.lora_rank}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info("=" * 60)

    global_step = 0
    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        policy_model.train()
        epoch_metrics = {
            'loss': [], 'delta': [], 'accuracy': [],
            'chosen_reward': [], 'rejected_reward': [],
        }

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=True,
        )

        for batch in pbar:
            chosen = batch['chosen'].to(device)      # (B, T, D)
            rejected = batch['rejected'].to(device)   # (B, T, D)
            condition = batch['condition']

            # Encoder 前向（共享，不计算梯度）
            with torch.no_grad():
                encoder_outputs = prepare_encoder_outputs(
                    policy_model, condition, device
                )

            # DPO Loss
            loss, metrics = dpo_loss_fn(
                model=policy_model,
                ref_model=ref_model,
                chosen=chosen,
                rejected=rejected,
                encoder_outputs=encoder_outputs,
                action_len=action_len,
                action_overlap=action_overlap,
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            metrics['dpo/grad_norm'] = grad_norm.item()

            optimizer.step()
            scheduler.step()

            # 记录指标
            global_step += 1
            epoch_metrics['loss'].append(metrics['dpo/loss'])
            epoch_metrics['delta'].append(metrics['dpo/delta_mean'])
            epoch_metrics['accuracy'].append(metrics['dpo/accuracy'])
            epoch_metrics['chosen_reward'].append(metrics['dpo/chosen_reward'])
            epoch_metrics['rejected_reward'].append(metrics['dpo/rejected_reward'])

            # 进度条
            pbar.set_postfix({
                'loss': f"{metrics['dpo/loss']:.4f}",
                'acc': f"{metrics['dpo/accuracy']:.2f}",
                'Δ': f"{metrics['dpo/delta_mean']:.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # TensorBoard
            if tb_writer and global_step % args.log_every == 0:
                for key, val in metrics.items():
                    tb_writer.add_scalar(key, val, global_step)
                tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        # ---- Epoch 结束 ----
        avg_loss = np.mean(epoch_metrics['loss'])
        avg_acc = np.mean(epoch_metrics['accuracy'])
        avg_delta = np.mean(epoch_metrics['delta'])

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | "
            f"Δ: {avg_delta:.4f} | "
            f"Time: {elapsed/60:.1f}min"
        )

        # 保存 checkpoint
        lora_ckpt_path = os.path.join(
            args.output_dir, f"lora_epoch_{epoch+1}.pt"
        )
        save_lora(
            policy_model.model_decoder,
            lora_ckpt_path,
            extra_info={
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'delta': avg_delta,
                'global_step': global_step,
                'args': vars(args),
            }
        )

        # 保存最佳模型
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            best_path = os.path.join(args.output_dir, "lora_best.pt")
            save_lora(
                policy_model.model_decoder,
                best_path,
                extra_info={
                    'epoch': epoch + 1,
                    'accuracy': best_accuracy,
                }
            )
            logger.info(f"  → New best accuracy: {best_accuracy:.2%}")

        if tb_writer:
            tb_writer.add_scalar('epoch/loss', avg_loss, epoch + 1)
            tb_writer.add_scalar('epoch/accuracy', avg_acc, epoch + 1)
            tb_writer.add_scalar('epoch/delta', avg_delta, epoch + 1)

    # ---- 训练完成 ----
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"  Total time: {total_time/60:.1f} min")
    logger.info(f"  Best accuracy: {best_accuracy:.2%}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # 保存合并后的模型权重（推理用）
    if args.save_merged:
        logger.info("Merging LoRA weights and saving full model...")
        merge_lora(policy_model.model_decoder)
        merged_path = os.path.join(args.output_dir, "model_dpo_merged.pth")
        torch.save(policy_model.state_dict(), merged_path)
        logger.info(f"Merged model saved to {merged_path}")

    if tb_writer:
        tb_writer.close()


# ==============================================================
# CLI
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DPO Fine-tuning for Flow-Planner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型
    parser.add_argument('--config_path', type=str, required=True,
                        help='Model config YAML path')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Model checkpoint path')

    # 数据
    parser.add_argument('--preference_path', type=str, required=True,
                        help='Preference pairs .npz path')
    parser.add_argument('--min_score_gap', type=float, default=2.0,
                        help='Minimum score gap for preference pairs')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Max preference pairs to use (for debugging)')

    # 训练
    parser.add_argument('--output_dir', type=str, default='checkpoints/dpo',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature')
    parser.add_argument('--num_workers', type=int, default=4)

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_targets', type=str, default=None,
                        help='Comma-separated target module names (default: auto)')

    # 日志
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log to TensorBoard every N steps')
    parser.add_argument('--save_merged', action='store_true',
                        help='Save merged model after training')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_dpo(args)

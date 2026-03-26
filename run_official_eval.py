#!/usr/bin/env python3
"""
官方权重开环验证：全量 Val 数据集
验证 HuggingFace 官方权重 (ttwhy/flow-planner) 的开环 ADE/FDE
使用尽可能多的数据（全部 28845 场景），对比论文声称的效果
"""
import torch, sys, os, json, time, numpy as np
from datetime import datetime
sys.path.insert(0, '/home/gcjms/Flow-Planner')
from omegaconf import OmegaConf
from hydra.utils import instantiate
from flow_planner.data.dataset.nuplan import NuPlanDataset
from flow_planner.data.utils.collect import collect_batch
from torch.utils.data import DataLoader

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

device = 'cuda'
MAX_SAMPLES = 10000  # 10000 场景，~3小时
OUTPUT_DIR = '/home/gcjms/Flow-Planner/risk_outputs/official_eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ BUILD MODEL ============
log("Building model (official weights)...")
cfg = OmegaConf.load('/home/gcjms/Flow-Planner/checkpoints/model_config.yaml')
OmegaConf.update(cfg, "device", device)
OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
model = instantiate(cfg.model, device=device)
ckpt = torch.load('/home/gcjms/Flow-Planner/checkpoints/model.pth', map_location='cpu', weights_only=True)
model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
model = model.to(device).eval()
log(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ============ LOAD DATA ============
log("Loading data...")
VAL_DIR = '/mnt/d/flow_planner_backup/processed/val'
VAL_LIST = '/mnt/d/flow_planner_backup/processed/val/flow_planner_training.json'

with open(VAL_LIST) as f:
    full_list = json.load(f)
log(f"Total val files: {len(full_list)}")

subset = full_list[:MAX_SAMPLES]
temp_json = os.path.join(OUTPUT_DIR, '_temp_list.json')
with open(temp_json, 'w') as f:
    json.dump(subset, f)

dataset = NuPlanDataset(
    data_dir=VAL_DIR, data_list=temp_json,
    past_neighbor_num=32, predicted_neighbor_num=0,
    future_len=80, future_downsampling_method='uniform',
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                    drop_last=False, collate_fn=collect_batch)
log(f"Dataset: {len(dataset)} samples")

# ============ EVALUATION ============
log("=" * 60)
log("Starting Official Weights Open-Loop Evaluation")
log(f"CFG weights to test: [1.5, 2.5] (with and without CFG)")
log("=" * 60)

# Test multiple CFG configs
configs = [
    ("no_cfg", False, None),
    ("w=1.5", True, 1.5),
    ("w=2.5", True, 2.5),
]

all_ades = {name: [] for name, _, _ in configs}
all_fdes = {name: [] for name, _, _ in configs}
# Per-timestep ADE at 1s, 2s, 4s, 8s
timestep_ades = {name: {1: [], 2: [], 4: [], 8: []} for name, _, _ in configs}

start = time.time()

for sample_idx, data in enumerate(loader):
    data = data.to(device)
    gt = data.ego_future  # (B, T, D)
    
    for name, use_cfg, cfg_weight in configs:
        with torch.no_grad():
            pred = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight)
        
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        
        pred_xy = pred[:, :, :2].float()
        gt_xy = gt[:, :, :2].float()
        min_T = min(pred_xy.shape[1], gt_xy.shape[1])
        
        # Per-point displacement error
        de = torch.norm(pred_xy[:, :min_T, :] - gt_xy[:, :min_T, :], dim=-1)  # (B, T)
        
        # ADE (全程平均)
        ade = de.mean().item()
        all_ades[name].append(ade)
        
        # FDE (终点误差)
        fde = de[:, -1].mean().item()
        all_fdes[name].append(fde)
        
        # Per-timestep ADE: 1s=10pts, 2s=20pts, 4s=40pts, 8s=80pts
        for sec, pts in [(1, 10), (2, 20), (4, 40), (8, 80)]:
            if min_T >= pts:
                t_ade = de[:, :pts].mean().item()
                timestep_ades[name][sec].append(t_ade)
    
    if (sample_idx + 1) % 200 == 0:
        elapsed = time.time() - start
        eta = elapsed / (sample_idx + 1) * (len(dataset) - sample_idx - 1)
        log(f"  {sample_idx+1}/{len(dataset)} ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)")
        for name, _, _ in configs:
            ades = all_ades[name]
            fdes = all_fdes[name]
            log(f"    {name:8s}: ADE={np.mean(ades):.3f}, FDE={np.mean(fdes):.3f}")
    
    torch.cuda.empty_cache()

elapsed = time.time() - start
log(f"\nEvaluation done in {elapsed/60:.1f} min")

# ============ RESULTS ============
log("=" * 60)
log("OFFICIAL WEIGHTS OPEN-LOOP RESULTS")
log("=" * 60)

print(f"\n{'Config':>10s} | {'ADE (m)':>8s} | {'FDE (m)':>8s} | {'ADE@1s':>7s} | {'ADE@2s':>7s} | {'ADE@4s':>7s} | {'ADE@8s':>7s}")
print("-" * 75)
for name, _, _ in configs:
    ade = np.mean(all_ades[name])
    fde = np.mean(all_fdes[name])
    t1 = np.mean(timestep_ades[name][1]) if timestep_ades[name][1] else float('nan')
    t2 = np.mean(timestep_ades[name][2]) if timestep_ades[name][2] else float('nan')
    t4 = np.mean(timestep_ades[name][4]) if timestep_ades[name][4] else float('nan')
    t8 = np.mean(timestep_ades[name][8]) if timestep_ades[name][8] else float('nan')
    print(f"{name:>10s} | {ade:8.3f} | {fde:8.3f} | {t1:7.3f} | {t2:7.3f} | {t4:7.3f} | {t8:7.3f}")

print(f"\n论文参考 (FlowPlanner with refinement):")
print(f"  Val14 NR-CLS: 94.31  (闭环，需要 nuPlan 仿真器)")
print(f"  注意：以上开环 ADE/FDE 不能直接与闭环分数比较")

# ADE distribution
for name, _, _ in configs:
    ades = np.array(all_ades[name])
    print(f"\n{name} ADE 分布:")
    print(f"  Mean={ades.mean():.3f}, Median={np.median(ades):.3f}, Std={ades.std():.3f}")
    print(f"  P25={np.percentile(ades, 25):.3f}, P75={np.percentile(ades, 75):.3f}, P95={np.percentile(ades, 95):.3f}")
    print(f"  <1m: {(ades<1).sum()}/{len(ades)} ({(ades<1).mean()*100:.1f}%)")
    print(f"  <2m: {(ades<2).sum()}/{len(ades)} ({(ades<2).mean()*100:.1f}%)")
    print(f"  >5m: {(ades>5).sum()}/{len(ades)} ({(ades>5).mean()*100:.1f}%)")

# Save
np.savez(os.path.join(OUTPUT_DIR, 'official_eval_results.npz'),
         **{f'ade_{name}': np.array(all_ades[name]) for name, _, _ in configs},
         **{f'fde_{name}': np.array(all_fdes[name]) for name, _, _ in configs},
         configs=[name for name, _, _ in configs],
         num_samples=len(dataset))

log(f"\n{'='*60}")
log(f"✅ OFFICIAL WEIGHTS EVALUATION DONE!")
log(f"{'='*60}")
log(f"Saved: {OUTPUT_DIR}/official_eval_results.npz")

#!/usr/bin/env python3
"""
大规模 Best-of-N 开环验证
2000 场景 × N={1,3,5,10}，预计 ~6 小时（2060 上）
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
NUM_SAMPLES = 2000
N_CANDIDATES = [1, 3, 5, 10]
OUTPUT_DIR = '/home/gcjms/Flow-Planner/risk_outputs/best_of_n'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ BUILD MODEL ============
log("Building model...")
cfg = OmegaConf.load('/home/gcjms/Flow-Planner/checkpoints/model_config.yaml')
OmegaConf.update(cfg, "device", device)
OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
model = instantiate(cfg.model, device=device)
ckpt = torch.load('/home/gcjms/Flow-Planner/checkpoints/model.pth', map_location='cpu', weights_only=True)
model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
model = model.to(device).eval()
log(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

# ============ LOAD DATA ============
log("Loading data...")
VAL_DIR = '/mnt/d/flow_planner_backup/processed/val'
VAL_LIST = '/mnt/d/flow_planner_backup/processed/val/flow_planner_training.json'

with open(VAL_LIST) as f:
    full_list = json.load(f)
subset = full_list[:NUM_SAMPLES]

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
log(f"N candidates: {N_CANDIDATES}")
log(f"Total inferences: {len(dataset) * sum(N_CANDIDATES)}")

# ============ RUN VALIDATION ============
log("=" * 60)
log("Starting Best-of-N Large-Scale Validation")
log("=" * 60)

results = {N: [] for N in N_CANDIDATES}  # ADE per scenario
times = {N: [] for N in N_CANDIDATES}    # inference time per scenario
start = time.time()

for sample_idx, data in enumerate(loader):
    data = data.to(device)
    gt = data.ego_future

    for N in N_CANDIDATES:
        t0 = time.time()
        with torch.no_grad():
            pred = model(data, mode='inference', use_cfg=True, cfg_weight=1.5,
                        num_candidates=N)
        dt = time.time() - t0

        if pred.ndim == 4:
            pred = pred.squeeze(1)
        min_T = min(pred.shape[1], gt.shape[1])
        ade = torch.mean(torch.norm(
            pred[:, :min_T, :2].float() - gt[:, :min_T, :2].float(), dim=-1
        )).item()
        results[N].append(ade)
        times[N].append(dt)

    if (sample_idx + 1) % 100 == 0:
        elapsed = time.time() - start
        eta = elapsed / (sample_idx + 1) * (len(dataset) - sample_idx - 1)
        log(f"  {sample_idx+1}/{len(dataset)} ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)")
        # Print interim results
        for N in N_CANDIDATES:
            ades = results[N]
            log(f"    N={N:2d}: ADE={np.mean(ades):.4f} ± {np.std(ades):.4f}, "
                f"avg_time={np.mean(times[N]):.3f}s")

    torch.cuda.empty_cache()

elapsed = time.time() - start
log(f"\nValidation done in {elapsed/60:.1f} min")

# ============ ANALYSIS ============
log("=" * 60)
log("Analyzing results...")
log("=" * 60)

print(f"\n{'N':>3s} | {'Mean ADE':>9s} | {'Std ADE':>8s} | {'Median ADE':>10s} | {'Mean Time':>10s}")
print("-" * 55)
for N in N_CANDIDATES:
    ades = np.array(results[N])
    ts = np.array(times[N])
    print(f"{N:3d} | {ades.mean():9.4f} | {ades.std():8.4f} | {np.median(ades):10.4f} | {ts.mean():8.3f}s")

# Improvement analysis
base_ades = np.array(results[1])
for N in N_CANDIDATES[1:]:
    n_ades = np.array(results[N])
    improved = (n_ades < base_ades).sum()
    degraded = (n_ades > base_ades).sum()
    avg_diff = (base_ades - n_ades).mean()
    print(f"\nN={N} vs N=1:")
    print(f"  Improved: {improved}/{len(base_ades)} ({improved/len(base_ades)*100:.1f}%)")
    print(f"  Degraded: {degraded}/{len(base_ades)} ({degraded/len(base_ades)*100:.1f}%)")
    print(f"  Avg ADE change: {avg_diff:+.4f}m")

    # Breakdown by ADE range
    low_mask = base_ades < 1.0
    mid_mask = (base_ades >= 1.0) & (base_ades < 3.0)
    high_mask = base_ades >= 3.0
    for mask, label in [(low_mask, "Easy(ADE<1)"), (mid_mask, "Medium(1-3)"), (high_mask, "Hard(ADE>3)")]:
        if mask.sum() > 0:
            diff = (base_ades[mask] - n_ades[mask]).mean()
            imp = (n_ades[mask] < base_ades[mask]).sum()
            print(f"  {label}: {imp}/{mask.sum()} improved, avg change = {diff:+.4f}m")

# Save results
np.savez(os.path.join(OUTPUT_DIR, 'best_of_n_results.npz'),
         **{f'ade_N{N}': np.array(results[N]) for N in N_CANDIDATES},
         **{f'time_N{N}': np.array(times[N]) for N in N_CANDIDATES},
         n_candidates=np.array(N_CANDIDATES),
         num_samples=NUM_SAMPLES)

log(f"\n{'='*60}")
log(f"✅ BEST-OF-N VALIDATION DONE!")
log(f"{'='*60}")
log(f"Saved: {OUTPUT_DIR}/best_of_n_results.npz")
log(f"Total time: {elapsed/60:.1f} min")

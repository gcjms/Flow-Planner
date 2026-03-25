#!/usr/bin/env python3
"""
消噪 Grid Search：每个场景×每个w跑N次取平均，验证w是否真的无关紧要
"""
import torch, sys, os, json, time, numpy as np
sys.path.insert(0, '/home/gcjms/Flow-Planner')
from omegaconf import OmegaConf
from hydra.utils import instantiate
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

device = 'cuda'
NUM_RUNS = 5        # 每个场景×每个w跑5次取平均
MAX_SAMPLES = 500   # 500个场景
W_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
OUTPUT_DIR = '/home/gcjms/Flow-Planner/risk_outputs/denoised'
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
from flow_planner.data.dataset.nuplan import NuPlanDataset
from flow_planner.data.utils.collect import collect_batch
from torch.utils.data import DataLoader

VAL_DIR = '/mnt/d/flow_planner_backup/processed/val'
VAL_LIST = '/mnt/d/flow_planner_backup/processed/val/flow_planner_training.json'

with open(VAL_LIST) as f:
    full_list = json.load(f)
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
log(f"Config: {NUM_RUNS} runs × {len(W_CANDIDATES)} w values = {NUM_RUNS * len(W_CANDIDATES)} inferences per scenario")
log(f"Total: {len(dataset) * NUM_RUNS * len(W_CANDIDATES)} inferences")

# ============ DENOISED GRID SEARCH ============
log("="*60)
log(f"Starting Denoised Grid Search")
log("="*60)

# ade_all[w_idx][sample_idx] = list of ADE from N runs
ade_all = {w: [] for w in W_CANDIDATES}
start = time.time()

for sample_idx, data in enumerate(loader):
    data = data.to(device)
    gt = data.ego_future
    
    for w in W_CANDIDATES:
        run_ades = []
        for run in range(NUM_RUNS):
            with torch.no_grad():
                pred = model(data, mode='inference', use_cfg=True, cfg_weight=w)
            if pred.ndim == 4:
                pred = pred.squeeze(1)
            min_T = min(pred.shape[1], gt.shape[1])
            ade = torch.mean(torch.norm(
                pred[:, :min_T, :2].float() - gt[:, :min_T, :2].float(), dim=-1
            )).item()
            run_ades.append(ade)
        
        avg_ade = np.mean(run_ades)
        ade_all[w].append(avg_ade)
    
    if (sample_idx + 1) % 50 == 0:
        elapsed = time.time() - start
        eta = elapsed / (sample_idx + 1) * (len(dataset) - sample_idx - 1)
        log(f"  {sample_idx+1}/{len(dataset)} ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)")
    
    torch.cuda.empty_cache()

elapsed = time.time() - start
log(f"Grid Search done in {elapsed/60:.1f} min")

# ============ ANALYSIS ============
log("="*60)
log("Analyzing denoised results...")
log("="*60)

# Build denoised ADE matrix
ade_matrix = np.array([ade_all[w] for w in W_CANDIDATES]).T  # (N, 8)
optimal_w_idx = np.argmin(ade_matrix, axis=1)
optimal_w = np.array([W_CANDIDATES[i] for i in optimal_w_idx])

log(f"\n1. Mean ADE per w (denoised, {NUM_RUNS} runs avg):")
for i, w in enumerate(W_CANDIDATES):
    m = ade_matrix[:, i].mean()
    best = "  ← BEST" if m == ade_matrix.mean(axis=0).min() else ""
    log(f"   w={w:.1f}: ADE={m:.4f}{best}")

log(f"\n2. Optimal w distribution (denoised):")
for w in W_CANDIDATES:
    count = (optimal_w == w).sum()
    pct = count / len(optimal_w) * 100
    bar = '█' * int(pct / 2)
    log(f"   w={w:.1f}: {count:4d} ({pct:5.1f}%) {bar}")
log(f"   Mean: {optimal_w.mean():.2f} ± {optimal_w.std():.2f}")

# Chi-squared test for uniformity
from scipy import stats
observed = np.array([(optimal_w == w).sum() for w in W_CANDIDATES])
expected = np.full_like(observed, len(optimal_w) / len(W_CANDIDATES))
chi2, p_value = stats.chisquare(observed, expected)
log(f"\n3. Uniformity test (chi-squared):")
log(f"   chi2={chi2:.2f}, p-value={p_value:.4f}")
if p_value < 0.05:
    log(f"   ✅ Distribution is NOT uniform (p<0.05) → w matters!")
else:
    log(f"   ❌ Distribution is uniform (p≥0.05) → w does not matter")

# ADE improvement
best_ade = ade_matrix.min(axis=1)
global_best_idx = ade_matrix.mean(axis=0).argmin()
global_ade = ade_matrix[:, global_best_idx]
diff = global_ade - best_ade
log(f"\n4. Potential improvement:")
log(f"   Global best w={W_CANDIDATES[global_best_idx]}: ADE={global_ade.mean():.4f}")
log(f"   Oracle adaptive: ADE={best_ade.mean():.4f}")
log(f"   Improvement: {diff.mean():.4f}m ({diff.mean()/global_ade.mean()*100:.1f}%)")

# Feature correlation (extract features and check)
log(f"\n5. Feature correlation (denoised optimal w):")
from flow_planner.risk.risk_features import extract_risk_features_from_npz, RISK_FEATURE_NAMES, NUM_RISK_FEATURES

features_list = []
for fname in subset:
    try:
        f = extract_risk_features_from_npz(os.path.join(VAL_DIR, fname))
        features_list.append(f)
    except:
        features_list.append(np.zeros(NUM_RISK_FEATURES))

features = np.stack(features_list)
for i, name in enumerate(RISK_FEATURE_NAMES):
    valid = (features[:, i] < 900) & np.isfinite(features[:, i])
    if valid.sum() > 50:
        corr = np.corrcoef(features[valid, i], optimal_w[valid])[0, 1]
        star = " ★" if abs(corr) > 0.1 else ""
        log(f"   {str(name):25s}: r={corr:+.4f}{star}")

# Save
np.savez(os.path.join(OUTPUT_DIR, 'denoised_results.npz'),
         ade_matrix=ade_matrix, optimal_w=optimal_w,
         w_candidates=np.array(W_CANDIDATES),
         features=features, feature_names=np.array(RISK_FEATURE_NAMES),
         num_runs=NUM_RUNS, num_samples=MAX_SAMPLES)

log(f"\n{'='*60}")
log(f"🎉 DENOISED GRID SEARCH DONE!")
log(f"{'='*60}")
log(f"Saved: {OUTPUT_DIR}/denoised_results.npz")
log(f"Total time: {elapsed/60:.1f} min")

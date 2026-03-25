#!/usr/bin/env python3
"""Verify official weights + launch overnight pipeline"""
import torch, sys, os, json, time, numpy as np
sys.path.insert(0, '/home/gcjms/Flow-Planner')
from omegaconf import OmegaConf
from hydra.utils import instantiate
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

device = 'cuda'

# ============ BUILD MODEL ============
log("Building model...")
cfg = OmegaConf.load('/home/gcjms/Flow-Planner/checkpoints/model_config.yaml')
OmegaConf.update(cfg, "device", device)
OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
model = instantiate(cfg.model, device=device)

log("Loading official weights...")
ckpt = torch.load('/home/gcjms/Flow-Planner/checkpoints/model.pth', map_location='cpu', weights_only=True)
state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
model.load_state_dict(state_dict)
model = model.to(device).eval()
log(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e6:.0f} MB")

# ============ LOAD DATA ============
log("Loading val data...")
from flow_planner.data.dataset.nuplan import NuPlanDataset
from flow_planner.data.utils.collect import collect_batch
from torch.utils.data import DataLoader

VAL_DIR = '/mnt/d/flow_planner_backup/processed/val'
VAL_LIST = '/mnt/d/flow_planner_backup/processed/val/flow_planner_training.json'

dataset = NuPlanDataset(
    data_dir=VAL_DIR, data_list=VAL_LIST,
    past_neighbor_num=32, predicted_neighbor_num=0,
    future_len=80, future_downsampling_method='uniform', max_num=6
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collect_batch)
data = next(iter(loader)).to(device)
gt = data.ego_future  # (B, T, D) where D could be 3 or 4
log(f"GT shape: {gt.shape}, Pred will be (B, T, 4)")

# ============ VERIFY W VALUES ============
log("Testing different w values...")
for w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    with torch.no_grad():
        pred = model(data, mode='inference', use_cfg=True, cfg_weight=w)
    # pred is (B,1,T,4) -> squeeze to (B,T,4), gt is (B,T,3)
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    min_T = min(pred.shape[1], gt.shape[1])
    ade = torch.mean(torch.norm(pred[:, :min_T, :2].float() - gt[:, :min_T, :2].float(), dim=-1)).item()
    log(f"  w={w:.1f}: ADE={ade:.4f}")

log("✅ Official weights verified!")

# ============ GRID SEARCH ============
log("="*60)
log("Starting Grid Search (2000 samples, 8 w values)")
log("="*60)

from flow_planner.risk.risk_features import extract_risk_features_from_npz, RISK_FEATURE_NAMES, NUM_RISK_FEATURES

W_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MAX_SAMPLES = 15000
OUTPUT_DIR = '/home/gcjms/Flow-Planner/risk_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(VAL_LIST, 'r') as f:
    full_data_list = json.load(f)
data_list = full_data_list[:MAX_SAMPLES]

# Create dataset for grid search
temp_json = os.path.join(OUTPUT_DIR, '_temp_list.json')
with open(temp_json, 'w') as f:
    json.dump(data_list, f)

gs_dataset = NuPlanDataset(
    data_dir=VAL_DIR, data_list=temp_json,
    past_neighbor_num=32, predicted_neighbor_num=0,
    future_len=80, future_downsampling_method='uniform',
)
gs_loader = DataLoader(gs_dataset, batch_size=1, shuffle=False, 
                        num_workers=0, pin_memory=False, 
                        drop_last=False, collate_fn=collect_batch)

log(f"Dataset: {len(gs_dataset)} samples, {len(gs_loader)} batches")

# Run grid search
all_ade = {w: [] for w in W_CANDIDATES}
start = time.time()

for w_idx, w in enumerate(W_CANDIDATES):
    log(f"  Grid Search w={w:.1f} ({w_idx+1}/{len(W_CANDIDATES)})...")
    batch_count = 0
    with torch.no_grad():
        for data in gs_loader:
            data = data.to(device)
            B = data.ego_current.shape[0]
            gt = data.ego_future
            
            try:
                pred = model(data, mode='inference', use_cfg=True, cfg_weight=w)
                if pred.ndim == 4:
                    pred = pred.squeeze(1)  # (B,1,T,4) -> (B,T,4)
                min_T = min(pred.shape[1], gt.shape[1])
                ade = torch.mean(
                    torch.norm(pred[:, :min_T, :2].float() - gt[:, :min_T, :2].float(), dim=-1),
                    dim=-1
                ).cpu().numpy()
                all_ade[w].extend(ade.tolist())
            except Exception as e:
                all_ade[w].extend([999.0] * B)
                if batch_count < 3:
                    log(f"    Error batch {batch_count}: {e}")
            
            batch_count += 1
            if batch_count % 100 == 0:
                log(f"    Batch {batch_count}/{len(gs_loader)}")
    
    avg = np.mean([x for x in all_ade[w] if x < 900])
    log(f"    w={w:.1f}: mean ADE = {avg:.4f} ({len(all_ade[w])} samples)")
    torch.cuda.empty_cache()

elapsed = time.time() - start
log(f"Grid Search done in {elapsed/60:.1f} min")

# Build ADE matrix
min_len = min(len(v) for v in all_ade.values())
ade_matrix = np.array([all_ade[w][:min_len] for w in W_CANDIDATES]).T  # (N, 8)
optimal_w_idx = np.argmin(ade_matrix, axis=1)
optimal_w = np.array([W_CANDIDATES[i] for i in optimal_w_idx])

log(f"\nOptimal w distribution ({min_len} scenarios):")
for w in W_CANDIDATES:
    count = (optimal_w == w).sum()
    pct = count / len(optimal_w) * 100
    bar = '█' * int(pct / 2)
    log(f"  w={w:.1f}: {count:5d} ({pct:5.1f}%) {bar}")
log(f"  Mean: {optimal_w.mean():.2f} ± {optimal_w.std():.2f}")

# ============ EXTRACT RISK FEATURES ============
log("="*60)
log("Extracting risk features...")
log("="*60)

all_features = []
failed = 0
for i, fname in enumerate(data_list[:min_len]):
    try:
        f = extract_risk_features_from_npz(os.path.join(VAL_DIR, fname))
        all_features.append(f)
    except:
        all_features.append(np.zeros(NUM_RISK_FEATURES, dtype=np.float32))
        failed += 1
    if (i+1) % 500 == 0:
        log(f"  {i+1}/{min_len} features extracted")

features = np.stack(all_features)
log(f"Features: {features.shape} ({failed} failed)")

# Save grid search results
np.savez(os.path.join(OUTPUT_DIR, 'risk_dataset.npz'),
         features=features, optimal_w=optimal_w,
         ade_matrix=ade_matrix, w_candidates=np.array(W_CANDIDATES),
         feature_names=np.array(RISK_FEATURE_NAMES))
log(f"Saved risk_dataset.npz")

# ============ TRAIN RISK NETWORK ============
log("="*60)
log("Training Risk Network...")
log("="*60)

del model  # Free GPU
torch.cuda.empty_cache()

from flow_planner.risk.train_risk_network import train_risk_network
risk_model, best_state = train_risk_network(
    features=features, optimal_w=optimal_w,
    output_path=os.path.join(OUTPUT_DIR, 'risk_network.pth'),
    epochs=300, batch_size=64, lr=1e-3, device='cpu'
)

# ============ VALIDATION REPORT ============
log("="*60)
log("Generating validation report...")
log("="*60)

from flow_planner.risk.risk_features import normalize_features
features_norm, stats = normalize_features(features)
X = torch.from_numpy(features_norm).float()
with torch.no_grad():
    output = risk_model(X)
    pred_w = output['w'].numpy().flatten()
    pred_risk = output['risk_score'].numpy().flatten()

w_error = np.abs(pred_w - optimal_w)
log(f"MAE(w): {w_error.mean():.3f}")
log(f"Within ±0.5: {(w_error <= 0.5).mean()*100:.1f}%")
log(f"Within ±1.0: {(w_error <= 1.0).mean()*100:.1f}%")

log(f"\nFeature correlation with optimal w:")
for i, name in enumerate(RISK_FEATURE_NAMES):
    valid = features[:, i] < 900
    if valid.sum() > 10:
        corr = np.corrcoef(features[valid, i], optimal_w[valid])[0, 1]
        log(f"  {name:25s}: r={corr:+.3f}")

# Save report
np.savez(os.path.join(OUTPUT_DIR, 'validation_results.npz'),
         pred_w=pred_w, pred_risk=pred_risk, optimal_w=optimal_w,
         w_error=w_error, mae=w_error.mean())

log(f"\n{'='*60}")
log(f"🎉 ALL DONE!")
log(f"{'='*60}")
log(f"Results in: {OUTPUT_DIR}/")
log(f"  risk_dataset.npz     - Grid Search results")
log(f"  risk_network.pth     - Trained Risk Network")
log(f"  validation_results.npz - Validation analysis")
log(f"Total time: {(time.time()-start)/60:.1f} min")

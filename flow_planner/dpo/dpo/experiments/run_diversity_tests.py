import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/gcjms/Flow-Planner')
from omegaconf import OmegaConf
from hydra.utils import instantiate
from flow_planner.dpo.bev_renderer import BEVRenderer

class DataBunch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def load_scenario(f, device):
    npz = np.load(f)
    print(f"Loading {f}")
    data = {}
    key_mapping = {
        'ego_agent_past': 'ego_past', 'ego_current_state': 'ego_current', 'neighbor_agents_past': 'neighbor_past',
        'ego_agent_future': 'ego_future', 'neighbor_agents_future': 'neighbor_future',
        'lanes': 'lanes', 'route_lanes': 'routes', 'static_objects': 'map_objects',
        'lanes_speed_limit': 'lanes_speedlimit', 'route_lanes_speed_limit': 'routes_speedlimit',
        'lanes_has_speed_limit': 'lanes_has_speedlimit', 'route_lanes_has_speed_limit': 'routes_has_speedlimit',
    }
    for npz_k, model_k in key_mapping.items():
        if npz_k in npz:
            t = torch.from_numpy(npz[npz_k])
            if 'has' in npz_k: val = t.to(torch.bool).unsqueeze(0).to(device)
            else: val = t.to(torch.float32).unsqueeze(0).to(device)
            data[model_k] = val
    return DataBunch(**data), npz

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "/mnt/d/flow_planner_backup/processed/val/"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    np.random.seed(42)
    np.random.shuffle(files)

    # Load Model
    cfg = OmegaConf.load("/home/gcjms/Flow-Planner/checkpoints/model_config.yaml")
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)
    
    model = instantiate(cfg.model)
    ckpt = torch.load("/home/gcjms/Flow-Planner/checkpoints/model.pth", map_location='cpu')
    if 'ema_state_dict' in ckpt: sd = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt: sd = ckpt['state_dict']
    else: sd = ckpt
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    model = model.to(device)
    model.eval()

    renderer = BEVRenderer(image_size=(800, 800), view_range=60.0)

    # Let's find 3 different scenarios: 1 Left turn, 1 Right Turn, 1 Straight Fast
    chosen_files = []
    for f in files:
        npz = np.load(f)
        if 'ego_agent_future' in npz and len(npz['ego_agent_future']) > 0:
            final_pos = npz['ego_agent_future'][-1]
            if len(chosen_files) == 0 and final_pos[1] > 10.0 and final_pos[0] > 5.0: # Left
                chosen_files.append(f)
            elif len(chosen_files) == 1 and final_pos[1] < -10.0 and final_pos[0] > 5.0: # Right
                chosen_files.append(f)
            elif len(chosen_files) == 2 and abs(final_pos[1]) < 3.0 and final_pos[0] > 30.0: # Fast straight
                chosen_files.append(f)
        if len(chosen_files) == 3: break

    os.makedirs("/tmp/diversity_images", exist_ok=True)
    
    with torch.no_grad():
        for scene_idx, f in enumerate(chosen_files):
            data, npz = load_scenario(f, device)
            
            candidates = []
            for i in range(5):
                torch.manual_seed(i * 1000)
                pred = model(data, mode='inference', use_cfg=False, cfg_weight=0.0, num_candidates=1)
                candidates.append(pred.cpu().numpy().reshape(-1, 4))
            
            candidates = np.stack(candidates, axis=0) # (5, 80, 4)
            print(f"\\n--- Scene {scene_idx} ---")
            for i in range(5):
                dist = np.linalg.norm(candidates[i, :, :2] - candidates[0, :, :2], axis=-1).mean()
                print(f"Traj {i} vs Traj 0 mean L2 distance: {dist:.6f} meters")
                print(f"   Endpoint {i}: X={candidates[i, -1, 0]:.3f}, Y={candidates[i, -1, 1]:.3f}")

            neighbors = data.__dict__.get('neighbor_past', None)
            if neighbors is not None: neighbors = neighbors.cpu().numpy()[0]
            lanes = data.__dict__.get('lanes', None)
            if lanes is not None: lanes = lanes.cpu().numpy()[0]

            img_path = f"/tmp/diversity_images/scene_{scene_idx}.png"
            # Offset labels slightly in rendering to prevent exact text overlap hiding them!
            renderer.render_scenario(
                candidates=candidates[..., :2],
                neighbors=neighbors,
                lanes=lanes,
                save_path=img_path,
                title=f"Scene {scene_idx} CFG=False Diversity Test"
            )

if __name__ == "__main__":
    main()

"""
Local Verification Script for BEV generation + Gemini VLM Scoring
"""
import os
import sys
import torch
import numpy as np
import logging

sys.path.insert(0, '/home/gcjms/Flow-Planner')

from omegaconf import OmegaConf
from hydra.utils import instantiate
from flow_planner.data.dataset.nuplan import NuPlanDataset
from torch.utils.data import DataLoader
from flow_planner.dpo.bev_renderer import BEVRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLM_Test")

class DataBunch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def repeat(self, repeats):
        res = DataBunch()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(res, k, v.repeat(repeats, *([1]*(v.ndim - 1))))
            else:
                setattr(res, k, v)
        return res

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # ==========================
    # 1. Load Data
    # ==========================
    logger.info("Loading 1 scene from processed dataset...")
    # Using the local backup on D drive
    data_path = "/mnt/d/flow_planner_backup/processed/val/us-nv-las-vegas-strip_0002960b87b8527f.npz"
    if not os.path.exists(data_path):
        logger.error(f"Cannot find data file: {data_path}")
        return

    npz_data = np.load(data_path)
    
    # Extract keys and convert to tensor with batch dimension (unsqueeze)
    data = {}
    # mapping from npz keys to model expected keys
    key_mapping = {
        'ego_agent_past': 'ego_past',
        'ego_current_state': 'ego_current',
        'neighbor_agents_past': 'neighbor_past',
        'ego_agent_future': 'ego_future',
        'neighbor_agents_future': 'neighbor_future',
        'lanes': 'lanes',
        'route_lanes': 'routes',
        'static_objects': 'map_objects',
        'lanes_speed_limit': 'lanes_speedlimit',
        'route_lanes_speed_limit': 'routes_speedlimit',
        'lanes_has_speed_limit': 'lanes_has_speedlimit',
        'route_lanes_has_speed_limit': 'routes_has_speedlimit',
    }
    
    for npz_k, model_k in key_mapping.items():
        if npz_k in npz_data:
            t = torch.from_numpy(npz_data[npz_k])
            if 'has' in npz_k:
                val = t.to(torch.bool).unsqueeze(0).to(device)
            else:
                val = t.to(torch.float32).unsqueeze(0).to(device)
            data[model_k] = val
    
    data = DataBunch(**data)

    logger.info("Successfully loaded 1 scene manually.")

    # ==========================
    # 2. Load Model
    # ==========================
    config_path = "/home/gcjms/Flow-Planner/checkpoints/model_config.yaml"
    ckpt_path = "/home/gcjms/Flow-Planner/checkpoints/model.pth"

    logger.info("Loading Flow-Planner model...")
    cfg = OmegaConf.load(config_path)
    
    # Inject dummy nodes to resolve interpolations in model config
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)
    
    model = instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'ema_state_dict' in ckpt:
        sd = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    state_dict = {k.replace('module.', ''): v for k, v in sd.items()}

    _, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
    model = model.to(device)
    model.eval()

    # ==========================
    # 3. Generate N=5 Candidates
    # ==========================
    logger.info("Generating N=5 candidates...")
    candidates = []
    num_candidates = 5
    with torch.no_grad():
        for i in range(num_candidates):
            torch.manual_seed(i * 100)  # different seed to get different trajectories
            # 1.8 is the optimal CFG weight from previous experiments
            pred = model(data, mode='inference', use_cfg=True, cfg_weight=1.8, num_candidates=1)
            # pred is a tensor shaped (B, N, T, 4) or similar
            candidates.append(pred.cpu().numpy().reshape(-1, 4))

    candidates = np.stack(candidates, axis=0)  # (K, T, D)
    logger.info(f"Candidates generated: shape {candidates.shape}")

    # ==========================
    # 4. Render BEV
    # ==========================
    logger.info("Rendering BEV Image...")
    # Extract condition items
    neighbors = data.__dict__.get('neighbor_past', None)
    if neighbors is not None:
        neighbors = neighbors.cpu().numpy()[0]
    lanes = data.__dict__.get('lanes', None)
    if lanes is not None:
        lanes = lanes.cpu().numpy()[0]

    renderer = BEVRenderer(image_size=(800, 800), view_range=50.0)
    image_path = "/home/gcjms/Flow-Planner/dpo_data/bev_images/vlm_test.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    renderer.render_scenario(
        candidates=candidates[..., :2],
        neighbors=neighbors,
        lanes=lanes,
        save_path=image_path,
        title="Gemini VLM Ranking Test",
    )
    logger.info(f"BEV image saved to {image_path}")

    # ==========================
    # 5. Call Gemini VLM
    # ==========================
    logger.info("Calling Gemini API via google.genai...")
    from google import genai
    from PIL import Image

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    if not api_key:
        logger.error("No API key provided!")
        return

    client = genai.Client(api_key=api_key)

    prompt = """你是顶级自动驾驶安全专家。
下面是一张自动驾驶车辆的鸟瞰图（BEV）。
1. 蓝色大三角形表示【自车】。
2. 红色带虚线尾迹的方块表示【周围的邻居车】。
3. 灰色线条是【车道线】。
4. 图中标注了 5 条带颜色的行驶轨迹（从 #1 到 #5），表示自车未来可能的行驶路线。

请根据以下标准对轨迹进行评价：
1. 安全性：绝对不能与路上的红色邻居车发生任何碰撞或靠得太近。
2. 合规性：尽量遵守车道线行驶，不要随意越线。
3. 平滑性：轨迹平顺，没有剧烈的急转弯或突兀的停止。

最后，请给出这 5 条轨迹的排名（从最优到最差），并简要解释为什么。
格式要求：最后一行必须是具体的排名，例如：排名结果：1, 3, 2, 5, 4"""

    try:
        img = Image.open(image_path)
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, img]
        )
        logger.info(f"Gemini Response:\n{response.text}")
        
        # Save output to text file for review
        with open("/home/gcjms/Flow-Planner/dpo_data/gemini_response.txt", "w") as f:
            f.write(response.text)
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")

if __name__ == "__main__":
    main()

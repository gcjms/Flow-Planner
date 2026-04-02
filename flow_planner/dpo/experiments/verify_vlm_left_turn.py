import os
import sys
import time
import torch
import numpy as np
import logging

sys.path.insert(0, '/home/gcjms/Flow-Planner')

from omegaconf import OmegaConf
from hydra.utils import instantiate
from flow_planner.dpo.bev_renderer import BEVRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLM_Time_Test")

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

    data_dir = "/mnt/d/flow_planner_backup/processed/val/"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    np.random.seed(42)
    np.random.shuffle(files)

    logger.info("Searching for an unprotected left-turn scenario...")
    selected_file = None
    selected_data = None
    npz_data = None

    key_mapping = {
        'ego_agent_past': 'ego_past', 'ego_current_state': 'ego_current', 'neighbor_agents_past': 'neighbor_past',
        'ego_agent_future': 'ego_future', 'neighbor_agents_future': 'neighbor_future',
        'lanes': 'lanes', 'route_lanes': 'routes', 'static_objects': 'map_objects',
        'lanes_speed_limit': 'lanes_speedlimit', 'route_lanes_speed_limit': 'routes_speedlimit',
        'lanes_has_speed_limit': 'lanes_has_speedlimit', 'route_lanes_has_speed_limit': 'routes_has_speedlimit',
    }

    t0_data = time.time()
    for f in files:
        npz = np.load(f)
        # Check ego future (e.g. left turn means Y increases significantly positively, > 10m; X also increases)
        # Ego coordinate system: initially moving +X. Left turn is +Y
        if 'ego_agent_future' in npz:
            ego_fut = npz['ego_agent_future']
            if len(ego_fut) > 0:
                final_pos = ego_fut[-1]
                if final_pos[1] > 10.0 and final_pos[0] > 5.0:
                    # Also require some neighbors
                    if 'neighbor_agents_past' in npz and npz['neighbor_agents_past'].shape[0] > 5:
                        selected_file = f
                        npz_data = npz
                        break

    if not selected_file:
        logger.error("No left turn scenario found!")
        return
    t1_data = time.time()
    logger.info(f"Found scenario {selected_file} in {t1_data - t0_data:.2f}s")

    data = {}
    for npz_k, model_k in key_mapping.items():
        if npz_k in npz_data:
            t = torch.from_numpy(npz_data[npz_k])
            if 'has' in npz_k:
                val = t.to(torch.bool).unsqueeze(0).to(device)
            else:
                val = t.to(torch.float32).unsqueeze(0).to(device)
            data[model_k] = val
    data = DataBunch(**data)

    config_path = "/home/gcjms/Flow-Planner/checkpoints/model_config.yaml"
    ckpt_path = "/home/gcjms/Flow-Planner/checkpoints/model.pth"

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "data.dataset.train.future_downsampling_method", "uniform", force_add=True)
    OmegaConf.update(cfg, "data.dataset.train.predicted_neighbor_num", 0, force_add=True)
    OmegaConf.update(cfg, "normalization_stats", cfg.get("normalization_stats"), force_add=True)
    
    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'ema_state_dict' in ckpt: sd = ckpt['ema_state_dict']
    elif 'state_dict' in ckpt: sd = ckpt['state_dict']
    else: sd = ckpt
    state_dict = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info("Generating N=5 candidates...")
    candidates = []
    num_candidates = 5
    
    t0_gen = time.time()
    with torch.no_grad():
        for i in range(num_candidates):
            torch.manual_seed(i * 100)
            pred = model(data, mode='inference', use_cfg=False, cfg_weight=0.0, num_candidates=1)
            candidates.append(pred.cpu().numpy().reshape(-1, 4))
    t1_gen = time.time()
    logger.info(f"Generation took: {t1_gen - t0_gen:.2f}s -> {(t1_gen - t0_gen)/num_candidates:.2f}s per trajectory")

    candidates = np.stack(candidates, axis=0)  # (K, T, D)

    logger.info("Rendering BEV Image...")
    t0_render = time.time()
    neighbors = data.__dict__.get('neighbor_past', None)
    if neighbors is not None: neighbors = neighbors.cpu().numpy()[0]
    lanes = data.__dict__.get('lanes', None)
    if lanes is not None: lanes = lanes.cpu().numpy()[0]

    renderer = BEVRenderer(image_size=(800, 800), view_range=50.0)
    image_path = "/home/gcjms/Flow-Planner/dpo_data/bev_images/vlm_test_left_turn.png"
    
    renderer.render_scenario(
        candidates=candidates[..., :2],
        neighbors=neighbors,
        lanes=lanes,
        save_path=image_path,
        title="Gemini VLM Left Turn Interaction",
    )
    t1_render = time.time()
    logger.info(f"Rendering took: {t1_render - t0_render:.2f}s")
    
    logger.info("Calling Gemini API via google.genai...")
    from google import genai
    from PIL import Image

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    t0_vlm = time.time()
    if api_key:
        client = genai.Client(api_key=api_key)
        prompt = """你是顶级自动驾驶安全专家。
下面是一张无保护左转路口的鸟瞰图（BEV）。
1. 蓝色大三角形表示【自车】。
2. 红色方块表示【周围的邻居车】，虚线尾迹是历史轨迹。
3. 灰色线条是【车道线】。
4. 图中标注了 5 条带颜色的候选行驶轨迹。
注：轨迹线上的*小圆点*表示时间刻度（每1秒一个点），圆点间距大说明速度快，间距小说明速度慢。

请根据安全性（避免冲突）、合规性和平滑性对这 5 条左转轨迹进行排名。
结合纵向速度（圆点间距）判断是否选择了合适的加速穿越或减速让行时机。
排名结果：1, 3, 2, 5, 4"""

        img = Image.open(image_path)
        try:
            response = client.models.generate_content(model='gemini-2.5-pro', contents=[prompt, img])
            logger.info(f"Gemini Response:\n{response.text}")
        except Exception as e:
            logger.error(f"VLM Error: {e}")
    t1_vlm = time.time()
    logger.info(f"VLM API took: {t1_vlm - t0_vlm:.2f}s")
    
    total_time = t1_vlm - t0_gen
    logger.info(f"TOTAL TIME (Gen + Render + VLM) for 1 scene (N=5): {total_time:.2f}s")

if __name__ == "__main__":
    main()

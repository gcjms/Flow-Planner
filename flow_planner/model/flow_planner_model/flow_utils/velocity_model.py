"""
VelocityModel: 将 decoder 包装为 ODE solver 可用的速度函数
=====================================================
ODE solver 每一步需要调用 velocity_model(x_t, t) → 速度 v_t
这个模块负责:
  1. 调用 decoder 得到预测（可能是 velocity / x_start / noise）
  2. 将预测转换为速度场
  3. 如果启用 CFG，做有条件/无条件的加权融合
"""

from typing import Dict
import torch
from torch import nn
from flow_matching.utils import ModelWrapper
from flow_matching.path.scheduler.scheduler import Scheduler
from flow_matching.path.scheduler.schedule_transform import ScheduleTransformedModel
from flow_matching.path.affine import AffineProbPath

class VelocityModel(nn.Module):

    def __init__(self, model_fn, path, pred_transform_func: str, correct_xt_fn=None, use_cfg=True, cfg_weight=None):
        """
        Args:
            model_fn: decoder 函数, 输入 (x_t, t, **extras) 输出预测
            path: flow 的插值路径对象
            pred_transform_func: 「模型预测 → 速度场」的转换函数
                例如模型预测 x_start 时, 这个函数是 path.target_to_velocity
            correct_xt_fn: 可选的后处理函数（目前未使用）
            use_cfg: 是否启用 Classifier-Free Guidance
            cfg_weight: CFG 权重 w, 用于 v = (1-w)·v_uncond + w·v_cond
        """
        super().__init__()
        self.model_fn = model_fn
        self.path = path
        self.pred_transform_func = pred_transform_func
        self.correct_xt_fn = correct_xt_fn
        self.use_cfg = use_cfg
        self.cfg_weight = cfg_weight

    def forward(self, x, t, **model_extras):
        """
        ODE solver 每一步调用此函数获取速度。
        
        Args:
            x: 当前时间步的轨迹 x_t, shape (B, action_num, action_len, state_dim)
            t: 当前时间步, 标量
            model_extras: encoder 输出 (encodings, masks, routes_cond, cfg_flags 等)
                          注意: 这些已经在推理开始时被 repeat(2) 过了（如果 use_cfg）
        
        Returns:
            u: 速度场, shape (B, action_num, action_len, state_dim)
               ODE solver 会用这个速度更新: x_{t+dt} = x_t + dt · u
        """
        B, P, _, _ = x.shape

        # 将标量 t 转为张量
        t = t.unsqueeze(0).to(x.device)
        
        if self.use_cfg:
            # 将轨迹复制一份: 前半 B 个用有条件, 后半 B 个用无条件
            # (model_extras 里的 encodings/masks 等已经在 flow_planner.py 的
            #  forward_inference 中被 data.repeat(2) 处理过了)
            x = x.repeat(2, *[1] * (x.dim()-1))   # (B, ...) → (2B, ...)
        
        # 调用 decoder 得到预测 (可能是 velocity / x_start / noise)
        pred = self.model_fn(x, t, **model_extras)
        
        # 可选的后处理
        if self.correct_xt_fn is not None:
            pred = self.correct_xt_fn(pred)
        
        # 将预测转换为速度场
        # 例如: 模型预测 x_start，则 u = path.target_to_velocity(pred, x_t, t)
        u = self.pred_transform_func(pred, x, t)
        
        if self.use_cfg:
            # 拆分: 前半是有条件的速度，后半是无条件的速度
            u_cond, u_uncond = torch.chunk(u, 2)
            
            # CFG 加权公式:
            #   u = (1 - w) · u_uncond + w · u_cond
            #     = u_uncond + w · (u_cond - u_uncond)
            #
            # w=1.0: u = u_cond（纯有条件，等于没引导）
            # w=1.5: u = u_cond + 0.5·(u_cond - u_uncond)（增强条件方向）
            # w=0.0: u = u_uncond（纯无条件）
            u = (1 - self.cfg_weight) * u_uncond + self.cfg_weight * u_cond
        
        return u
"""
FlowODE: Flow Matching 的核心调度模块
======================================
负责训练时的插值采样和推理时的 ODE/SDE 求解。

核心思想 (Flow Matching):
  定义一条从噪声 x₀ 到数据 x₁ 的路径:
    x_t = t · x₁ + (1-t) · x₀     (CondOT 路径)
  
  训练时: 在路径上随机采一个点 x_t，让网络预测速度场 dx_t/dt = x₁ - x₀
  推理时: 从纯噪声 x₀ 出发，用 ODE solver 沿速度场积分到 t=1，得到生成结果

  SDE 模式 (参考 Flow-GRPO / FlowDrive):
    在 ODE Euler 步之间注入可控随机扰动，增加采样多样性。
    噪声 schedule σ(t) = σ_base · (1-t)，早期大扰动探索、后期小扰动保质量。
    用于 DPO/GRPO 的候选轨迹生成，部署时仍用确定性 ODE。
"""

import torch
from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel
from flow_planner.model.model_base import Scheduler
from flow_matching.solver.ode_solver import ODESolver

class FlowODE(Scheduler):
    
    def __init__(self,
                 path, 
                 time_sampler,
                 cfg_weight=1.5,
                 **sample_params
                 ):
        '''
        Args:
            path: 定义 flow 的插值路径（通常是 CondOT: x_t = t·x₁ + (1-t)·x₀）
                  提供 sample() 方法用于训练时的插值
                  提供 velocity_to_target() 等方法用于预测类型之间的转换
            time_sampler: 训练时的时间步采样器，采样 t ∈ [0, 1]
            cfg_weight: CFG 引导权重 w，推理时用于 v = (1-w)·v_uncond + w·v_cond
            sample_params: 推理采样参数，包含:
                - sample_temperature: 初始噪声的缩放系数（通常=1.0）
                - sample_steps: ODE 求解步数（如 10 步）
                - sample_method: ODE 求解器方法（如 'euler', 'midpoint'）
        '''
        self.path = path
        self.cfg_weight = cfg_weight
        self.time_sampler = time_sampler
        self.sample_params = sample_params
        self.translation_funcs = self._get_translation_funcs()
    
    def sample(self, x_data, target_type):
        """
        训练时的前向采样：在噪声和真实数据之间插值，生成训练样本和目标。
        
        Args:
            x_data: 真实轨迹 x₁, shape (B, P, T, D)
                    例如 (32, 1, 80, 4) — 80帧, 4维(x,y,cos_h,sin_h)
            target_type: 网络预测目标的类型
                - 'velocity': 预测速度场 dx_t/dt = x₁ - x₀
                - 'x_start':  预测干净数据 x₁
                - 'noise':    预测噪声 x₀
        
        Returns:
            x_t:    插值后的加噪轨迹 = t·x₁ + (1-t)·x₀, 作为 decoder 的输入
            target: 训练目标（取决于 target_type）
            t:      采样的时间步, shape (B,)
        
        流程:
            ① 随机采样时间步 t ∈ [0,1], 每个样本一个 t
            ② 随机采样高斯噪声 x₀ ~ N(0, I)
            ③ 沿路径插值: x_t = t·x₁ + (1-t)·x₀
            ④ 根据 target_type 取出对应的训练目标
        """
        B = x_data.shape[0]
        # ① 随机采样时间步 t, shape (B,)
        # t 接近 0 → x_t ≈ x₀（几乎纯噪声，任务最难）
        # t 接近 1 → x_t ≈ x₁（几乎无噪声，任务最简单）
        t = self.time_sampler.sample(B).to(x_data.device)
        
        # ② 随机采样噪声 x₀ ~ N(0, I)，形状与真实数据相同
        x_0 = torch.randn_like(x_data, device=x_data.device)
        
        # ③ 路径插值: x_t = t·x₁ + (1-t)·x₀
        # path_sample 包含: x_t (插值点), x_0, x_1, dx_t (速度场)
        path_sample = self.path.sample(x_0=x_0, x_1=x_data, t=t)
        
        # ④ 根据预测类型取训练目标
        if target_type == 'velocity':
            # 速度场: dx_t/dt = x₁ - x₀（CondOT 路径下是常数）
            target = path_sample.dx_t
        elif target_type == 'x_start':
            # 直接预测干净数据 x₁
            target = path_sample.x_1
        elif target_type == 'noise':
            # 预测噪声 x₀（类似 DDPM）
            target = x_0
            
        return path_sample.x_t, target, t
    
    def generate(self, x_init, model_fn, model_pred_type, use_cfg, **model_extra):
        """
        推理时的轨迹生成：从纯噪声出发，用 ODE solver 沿速度场积分到 t=1。
        
        Args:
            x_init: 初始噪声 x₀ ~ N(0, I), shape (B, action_num, action_len, state_dim)
            model_fn: decoder 函数，输入 (x_t, t) 输出预测
            model_pred_type: 模型预测的类型 ('velocity'/'x_start'/'noise')
            use_cfg: 是否使用 Classifier-Free Guidance
            model_extra: decoder 需要的额外输入 (encodings, masks, routes_cond 等)
        
        Returns:
            sample: 生成的轨迹, shape (B, action_num, action_len, state_dim)
        
        流程:
            ① 将模型预测转换为速度场（如果模型预测的是 x_start，需要转换）
            ② 包装成 VelocityModel（处理 CFG 加权逻辑）
            ③ 用 ODE solver 从 t=0 积分到 t=1:
               for t in [0, 0.1, 0.2, ..., 1.0]:  (假设 10 步)
                   v = velocity_model(x_t, t)      # 预测速度（含 CFG）
                   x_{t+dt} = x_t + dt · v         # 沿速度方向走一步
            ④ 返回 t=1 时的 x₁ 作为最终生成结果
        """
        # 获取「模型预测类型 → 速度场」的转换函数
        # 例如 model_pred_type='x_start' → 需要 target_to_velocity 转换
        velocity_func = self.translation_funcs[(model_pred_type, 'velocity')]
        
        # 包装 decoder 为速度模型（内部处理 CFG 的有条件/无条件加权）
        velocity_model = VelocityModel(model_fn, self.path, velocity_func, use_cfg=use_cfg, cfg_weight=self.cfg_weight)
        
        # 创建 ODE 求解器
        solver = ODESolver(velocity_model=velocity_model)
        
        # 缩放初始噪声（temperature=1.0 时不变）
        x_init = x_init * self.sample_params['sample_temperature']
        
        # 每步的步长, 例如 sample_steps=10 → step_size=0.1
        step_size = 1.0 / self.sample_params['sample_steps']
        
        # ODE 求解: 从 t=0 (噪声) 走到 t=1 (数据)
        # method 可以是 'euler'(一阶), 'midpoint'(二阶) 等
        sample = solver.sample(x_init=x_init,
                               step_size=step_size,
                               method=self.sample_params['sample_method'],
                               **model_extra)
        
        return sample

    def generate_sde(self, x_init, model_fn, model_pred_type, use_cfg,
                     cfg_weight=None, sigma_base=0.3, sde_steps=20,
                     noise_schedule='linear', **model_extra):
        """
        SDE 采样：在 Euler 积分的每一步注入可控随机扰动，增加轨迹多样性。

        参考:
          - Flow-GRPO (arXiv:2505.05470): ODE→SDE 转换，用于 RL 探索
          - FlowDrive (arXiv:2509.21961): 流步间扰动注入，用于轨迹多样性

        数学:
          标准 ODE:  x_{t+dt} = x_t + dt · v(x_t, t)
          SDE 模式:  x_{t+dt} = x_t + dt · v(x_t, t) + σ(t) · √dt · ε

          噪声 schedule:
            linear:  σ(t) = σ_base · (1 - t)         — 早期大、后期小
            cosine:  σ(t) = σ_base · cos(π·t / 2)    — 更平滑的衰减
            constant: σ(t) = σ_base                   — 全程均匀（不推荐）

        Args:
            x_init: 初始噪声, shape (B, action_num, action_len, state_dim)
            model_fn: decoder 函数
            model_pred_type: 预测类型 ('velocity'/'x_start'/'noise')
            use_cfg: 是否使用 CFG
            cfg_weight: CFG 权重，None 时使用 self.cfg_weight
            sigma_base: 噪声强度基准值，推荐 0.1~0.5
            sde_steps: SDE 积分步数，推荐 20~50（比部署时多）
            noise_schedule: 噪声衰减策略 ('linear'/'cosine'/'constant')
            model_extra: decoder 的额外输入 (encodings, masks 等)

        Returns:
            x: 生成的轨迹, shape 同 x_init
        """
        import math

        velocity_func = self.translation_funcs[(model_pred_type, 'velocity')]
        w = cfg_weight if cfg_weight is not None else self.cfg_weight
        velocity_model = VelocityModel(
            model_fn, self.path, velocity_func,
            use_cfg=use_cfg, cfg_weight=w,
        )

        x = x_init * self.sample_params['sample_temperature']
        dt = 1.0 / sde_steps

        for i in range(sde_steps):
            t_val = i * dt
            t_tensor = torch.tensor(t_val, dtype=x.dtype, device=x.device)

            with torch.no_grad():
                v = velocity_model(x, t_tensor, **model_extra)

            x = x + dt * v

            if sigma_base > 0 and i < sde_steps - 1:
                t_next = (i + 1) * dt
                if noise_schedule == 'linear':
                    sigma_t = sigma_base * (1.0 - t_next)
                elif noise_schedule == 'cosine':
                    sigma_t = sigma_base * math.cos(math.pi * t_next / 2.0)
                else:
                    sigma_t = sigma_base

                noise = torch.randn_like(x)
                x = x + sigma_t * (dt ** 0.5) * noise

        return x

    def identity(self, x, xt, t):
        """恒等变换: 当模型预测类型与目标类型相同时使用"""
        return x
    
    def _get_translation_funcs(self):
        """
        预测类型之间的转换函数映射表。
        
        由于模型可以预测不同的目标 (velocity/x_start/noise)，
        但 ODE solver 需要的是速度场，所以需要转换函数。
        
        三种预测类型在 CondOT 路径 x_t = t·x₁ + (1-t)·x₀ 下的关系:
            velocity(速度场): v = x₁ - x₀
            x_start(干净数据): x₁ = (x_t - (1-t)·x₀) / t = x_t/t + (1-1/t)·x₀
            noise(噪声):       x₀ = (x_t - t·x₁) / (1-t)
        
        键格式: (模型预测类型, 需要转换到的类型) → 转换函数
        例如: ('x_start', 'velocity') → path.target_to_velocity
              即: 知道 x₁ 的预测值，转换为速度 v = (x₁ - x_t) / (1-t) + ...
        """
        return {('velocity', 'x_start'): self.path.velocity_to_target,
                ('velocity', 'noise'): self.path.velocity_to_epsilon,
                ('x_start', 'velocity'): self.path.target_to_velocity,
                ('x_start', 'noise'): self.path.target_to_epsilon,
                ('noise', 'velocity'): self.path.epsilon_to_velocity,
                ('noise', 'x_start'): self.path.epsilon_to_target,
                ('velocity', 'velocity'): self.identity,
                ('x_start', 'x_start'): self.identity,
                ('noise', 'noise'): self.identity}
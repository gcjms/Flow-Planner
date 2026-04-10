"""
FlowPlanner Decoder 模块
========================
核心架构: 基于 DiT (Diffusion Transformer) 的轨迹解码器

整体流程:
  加噪轨迹 x_t → preproj → 与 encoder 输出拼接 → DiT blocks → PostFusion → FinalLayer → 预测轨迹

关键概念:
  - action token: 将 80 帧未来轨迹切分为多个重叠的 token（例如 7 个 20 帧 token，重叠 10 帧）
  - 多模态: agents(邻居+静态)、lanes(车道线)、trajectory(轨迹 token) 三个模态
  - 条件: 时间步 t、路由条件 routes_cond、action 位置编码、CFG 嵌入
"""

import torch
import torch.nn as nn
from timm.layers import Mlp
from flow_planner.model.modules.decoder_modules import FinalLayer, PostFusion
from flow_planner.model.model_utils.tool_func import sinusoidal_positional_encoding
from flow_planner.model.modules.decoder_modules import RMSNorm, FeedForward, AdaptiveLayerNorm
from flow_planner.model.flow_planner_model.global_attention import JointAttention


class FlowPlannerDecoder(nn.Module):
    """
    FlowPlanner 的轨迹解码器。

    输入:
        x:  加噪的轨迹 tokens, shape (B, action_num, action_len, state_dim)
            例如 (32, 7, 20, 4) — 7 个 action token, 每个 20 帧, 4 维(x,y,cos_h,sin_h)
        t:  flow matching 时间步, shape (B,), 范围 [0, 1]
        model_extra: encoder 输出的字典, 包含:
            - encodings: (agents_encoding, lanes_encoding) — 编码后的场景 token
            - masks: (agents_mask, lanes_mask) — 有效性 mask
            - routes_cond: (B, hidden_dim) — 路由全局条件向量
            - token_dist: (B, N, N) — token 间欧氏距离矩阵 (用于 attention bias)
            - cfg_flags: (B,) — CFG 标志 (1=有条件, 0=无条件)

    输出:
        prediction: (B, action_num, action_len, state_dim) — 预测的去噪轨迹 tokens
    """
    def __init__(
            self,
            hidden_dim,           # 隐藏层维度, 例如 256
            depth,                # DiT block 的层数, 例如 6
            t_embedder,           # 时间步嵌入器 (TimestepEmbedder)
            agents_hidden_dim=192,  # agents 子编码器的输出维度
            lane_hidden_dim=192,    # lanes 子编码器的输出维度
            heads=6,              # 多头注意力的头数
            preproj_hidden=256,   # 轨迹预投影的中间维度
            enable_attn_dist=False,  # 是否启用距离感知的 attention bias
            act_pe_type: str = 'learnable',  # action 位置编码类型: learnable/fixed_sin/none
            goal_dim: int = 0,    # goal point 维度: 0=不启用, 2=(x,y)
            device: str = 'cuda',
            **planner_params      # 包含 action_len, state_dim, future_len, action_overlap 等
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # output_dim = 每个 action token 展平后的维度 = action_len * state_dim
        # 例如 20 * 4 = 80
        self.output_dim = planner_params['action_len'] * planner_params['state_dim']
        
        # 计算 action token 数量
        # 例如: future_len=80, action_len=20, overlap=10 → action_num = (80-10)/(20-10) = 7
        action_num = (planner_params['future_len'] - planner_params['action_overlap']) // (planner_params['action_len'] - planner_params['action_overlap'])
        self.action_num = int(action_num)
        
        # 总 token 数 = 邻居数(32) + 静态物体数(5) + 车道数(70) + action_num
        # 用于 JointAttention 中的全局 attention
        self.token_num = planner_params['neighbor_num'] + planner_params['static_num'] + planner_params['lane_num'] + self.action_num
        self.action_len = planner_params['action_len']
        self.action_overlap = planner_params['action_overlap']
        self.state_dim = planner_params['state_dim']
        
        # ============ 核心模块 ============
        
        # DiT: 多模态 Transformer, 包含 depth 个 FlowPlannerDiTBlock
        # 三个模态: agents(agents_hidden_dim), lanes(lane_hidden_dim), trajectory(hidden_dim)
        self.dit = FlowPlannerDiT(
            depth=depth,
            dim_modalities=(
                agents_hidden_dim,    # 模态1: agents (邻居+静态物体)
                lane_hidden_dim,      # 模态2: lanes (车道线)
                hidden_dim            # 模态3: trajectory (轨迹 token)
            ),
            dim_cond=hidden_dim,      # 条件向量维度
            heads=heads,
            dim_head=int(hidden_dim/heads),
            enable_attn_dist=enable_attn_dist,
            token_num=self.token_num
        )
        
        # PostFusion: DiT 之后进一步融合 — 用 agents+lanes 的信息增强轨迹 token
        self.post_fusion = PostFusion(hidden_dim=hidden_dim, heads=heads, action_num=self.action_num)
        
        # 时间步嵌入器: 标量 t → (hidden_dim,) 向量
        self.t_embedder = t_embedder
        
        # 轨迹预投影: 将展平的加噪轨迹投影到隐空间
        # action_len*state_dim → preproj_hidden → hidden_dim
        self.preproj = Mlp(in_features=self.output_dim, hidden_features=preproj_hidden, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        
        # 最终输出层: hidden_dim → action_len*state_dim (带 AdaLN 调制)
        self.final_layer = FinalLayer(hidden_dim, self.output_dim)

        # CFG 嵌入: 2 个学习的嵌入向量
        # index 0 = 无条件(masked), index 1 = 有条件(unmasked)
        self.cfg_embedding = nn.Embedding(2, hidden_dim)

        # Goal Point Conditioning (GoalFlow-style)
        # goal_dim=0: 不启用 (向后兼容); goal_dim=2: (x,y) goal point
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_proj = nn.Sequential(
                nn.Linear(goal_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # 初始化最后一层为零，让 goal 在训练初期影响最小
            nn.init.zeros_(self.goal_proj[-1].weight)
            nn.init.zeros_(self.goal_proj[-1].bias)
        
        # Action 位置编码: 让模型区分第 0, 1, 2, ... 个 action token
        self.act_pe_type = act_pe_type
        self.load_action_pe(act_pe_type)
        
        # 如果 encoder 各子模块的输出维度与 decoder hidden_dim 不同, 需要投影
        self.agents_in_proj = nn.Linear(agents_hidden_dim, hidden_dim) if agents_hidden_dim != hidden_dim \
            else nn.Identity()
        self.lane_in_proj = nn.Linear(lane_hidden_dim, hidden_dim) if lane_hidden_dim != hidden_dim \
            else nn.Identity()
        
        self.planner_params = planner_params
        
        self.device = device
        
        # self.initialize_weights()

    def load_action_pe(self, act_pe_type: str):
        """
        加载 action 位置编码。
        
        每个 action token 代表未来轨迹的一段时间窗口，
        位置编码帮助模型知道"这是第几个 action token"。
        
        类型:
          - learnable: 可学习的参数 (默认)
          - fixed_sin: 固定的正弦位置编码
          - none:      不使用位置编码
        """
        if act_pe_type == 'learnable':
            # 可学习的位置编码, shape (action_num, hidden_dim)
            self.action_pe = nn.Parameter(torch.Tensor(self.action_num, self.hidden_dim))
            nn.init.normal_(self.action_pe, mean=0.0, std=1.0)
        elif act_pe_type == 'fixed_sin':
            # 固定正弦编码, 基于每个 action token 的中心时间点
            action_t = (torch.arange(0, self.action_num) * (self.action_len - self.action_overlap) + self.action_len / 2)
            action_pe = sinusoidal_positional_encoding(action_t, self.hidden_dim)
            self.register_buffer('action_pe', action_pe)
        elif act_pe_type == 'none':
            action_pe = torch.zeros((self.action_num, self.hidden_dim))
            self.register_buffer('action_pe', action_pe)
        else:
            raise ValueError(f'Unexpected action embedding type {act_pe_type}')
    
    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
        self.apply(basic_init)
    
    def forward(self, x, t, **model_extra):
        """
        Decoder 前向传播。
        
        Args:
            x: (B, P, action_len, state_dim) — 加噪轨迹 tokens
               例如 (32, 7, 20, 4), 其中 P=action_num
            t: (B,) — flow matching 时间步
            model_extra: encoder 输出, 包含 encodings, masks, routes_cond, token_dist, cfg_flags
        
        Returns:
            prediction: (B, P, action_len, state_dim) — 预测的去噪轨迹
        
        详细步骤:
            Step A: 轨迹预投影 — 将每个 action token 从 (action_len*state_dim) 投影到 hidden_dim
            Step B: 构建条件向量 — 时间嵌入 + 路由条件 + action位置编码 + CFG嵌入
            Step C: 准备多模态输入 — 将 encoder 的 agents/lanes 编码与轨迹 token 拼接
            Step D: DiT 多模态 Transformer — N 层 JointAttention + FFN (详见 FlowPlannerDiTBlock)
            Step E: PostFusion — 用 agents+lanes token 做 cross-attention 增强轨迹 token
            Step F: FinalLayer — AdaLN 调制 + MLP 输出最终预测
        """

        B, P, _, _ = x.shape  # B=batch, P=action_num
        x = x.to(torch.float32)
        
        # ===== Step A: 轨迹预投影 =====
        # (B, P, action_len, state_dim) → (B, P, action_len*state_dim)
        x = x.reshape(B, P, -1)
        # (B, P, action_len*state_dim) → (B, P, hidden_dim)
        # 例如 (32, 7, 80) → (32, 7, 256)
        
        # 取出 encoder 输出
        encodings = list(model_extra['encodings'])  # [agents_enc, lanes_enc]
        attn_dist = model_extra['token_dist']       # (B, total_tokens, total_tokens) 距离矩阵
        masks = list(model_extra['masks'])           # [agents_mask, lanes_mask]
        
        # ===== Step B: 构建条件向量 =====
        # CFG 嵌入: 根据 flag 查表得到条件/无条件嵌入
        cfg_flags = model_extra['cfg_flags'].reshape(B)        # (B,)
        cfg_embedding = self.cfg_embedding(cfg_flags).unsqueeze(-2)  # (B, 1, hidden_dim)
        
        # 路由条件: 来自 RouteEncoder 的全局路由向量
        routes_cond = model_extra['routes_cond']  # (B, hidden_dim)

        # 轨迹预投影
        x = self.preproj(x)  # (B, P, hidden_dim)
        
        # ===== Step C: 准备多模态输入 =====
        # 将轨迹 token 作为第三个模态加入
        encodings.append(x)       # [agents_enc, lanes_enc, traj_enc]
        masks.append(None)        # 轨迹 token 没有 mask (全部有效)
        
        # 时间嵌入: 标量 t → 向量
        time_cond = self.t_embedder(t).unsqueeze(1)            # (B, 1, hidden_dim)
        routes_cond = routes_cond.unsqueeze(1)                  # (B, 1, hidden_dim)
        action_pe = self.action_pe.unsqueeze(0).repeat(B, 1, 1) # (B, P, hidden_dim)

        # Goal embedding: 从 model_extra 中取 goal_point (B, 2)
        goal_embedding = 0
        if self.goal_dim > 0 and 'goal_point' in model_extra and model_extra['goal_point'] is not None:
            gp = model_extra['goal_point'].float()  # (B, 2)
            goal_embedding = self.goal_proj(gp).unsqueeze(1)  # (B, 1, hidden_dim)
            # CFG masking: unconditioned samples (cfg_flags=0) 不给 goal
            goal_mask = cfg_flags.float().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            goal_embedding = goal_embedding * goal_mask

        # 综合条件 y: 用于 FinalLayer 的 AdaLN 调制
        y = time_cond + routes_cond + action_pe + cfg_embedding + goal_embedding  # (B, P, hidden_dim)

        # ===== Step D: DiT 多模态 Transformer =====
        # 输入 3 个模态: agents(32+5=37个token), lanes(70个token), trajectory(P个token)
        # 每层 DiTBlock: AdaLN → JointAttention(全局) → gate+residual → AdaLN → FFN → gate+residual
        token_tuple = self.dit(
            modality_tokens=encodings,    # [agents(B,37,dim), lanes(B,70,dim), traj(B,P,dim)]
            modality_masks=masks,         # [agents_valid, lanes_valid, None]
            time_cond=time_cond,          # (B, 1, hidden_dim)
            routes_cond=routes_cond,      # (B, 1, hidden_dim)
            action_encoding=action_pe,    # (B, P, hidden_dim) — 仅用于 trajectory 模态
            cfg_embedding=cfg_embedding,  # (B, 1, hidden_dim) — 仅用于 trajectory 模态
            goal_embedding=goal_embedding,  # (B, 1, hidden_dim) or 0 — goal point 条件
            attn_dist=attn_dist           # (B, N, N) 距离矩阵用于 attention bias
        )
        
        # DiT 输出: 3 个模态各自更新后的 token
        agents_token, lanes_token, x_token = token_tuple
        # agents_token: (B, 37, agents_hidden_dim)
        # lanes_token:  (B, 70, lane_hidden_dim)
        # x_token:      (B, P, hidden_dim)

        # ===== Step E: PostFusion — 后融合 =====
        # 将 agents 和 lanes 投影到统一的 hidden_dim
        agents_token = self.agents_in_proj(agents_token)  # (B, 37, hidden_dim)
        lanes_token = self.lane_in_proj(lanes_token)      # (B, 70, hidden_dim)

        # 拼接为 kv tokens
        kv_token = torch.cat([agents_token, lanes_token], dim=1)  # (B, 107, hidden_dim)
        # 构建 key_padding_mask (True=需要被mask的无效位置)
        key_mapping_mask = torch.cat([masks[i] for i in range(len(masks)-1)], dim=1)  # (B, 107)

        # PostFusion: x_token 与 kv_token 做联合 self-attention → mean pool → MLP → 残差
        x_token = self.post_fusion(x_token, kv_token, ~key_mapping_mask)  # (B, P, hidden_dim)

        # ===== Step F: FinalLayer — 输出预测 =====
        # AdaLN 调制: 用条件 y 生成 shift/scale 来调制 LayerNorm
        # 然后通过 MLP: hidden_dim → hidden_dim*4 → output_dim
        prediction = self.final_layer(x_token, y)  # (B, P, action_len*state_dim)

        # reshape 回轨迹维度
        prediction = prediction.reshape(B, P, -1, self.planner_params['state_dim'])
        # → (B, P, action_len, state_dim), 例如 (32, 7, 20, 4)

        return prediction


# ==============================================================
# FlowPlannerDiTBlock: DiT 的单层模块
# ==============================================================

class FlowPlannerDiTBlock(nn.Module):
    """
    多模态 DiT Block (类似 MMDiT)。
    
    对多个模态 (agents, lanes, trajectory) 做联合处理:
      1. 各模态独立做 AdaptiveLayerNorm (用各自的条件调制)
      2. 所有模态 token 拼在一起做全局 JointAttention
      3. 各模态独立做 FeedForward
      4. 每一步都有 gated residual connection
    
    关键设计:
      - 各模态可以有不同的隐藏维度 (agents_dim ≠ lanes_dim ≠ traj_dim)
      - JointAttention 内部会投影到统一维度再做 attention, 然后投影回各自维度
      - gate 参数通过条件向量动态生成, 初始化为 1 (即初始状态接近恒等映射)
    """
    def __init__(
        self,
        *,
        dim_modalities: tuple[int, ...],  # 每个模态的维度, 例如 (192, 192, 256)
        dim_cond = None,          # 条件向量维度
        dim_head = 64,            # 每个注意力头的维度
        heads = 8,                # 注意力头数
        enable_attn_dist = False, # 是否使用距离感知 attention bias
        ff_kwargs: dict = dict(),
        token_num: int = 118,     # 总 token 数 (用于距离矩阵)
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # Gate 投影: 条件向量 → 2 * dim (分别用于 attention gate 和 ffn gate)
        # 初始化: weight=0, bias=1 → 初始 gate≈1, 网络初始行为接近恒等
        self.modalities_gate_proj = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_cond, dim * 2)
            )
        for dim in dim_modalities])

        for layer in self.modalities_gate_proj:
            nn.init.zeros_(layer[-1].weight)
            nn.init.constant_(layer[-1].bias, 1.)

        # AdaptiveLayerNorm: 用条件生成 gamma/beta 来调制 LayerNorm
        # 在 attention 前使用
        self.attn_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])

        # JointAttention: 所有模态的 token 拼在一起做全局多头注意力
        # 每个模态独立做 Q,K,V 投影 → 拼接 → attention → 拆分 → 各自投影回原维度
        self.joint_attn = JointAttention(
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            enable_attn_dist = enable_attn_dist,
            token_num=token_num
        )

        # FFN 前的 AdaptiveLayerNorm
        self.ff_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])
        # 各模态独立的 FeedForward 网络
        self.feedforwards = nn.ModuleList([FeedForward(dim, **ff_kwargs) for dim in dim_modalities])

    def forward(
        self,
        *,
        modality_tokens,    # [agents_tokens, lanes_tokens, traj_tokens]
        modality_masks = None,    # [agents_mask, lanes_mask, None]
        modality_conds = None,    # [cond_for_agents, cond_for_lanes, cond_for_traj]
        attn_dist = None          # (B, N, N) token 间距离矩阵
    ):
        """
        单层 DiTBlock 前向传播。
        
        流程:
          1. 从条件向量生成 gate 系数 (attn_gate, ffn_gate)
          2. AdaLN → JointAttention → gate*output + residual
          3. AdaLN → FeedForward → gate*output + residual
        """
        assert len(modality_tokens) == self.num_modalities
 
        # ---- 生成 gate 系数 ----
        # 每个模态的条件 → Linear → 拆分为 attn_gate 和 ffn_gate
        attn_gammas = []
        ff_gammas = []
        for proj, cond in zip(self.modalities_gate_proj, modality_conds):
            gamma = proj(cond)                # (B, ?, dim*2)
            attn_g, ff_g = gamma.chunk(2, dim=-1)  # 各 (B, ?, dim)
            attn_gammas.append(attn_g)
            ff_gammas.append(ff_g)
        
        # ---- Attention 分支 ----
        # 保存残差
        modality_tokens_attn_res = [token.clone() for token in modality_tokens]
        # AdaptiveLayerNorm: 根据条件调制 LayerNorm 的 gamma/beta
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.attn_layernorms, modality_tokens, modality_conds)]
        # JointAttention: 所有模态 token 拼接做全局 attention
        # agents(37) + lanes(70) + traj(P) 个 token 一起做 attention
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks, attn_dist = attn_dist)
        # gate * attention_output
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, attn_gammas)]
        # 残差连接
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_attn_res)]

        # ---- FFN 分支 ----
        # 保存残差
        modality_tokens_ffn_res = [token.clone() for token in modality_tokens]
        # AdaptiveLayerNorm
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.ff_layernorms, modality_tokens, modality_conds)]
        # 各模态独立的 FeedForward
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]
        # gate * ffn_output
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, ff_gammas)]
        # 残差连接
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_ffn_res)]

        return modality_tokens


# ==============================================================
# FlowPlannerDiT: 堆叠 N 层 DiTBlock
# ==============================================================

class FlowPlannerDiT(nn.Module):
    """
    FlowPlanner 的完整 DiT (Diffusion Transformer)。
    
    由 depth 个 FlowPlannerDiTBlock 堆叠而成，最后对每个模态做 RMSNorm。
    
    关键: 不同模态接收不同的条件:
      - agents, lanes: cond = time_cond + routes_cond
      - trajectory:    cond = time_cond + routes_cond + action_pe + cfg_embedding
    """
    def __init__(
        self,
        *,
        depth,                    # DiT block 的层数
        dim_modalities,           # (agents_dim, lanes_dim, traj_dim)
        enable_attn_dist = False,
        **block_kwargs
    ):
        super().__init__()

        # 堆叠 depth 个 DiTBlock
        blocks = [FlowPlannerDiTBlock(dim_modalities = dim_modalities, enable_attn_dist = enable_attn_dist, **block_kwargs) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)

        # 每个模态各自的 RMSNorm (最终归一化)
        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = nn.ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens,      # [agents_enc, lanes_enc, traj_enc]
        modality_masks = None,  # [agents_mask, lanes_mask, None]
        time_cond = None,       # (B, 1, hidden_dim) 时间嵌入
        routes_cond = None,     # (B, 1, hidden_dim) 路由条件
        action_encoding = None, # (B, P, hidden_dim) action 位置编码
        cfg_embedding = None,   # (B, 1, hidden_dim) CFG 嵌入
        goal_embedding = 0,     # (B, 1, hidden_dim) or 0 — goal point 条件
        attn_dist = None,       # (B, N, N) 距离矩阵
    ):
        """
        完整 DiT 前向传播。
        
        为每个模态构建各自的条件向量:
          - agents / lanes 的条件: time + routes (它们不需要知道 action 的位置和 CFG 状态)
          - trajectory 的条件: time + routes + action_pe + cfg + goal
        
        然后通过 depth 个 DiTBlock，最后 RMSNorm。
        """
        # 构建每个模态的条件向量
        # 前 N-1 个模态 (agents, lanes): 只用 time + routes
        other_modality_conds = [time_cond + routes_cond] * (len(modality_tokens) - 1)
        # 最后一个模态 (trajectory): time + routes + action_pe + cfg + goal
        ego_traj_conds = [time_cond + routes_cond + action_encoding + cfg_embedding + goal_embedding]
        modality_conds = other_modality_conds + ego_traj_conds

        # 通过每一层 DiTBlock
        for block in self.blocks:
            modality_tokens = block(
                modality_tokens = modality_tokens,
                modality_masks = modality_masks,
                modality_conds = modality_conds,
                attn_dist = attn_dist
            )

        # 最终 RMSNorm
        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]
        
        return tuple(modality_tokens)
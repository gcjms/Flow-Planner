"""
Risk Network: 风险评估网络
============================
基于驾驶安全指标预测最优 CFG 权重 w。

输入: 12 维风险特征向量（TTC, THW, DRAC, 距离, 速度等）
输出: 风险分数 r ∈ [0, 1] → 映射为 w = w_min + r × (w_max - w_min)

参数量: ~5K（相比主模型 14.28M 可忽略不计）
"""

import torch
import torch.nn as nn
import numpy as np
from flow_planner.risk.risk_features import NUM_RISK_FEATURES


class RiskNetwork(nn.Module):
    """
    风险评估网络：场景安全特征 → CFG 引导权重 w
    
    Architecture:
        Linear(12, 64) → ReLU → Dropout → 
        Linear(64, 32) → ReLU → Dropout → 
        Linear(32, 1) → Sigmoid → Scale to [w_min, w_max]
    
    Args:
        input_dim: 输入特征维度 (默认 12)
        hidden_dim: 隐藏层维度 (默认 64)
        w_min: CFG 权重下界 (默认 0.5)
        w_max: CFG 权重上界 (默认 4.0)
        dropout: Dropout 比例 (默认 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = NUM_RISK_FEATURES,
        hidden_dim: int = 64,
        w_min: float = 0.5,
        w_max: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 特征归一化参数（训练时设置）
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # 最后一层的 bias 初始化，使初始输出 ≈ 0.5（即 w ≈ (w_min+w_max)/2）
        last_linear = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.bias)
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """设置特征归一化参数"""
        self.feature_mean = torch.from_numpy(mean).float()
        self.feature_std = torch.from_numpy(std).float()
    
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """归一化输入特征"""
        return (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device)
    
    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, input_dim) 原始风险特征
        
        Returns:
            dict with:
                'w': (B, 1) 预测的 CFG 权重
                'risk_score': (B, 1) 风险分数 ∈ [0, 1]
        """
        x = self.normalize(features)
        risk_score = self.mlp(x)  # (B, 1), range [0, 1]
        w = self.w_min + risk_score * (self.w_max - self.w_min)
        
        return {
            'w': w,
            'risk_score': risk_score,
        }
    
    def predict_w(self, features: torch.Tensor) -> torch.Tensor:
        """便捷接口：直接返回 w 值"""
        with torch.no_grad():
            return self.forward(features)['w']
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveODESteps:
    """
    基于风险分数动态调整 ODE 求解步数。
    
    risk_score 高 → 步数多（精度高，安全性强）
    risk_score 低 → 步数少（速度快，效率高）
    """
    
    def __init__(self, min_steps: int = 2, max_steps: int = 6):
        self.min_steps = min_steps
        self.max_steps = max_steps
    
    def get_steps(self, risk_score: float) -> int:
        """根据风险分数返回 ODE 步数"""
        steps = self.min_steps + int(risk_score * (self.max_steps - self.min_steps))
        return max(self.min_steps, min(self.max_steps, steps))


def load_risk_network(checkpoint_path: str, device: str = 'cuda') -> RiskNetwork:
    """加载训练好的 Risk Network"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = RiskNetwork(
        input_dim=checkpoint.get('input_dim', NUM_RISK_FEATURES),
        hidden_dim=checkpoint.get('hidden_dim', 64),
        w_min=checkpoint.get('w_min', 0.5),
        w_max=checkpoint.get('w_max', 4.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'feature_mean' in checkpoint:
        model.set_normalization(checkpoint['feature_mean'], checkpoint['feature_std'])
    
    model.to(device)
    model.eval()
    return model

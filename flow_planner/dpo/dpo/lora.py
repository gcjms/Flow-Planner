"""
LoRA (Low-Rank Adaptation) for Flow-Planner Decoder
=====================================================
手写 LoRA 实现，不依赖 peft 等第三方库。

核心思想：
  W' = W + (α/r) · B @ A
  - W: 冻结的原始权重 (d_out × d_in)
  - A: 低秩矩阵 (r × d_in), Kaiming 初始化
  - B: 低秩矩阵 (d_out × r), 零初始化
  - α: 缩放系数 (默认 16)
  - r: 秩 (默认 4)

由于 B 初始化为 0，注入 LoRA 后模型输出不变。

用法：
  from flow_planner.dpo.lora import inject_lora, get_lora_params, merge_lora

  # 1. 注入 LoRA
  lora_info = inject_lora(model.model_decoder, target_modules=['in_proj', 'out_proj', 'ffn', 'proj', 'adaLN'])

  # 2. 冻结非 LoRA 参数
  for p in model.parameters():
      p.requires_grad = False
  for p in get_lora_params(model):
      p.requires_grad = True

  # 3. 训练 ...

  # 4. 推理前合并
  merge_lora(model)
"""

import os
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA 包裹层：替换原始 nn.Linear，添加低秩旁路。

    前向计算：
      y = W @ x + bias + (α/r) · B @ A @ x

    Args:
        original_linear: 原始的 nn.Linear 层
        rank: LoRA 的秩 r
        alpha: 缩放系数 α
        dropout: LoRA 旁路上的 dropout
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 保留原始权重（冻结）
        self.weight = original_linear.weight  # nn.Parameter, 后续设 requires_grad=False
        self.bias = original_linear.bias      # 可能为 None

        # LoRA 低秩分解矩阵
        # A: (rank, in_features) — 降维投影
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        # B: (out_features, rank) — 升维投影
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Kaiming 初始化 A，B 初始化为 0 → 初始 LoRA 贡献为 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        # 可选 dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 标记是否已合并
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性变换
        result = F.linear(x, self.weight, self.bias)

        if not self._merged:
            # LoRA 旁路: x → dropout → A → B → scale
            lora_out = self.lora_dropout(x)
            lora_out = F.linear(lora_out, self.lora_A)   # x @ A^T → (*, rank)
            lora_out = F.linear(lora_out, self.lora_B)   # → (*, out_features)
            result = result + lora_out * self.scaling

        return result

    def merge(self):
        """将 LoRA 权重合并到原始权重中（推理加速）"""
        if self._merged:
            return
        with torch.no_grad():
            # W' = W + (α/r) · B @ A
            self.weight.add_(self.scaling * (self.lora_B @ self.lora_A))
        self._merged = True

    def unmerge(self):
        """撤销合并（恢复到训练状态）"""
        if not self._merged:
            return
        with torch.no_grad():
            self.weight.sub_(self.scaling * (self.lora_B @ self.lora_A))
        self._merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, merged={self._merged}"
        )


def _match_target(name: str, target_modules: List[str]) -> bool:
    """检查模块名称是否匹配任意目标模式"""
    for target in target_modules:
        if target in name:
            return True
    return False


def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Dict[str, dict]:
    """
    向模型中匹配的 nn.Linear 层注入 LoRA。

    Args:
        model: 要注入 LoRA 的模型（通常是 model.model_decoder）
        target_modules: 目标模块名称列表，匹配规则为 substring match
                       默认: ['in_proj', 'out_proj', 'ffn', 'proj', 'adaLN_modulation']
        rank: LoRA 秩
        alpha: LoRA 缩放系数
        dropout: LoRA dropout

    Returns:
        injection_info: 注入信息字典 {module_path: {in, out, rank, params}}
    """
    if target_modules is None:
        target_modules = ['in_proj', 'out_proj', 'ffn', 'proj', 'adaLN_modulation']

    injection_info = {}
    replaced_count = 0

    # 收集所有需要替换的 (parent, attr_name, linear_module) 三元组
    replacements = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _match_target(full_name, target_modules):
            continue

        # 找到父模块和属性名
        parts = full_name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = model

        replacements.append((full_name, parent, attr_name, module))

    # 执行替换
    for full_name, parent, attr_name, linear in replacements:
        lora_linear = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)

        # 冻结原始权重
        lora_linear.weight.requires_grad = False
        if lora_linear.bias is not None:
            lora_linear.bias.requires_grad = False

        # 替换模块
        setattr(parent, attr_name, lora_linear)

        lora_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
        injection_info[full_name] = {
            'in_features': linear.in_features,
            'out_features': linear.out_features,
            'rank': rank,
            'lora_params': lora_params,
        }
        replaced_count += 1

    total_lora_params = sum(info['lora_params'] for info in injection_info.values())
    total_model_params = sum(p.numel() for p in model.parameters())

    logger.info(
        f"LoRA injected: {replaced_count} layers, "
        f"{total_lora_params:,} LoRA params / {total_model_params:,} total "
        f"({100 * total_lora_params / total_model_params:.2f}%)"
    )

    return injection_info


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """返回模型中所有 LoRA 可训练参数"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def get_lora_state_dict(model: nn.Module) -> dict:
    """提取模型中所有 LoRA 参数的 state dict"""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()
    return lora_state


def merge_lora(model: nn.Module):
    """将所有 LoRA 权重合并到原始权重（推理用）"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module):
    """撤销所有 LoRA 合并（恢复训练状态）"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def save_lora(model: nn.Module, path: str, extra_info: Optional[dict] = None):
    """
    只保存 LoRA 权重到文件。

    Args:
        model: 注入了 LoRA 的模型
        path: 保存路径 (.pt)
        extra_info: 额外信息（如 epoch, loss 等），会一并保存
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    save_dict = {
        'lora_state_dict': get_lora_state_dict(model),
    }
    if extra_info:
        save_dict['extra_info'] = extra_info

    torch.save(save_dict, path)
    logger.info(f"LoRA weights saved to {path}")


def load_lora(model: nn.Module, path: str, strict: bool = True) -> dict:
    """
    加载 LoRA 权重到已注入 LoRA 的模型。

    Args:
        model: 已经通过 inject_lora 注入了 LoRA 的模型
        path: LoRA 权重文件路径
        strict: 是否严格匹配所有 key

    Returns:
        extra_info: 保存时附带的额外信息
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    lora_state = checkpoint['lora_state_dict']

    # 加载到模型
    model_modules = dict(model.named_modules())
    loaded_count = 0

    for key, value in lora_state.items():
        # key 格式: "module.path.lora_A" 或 "module.path.lora_B"
        parts = key.rsplit('.', 1)
        if len(parts) != 2:
            if strict:
                raise KeyError(f"Unexpected LoRA key format: {key}")
            continue

        module_name, param_name = parts
        if module_name not in model_modules:
            if strict:
                raise KeyError(f"Module {module_name} not found in model")
            continue

        module = model_modules[module_name]
        if not isinstance(module, LoRALinear):
            if strict:
                raise TypeError(f"Module {module_name} is not LoRALinear")
            continue

        getattr(module, param_name).data.copy_(value)
        loaded_count += 1

    logger.info(f"Loaded {loaded_count} LoRA parameters from {path}")

    return checkpoint.get('extra_info', {})


def print_lora_summary(model: nn.Module):
    """打印 LoRA 注入摘要"""
    total_params = 0
    total_trainable = 0
    lora_layers = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            n_params = module.lora_A.numel() + module.lora_B.numel()
            lora_layers.append({
                'name': name,
                'shape': f"({module.out_features}, {module.in_features})",
                'rank': module.rank,
                'lora_params': n_params,
                'merged': module._merged,
            })
            total_trainable += n_params

    for p in model.parameters():
        total_params += p.numel()

    print("=" * 70)
    print("LoRA Summary")
    print("=" * 70)
    for layer in lora_layers:
        print(f"  {layer['name']:50s} | {layer['shape']:15s} | rank={layer['rank']} | params={layer['lora_params']:,}")
    print("-" * 70)
    print(f"  Total LoRA params:   {total_trainable:>12,}")
    print(f"  Total model params:  {total_params:>12,}")
    print(f"  LoRA ratio:          {100 * total_trainable / total_params:>11.2f}%")
    print("=" * 70)

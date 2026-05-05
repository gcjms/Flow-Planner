from __future__ import annotations

from typing import Any, Dict


def unwrap_state_dict(ckpt: Any) -> Dict[str, Any]:
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in (
            'ema_state_dict',
            'state_dict',
            'model_state_dict',
            'anchor_predictor_state_dict',
            'goal_predictor_state_dict',
            'model',
        ):
            value = ckpt.get(key)
            if isinstance(value, dict):
                state_dict = value
                break

    if not isinstance(state_dict, dict):
        raise TypeError(f'Expected checkpoint dict, got {type(state_dict)!r}')

    for prefix in ('module.', 'model.', 'goal_predictor.', 'anchor_predictor.'):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def extract_predictor_head_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith('head.') for k in state_dict):
        return {
            k[len('head.'):]: v
            for k, v in state_dict.items()
            if k.startswith('head.')
        }
    return state_dict

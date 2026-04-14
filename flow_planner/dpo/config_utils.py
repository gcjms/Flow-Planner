from pathlib import Path
from typing import Any, Iterable, List

from omegaconf import DictConfig, OmegaConf


def _set_nested_value(target: dict, group: str, value: Any) -> dict:
    current = target
    parts = group.split("/")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    return target


def _candidate_search_roots(config_path: Path, raw_cfg: DictConfig) -> List[Path]:
    roots: List[Path] = [config_path.parent]

    project_root = raw_cfg.get("project_root")
    if project_root:
        roots.append(Path(project_root) / "flow_planner" / "script")

    roots.append(Path(__file__).resolve().parents[1] / "script")

    unique_roots: List[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _load_group_config(search_roots: Iterable[Path], group: str, option: str) -> DictConfig:
    rel_path = Path(group) / f"{option}.yaml"
    for root in search_roots:
        candidate = root / rel_path
        if candidate.exists():
            return OmegaConf.load(candidate)
    raise FileNotFoundError(f"Could not resolve Hydra default '{group}: {option}'")


def load_composed_config(config_path: str) -> DictConfig:
    """
    Load a Hydra entry config and expand its defaults into a concrete config.

    This is needed for copied checkpoint configs like `checkpoints/config_goal.yaml`,
    which are no longer stored beside the original Hydra config tree.
    """
    path = Path(config_path).resolve()
    raw_cfg = OmegaConf.load(path)
    defaults = raw_cfg.get("defaults")

    if not defaults:
        return raw_cfg

    search_roots = _candidate_search_roots(path, raw_cfg)
    body_cfg = OmegaConf.masked_copy(raw_cfg, [key for key in raw_cfg.keys() if key != "defaults"])

    composed = OmegaConf.create()
    merged_self = False

    for entry in defaults:
        if entry == "_self_":
            composed = OmegaConf.merge(composed, body_cfg)
            merged_self = True
            continue

        if not isinstance(entry, (dict, DictConfig)):
            continue

        group, option = next(iter(entry.items()))
        if option is None:
            continue

        group_cfg = _load_group_config(search_roots, group, str(option))
        wrapped_cfg = OmegaConf.create(_set_nested_value({}, group, group_cfg))
        composed = OmegaConf.merge(composed, wrapped_cfg)

    if not merged_self:
        composed = OmegaConf.merge(composed, body_cfg)

    return composed

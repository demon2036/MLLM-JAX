from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$(\w+|\{(\w+)\})")


def _yaml_parse_scalar(text: str) -> Any:
    if text == "":
        return ""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if not isinstance(obj, str):
        return obj

    def repl(match: re.Match[str]) -> str:
        var_name = match.group(2) or match.group(1)
        if var_name.startswith("{") and var_name.endswith("}"):
            var_name = var_name[1:-1]
        return os.environ.get(var_name, "")

    replaced = _ENV_PATTERN.sub(repl, obj)
    if replaced == obj:
        return obj
    return _yaml_parse_scalar(replaced)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _set_by_path(cfg: dict[str, Any], key_path: str, value: Any) -> None:
    keys = [k for k in key_path.split(".") if k]
    if not keys:
        raise ValueError("Empty override key")
    cur: dict[str, Any] = cfg
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        _set_by_path(cfg, key.strip(), _yaml_parse_scalar(raw.strip()))


def load_config(
    default_config: dict[str, Any],
    config_path: str | None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML config with a dict default + dotted-path overrides.

    The semantics are intentionally simple and shared across plugin CLIs:
    - Start from a deep copy of `default_config`.
    - Merge YAML file content (dict root) recursively.
    - Expand $ENV and ${ENV} strings (and parse scalars via YAML).
    - Apply overrides like: `train.optimizer.name=adamw`.
    """
    cfg = deepcopy(default_config)

    if config_path:
        data = yaml.safe_load(Path(config_path).read_text())
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"YAML root must be a dict, got: {type(data).__name__}")
        data = _expand_env_vars(data)
        _deep_update(cfg, data)

    if overrides:
        _apply_overrides(cfg, overrides)

    return cfg


__all__ = ["load_config"]


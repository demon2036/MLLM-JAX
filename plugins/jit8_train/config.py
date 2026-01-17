from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$(\w+|\{(\w+)\})")


DEFAULT_CONFIG: dict[str, Any] = {
    # Data/model
    "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_name": "openai/gsm8k",
    "dataset_config": "main",
    "dataset_split": "train",
    # Parallelism
    "mesh_dp": "-1,1,1",
    "mesh_fsdp": "1,-1,1",
    # Train loop
    "training_steps": 400,
    "batch_size": 8,
    "grad_accum_steps": 8,
    "num_pre_q": 16,
    "ppo_steps": 2,
    "ema_decay": 0.9,
    # Sequence lengths
    "max_length_sample": 1024,
    "max_length_extra": 512,
    "global_length": 512,
    # Rewards
    "reward_funcs_weights": [1.0, 0.5, 0.5],
    # Logging
    "wandb_enabled": True,
    "wandb_project": "grop-gsm8k",
    "wandb_name": "test",
    # JAX knobs
    "jax_compilation_cache_dir": None,
    "params_dtype": "bfloat16",
    # Debugging / safety
    "validate_schema": False,
}


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


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)

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

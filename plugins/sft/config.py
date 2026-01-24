from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$(\w+|\{(\w+)\})")


DEFAULT_CONFIG: dict[str, Any] = {
    # Execution backend:
    # - "jax": TPU-friendly JAX/Flax training + eval (recommended)
    # - "hf": legacy PyTorch + transformers.Trainer (deprecated)
    "backend": "jax",
    "base_model": "Qwen/Qwen2.5-0.5B",
    "output_dir": "runs/sid_sft",
    "seed": 42,
    "jax": {
        # Mesh dims for ("dp", "fsdp", "tp"). Accepts either "1,-1,1" or named dims like "dp:1,fsdp:-1,tp:1".
        "mesh_shape": "1,-1,1",
        # Dtypes: "float32" | "bfloat16" | "float16"
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        # Cache length used for constrained generation eval (prompt_len + max_new_tokens must fit).
        "max_cache_length": 2048,
    },
    "data": {
        "category": "Industrial_and_Scientific",
        "train_file": "workdir/MiniOneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "eval_file": "workdir/MiniOneRec/data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "test_file": "workdir/MiniOneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "info_file": "workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt",
        "sid_index_path": "workdir/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json",
        "item_meta_path": "workdir/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json",
        "max_len": 512,
        "sample_train": -1,
        "sample_eval": -1,
        "sample_test": -1,
    },
    "tasks": {
        "sid_next_item": True,
        "sid_item_alignment": True,
        "fusion_seq_rec": True,
    },
    "train": {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        # Optimizer: "adamw" | "lion"
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "num_train_epochs": 1,
        # If > 0, overrides num_train_epochs.
        "max_steps": -1,
        "warmup_steps": 0,
        "logging_steps": 10,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 1,
        "group_by_length": False,
        "freeze_LLM": False,
        "train_from_scratch": False,
        "resume_from_checkpoint": None,
        "early_stopping_patience": 3,
        # Dtype flags (Trainer will pick the appropriate one).
        "bf16": False,
        "fp16": False,
    },
    "eval": {
        "enabled": True,
        "batch_size": 4,
        "num_beams": 50,
        "max_new_tokens": 64,
        "length_penalty": 0.0,
        "topk": [1, 3, 5, 10, 20, 50],
        "constrained": True,
        "save_predictions_json": True,
    },
    "wandb": {
        "project": "minionerec-sid-sft",
        "mode": "online",
        "name": None,
    },
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


__all__ = ["DEFAULT_CONFIG", "load_config"]

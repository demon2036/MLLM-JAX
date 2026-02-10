from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        raise ValueError("Config path is required (pass --config ...).")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root type: {type(cfg)} (expected mapping)")
    return copy.deepcopy(cfg)


__all__ = ["load_config"]


from __future__ import annotations

from typing import Any


def ensure_rope_theta(config: Any) -> Any:
    """Best-effort compatibility shim for RoPE config fields.

    Newer `transformers` versions (>=5) moved Qwen2 RoPE parameters under
    `rope_scaling` / `rope_parameters` dicts, while some model code paths still
    expect a top-level `config.rope_theta`.
    """
    if hasattr(config, "rope_theta"):
        return config

    for key in ("rope_parameters", "rope_scaling"):
        value = getattr(config, key, None)
        if isinstance(value, dict) and "rope_theta" in value:
            try:
                rope_theta = float(value["rope_theta"])
            except Exception:
                rope_theta = value["rope_theta"]
            setattr(config, "rope_theta", rope_theta)
            return config

    return config


__all__ = ["ensure_rope_theta"]


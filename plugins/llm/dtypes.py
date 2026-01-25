from __future__ import annotations

from typing import Any


def parse_dtype(name: str) -> Any:
    import jax.numpy as jnp

    n = str(name or "float32").strip().lower()
    if n in {"float32", "f32"}:
        return jnp.float32
    if n in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if n in {"float16", "f16"}:
        return jnp.float16
    raise ValueError(f"Unsupported dtype: {name!r}")


__all__ = ["parse_dtype"]


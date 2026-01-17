"""Reusable training utilities (plugins-first).

This package is intentionally small and non-invasive:
- It defines contracts and helpers that higher-level training plugins (e.g. `jit8_train`)
  can reuse.
- It does not modify upstream code under `MLLM_JAX/` or root training scripts.
"""


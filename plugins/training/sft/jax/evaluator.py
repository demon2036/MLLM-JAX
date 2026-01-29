from __future__ import annotations

"""Compatibility shim for MiniOneRec JAX evaluator (plugin-owned path)."""

from projects.minionerec.sft.jax.evaluator import SidNextItemJaxEvaluator, evaluate_sid_next_item_jax

__all__ = ["SidNextItemJaxEvaluator", "evaluate_sid_next_item_jax"]

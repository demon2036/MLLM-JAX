"""Training-time Flax modules that extend/override upstream modules (plugins-first).

This package is intentionally small: keep custom code here and avoid invasive
edits to upstream folders (e.g. `MLLM_JAX/`).
"""

from plugins.training.train_modules.grpo_pallas import TrainGRPOModulePallas

__all__ = ["TrainGRPOModulePallas"]


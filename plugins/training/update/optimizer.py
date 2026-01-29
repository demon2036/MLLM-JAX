"""DEPRECATED: use `plugins.training.core.optim.optimizer`.

This module is a compatibility shim for older import paths.
"""

from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig, build_lr_schedule, build_tx

__all__ = ["LRScheduleConfig", "OptimizerConfig", "build_lr_schedule", "build_tx"]


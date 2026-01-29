"""DEPRECATED: use `plugins.training.core.logging.wandb`.

This module is a compatibility shim for older import paths.
"""

from plugins.training.core.logging.wandb import maybe_init_wandb

__all__ = ["maybe_init_wandb"]


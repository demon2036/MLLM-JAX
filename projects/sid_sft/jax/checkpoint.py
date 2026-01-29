"""DEPRECATED: use `plugins.training.core.checkpoint.msgpack`.

This module is kept as a compatibility shim for older import paths.
"""

from plugins.training.core.checkpoint.msgpack import load_checkpoint, save_checkpoint

__all__ = ["load_checkpoint", "save_checkpoint"]


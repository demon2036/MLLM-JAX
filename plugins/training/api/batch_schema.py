"""DEPRECATED: use `plugins.api.training.schemas.grpo_batch`.

This module is a compatibility shim for older import paths.
"""

from plugins.api.training.schemas.grpo_batch import BatchSchemaError, validate_grpo_batch

__all__ = ["BatchSchemaError", "validate_grpo_batch"]


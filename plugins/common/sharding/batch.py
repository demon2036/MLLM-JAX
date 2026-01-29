"""DEPRECATED: use `plugins.training.core.sharding.batch`.

This module is a compatibility shim for older import paths.
"""

from plugins.training.core.sharding.batch import local_from_global, make_form_training_global_array

__all__ = ["local_from_global", "make_form_training_global_array"]


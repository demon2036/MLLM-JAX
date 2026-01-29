"""DEPRECATED: use `plugins.training.core.io.hf_safetensors`.

This module is a compatibility shim for older import paths.
"""

from plugins.training.core.io.hf_safetensors import load_hf_safetensors_state_dict

__all__ = ["load_hf_safetensors_state_dict"]


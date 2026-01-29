"""DEPRECATED: use `plugins.training.core.runtime.env`.

This module is a compatibility shim for older import paths.
"""

from plugins.training.core.runtime.env import load_dotenv_if_present

__all__ = ["load_dotenv_if_present"]


"""Stub `bitsandbytes` for CPU/TPU environments.

MiniOneRec's upstream scripts import `bitsandbytes` unconditionally even when
running on CPU/TPU. On TPU VMs we usually don't have CUDA, so installing the
real package is unnecessary (and often fails). This stub exists only to satisfy
imports for reference runs; it is not a functional replacement.
"""

__all__ = ["__version__"]
__version__ = "0.0.0-stub"


"""Sampling backends (engines that implement prompt -> tokens/text).

Canonical implementations live here. Prefer importing from this package in new
code; `plugins.sample.mllm_sampler` remains as a compatibility re-export.
"""

from plugins.sample.backends.mllm_jax_sampler import Sampler, SampleState, get_model, get_params

__all__ = ["Sampler", "SampleState", "get_model", "get_params"]


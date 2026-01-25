from __future__ import annotations

from plugins.sample.optimizations.fast_sampler_generate import patch_sampler_generate_fast
from plugins.sample.optimizations.qwen2_decode_attention import patch_qwen2_attention_decode_fast

__all__ = ["patch_sampler_generate_fast", "patch_qwen2_attention_decode_fast"]


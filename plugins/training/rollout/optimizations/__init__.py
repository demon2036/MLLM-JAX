from __future__ import annotations

from plugins.sample.optimizations import patch_sampler_generate_fast, patch_qwen2_attention_decode_fast

__all__ = ["patch_sampler_generate_fast", "patch_qwen2_attention_decode_fast"]

"""LLM platform utilities (model/tokenizer/weights/sharding).

This package is the shared "LLM layer" used across SFT + RL runners so we don't
duplicate:
- HF config/tokenizer loading
- vocab padding/resizing
- weight loading (safetensors-first, torch fallback)
- params sharding/placement
"""

from plugins.llm.bundle import LlmBundle, build_llm_bundle

__all__ = ["LlmBundle", "build_llm_bundle"]


from __future__ import annotations

"""Compatibility shim for MiniOneRec SID token utilities."""

from projects.minionerec.sft.tokens import (
    TokenExtensionResult,
    TokenExtender,
    freeze_llm_only_train_new_embeddings,
    maybe_extend_tokenizer,
    maybe_extend_tokenizer_and_model,
)

__all__ = [
    "TokenExtensionResult",
    "TokenExtender",
    "freeze_llm_only_train_new_embeddings",
    "maybe_extend_tokenizer",
    "maybe_extend_tokenizer_and_model",
]

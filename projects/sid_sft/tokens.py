from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TokenExtender:
    """Collect new SID tokens from a MiniOneRec `.index.json` file."""

    def __init__(self, sid_index_path: str):
        self.sid_index_path = sid_index_path
        self._indices: dict[str, list[str]] | None = None
        self._new_tokens: list[str] | None = None

    def _load(self) -> None:
        self._indices = json.loads(Path(self.sid_index_path).read_text())

    def get_new_tokens(self) -> list[str]:
        if self._new_tokens is not None:
            return self._new_tokens
        if self._indices is None:
            self._load()
        assert self._indices is not None
        tokens: set[str] = set()
        for index in self._indices.values():
            for token in index:
                tokens.add(str(token))
        self._new_tokens = sorted(tokens)
        return self._new_tokens


@dataclass(frozen=True)
class TokenExtensionResult:
    original_vocab_size: int
    added_tokens: list[str]
    new_vocab_size: int


def maybe_extend_tokenizer_and_model(*, tokenizer: Any, model: Any | None, sid_index_path: str | None) -> TokenExtensionResult:
    original_vocab_size = int(len(tokenizer))
    if not sid_index_path:
        return TokenExtensionResult(original_vocab_size=original_vocab_size, added_tokens=[], new_vocab_size=original_vocab_size)
    if not Path(sid_index_path).is_file():
        return TokenExtensionResult(original_vocab_size=original_vocab_size, added_tokens=[], new_vocab_size=original_vocab_size)

    extender = TokenExtender(sid_index_path)
    new_tokens = extender.get_new_tokens()
    existing = set(getattr(tokenizer, "get_vocab")().keys())
    to_add = [t for t in new_tokens if t not in existing]

    if to_add:
        tokenizer.add_tokens(to_add)
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))

    return TokenExtensionResult(original_vocab_size=original_vocab_size, added_tokens=to_add, new_vocab_size=int(len(tokenizer)))


def maybe_extend_tokenizer(*, tokenizer: Any, sid_index_path: str | None) -> TokenExtensionResult:
    """Tokenizer-only SID token extension (JAX backend-friendly)."""
    return maybe_extend_tokenizer_and_model(tokenizer=tokenizer, model=None, sid_index_path=sid_index_path)


def freeze_llm_only_train_new_embeddings(*, model: Any, original_vocab_size: int) -> None:
    """Freeze all weights and mask grads to train only newly-added embedding rows."""
    for param in model.parameters():
        param.requires_grad = False

    embedding_layer = model.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    original_vocab_size = int(original_vocab_size)

    def mask_grad(grad):
        grad[:original_vocab_size].zero_()
        return grad

    embedding_layer.weight.register_hook(mask_grad)

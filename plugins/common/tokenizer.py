from __future__ import annotations

from typing import Any


def prepare_tokenizer(tokenizer: Any, *, padding_side: str = "right") -> tuple[Any, int]:
    """Normalize tokenizer defaults used across runners.

    - Ensures a usable `pad_token_id` (falls back to EOS token when possible).
    - Sets `padding_side` (default: right).
    """
    try:
        tokenizer.padding_side = str(padding_side)
    except Exception:
        pass

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is not None:
            try:
                tokenizer.pad_token = eos_token
            except Exception:
                pass
        if eos_token_id is not None:
            try:
                tokenizer.pad_token_id = eos_token_id
            except Exception:
                pass
        pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = 0
    return tokenizer, int(pad_token_id)


__all__ = ["prepare_tokenizer"]


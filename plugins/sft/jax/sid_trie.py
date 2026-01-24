from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SidTrie:
    """Depth-3 SID trie for constrained decoding.

    Generation steps after the prompt's `### Response:\n` prefix:
    - step0: token_1 must be in `first_ids`
    - step1: token_2 must be allowed by token_1 (lookup in `second_table`)
    - step2: token_3 must be allowed by (token_1, token_2) (lookup in `third_table`)
    - step3: EOS only

    Tables are padded with `pad_id` (-1) and require masking at runtime.
    """

    pad_id: int
    eos_token_id: int
    vocab_size: int

    first_ids: np.ndarray  # int32 [N1] sorted

    second_keys: np.ndarray  # int32 [N1] sorted (== first_ids)
    second_table: np.ndarray  # int32 [N1, M2] padded with pad_id

    third_table: np.ndarray  # int32 [N1, M2, M3] padded with pad_id


def _token_to_id(tokenizer: Any, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        return int(tokenizer.convert_tokens_to_ids(token))
    # Fallback: encode single token without specials; expect 1 id.
    ids = list(tokenizer.encode(token, add_special_tokens=False))
    if len(ids) != 1:
        raise ValueError(f"Token {token!r} encoded to {ids}, expected a single id")
    return int(ids[0])


def build_sid_trie_from_index(
    *,
    tokenizer: Any,
    sid_index_path: str,
    eos_token_id: int,
    pad_id: int = -1,
) -> SidTrie:
    indices = json.loads(Path(sid_index_path).read_text(encoding="utf-8"))
    if not isinstance(indices, dict):
        raise TypeError("sid_index_path must contain a JSON object mapping item_id->sid_tokens")

    first: set[int] = set()
    second: dict[int, set[int]] = {}
    third: dict[tuple[int, int], set[int]] = {}

    for _item_id, toks in indices.items():
        if not isinstance(toks, list) or len(toks) < 3:
            continue
        t1 = _token_to_id(tokenizer, str(toks[0]))
        t2 = _token_to_id(tokenizer, str(toks[1]))
        t3 = _token_to_id(tokenizer, str(toks[2]))
        first.add(t1)
        second.setdefault(t1, set()).add(t2)
        third.setdefault((t1, t2), set()).add(t3)

    first_ids = np.asarray(sorted(first), dtype=np.int32)
    if first_ids.size == 0:
        raise ValueError("No SID tokens found in sid_index_path (expected lists of 3 tokens).")

    # Second-level table keyed by token_1.
    second_keys = first_ids
    max_second = max(len(second.get(int(t1), set())) for t1 in second_keys)
    second_table = np.full((len(second_keys), int(max_second)), int(pad_id), dtype=np.int32)
    for i, t1 in enumerate(second_keys):
        vals = sorted(second.get(int(t1), set()))
        if vals:
            second_table[i, : len(vals)] = np.asarray(vals, dtype=np.int32)

    # Third-level table keyed by (token_1 row, token_2 col).
    max_third = 0
    for i, t1 in enumerate(second_keys):
        for j in range(int(max_second)):
            t2 = int(second_table[i, j])
            if t2 == int(pad_id):
                continue
            max_third = max(max_third, len(third.get((int(t1), t2), set())))
    if max_third <= 0:
        raise ValueError("No third-level SID expansions found (unexpected empty SID trie).")

    third_table = np.full((len(second_keys), int(max_second), int(max_third)), int(pad_id), dtype=np.int32)
    for i, t1 in enumerate(second_keys):
        for j in range(int(max_second)):
            t2 = int(second_table[i, j])
            if t2 == int(pad_id):
                continue
            vals = sorted(third.get((int(t1), t2), set()))
            if vals:
                third_table[i, j, : len(vals)] = np.asarray(vals, dtype=np.int32)

    vocab_size = int(len(tokenizer))
    return SidTrie(
        pad_id=int(pad_id),
        eos_token_id=int(eos_token_id),
        vocab_size=vocab_size,
        first_ids=first_ids,
        second_keys=second_keys,
        second_table=second_table,
        third_table=third_table,
    )


__all__ = ["SidTrie", "build_sid_trie_from_index"]

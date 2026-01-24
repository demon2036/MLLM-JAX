from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from transformers.generation import LogitsProcessor


def load_valid_sids_from_info(info_file: str) -> list[str]:
    lines = Path(info_file).read_text(encoding="utf-8").splitlines()
    sids: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        # expected format: sid \t title \t item_id
        sid = line.split("\t", 1)[0].strip()
        if sid:
            sids.append(sid)
    return sids


@dataclass(frozen=True)
class SidConstraint:
    prefix_token_count: int
    allowed_next_tokens: dict[tuple[int, ...], list[int]]
    valid_sid_set: set[str]

    def prefix_allowed_tokens_fn(self) -> Callable[[int, list[int]], list[int]]:
        allowed = self.allowed_next_tokens

        def fn(_batch_id: int, hash_key: list[int]) -> list[int]:
            return allowed.get(tuple(int(x) for x in hash_key), [])

        return fn


def build_sid_constraint(
    *,
    tokenizer: Any,
    valid_sids: list[str],
    eos_token_id: int,
    prefix_text: str = "### Response:\n",
    append_newline: bool = True,
) -> SidConstraint:
    prefix_ids = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    prefix_len = int(len(prefix_ids))
    if prefix_len <= 0:
        raise ValueError("prefix_text produced empty token ids; cannot build constraint.")

    allowed: dict[tuple[int, ...], set[int]] = {}
    for sid in valid_sids:
        sid_text = sid + ("\n" if append_newline else "")
        sid_ids = list(tokenizer.encode(sid_text, add_special_tokens=False))
        seq = prefix_ids + sid_ids + [int(eos_token_id)]

        for i in range(prefix_len, len(seq)):
            if i == prefix_len:
                key = tuple(int(x) for x in seq[:i])  # prefix
            else:
                key = tuple(int(x) for x in seq[prefix_len:i])  # generated suffix (SID partial)
            allowed.setdefault(key, set()).add(int(seq[i]))

    allowed_lists = {k: sorted(v) for k, v in allowed.items()}
    return SidConstraint(
        prefix_token_count=prefix_len,
        allowed_next_tokens=allowed_lists,
        valid_sid_set=set(valid_sids),
    )


class ConstrainedLogitsProcessor(LogitsProcessor):
    """Stateful constrained decoding helper (beam-search friendly).

    This mirrors the upstream MiniOneRec approach: at step 0 we look at the tail
    of the prompt (prefix tokens) and then progressively constrain the generated
    suffix to match one of the valid SID token sequences.
    """

    def __init__(
        self,
        *,
        prefix_allowed_tokens_fn: Callable[[int, list[int]], list[int]],
        num_beams: int,
        prefix_token_count: int,
        eos_token_id: int,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = int(num_beams)
        self._prefix_token_count = int(prefix_token_count)
        self._eos_token_id = int(eos_token_id)
        self._count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float("-inf"))

        beams = input_ids.view(-1, self._num_beams, input_ids.shape[-1])
        for batch_id, beam_sent in enumerate(beams):
            for beam_id, sent in enumerate(beam_sent):
                if self._count == 0:
                    hash_key = sent[-self._prefix_token_count :].tolist()
                else:
                    hash_key = sent[-self._count :].tolist()

                allowed = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                if not allowed:
                    # If the hypothesis is already invalid, force EOS so it terminates.
                    mask[batch_id * self._num_beams + beam_id, self._eos_token_id] = 0
                    continue
                mask[batch_id * self._num_beams + beam_id, allowed] = 0

        self._count += 1
        return scores + mask


def compute_prefix_index(tokenizer: Any, *, prefix_text: str = "### Response:\n") -> int:
    """Compute the token count for the decoding prefix (model/tokenizer-specific)."""
    prefix_ids = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    if not prefix_ids:
        raise ValueError("prefix_text produced empty token ids")
    return int(len(prefix_ids))


from __future__ import annotations

from typing import Sequence

import numpy as np


def _find_subsequence(haystack: np.ndarray, needle: np.ndarray) -> int | None:
    haystack = np.asarray(haystack).reshape(-1)
    needle = np.asarray(needle).reshape(-1)
    if needle.size == 0:
        raise ValueError("needle must be non-empty")
    if haystack.size < needle.size:
        return None
    m = int(needle.size)
    for i in range(int(haystack.size) - m + 1):
        if np.array_equal(haystack[i : i + m], needle):
            return int(i)
    return None


def build_adv_zero_think_window_mask(
    *,
    input_ids: np.ndarray,
    labels: np.ndarray,
    tag_token_ids: Sequence[int],
    window_tokens: int,
    start_after_tag: bool = True,
    no_think_policy: str = "first_tokens",
    sequence_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a token mask for adv==0 samples focused on a <think> window.

    This is a host-side helper. It does NOT depend on JAX and is intended to
    run inside the Python training loop (before sharding to devices).

    Semantics
    ---------
    For each selected sequence (where `sequence_mask[i]` is True):
      - Find the first occurrence of `tag_token_ids` within the completion
        region (where `labels>0`).
      - If found, start the window either at the tag start or right after the
        tag (controlled by `start_after_tag`).
      - Select the next `window_tokens` completion tokens and set them to 1.0
        in the output mask.
      - If not found:
        - no_think_policy == "first_tokens": select the first `window_tokens`
          completion tokens.
        - no_think_policy == "mask_all": select nothing.

    Returns
    -------
    window_mask:
      float32 array of shape [B, L] with 0/1 entries.
    tag_found:
      bool array of shape [B] indicating whether the tag was found (only for
      sequences where `sequence_mask` is True).
    """
    input_ids = np.asarray(input_ids)
    labels = np.asarray(labels)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D [B, L], got shape={input_ids.shape}")
    if labels.shape != input_ids.shape:
        raise ValueError(f"labels must match input_ids shape, got {labels.shape} vs {input_ids.shape}")

    tag_ids = np.asarray(list(tag_token_ids), dtype=np.int64).reshape(-1)
    if tag_ids.size == 0:
        raise ValueError("tag_token_ids must be non-empty")

    k = int(window_tokens)
    if k < 1:
        raise ValueError(f"window_tokens must be >= 1, got {window_tokens!r}")

    policy = str(no_think_policy or "").strip().lower()
    if policy not in {"first_tokens", "mask_all"}:
        raise ValueError("no_think_policy must be one of: first_tokens, mask_all")

    if sequence_mask is None:
        seq_mask = np.ones((int(input_ids.shape[0]),), dtype=bool)
    else:
        seq_mask = np.asarray(sequence_mask).reshape(-1).astype(bool)
        if seq_mask.shape != (int(input_ids.shape[0]),):
            raise ValueError(f"sequence_mask must have shape [B], got {seq_mask.shape}")

    bsz, seqlen = input_ids.shape
    window_mask = np.zeros((int(bsz), int(seqlen)), dtype=np.float32)
    tag_found = np.zeros((int(bsz),), dtype=bool)

    for i in range(int(bsz)):
        if not bool(seq_mask[i]):
            continue

        completion_positions = np.flatnonzero(labels[i] > 0)
        if completion_positions.size == 0:
            continue

        completion_ids = input_ids[i, completion_positions]
        match_idx = _find_subsequence(completion_ids, tag_ids)

        if match_idx is None:
            if policy == "mask_all":
                continue
            start_pos = 0
        else:
            tag_found[i] = True
            start_pos = int(match_idx) + (int(tag_ids.size) if bool(start_after_tag) else 0)

        if start_pos >= int(completion_positions.size):
            continue

        chosen = completion_positions[start_pos : start_pos + int(k)]
        if chosen.size > 0:
            window_mask[i, chosen] = 1.0

    return window_mask, tag_found


__all__ = ["build_adv_zero_think_window_mask"]


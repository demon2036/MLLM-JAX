from __future__ import annotations

import jax.numpy as jnp


def build_token_focus_mask(
    logps: jnp.ndarray,
    valid_mask: jnp.ndarray,
    *,
    prob_threshold: float,
    max_tokens_per_sequence: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a token-level mask that keeps only early low-probability tokens.

    Semantics
    ---------
    For each sequence (row), mark tokens as eligible when:
      - valid_mask > 0 (typically completion tokens), and
      - prob(token) < prob_threshold

    Then keep only the first `max_tokens_per_sequence` eligible tokens when
    scanning left-to-right.

    Notes
    -----
    - This function uses `logps` instead of explicit `prob` for numerical
      stability: `prob < p` ⇔ `logp < log(p)`.
    - Returned `focus_mask` is float32 with shape [B, T] (0.0 or 1.0).
    - `selected_counts` is int32 with shape [B].
    """
    p = float(prob_threshold)
    if not (0.0 < p < 1.0):
        raise ValueError(f"prob_threshold must be in (0, 1), got {prob_threshold!r}")
    k = int(max_tokens_per_sequence)
    if k < 1:
        raise ValueError(f"max_tokens_per_sequence must be >= 1, got {max_tokens_per_sequence!r}")

    logp_threshold = jnp.log(jnp.asarray(p, dtype=logps.dtype))
    eligible = jnp.logical_and(valid_mask > 0, logps < logp_threshold)
    # Keep the first K eligible tokens per sequence.
    eligible_count_prefix = jnp.cumsum(eligible.astype(jnp.int32), axis=-1)
    keep = jnp.logical_and(eligible, eligible_count_prefix <= k)

    focus_mask = keep.astype(jnp.float32)
    selected_counts = focus_mask.sum(axis=-1).astype(jnp.int32)
    return focus_mask, selected_counts


__all__ = ["build_token_focus_mask"]

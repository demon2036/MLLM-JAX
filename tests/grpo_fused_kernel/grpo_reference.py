# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

# Pure-Python reference shapes (no JAX required).
LogitsPy = list[list[list[float]]]
LogpPy = list[list[float]]
IdsPy = list[list[int]]
AdvantagesPy = list[float]
MaskPy = list[list[int]]

LossPy = list[list[float]]
KlPy = list[list[float]] | None
IsClippedPy = list[list[bool]]


def _to_list(x: object) -> object:
    """Convert array-likes to nested Python lists (best-effort)."""

    if x is None:
        return None
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    return x


def _logsumexp(values: Sequence[float]) -> float:
    # Stable logsumexp for small vectors.
    m = max(values)
    if m == -math.inf:
        return -math.inf
    s = 0.0
    for v in values:
        s += math.exp(v - m)
    return m + math.log(s)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def grpo_loss_reference_python(
    logits: LogitsPy,
    old_logp: LogpPy | None,
    ref_logp: LogpPy | None,
    completion_ids: IdsPy,
    advantages: AdvantagesPy,
    completion_mask: MaskPy | None = None,
    temperature: float = 1.0,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> tuple[LossPy, KlPy, IsClippedPy]:
    """Reference GRPO loss using pure Python lists.

    This intentionally favors portability (no JAX import required) over speed.
    Shapes follow the Liger-style API:

    - logits: [B, L+1, V]
    - completion_ids: [B, L]
    - advantages: [B]
    - old_logp/ref_logp (optional): [B, L]
    - completion_mask (optional): [B, L] (0/1)

    Returns:
      (loss, kl, is_clipped) where loss/is_clipped are [B, L]. kl is None if
      beta == 0.
    """

    bsz = len(logits)
    l_add_1 = len(logits[0])
    seq_len = l_add_1 - 1

    loss_out = [[0.0 for _ in range(seq_len)] for _ in range(bsz)]
    is_clipped_out = [[False for _ in range(seq_len)] for _ in range(bsz)]
    kl_out = [[0.0 for _ in range(seq_len)] for _ in range(bsz)] if beta != 0.0 else None

    if beta != 0.0 and ref_logp is None:
        raise ValueError("ref_logp must be provided when beta != 0")

    for b in range(bsz):
        adv = float(advantages[b])
        for t in range(seq_len):
            if completion_mask is not None and int(completion_mask[b][t]) == 0:
                continue

            token_id = int(completion_ids[b][t])
            scaled = [float(x) / float(temperature) for x in logits[b][t]]
            lse = _logsumexp(scaled)
            logp = scaled[token_id] - lse
            old = logp if old_logp is None else float(old_logp[b][t])
            ratio = math.exp(logp - old)
            clipped_ratio = _clamp(ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))

            per_token_loss1 = ratio * adv
            per_token_loss2 = clipped_ratio * adv
            per_token_loss = -min(per_token_loss1, per_token_loss2)

            is_low_clipped = (ratio < 1.0 - float(eps_low)) and (adv < 0.0)
            is_high_clipped = (ratio > 1.0 + float(eps_high)) and (adv > 0.0)
            is_clipped = is_low_clipped or is_high_clipped

            if beta != 0.0:
                assert ref_logp is not None
                ref = float(ref_logp[b][t])
                kl = math.exp(ref - logp) - (ref - logp) - 1.0
                per_token_loss += float(beta) * kl
                assert kl_out is not None
                kl_out[b][t] = kl

            loss_out[b][t] = per_token_loss
            is_clipped_out[b][t] = is_clipped

    return loss_out, kl_out, is_clipped_out


def grpo_loss_reference_jax(
    logits: object,
    old_logp: object | None,
    ref_logp: object | None,
    completion_ids: object,
    advantages: object,
    completion_mask: object | None = None,
    temperature: float = 1.0,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> tuple[object, object | None, object]:
    """Reference GRPO loss using JAX ops (CPU/GPU/TPU)."""

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(logits)
    completion_ids = jnp.asarray(completion_ids)
    advantages = jnp.asarray(advantages, dtype=jnp.float32)
    completion_mask = None if completion_mask is None else jnp.asarray(completion_mask)

    seq_len = logits.shape[1] - 1

    # bf16 logsumexp/gather can be numerically fragile; do reductions in fp32.
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse

    if old_logp is None:
        old = logp
    else:
        old = jnp.asarray(old_logp, dtype=logp.dtype)

    ratio = jnp.exp(logp - old)
    clipped_ratio = jnp.clip(ratio, 1.0 - eps_low, 1.0 + eps_high)
    adv = advantages[:, None]

    per_token_loss1 = ratio * adv
    per_token_loss2 = clipped_ratio * adv
    per_token_loss = -jnp.minimum(per_token_loss1, per_token_loss2)

    is_low_clipped = (ratio < 1.0 - eps_low) & (adv < 0.0)
    is_high_clipped = (ratio > 1.0 + eps_high) & (adv > 0.0)
    is_clipped = is_low_clipped | is_high_clipped

    if beta != 0.0:
        if ref_logp is None:
            raise ValueError("ref_logp must be provided when beta != 0")
        ref = jnp.asarray(ref_logp, dtype=logp.dtype)
        kl = jnp.exp(ref - logp) - (ref - logp) - 1.0
        per_token_loss = per_token_loss + beta * kl
    else:
        kl = None

    if completion_mask is not None:
        keep = completion_mask.astype(bool)
        per_token_loss = jnp.where(keep, per_token_loss, 0.0)
        is_clipped = jnp.where(keep, is_clipped, False)
        if kl is not None:
            kl = jnp.where(keep, kl, 0.0)

    return per_token_loss, kl, is_clipped


def grpo_loss_reference(
    logits: object,
    old_logp: object | None,
    ref_logp: object | None,
    completion_ids: object,
    advantages: object,
    completion_mask: object | None = None,
    temperature: float = 1.0,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    *,
    backend: str | None = None,
) -> tuple[object, object | None, object]:
    """Backend-switching wrapper.

    backend:
      - None: heuristic (JAX array inputs -> jax, otherwise python)
      - "jax": force JAX implementation
      - "python": force pure-Python implementation
    """

    if backend == "python":
        return grpo_loss_reference_python(
            cast(LogitsPy, _to_list(logits)),
            cast(LogpPy | None, _to_list(old_logp)),
            cast(LogpPy | None, _to_list(ref_logp)),
            cast(IdsPy, _to_list(completion_ids)),
            cast(AdvantagesPy, _to_list(advantages)),
            completion_mask=cast(MaskPy | None, _to_list(completion_mask)),
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )
    if backend == "jax":
        return grpo_loss_reference_jax(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )

    # Heuristic: JAX arrays often have a type module like "jax.*".
    if type(logits).__module__.startswith("jax"):
        return grpo_loss_reference_jax(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )

    return grpo_loss_reference_python(
        cast(LogitsPy, _to_list(logits)),
        cast(LogpPy | None, _to_list(old_logp)),
        cast(LogpPy | None, _to_list(ref_logp)),
        cast(IdsPy, _to_list(completion_ids)),
        cast(AdvantagesPy, _to_list(advantages)),
        completion_mask=cast(MaskPy | None, _to_list(completion_mask)),
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
    )


__all__ = [
    "grpo_loss_reference",
    "grpo_loss_reference_jax",
    "grpo_loss_reference_python",
]

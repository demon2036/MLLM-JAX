"""Runtime patches for sglang-jax to make it usable in this repo's training loop.

We avoid modifying the upstream `workdir/sglang-jax` checkout. Instead we apply
small monkey-patches at runtime from `plugins/`.
"""

from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)


def patch_sglang_sampler_penalty_cond_sharding() -> bool:
    """Fix a TPU crash in sglang-jax Sampler when `do_penalties=False`.

    Some sglang-jax versions build `logits` without an explicit sharding annotation,
    but the `_apply_linear_penalty` branch produces `@tensor`-sharded logits. JAX
    requires `lax.cond` branches to return identical abstract types (including
    sharding), otherwise it raises:

      TypeError: cond branches must have equal output types but they differ.

    We enforce `@tensor` sharding on `logits_output.next_token_logits` before the
    penalty cond so both branches see a consistently sharded logits input.
    """

    try:
        import jax
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.layers.sampler import Sampler as SglangSampler
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-jax sampler patch skipped (import failed): %s", e)
        return False

    if getattr(SglangSampler, "_mllm_jax_patched_penalty_cond_sharding", False):
        return True

    orig_call = SglangSampler.__call__

    def patched_call(self, logits_output, sampling_metadata, use_sort_for_toppk_minp):  # noqa: ANN001
        try:
            logits = getattr(logits_output, "next_token_logits", None)
            mesh = getattr(self, "mesh", None)
            if logits is not None and mesh is not None:
                target = NamedSharding(mesh, P(None, "tensor"))
                logits = jax.lax.with_sharding_constraint(logits, target)
                logits_output = dataclasses.replace(logits_output, next_token_logits=logits)
        except Exception:
            # Best-effort patch: if anything goes wrong, fall back to upstream behavior.
            pass
        return orig_call(self, logits_output, sampling_metadata, use_sort_for_toppk_minp)

    SglangSampler.__call__ = patched_call  # type: ignore[assignment]
    setattr(SglangSampler, "_mllm_jax_patched_penalty_cond_sharding", True)
    logger.info("Applied sglang-jax Sampler penalty sharding patch.")
    return True


__all__ = ["patch_sglang_sampler_penalty_cond_sharding"]


"""Runtime patches for sglang-jax to make it usable in this repo's training loop.

We avoid modifying the upstream `workdir/sglang-jax` checkout. Instead we apply
small monkey-patches at runtime from `plugins/`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_sglang_sampler_penalty_cond_sharding() -> bool:
    """Fix a TPU crash in sglang-jax Sampler when `do_penalties=False`.

    Root cause (sglang-jax):
    - `SamplingMetadata.linear_penalty` is always a sharded `jax.Array` with
      `PartitionSpec(None, "tensor")` (even when it's all zeros).
    - `SamplingMetadata.do_penalties` can be False, so `Sampler.__call__` uses:

        lax.cond(do_penalties, _apply_linear_penalty, lambda: logits)

      which makes the `True` branch return `@tensor`-sharded logits (due to the
      sharded penalty add) but the `False` branch return an *unsharded* logits.
      JAX requires both `lax.cond` branches to return identical abstract types
      (including sharding), so this crashes on TPU with explicit sharding.

    Patch strategy:
    - Replace the penalty `lax.cond` with an unconditional penalty add.
      When penalties are disabled, `linear_penalty` is the cached all-zero buffer,
      so semantics are unchanged while sharding becomes consistent.
    """

    try:
        import jax
        from jax import lax

        from sgl_jax.srt.constrained.bitmask_ops import apply_token_bitmask
        from sgl_jax.srt.layers.sampler import Sampler as SglangSampler
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-jax sampler patch skipped (import failed): %s", e)
        return False

    if getattr(SglangSampler, "_mllm_jax_patched_penalty_cond_sharding", False):
        return True

    def patched_call(self, logits_output, sampling_metadata, use_sort_for_toppk_minp):  # noqa: ANN001
        # Unconditionally apply linear penalty (all-zero penalty when disabled).
        logits = self._apply_linear_penalty((logits_output.next_token_logits, sampling_metadata))

        # Apply grammar-constrained vocab mask.
        logits = lax.cond(
            sampling_metadata.apply_vocab_mask,
            lambda operands: apply_token_bitmask(operands[0], operands[1]),
            lambda operands: operands[0],
            (logits, sampling_metadata.vocab_mask),
        )

        _, rng = jax.random.split(self.rngs.params())
        operands = (logits, sampling_metadata, rng)
        regular_fn = lambda op: self._regular_sampling((*op, use_sort_for_toppk_minp))
        batch_next_token_ids, logprobs = lax.cond(
            sampling_metadata.is_all_greedy,
            self._greedy_sampling,
            regular_fn,
            operands,
        )

        logprob_operands = (
            logits_output,
            sampling_metadata,
            batch_next_token_ids,
            logprobs,
        )
        new_logits_output = None
        if sampling_metadata.return_logprob:
            new_logits_output = self._process_logprob_results(logprob_operands)

        return batch_next_token_ids, logprobs, new_logits_output

    SglangSampler.__call__ = patched_call  # type: ignore[assignment]
    setattr(SglangSampler, "_mllm_jax_patched_penalty_cond_sharding", True)
    logger.info("Applied sglang-jax Sampler penalty sharding patch.")
    return True


__all__ = ["patch_sglang_sampler_penalty_cond_sharding"]

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from plugins.training.kernels.tiled_cross_entropy_pallas import (
    CrossEntropyKernelConfig,
    cross_entropy_per_token_pallas,
    cross_entropy_per_token_reference,
)


def test_tiled_cross_entropy_pallas_matches_reference_forward_and_backward():
    key = jax.random.PRNGKey(0)

    batch = 2
    time = 5
    vocab = 17

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    labels = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    labels = labels.at[0, 0].set(-100)
    labels = labels.at[1, 3].set(-100)

    per_loss_ref, per_logp_ref = cross_entropy_per_token_reference(
        logits,
        labels,
        ignore_index=-100,
        temperature=1.0,
    )

    cfg = CrossEntropyKernelConfig(block_size=8, time_block=4, ignore_index=-100, temperature=1.0)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)
    per_loss_k, per_logp_k = cross_entropy_per_token_pallas(
        logits=logits,
        labels=labels,
        cfg=cfg,
        interpret=interpret,
        debug=False,
    )

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-5
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-5

    def loss_ref_fn(l):
        per_loss, _ = cross_entropy_per_token_reference(
            l,
            labels,
            ignore_index=-100,
            temperature=1.0,
        )
        return jnp.sum(per_loss)

    def loss_k_fn(l):
        per_loss, _ = cross_entropy_per_token_pallas(
            logits=l,
            labels=labels,
            cfg=cfg,
            interpret=interpret,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_k = jax.grad(loss_k_fn)(logits).astype(jnp.float32)

    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-5

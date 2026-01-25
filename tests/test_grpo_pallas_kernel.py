import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_reference,
)


@pytest.mark.parametrize("vocab", [15, 24])
def test_grpo_pallas_kernel_matches_reference_forward_and_backward(vocab: int):
    key = jax.random.PRNGKey(0)

    batch = 2
    time = 3

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    chosen_ids = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    _, logps_seed = grpo_per_token_loss_reference(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=jnp.zeros((batch, time), dtype=jnp.float32),
        advantages=advantages,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
    )
    old_per_token_logps = jax.lax.stop_gradient(logps_seed + 0.3)

    per_loss_ref, per_logp_ref = grpo_per_token_loss_reference(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
    )

    cfg = GRPOKernelConfig(block_size=8, epsilon_low=0.2, epsilon_high=0.2, temperature=1.0)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)
    per_loss_k, per_logp_k = grpo_per_token_loss_pallas(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        cfg=cfg,
        interpret=interpret,
        debug=False,
    )

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-5
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-5

    def loss_ref_fn(l):
        per_loss, _ = grpo_per_token_loss_reference(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=0.2,
            epsilon_high=0.2,
            temperature=1.0,
        )
        return jnp.sum(per_loss)

    def loss_k_fn(l):
        per_loss, _ = grpo_per_token_loss_pallas(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_k = jax.grad(loss_k_fn)(logits).astype(jnp.float32)

    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-5

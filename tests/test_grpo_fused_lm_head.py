import jax
import jax.numpy as jnp
import numpy as np

from plugins.training.kernels.grpo_fused_lm_head import (
    GRPOLmHeadFusedConfig,
    grpo_per_token_loss_fused_lm_head,
    grpo_per_token_loss_lm_head_reference,
)


def test_grpo_fused_lm_head_matches_reference_forward_and_backward():
    key = jax.random.PRNGKey(0)

    batch = 2
    time = 3
    hidden = 4
    vocab = 17

    key_h, key_w, key_ids, key_adv = jax.random.split(key, 4)
    hidden_states = jax.random.normal(key_h, (batch, time, hidden), dtype=jnp.float32)
    lm_head_kernel = jax.random.normal(key_w, (hidden, vocab), dtype=jnp.float32)
    chosen_ids = jax.random.randint(key_ids, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key_adv, (batch,), dtype=jnp.float32)

    _, logps_seed = grpo_per_token_loss_lm_head_reference(
        hidden_states=hidden_states,
        lm_head_kernel=lm_head_kernel,
        chosen_ids=chosen_ids,
        old_per_token_logps=jnp.zeros((batch, time), dtype=jnp.float32),
        advantages=advantages,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
    )
    old_per_token_logps = jax.lax.stop_gradient(logps_seed + 0.3)

    per_loss_ref, per_logps_ref = grpo_per_token_loss_lm_head_reference(
        hidden_states=hidden_states,
        lm_head_kernel=lm_head_kernel,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
    )

    cfg = GRPOLmHeadFusedConfig(vocab_block_size=8, epsilon_low=0.2, epsilon_high=0.2, temperature=1.0)
    per_loss_fused, per_logps_fused = grpo_per_token_loss_fused_lm_head(
        hidden_states=hidden_states,
        lm_head_kernel=lm_head_kernel,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        cfg=cfg,
        use_self_old=False,
    )

    np.testing.assert_allclose(np.asarray(per_loss_ref), np.asarray(per_loss_fused), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(per_logps_ref), np.asarray(per_logps_fused), rtol=1e-2, atol=1e-2)

    def loss_ref_fn(h, w):
        per_loss, _ = grpo_per_token_loss_lm_head_reference(
            hidden_states=h,
            lm_head_kernel=w,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=0.2,
            epsilon_high=0.2,
            temperature=1.0,
        )
        return jnp.sum(per_loss)

    def loss_fused_fn(h, w):
        per_loss, _ = grpo_per_token_loss_fused_lm_head(
            hidden_states=h,
            lm_head_kernel=w,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            use_self_old=False,
        )
        return jnp.sum(per_loss)

    grad_ref_h, grad_ref_w = jax.grad(loss_ref_fn, argnums=(0, 1))(hidden_states, lm_head_kernel)
    grad_fused_h, grad_fused_w = jax.grad(loss_fused_fn, argnums=(0, 1))(hidden_states, lm_head_kernel)

    np.testing.assert_allclose(np.asarray(grad_ref_h), np.asarray(grad_fused_h), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.asarray(grad_ref_w), np.asarray(grad_fused_w), rtol=1e-2, atol=1e-2)

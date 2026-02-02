# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false

import contextlib

import pytest


def test_imports() -> None:
    from plugins.training.rl.grpo import grpo_loss_logp_entropy

    assert grpo_loss_logp_entropy is not None


def test_fused_matches_fallback_forward_jax_cpu_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from plugins.training.rl.grpo import FUSED_AVAILABLE, grpo_loss_logp_entropy

    if not FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    cpu_devices = jax.devices("cpu")
    cpu_device = cpu_devices[0] if cpu_devices else None

    bsz, seq_len, vocab = 2, 8, 64
    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2
    beta = 0.1

    device_ctx = jax.default_device(cpu_device) if cpu_device is not None else contextlib.nullcontext()
    with device_ctx:
        key = jax.random.PRNGKey(0)
        k_logits, k_ids, k_adv, k_mask = jax.random.split(key, 4)
        logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(
            jnp.bfloat16
        )
        completion_ids = jax.random.randint(
            k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32
        )
        advantages = jax.random.normal(k_adv, (bsz,), dtype=jnp.float32)
        completion_mask = (jax.random.uniform(k_mask, (bsz, seq_len)) > 0.3).astype(jnp.int32)

        scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
        token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
        lse = jax.nn.logsumexp(scaled, axis=-1)
        logp = token_logits - lse
        old_logp = jax.lax.stop_gradient(logp + 0.5)
        ref_logp = jax.lax.stop_gradient(logp + 0.3)

        fused_loss, fused_logp, fused_entropy = grpo_loss_logp_entropy(
            logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            use_fused=True,
        )
        fallback_loss, fallback_logp, fallback_entropy = grpo_loss_logp_entropy(
            logits,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            use_fused=False,
        )

        assert fused_loss.shape == fallback_loss.shape == (bsz, seq_len)
        assert fused_logp.shape == fallback_logp.shape == (bsz, seq_len)
        assert fused_entropy.shape == fallback_entropy.shape == (bsz, seq_len)

        # bf16 logits -> fp32 reductions; allow small tolerance differences in fused reductions.
        assert jnp.allclose(fused_logp, fallback_logp, atol=1e-3, rtol=1e-3)
        assert jnp.allclose(fused_entropy, fallback_entropy, atol=1e-3, rtol=1e-3)
        assert jnp.allclose(fused_loss, fallback_loss, atol=2e-3, rtol=2e-3)


def test_fused_matches_fallback_grad_logits_jax_cpu_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from plugins.training.rl.grpo import FUSED_AVAILABLE, grpo_loss_logp_entropy

    if not FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    cpu_devices = jax.devices("cpu")
    cpu_device = cpu_devices[0] if cpu_devices else None

    bsz, seq_len, vocab = 2, 8, 64
    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2

    device_ctx = jax.default_device(cpu_device) if cpu_device is not None else contextlib.nullcontext()
    with device_ctx:
        key = jax.random.PRNGKey(0)
        k_logits, k_ids, k_adv, k_mask = jax.random.split(key, 4)
        logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(
            jnp.bfloat16
        )
        completion_ids = jax.random.randint(
            k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32
        )
        advantages = jax.random.normal(k_adv, (bsz,), dtype=jnp.float32)
        completion_mask = (jax.random.uniform(k_mask, (bsz, seq_len)) > 0.3).astype(jnp.int32)

        def scalar_loss_fused(logits_in):
            per_token_loss, _logp, _entropy = grpo_loss_logp_entropy(
                logits_in,
                old_logp=None,
                ref_logp=None,
                completion_ids=completion_ids,
                advantages=advantages,
                completion_mask=completion_mask,
                temperature=temperature,
                beta=0.0,
                eps_low=eps_low,
                eps_high=eps_high,
                use_fused=True,
            )
            return jnp.sum(per_token_loss)

        def scalar_loss_fallback(logits_in):
            per_token_loss, _logp, _entropy = grpo_loss_logp_entropy(
                logits_in,
                old_logp=None,
                ref_logp=None,
                completion_ids=completion_ids,
                advantages=advantages,
                completion_mask=completion_mask,
                temperature=temperature,
                beta=0.0,
                eps_low=eps_low,
                eps_high=eps_high,
                use_fused=False,
            )
            return jnp.sum(per_token_loss)

        fused_val_grad = jax.jit(jax.value_and_grad(scalar_loss_fused))
        fallback_val_grad = jax.jit(jax.value_and_grad(scalar_loss_fallback))

        fused_loss, fused_grads = fused_val_grad(logits)
        fallback_loss, fallback_grads = fallback_val_grad(logits)

        assert jnp.all(jnp.isfinite(fused_loss)).item()
        assert jnp.all(jnp.isfinite(fallback_loss)).item()
        assert jnp.all(jnp.isfinite(fused_grads)).item()
        assert jnp.all(jnp.isfinite(fallback_grads)).item()

        assert fused_grads.shape == fallback_grads.shape == logits.shape

        # Loss only depends on the first L positions.
        assert jnp.all(fused_grads[:, -1, :] == 0).item()
        assert jnp.all(fallback_grads[:, -1, :] == 0).item()

        assert jnp.allclose(fused_loss, fallback_loss, atol=2e-3, rtol=2e-3)
        assert jnp.allclose(
            fused_grads.astype(jnp.float32),
            fallback_grads.astype(jnp.float32),
            atol=5e-3,
            rtol=5e-3,
        )

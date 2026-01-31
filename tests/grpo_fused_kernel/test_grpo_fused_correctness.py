# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false

import pytest

from tests.grpo_fused_kernel import grpo_fused_pallas, grpo_reference

def test_imports_cpu_only() -> None:
    assert grpo_reference is not None
    assert grpo_fused_pallas is not None


def test_reference_and_fused_match_on_tiny_python_backend() -> None:
    bsz, seq_len, vocab = 2, 3, 5
    logits = [[[0.0 for _ in range(vocab)] for _ in range(seq_len + 1)] for _ in range(bsz)]
    completion_ids = [[0 for _ in range(seq_len)] for _ in range(bsz)]
    advantages = [1.0, -1.0]
    completion_mask = [[1 for _ in range(seq_len)] for _ in range(bsz)]

    ref_loss_obj, ref_kl_obj, ref_is_clipped_obj = grpo_reference.grpo_loss_reference(
        logits,
        old_logp=None,
        ref_logp=None,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=1.0,
        beta=0.0,
        eps_low=0.2,
        eps_high=0.2,
        backend="python",
    )

    fused_loss_obj, fused_kl_obj, fused_is_clipped_obj = grpo_fused_pallas.grpo_loss_fused_pallas(
        logits,
        old_logp=None,
        ref_logp=None,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=1.0,
        beta=0.0,
        eps_low=0.2,
        eps_high=0.2,
        backend="python",
    )

    assert ref_kl_obj is None
    assert fused_kl_obj is None
    assert isinstance(ref_loss_obj, list)
    assert isinstance(fused_loss_obj, list)
    assert isinstance(ref_is_clipped_obj, list)
    assert isinstance(fused_is_clipped_obj, list)

    ref_loss = ref_loss_obj
    ref_is_clipped = ref_is_clipped_obj
    fused_loss = fused_loss_obj
    fused_is_clipped = fused_is_clipped_obj
    assert ref_loss == fused_loss
    assert ref_is_clipped == fused_is_clipped

    assert len(ref_loss) == bsz
    assert len(ref_loss[0]) == seq_len

    # With zero logits and old_logp=None, ratio==1 and the objective reduces to -adv.
    assert ref_loss[0] == pytest.approx([-1.0] * seq_len)
    assert ref_loss[1] == pytest.approx([1.0] * seq_len)


def test_reference_jax_bf16_grad_smoke() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    bsz, seq_len, vocab = 2, 4, 32
    key = jax.random.PRNGKey(0)
    k_logits, k_ids = jax.random.split(key, 2)

    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    completion_ids = jax.random.randint(k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32)
    advantages = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    completion_mask = jnp.ones((bsz, seq_len), dtype=jnp.int32)

    def scalar_loss(logits_in):
        per_token_loss, _, _ = grpo_reference.grpo_loss_reference_jax(
            logits_in,
            old_logp=None,
            ref_logp=None,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=1.0,
            beta=0.0,
            eps_low=0.2,
            eps_high=0.2,
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    loss, grads = jax.value_and_grad(scalar_loss)(logits)
    assert jnp.all(jnp.isfinite(loss)).item()
    assert jnp.all(jnp.isfinite(grads)).item()


def test_fused_forward_matches_reference_jax_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    if not grpo_fused_pallas.FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    bsz, seq_len, vocab = 2, 8, 64
    key = jax.random.PRNGKey(0)
    k_logits, k_ids = jax.random.split(key, 2)
    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(jnp.bfloat16)
    completion_ids = jax.random.randint(k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32)
    completion_mask = jnp.asarray(
        [[1, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1]], dtype=jnp.int32
    )
    advantages = jnp.asarray([1.0, -1.0], dtype=jnp.float32)

    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2
    beta = 0.1

    # Compute a stable logp to construct old/ref logp inputs.
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse
    old_logp = logp + 0.5
    ref_logp = logp + 0.3

    ref_loss, ref_kl, ref_is_clipped = grpo_reference.grpo_loss_reference_jax(
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
    )
    fused_loss, fused_kl, fused_is_clipped = grpo_fused_pallas.grpo_loss_fused_pallas(
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
        backend="jax",
    )

    assert fused_kl is not None
    assert ref_kl is not None

    ref_loss = jnp.asarray(ref_loss)
    fused_loss = jnp.asarray(fused_loss)
    ref_kl = jnp.asarray(ref_kl)
    fused_kl = jnp.asarray(fused_kl)
    ref_is_clipped = jnp.asarray(ref_is_clipped)
    fused_is_clipped = jnp.asarray(fused_is_clipped)

    assert fused_loss.shape == ref_loss.shape == (bsz, seq_len)
    assert fused_kl.shape == ref_kl.shape == (bsz, seq_len)

    assert jnp.allclose(fused_loss, ref_loss, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(fused_kl, ref_kl, atol=1e-4, rtol=1e-4)
    assert jnp.array_equal(fused_is_clipped, ref_is_clipped)


def test_fused_backward_matches_reference_jax_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    if not grpo_fused_pallas.FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    bsz, seq_len, vocab = 2, 8, 64
    key = jax.random.PRNGKey(0)
    k_logits, k_ids = jax.random.split(key, 2)
    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(jnp.bfloat16)
    completion_ids = jax.random.randint(
        k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32
    )
    completion_mask = jnp.asarray(
        [[1, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1]], dtype=jnp.int32
    )
    advantages = jnp.asarray([1.0, -1.0], dtype=jnp.float32)

    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2
    beta = 0.1

    # Construct stable old/ref logp inputs (treated as constants for grad).
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse
    old_logp = jax.lax.stop_gradient(logp + 0.5)
    ref_logp = jax.lax.stop_gradient(logp + 0.3)

    def scalar_loss_ref(logits_in):
        per_token_loss, _, _ = grpo_reference.grpo_loss_reference_jax(
            logits_in,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    def scalar_loss_fused(logits_in):
        per_token_loss, _, _ = grpo_fused_pallas.grpo_loss_fused_pallas(
            logits_in,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            backend="jax",
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    # Ensure custom_vjp works under jit.
    ref_grads = jax.jit(jax.grad(scalar_loss_ref))(logits)
    fused_grads = jax.jit(jax.grad(scalar_loss_fused))(logits)

    assert fused_grads.shape == logits.shape
    assert ref_grads.shape == logits.shape

    # Loss only depends on the first L positions.
    assert jnp.all(fused_grads[:, -1, :] == 0).item()
    assert jnp.all(ref_grads[:, -1, :] == 0).item()

    # Compare bf16 grads with relaxed tolerances.
    assert jnp.allclose(
        fused_grads.astype(jnp.float32),
        ref_grads.astype(jnp.float32),
        atol=5e-3,
        rtol=5e-3,
    )


def test_fused_all_masked_tokens_zero_loss_and_grad_jax_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    if not grpo_fused_pallas.FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    bsz, seq_len, vocab = 2, 8, 64
    key = jax.random.PRNGKey(0)
    k_logits, k_ids = jax.random.split(key, 2)
    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    completion_ids = jax.random.randint(k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32)
    advantages = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    completion_mask = jnp.zeros((bsz, seq_len), dtype=jnp.int32)

    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2
    beta = 0.1

    # Construct stable old/ref logp inputs (treated as constants for grad).
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse
    old_logp = jax.lax.stop_gradient(logp + 0.1)
    ref_logp = jax.lax.stop_gradient(logp + 0.3)

    fused_loss, fused_kl, fused_is_clipped = grpo_fused_pallas.grpo_loss_fused_pallas(
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
        backend="jax",
    )
    assert fused_kl is not None

    fused_loss = jnp.asarray(fused_loss)
    fused_kl = jnp.asarray(fused_kl)
    fused_is_clipped = jnp.asarray(fused_is_clipped)
    assert fused_loss.shape == (bsz, seq_len)
    assert fused_kl.shape == (bsz, seq_len)

    assert jnp.array_equal(fused_loss, jnp.zeros_like(fused_loss))
    assert jnp.array_equal(fused_kl, jnp.zeros_like(fused_kl))
    assert jnp.array_equal(fused_is_clipped, jnp.zeros_like(fused_is_clipped))

    def scalar_loss_fused(logits_in):
        per_token_loss, _, _ = grpo_fused_pallas.grpo_loss_fused_pallas(
            logits_in,
            old_logp=old_logp,
            ref_logp=ref_logp,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            backend="jax",
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    fused_grads = jax.jit(jax.grad(scalar_loss_fused))(logits)
    assert jnp.all(fused_grads == 0).item()


def test_fused_beta_zero_returns_no_kl_and_grad_matches_reference_jax_interpret() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    if not grpo_fused_pallas.FUSED_AVAILABLE:
        pytest.skip("Pallas fused path not available")

    bsz, seq_len, vocab = 2, 8, 64
    key = jax.random.PRNGKey(0)
    k_logits, k_ids = jax.random.split(key, 2)
    logits = jax.random.normal(k_logits, (bsz, seq_len + 1, vocab), dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    completion_ids = jax.random.randint(k_ids, (bsz, seq_len), minval=0, maxval=vocab, dtype=jnp.int32)
    completion_mask = jnp.asarray(
        [[1, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1]], dtype=jnp.int32
    )
    advantages = jnp.asarray([1.0, -1.0], dtype=jnp.float32)

    temperature = 1.0
    eps_low = 0.2
    eps_high = 0.2
    beta = 0.0

    # Construct a stable old_logp input (treated as a constant for grad).
    scaled = logits[:, :seq_len, :].astype(jnp.float32) / temperature
    token_logits = jnp.take_along_axis(scaled, completion_ids[:, :, None], axis=-1)[:, :, 0]
    lse = jax.nn.logsumexp(scaled, axis=-1)
    logp = token_logits - lse
    old_logp = jax.lax.stop_gradient(logp + 0.1)

    ref_loss, ref_kl, _ref_is_clipped = grpo_reference.grpo_loss_reference_jax(
        logits,
        old_logp=old_logp,
        ref_logp=None,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
    )
    fused_loss, fused_kl, _fused_is_clipped = grpo_fused_pallas.grpo_loss_fused_pallas(
        logits,
        old_logp=old_logp,
        ref_logp=None,
        completion_ids=completion_ids,
        advantages=advantages,
        completion_mask=completion_mask,
        temperature=temperature,
        beta=beta,
        eps_low=eps_low,
        eps_high=eps_high,
        backend="jax",
    )

    assert ref_kl is None
    assert fused_kl is None

    ref_loss = jnp.asarray(ref_loss)
    fused_loss = jnp.asarray(fused_loss)
    assert fused_loss.shape == ref_loss.shape == (bsz, seq_len)

    def scalar_loss_ref(logits_in):
        per_token_loss, _, _ = grpo_reference.grpo_loss_reference_jax(
            logits_in,
            old_logp=old_logp,
            ref_logp=None,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    def scalar_loss_fused(logits_in):
        per_token_loss, _, _ = grpo_fused_pallas.grpo_loss_fused_pallas(
            logits_in,
            old_logp=old_logp,
            ref_logp=None,
            completion_ids=completion_ids,
            advantages=advantages,
            completion_mask=completion_mask,
            temperature=temperature,
            beta=beta,
            eps_low=eps_low,
            eps_high=eps_high,
            backend="jax",
        )
        return jnp.sum(jnp.asarray(per_token_loss))

    ref_grads = jax.jit(jax.grad(scalar_loss_ref))(logits)
    fused_grads = jax.jit(jax.grad(scalar_loss_fused))(logits)

    assert fused_grads.shape == logits.shape
    assert ref_grads.shape == logits.shape

    # Loss only depends on the first L positions.
    assert jnp.all(fused_grads[:, -1, :] == 0).item()
    assert jnp.all(ref_grads[:, -1, :] == 0).item()

    # Compare bf16 grads with relaxed tolerances.
    assert jnp.allclose(
        fused_grads.astype(jnp.float32),
        ref_grads.astype(jnp.float32),
        atol=5e-3,
        rtol=5e-3,
    )

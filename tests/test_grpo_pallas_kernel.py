import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_pallas_on_policy,
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

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-4
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-4

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

    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-4


@pytest.mark.parametrize("vocab", [15, 24])
def test_grpo_pallas_kernel_on_policy_matches_reference_forward_and_backward(vocab: int):
    key = jax.random.PRNGKey(0)

    batch = 2
    time = 3

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    chosen_ids = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    eps_low = 0.2
    eps_high = 0.2
    temperature = 1.0

    def ref_impl(l):
        l = l.astype(jnp.float32)
        logps = jnp.take_along_axis(
            jax.nn.log_softmax(l, axis=-1),
            chosen_ids[..., None],
            axis=-1,
        )[..., 0] / float(temperature)
        old = jax.lax.stop_gradient(logps)
        ratio = jnp.exp(logps - old)
        clipped_ratio = jnp.clip(ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))
        loss1 = ratio * advantages[..., None]
        loss2 = clipped_ratio * advantages[..., None]
        loss = (-jnp.minimum(loss1, loss2)).astype(jnp.float32)
        return loss, logps.astype(jnp.float32)

    per_loss_ref, per_logp_ref = ref_impl(logits)

    cfg = GRPOKernelConfig(block_size=8, epsilon_low=eps_low, epsilon_high=eps_high, temperature=temperature)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)
    per_loss_k, per_logp_k = grpo_per_token_loss_pallas_on_policy(
        logits=logits,
        chosen_ids=chosen_ids,
        advantages=advantages,
        cfg=cfg,
        interpret=interpret,
        debug=False,
    )

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-4
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-4

    def loss_ref_fn(l):
        per_loss, _ = ref_impl(l)
        return jnp.sum(per_loss)

    def loss_k_fn(l):
        per_loss, _ = grpo_per_token_loss_pallas_on_policy(
            logits=l,
            chosen_ids=chosen_ids,
            advantages=advantages,
            cfg=cfg,
            interpret=interpret,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_k = jax.grad(loss_k_fn)(logits).astype(jnp.float32)

    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-4


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU mosaic lowering")
def test_grpo_pallas_kernel_tpu_lowering_batch_gt1():
    key = jax.random.PRNGKey(0)

    batch = 4
    time = 8
    vocab = 256

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

    cfg = GRPOKernelConfig(block_size=128, epsilon_low=0.2, epsilon_high=0.2, temperature=1.0)
    per_loss_k, per_logp_k = grpo_per_token_loss_pallas(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        cfg=cfg,
        interpret=False,
        debug=False,
    )

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-4
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-4

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
            interpret=False,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_k = jax.grad(loss_k_fn)(logits).astype(jnp.float32)

    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-4


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU mosaic lowering")
def test_grpo_pallas_kernel_tpu_lowering_block2048_time128_bf16_on_policy():
    key = jax.random.PRNGKey(0)

    batch = 1
    time = 128
    vocab = 2048

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.bfloat16)
    chosen_ids = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    cfg = GRPOKernelConfig(
        block_size=2048,
        time_block=128,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
        compute_dtype="bf16",
    )

    def loss_k_fn(l):
        per_loss, _ = grpo_per_token_loss_pallas_on_policy(
            logits=l,
            chosen_ids=chosen_ids,
            advantages=advantages,
            cfg=cfg,
            interpret=False,
            debug=False,
        )
        return jnp.sum(per_loss)

    # Forward + backward should both lower via Mosaic (no interpret mode).
    _ = loss_k_fn(logits)
    _ = jax.grad(loss_k_fn)(logits)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU mosaic lowering")
def test_grpo_pallas_kernel_bwd_output_alias_logits_matches_reference():
    key = jax.random.PRNGKey(0)

    batch = 1
    time = 128
    vocab = 257

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    chosen_ids = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    eps_low = 0.2
    eps_high = 0.2
    temperature = 1.0

    _, logps_seed = grpo_per_token_loss_reference(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=jnp.zeros((batch, time), dtype=jnp.float32),
        advantages=advantages,
        epsilon_low=eps_low,
        epsilon_high=eps_high,
        temperature=temperature,
    )
    old_per_token_logps = jax.lax.stop_gradient(logps_seed + 0.3)

    cfg = GRPOKernelConfig(
        block_size=128,
        time_block=time,
        epsilon_low=eps_low,
        epsilon_high=eps_high,
        temperature=temperature,
        bwd_output_alias_logits=True,
        compute_dtype="f32",
    )

    def loss_ref_fn(l):
        per_loss, _ = grpo_per_token_loss_reference(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=eps_low,
            epsilon_high=eps_high,
            temperature=temperature,
        )
        return jnp.sum(per_loss)

    def loss_k_fn(l):
        per_loss, _ = grpo_per_token_loss_pallas(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg,
            interpret=False,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(loss_ref_fn)(logits)
    grad_k = jax.grad(loss_k_fn)(logits).astype(jnp.float32)
    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 1e-4


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU mosaic lowering")
def test_grpo_pallas_kernel_on_policy_matches_baseline_bf16_log_softmax_grad():
    key = jax.random.PRNGKey(0)

    batch = 1
    time = 128
    vocab = 2048

    eps_low = 0.2
    eps_high = 0.2
    temperature = 1.0

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.bfloat16)
    chosen_ids = jax.random.randint(key, (batch, time), 0, vocab, dtype=jnp.int32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    def baseline_impl(l):
        logps = jnp.take_along_axis(
            jax.nn.log_softmax(l, axis=-1),
            chosen_ids[..., None],
            axis=-1,
        )[..., 0] / float(temperature)
        old = jax.lax.stop_gradient(logps)
        ratio = jnp.exp(logps - old)
        clipped_ratio = jnp.clip(ratio, 1.0 - float(eps_low), 1.0 + float(eps_high))
        loss1 = ratio * advantages[..., None]
        loss2 = clipped_ratio * advantages[..., None]
        loss = (-jnp.minimum(loss1, loss2)).astype(jnp.float32)
        return loss, logps

    per_loss_ref, per_logp_ref = baseline_impl(logits)

    cfg = GRPOKernelConfig(
        block_size=2048,
        time_block=128,
        epsilon_low=eps_low,
        epsilon_high=eps_high,
        temperature=temperature,
        compute_dtype="bf16",
    )
    per_loss_k, per_logp_k = grpo_per_token_loss_pallas_on_policy(
        logits=logits,
        chosen_ids=chosen_ids,
        advantages=advantages,
        cfg=cfg,
        interpret=False,
        debug=False,
    )

    # Note: on-policy GRPO per-token loss is invariant to logp value shifts, but
    # gradients should still match the baseline bf16 log_softmax path closely.
    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-5
    assert jnp.max(jnp.abs(per_logp_ref.astype(jnp.float32) - per_logp_k.astype(jnp.float32))) <= 0.0625

    def baseline_loss_sum(l):
        per_loss, _ = baseline_impl(l)
        return jnp.sum(per_loss)

    def pallas_loss_sum(l):
        per_loss, _ = grpo_per_token_loss_pallas_on_policy(
            logits=l,
            chosen_ids=chosen_ids,
            advantages=advantages,
            cfg=cfg,
            interpret=False,
            debug=False,
        )
        return jnp.sum(per_loss)

    grad_ref = jax.grad(baseline_loss_sum)(logits).astype(jnp.float32)
    grad_k = jax.grad(pallas_loss_sum)(logits).astype(jnp.float32)
    assert jnp.max(jnp.abs(grad_ref - grad_k)) < 3e-4

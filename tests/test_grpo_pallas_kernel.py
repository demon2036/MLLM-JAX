import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    grpo_per_token_loss_pallas,
    grpo_per_token_loss_pallas_sharded,
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


def test_grpo_pallas_kernel_shard_map_single_device_matches_reference():
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device], dtype=object).reshape((1, 1, 1)), ("dp", "fsdp", "tp"))

    key = jax.random.PRNGKey(0)
    batch = 2
    time = 3
    vocab = 24

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

    cfg = GRPOKernelConfig(block_size=8, time_block=8, epsilon_low=0.2, epsilon_high=0.2, temperature=1.0)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)

    logits = jax.device_put(logits, NamedSharding(mesh, P(("dp", "fsdp"), None, None)))
    chosen_ids = jax.device_put(chosen_ids, NamedSharding(mesh, P(("dp", "fsdp"), None)))
    old_per_token_logps = jax.device_put(old_per_token_logps, NamedSharding(mesh, P(("dp", "fsdp"), None)))
    advantages = jax.device_put(advantages, NamedSharding(mesh, P(("dp", "fsdp"))))

    per_loss_k, per_logp_k = grpo_per_token_loss_pallas_sharded(
        logits=logits,
        chosen_ids=chosen_ids,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        mesh=mesh,
        cfg=cfg,
        interpret=interpret,
        debug=False,
    )

    assert jnp.max(jnp.abs(per_loss_ref - per_loss_k)) < 1e-5
    assert jnp.max(jnp.abs(per_logp_ref - per_logp_k)) < 1e-5


def test_grpo_pallas_train_module_sharded_matches_baseline_single_device():
    import flax.linen as nn

    from MLLM_JAX.train_modules import TrainGRPOModule
    from plugins.training.grpo.module import TrainGRPOModulePallas

    vocab = 16
    batch = 2
    seq_len = 5

    key = jax.random.PRNGKey(0)

    class DummyModel(nn.Module):
        vocab: int

        @nn.compact
        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            logits = nn.Embed(num_embeddings=self.vocab, features=self.vocab)(input_ids)
            return logits, None

    model = DummyModel(vocab=vocab)
    params = model.init(key, input_ids=jnp.zeros((batch, seq_len), dtype=jnp.int32), attention_mask=jnp.ones((batch, seq_len)))["params"]

    input_ids = jax.random.randint(key, (batch, seq_len), 0, vocab, dtype=jnp.int32)
    attention_mask = jnp.ones((batch, seq_len), dtype=jnp.int32)
    labels = jnp.ones((batch, seq_len), dtype=jnp.float32)
    advantages = jax.random.normal(key, (batch,), dtype=jnp.float32)

    logits, _ = model.apply({"params": params}, input_ids=input_ids, attention_mask=attention_mask)
    logits_for_loss = logits[..., :-1, :]
    chosen_ids = input_ids[:, 1:]
    per_token_logps_ref = jnp.take_along_axis(jax.nn.log_softmax(logits_for_loss, axis=-1), chosen_ids[..., None], axis=-1)[
        ..., 0
    ]
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps_ref + 0.3)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "advantages": advantages,
        "old_per_token_logps": old_per_token_logps,
    }

    baseline = TrainGRPOModule(model=model, pad_token_id=0, num_pre_Q=1)
    metrics_baseline = baseline.apply({"params": {"model": params}}, inputs)

    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device], dtype=object).reshape((1, 1, 1)), ("dp", "fsdp", "tp"))
    cfg = GRPOKernelConfig(block_size=8, time_block=8)
    pallas = TrainGRPOModulePallas(
        model=model,
        pad_token_id=0,
        num_pre_Q=1,
        mesh=mesh,
        kernel_cfg=cfg,
        kernel_interpret=True,
    )
    metrics_pallas = pallas.apply({"params": {"model": params}}, inputs)

    assert jnp.max(jnp.abs(metrics_baseline["loss"] - metrics_pallas["loss"])) < 1e-5
    assert jnp.max(jnp.abs(metrics_baseline["per_token_logps"] - metrics_pallas["per_token_logps"])) < 1e-5

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from plugins.training.kernels.entropy_pallas import (
    EntropyKernelConfig,
    entropy_per_token_pallas,
    entropy_per_token_pallas_sharded,
    entropy_per_token_reference,
)


def test_entropy_pallas_kernel_matches_reference():
    key = jax.random.PRNGKey(0)

    batch = 2
    time = 3
    vocab = 24

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    ref = entropy_per_token_reference(logits, temperature=1.0)

    cfg = EntropyKernelConfig(block_size=8, time_block=8, temperature=1.0)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)
    out = entropy_per_token_pallas(logits=logits, cfg=cfg, interpret=interpret, debug=False)

    assert jnp.max(jnp.abs(ref - out)) < 1e-5


def test_entropy_pallas_kernel_shard_map_single_device_matches_reference():
    device = jax.devices()[0]
    mesh = Mesh(np.asarray([device], dtype=object).reshape((1, 1, 1)), ("dp", "fsdp", "tp"))

    key = jax.random.PRNGKey(0)
    batch = 2
    time = 3
    vocab = 24

    logits = jax.random.normal(key, (batch, time, vocab), dtype=jnp.float32)
    ref = entropy_per_token_reference(logits, temperature=1.0)

    cfg = EntropyKernelConfig(block_size=8, time_block=8, temperature=1.0)
    interpret = pltpu.InterpretParams(out_of_bounds_reads="raise", random_seed=0)

    logits = jax.device_put(logits, NamedSharding(mesh, P(("dp", "fsdp"), None, None)))
    out = entropy_per_token_pallas_sharded(
        logits=logits,
        mesh=mesh,
        cfg=cfg,
        batch_axes=("dp", "fsdp"),
        vocab_axis=None,
        interpret=interpret,
        debug=False,
    )

    assert jnp.max(jnp.abs(ref - out)) < 1e-5


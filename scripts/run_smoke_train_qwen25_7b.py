import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoConfig

from MLLM_JAX.utils import get_jax_mesh2
from training2 import get_state, training_step


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def main() -> None:
    try:
        jax.distributed.initialize()
    except Exception:
        pass

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B")
    steps = _get_env_int("STEPS", 3)
    batch_size = _get_env_int("BATCH_SIZE", 1)
    seq_len = _get_env_int("SEQ_LEN", 129)

    # Use FSDP=all devices, TP=1.
    mesh_fsdp = get_jax_mesh2("1,-1,1")

    print(f"backend={jax.default_backend()} device_count={jax.device_count()}")
    print(f"model_path={model_path} steps={steps} batch_size={batch_size} seq_len={seq_len}")

    state, _sampler, _state_sharding = get_state(
        mesh_fsdp,
        training_steps=steps,
        model_path=model_path,
        grad_accum_steps=1,
        num_pre_q=1,
        max_lengths=seq_len,
        beta=0.0,
        create_sampler=False,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = int(getattr(config, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocab_size from config: {vocab_size}")

    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)

    labels = np.zeros((batch_size, seq_len), dtype=np.int32)
    labels[:, seq_len // 2 :] = 1

    advantages = np.ones((batch_size,), dtype=np.float32)
    total_valid_token_count = int(labels[:, 1:].sum())
    if total_valid_token_count <= 0:
        raise ValueError("total_valid_token_count must be > 0")

    old_per_token_logps = np.zeros((batch_size, seq_len - 1), dtype=np.float32)

    data_sharding = NamedSharding(mesh_fsdp, P())

    train_fn = jax.jit(training_step, donate_argnums=(0,))

    for step in range(steps):
        t0 = time.time()
        batch = {
            "input_ids": jax.device_put(jnp.asarray(input_ids), data_sharding),
            "attention_mask": jax.device_put(jnp.asarray(attention_mask), data_sharding),
            "labels": jax.device_put(jnp.asarray(labels), data_sharding),
            "advantages": jax.device_put(jnp.asarray(advantages), data_sharding),
            "old_per_token_logps": jax.device_put(jnp.asarray(old_per_token_logps), data_sharding),
            "total_valid_token_count": jax.device_put(jnp.asarray(total_valid_token_count), data_sharding),
        }

        state, metrics = train_fn(state, batch)
        loss = metrics["loss"]
        jax.block_until_ready(loss)
        dt = time.time() - t0
        loss_value = float(np.asarray(loss))
        print(f"step={step} loss={loss_value:.6f} dt={dt:.2f}s")

        per_token_logps = metrics.get("per_token_logps")
        if per_token_logps is not None:
            old_per_token_logps = np.asarray(per_token_logps)


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from MLLM_JAX.utils import get_jax_mesh2
from plugins.training.update.train_step import training_step

from plugins.sft.jax.data import batched, collate_sft_batch, iter_indices
from plugins.sft.jax.state import SftStateBundle, create_sft_state


@dataclass(frozen=True)
class SftTrainStats:
    steps: int
    micro_steps: int
    effective_batch_size: int
    final_loss: float


def _device_batch_size(mesh: Mesh, micro_batch_size_per_replica: int) -> int:
    dp = int(mesh.shape.get("dp", 1))
    fsdp = int(mesh.shape.get("fsdp", 1))
    replicas = dp * fsdp
    return int(micro_batch_size_per_replica) * replicas


def create_mesh_from_config(mesh_shape: str) -> Mesh:
    return get_jax_mesh2(str(mesh_shape))


def run_sft_train(
    *,
    mesh: Mesh,
    model: Any,
    params: Any,
    train_dataset: Any,
    pad_token_id: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    grad_accum_steps: int,
    micro_batch_size_per_replica: int,
    max_steps: int,
    seed: int,
    logging_steps: int,
    log_cb: Callable[[int, float, int], None] | None = None,
) -> tuple[Any, SftTrainStats]:
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0 for the JAX backend")

    micro_batch_size_per_replica = int(micro_batch_size_per_replica)
    if micro_batch_size_per_replica <= 0:
        raise ValueError("micro_batch_size_per_replica must be >= 1")

    bundle: SftStateBundle = create_sft_state(
        mesh=mesh,
        model=model,
        params=params,
        optimizer_name=optimizer_name,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_accum_steps=int(grad_accum_steps),
    )

    data_sharding = NamedSharding(mesh, P(("dp", "fsdp"), None))
    train_step_fn = jax.jit(training_step, donate_argnums=(0,), out_shardings=(bundle.sharding, None))

    global_micro_batch = _device_batch_size(mesh, micro_batch_size_per_replica)
    effective_batch = int(global_micro_batch) * int(grad_accum_steps)

    n = int(len(train_dataset))
    if n <= 0:
        raise ValueError("Empty train_dataset")

    # Simple cycling iterator over shuffled indices; avoids epoch bookkeeping.
    indices = iter_indices(n=n, seed=seed, shuffle=True)
    cursor = 0

    last_loss = float("nan")
    micro_steps = 0
    for step in range(1, max_steps + 1):
        losses = []
        for _ in range(int(grad_accum_steps)):
            if cursor + global_micro_batch > len(indices):
                indices = iter_indices(n=n, seed=seed + step, shuffle=True)
                cursor = 0
            batch_idx = indices[cursor : cursor + global_micro_batch]
            cursor += global_micro_batch

            examples = [train_dataset[i] for i in batch_idx]
            batch_np = collate_sft_batch(examples, pad_token_id=int(pad_token_id))
            batch = {k: jax.device_put(v, data_sharding) for k, v in batch_np.as_dict().items()}

            bundle_state, metrics = train_step_fn(bundle.state, batch)
            bundle = SftStateBundle(state=bundle_state, sharding=bundle.sharding)

            micro_steps += 1
            loss_val = float(np.array(metrics["loss"]))
            losses.append(loss_val)

        last_loss = float(np.mean(losses)) if losses else last_loss
        if int(logging_steps) > 0 and (step % int(logging_steps) == 0 or step == 1 or step == max_steps):
            print(f"[sft] step={step}/{max_steps} loss={last_loss:.6f} effective_bs={effective_batch}")
            if log_cb is not None:
                log_cb(int(step), float(last_loss), int(effective_batch))

    return bundle.state, SftTrainStats(
        steps=max_steps,
        micro_steps=micro_steps,
        effective_batch_size=effective_batch,
        final_loss=float(last_loss),
    )


__all__ = ["SftTrainStats", "create_mesh_from_config", "run_sft_train"]

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from plugins.training.mesh import create_mesh
from plugins.training.update.train_step import training_step
from plugins.training.update.optimizer import LRScheduleConfig, OptimizerConfig, build_tx

from plugins.sft.jax.sharding import get_partition_rules_llama, match_partition_rules

from plugins.sft.jax.data import collate_sft_batch, iter_indices


@dataclass(frozen=True)
class SftTrainStats:
    steps: int
    micro_steps: int
    effective_batch_size: int
    final_loss: float


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float32)
    denom = jnp.maximum(mask_f.sum(), 1.0)
    return (values.astype(jnp.float32) * mask_f).sum() / denom


class TrainSftModule(nn.Module):
    model: Any
    label_ignore_id: int = -100

    def __call__(self, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        logits, _cache = self.model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_attention = attention_mask[:, 1:]

        valid = jnp.logical_and(shift_labels != int(self.label_ignore_id), shift_attention.astype(bool))
        safe_labels = jnp.where(valid, shift_labels, 0).astype(jnp.int32)

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        chosen = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
        per_token_loss = -chosen

        loss = _masked_mean(per_token_loss, valid)
        token_count = valid.astype(jnp.int32).sum()
        return {
            "loss": loss,
            "token_count": token_count,
        }


class SftTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: Any | None = None


@dataclass(frozen=True)
class SftStateBundle:
    state: SftTrainState
    sharding: Any


def create_sft_state(
    *,
    mesh: Mesh,
    model: Any,
    params: Any,
    training_steps: int,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    grad_accum_steps: int,
    warmup_steps: int = 0,
    label_ignore_id: int = -100,
) -> SftStateBundle:
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")
    training_steps = int(training_steps)
    if training_steps <= 0:
        raise ValueError("training_steps must be > 0")

    train_module = TrainSftModule(model=model, label_ignore_id=int(label_ignore_id))
    tx_cfg = OptimizerConfig(
        name=str(optimizer_name),
        clip_norm=1.0,
        weight_decay=float(weight_decay),
        lr_schedule=LRScheduleConfig(
            type="warmup_linear",
            init_value=0.0,
            peak_value=float(learning_rate),
            end_value=0.0,
            warmup_steps=int(warmup_steps),
        ),
    )
    tx = build_tx(training_steps=training_steps, cfg=tx_cfg)

    def init_fn(p):
        grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p) if grad_accum_steps > 1 else None
        return SftTrainState.create(
            apply_fn=train_module.apply,
            params=p,
            tx=tx,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum,
        )

    state_shapes = jax.eval_shape(init_fn, params)
    partitions = match_partition_rules(get_partition_rules_llama(), state_shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    state = jax.jit(init_fn, out_shardings=shardings)(params)
    return SftStateBundle(state=state, sharding=shardings)


def _device_batch_size(mesh: Mesh, micro_batch_size_per_replica: int) -> int:
    dp = int(mesh.shape.get("dp", 1))
    fsdp = int(mesh.shape.get("fsdp", 1))
    replicas = dp * fsdp
    return int(micro_batch_size_per_replica) * replicas


def create_mesh_from_config(mesh_shape: str) -> Mesh:
    # Reuse the GRPO mesh logic: "auto" builds a safe host-local mesh on multi-host TPUs.
    return create_mesh(str(mesh_shape))


def run_sft_train(
    *,
    mesh: Mesh,
    model: Any,
    params: Any,
    train_dataset: Any,
    pad_token_id: int,
    pad_to_length: int | None,
    padding_side: str,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    grad_accum_steps: int,
    micro_batch_size_per_replica: int,
    max_steps: int,
    seed: int,
    logging_steps: int,
    warmup_steps: int = 0,
    log_cb: Callable[[int, float, int, float], None] | None = None,
    eval_every_steps: int = 0,
    eval_cb: Callable[[int, Any], None] | None = None,
    checkpoint_every_steps: int = 0,
    checkpoint_cb: Callable[[int, Any], None] | None = None,
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
        training_steps=max_steps,
        optimizer_name=optimizer_name,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_accum_steps=int(grad_accum_steps),
        warmup_steps=int(warmup_steps),
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
    run_t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()
        losses = []
        for _ in range(int(grad_accum_steps)):
            if cursor + global_micro_batch > len(indices):
                indices = iter_indices(n=n, seed=seed + step, shuffle=True)
                cursor = 0
            batch_idx = indices[cursor : cursor + global_micro_batch]
            cursor += global_micro_batch

            examples = [train_dataset[i] for i in batch_idx]
            batch_np = collate_sft_batch(
                examples,
                pad_token_id=int(pad_token_id),
                pad_to_length=pad_to_length,
                padding_side=str(padding_side or "right"),
            )
            batch = {k: jax.device_put(v, data_sharding) for k, v in batch_np.as_dict().items()}

            bundle_state, metrics = train_step_fn(bundle.state, batch)
            bundle = SftStateBundle(state=bundle_state, sharding=bundle.sharding)

            micro_steps += 1
            loss_val = float(np.array(metrics["loss"]))
            losses.append(loss_val)

        last_loss = float(np.mean(losses)) if losses else last_loss
        step_dt = float(time.perf_counter() - step_t0)
        wall_dt = float(time.perf_counter() - run_t0)
        if int(logging_steps) > 0 and (step % int(logging_steps) == 0 or step == 1 or step == max_steps):
            print(f"[sft] step={step}/{max_steps} loss={last_loss:.6f} effective_bs={effective_batch} dt={step_dt:.3f}s t={wall_dt:.1f}s")
            if log_cb is not None:
                log_cb(int(step), float(last_loss), int(effective_batch), float(step_dt))

        if checkpoint_cb is not None and int(checkpoint_every_steps) > 0 and step % int(checkpoint_every_steps) == 0:
            checkpoint_cb(int(step), bundle.state)

        if eval_cb is not None and int(eval_every_steps) > 0 and step % int(eval_every_steps) == 0:
            eval_cb(int(step), bundle.state)

    return bundle.state, SftTrainStats(
        steps=max_steps,
        micro_steps=micro_steps,
        effective_batch_size=effective_batch,
        final_loss=float(last_loss),
    )


__all__ = ["SftTrainStats", "create_mesh_from_config", "run_sft_train"]

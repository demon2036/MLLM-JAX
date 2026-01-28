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

from plugins.common.sharding.batch import make_form_training_global_array
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
    pad_policy: str = "fixed",
    pad_buckets: tuple[int, ...] = (128, 256, 512),
    pad_to_multiple_of: int = 8,
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
    log_extra_cb: Callable[[int, dict[str, float]], None] | None = None,
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

    train_step_fn = jax.jit(training_step, donate_argnums=(0,), out_shardings=(bundle.sharding, None))

    global_micro_batch = _device_batch_size(mesh, micro_batch_size_per_replica)
    effective_batch = int(global_micro_batch) * int(grad_accum_steps)

    process_count = int(jax.process_count())
    process_index = int(jax.process_index())
    if process_count <= 0:
        raise RuntimeError("Unexpected jax.process_count() <= 0")
    if int(global_micro_batch) % process_count != 0:
        raise ValueError(
            f"global_micro_batch={int(global_micro_batch)} must be divisible by process_count={process_count} "
            "to shard batches evenly across hosts."
        )
    local_micro_batch = int(global_micro_batch) // process_count

    process_allgather = None
    if process_count > 1:
        from jax.experimental.multihost_utils import process_allgather as _process_allgather

        process_allgather = _process_allgather

    _form_training_global_array = make_form_training_global_array(mesh)

    n = int(len(train_dataset))
    if n <= 0:
        raise ValueError("Empty train_dataset")

    pad_policy_norm = str(pad_policy or "fixed").strip().lower()
    if pad_policy_norm not in {"fixed", "max", "pow2_buckets"}:
        raise ValueError(f"Unknown pad_policy={pad_policy!r} (expected fixed|max|pow2_buckets)")

    pad_to_length = int(pad_to_length) if pad_to_length is not None and int(pad_to_length) > 0 else None
    if pad_policy_norm == "fixed" and pad_to_length is None:
        raise ValueError("pad_to_length must be set when pad_policy='fixed'")

    pad_to_multiple_of = int(pad_to_multiple_of)
    if pad_to_multiple_of <= 0:
        pad_to_multiple_of = 1

    buckets: list[int] = sorted({int(x) for x in pad_buckets if int(x) > 0})
    if pad_policy_norm == "pow2_buckets":
        if not buckets:
            raise ValueError("pad_buckets must be non-empty when pad_policy='pow2_buckets'")
        bad = [b for b in buckets if int(b) % int(pad_to_multiple_of) != 0]
        if bad:
            raise ValueError(f"pad_buckets must be divisible by pad_to_multiple_of={pad_to_multiple_of}; bad={bad}")

    # Simple cycling iterator over shuffled indices; avoids epoch bookkeeping.
    indices = iter_indices(n=n, seed=seed, shuffle=True)
    cursor = 0

    last_loss = float("nan")
    micro_steps = 0
    run_t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        step_t0 = time.perf_counter()
        losses = []
        pad_lens_step: list[int] = []
        max_lens_step: list[int] = []
        avg_lens_step: list[float] = []
        token_utils_step: list[float] = []
        for _ in range(int(grad_accum_steps)):
            if cursor + global_micro_batch > len(indices):
                indices = iter_indices(n=n, seed=seed + step, shuffle=True)
                cursor = 0
            batch_idx = indices[cursor : cursor + global_micro_batch]
            cursor += global_micro_batch

            # Each process owns `local_micro_batch` examples; the global batch
            # is formed by sharding across all hosts/devices.
            local_start = int(process_index) * int(local_micro_batch)
            local_idx = batch_idx[local_start : local_start + int(local_micro_batch)]
            if len(local_idx) != int(local_micro_batch):
                raise RuntimeError(
                    f"Unexpected local batch slice: got {len(local_idx)} indices, expected {int(local_micro_batch)}"
                )

            examples = [train_dataset[i] for i in local_idx]

            # Resolve per-micro-step padding length.
            #
            # NOTE: once the input pipeline switches to per-process local batches
            # (for multi-host efficiency), the true max length must be aggregated
            # across processes so every host uses an identical `pad_len`.
            pad_len = pad_to_length

            local_max_len = max(len(x["input_ids"]) for x in examples)
            global_max_len = int(local_max_len)
            if process_allgather is not None:
                gathered = np.asarray(process_allgather(np.asarray([int(local_max_len)], dtype=np.int32)))
                global_max_len = int(gathered.max())

            max_lens_step.append(int(global_max_len))

            if pad_to_length is not None and int(global_max_len) > int(pad_to_length):
                raise ValueError(
                    f"Batch max_len={int(global_max_len)} exceeds pad_to_length={int(pad_to_length)}; "
                    "increase data.max_len or ensure examples are truncated."
                )

            if pad_policy_norm == "max":
                pad_len = int(global_max_len)
            elif pad_policy_norm == "pow2_buckets":
                for b in buckets:
                    if int(b) >= int(global_max_len):
                        pad_len = int(b)
                        break
                if pad_len is None:
                    raise ValueError(
                        f"pad_policy='pow2_buckets' but global_max_len={int(global_max_len)} exceeds max bucket={int(buckets[-1])}. "
                        "Add a larger bucket or increase data.max_len."
                    )

            batch_np = collate_sft_batch(
                examples,
                pad_token_id=int(pad_token_id),
                pad_to_multiple_of=int(pad_to_multiple_of),
                pad_to_length=pad_len,
                padding_side=str(padding_side or "right"),
            )
            pad_len_actual = int(batch_np.input_ids.shape[1])
            pad_lens_step.append(int(pad_len_actual))

            local_token_count = int(batch_np.attention_mask.sum())
            global_token_count = int(local_token_count)
            if process_allgather is not None:
                gathered_tokens = np.asarray(process_allgather(np.asarray([int(local_token_count)], dtype=np.int64)))
                global_token_count = int(gathered_tokens.sum())

            denom = float(int(global_micro_batch) * int(pad_len_actual))
            token_util = float(global_token_count) / denom if denom > 0 else float("nan")
            token_utils_step.append(float(token_util))
            avg_lens_step.append(float(global_token_count) / float(int(global_micro_batch)))

            batch = jax.tree_util.tree_map_with_path(_form_training_global_array, batch_np.as_dict())

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
            if log_extra_cb is not None:
                def _mean(xs: list[float] | list[int]) -> float:
                    if not xs:
                        return float("nan")
                    return float(sum(float(x) for x in xs) / len(xs))

                log_extra_cb(
                    int(step),
                    {
                        "train/max_len_global": _mean(max_lens_step),
                        "train/pad_len": _mean(pad_lens_step),
                        "train/avg_seq_len": _mean(avg_lens_step),
                        "train/token_utilization": _mean(token_utils_step),
                    },
                )

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

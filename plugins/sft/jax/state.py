from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding

from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

from plugins.sft.jax.sft_module import TrainSftModule

from plugins.training.update.optimizer import LRScheduleConfig, OptimizerConfig, build_tx


class SftTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None


@dataclass(frozen=True)
class SftStateBundle:
    state: SftTrainState
    sharding: Any


def create_sft_state(
    *,
    mesh: Mesh,
    model: Any,
    params: ArrayTree,
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


__all__ = ["SftTrainState", "SftStateBundle", "create_sft_state"]

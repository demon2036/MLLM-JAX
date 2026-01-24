from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding

from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

from plugins.sft.jax.sft_module import TrainSftModule


class SftTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None


def _build_optimizer(*, name: str, lr: float, weight_decay: float) -> optax.GradientTransformation:
    name_norm = str(name or "adamw").strip().lower()
    lr = float(lr)
    weight_decay = float(weight_decay)

    if name_norm in {"adamw", "adam"}:
        tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    elif name_norm in {"lion"}:
        tx = optax.lion(learning_rate=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name!r} (expected adamw|lion)")

    # Keep it conservative by default; this repo does not have a global knob yet.
    return optax.chain(optax.clip_by_global_norm(1.0), tx)


@dataclass(frozen=True)
class SftStateBundle:
    state: SftTrainState
    sharding: Any


def create_sft_state(
    *,
    mesh: Mesh,
    model: Any,
    params: ArrayTree,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    grad_accum_steps: int,
    label_ignore_id: int = -100,
) -> SftStateBundle:
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    train_module = TrainSftModule(model=model, label_ignore_id=int(label_ignore_id))
    tx = _build_optimizer(name=optimizer_name, lr=float(learning_rate), weight_decay=float(weight_decay))

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


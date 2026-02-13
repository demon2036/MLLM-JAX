from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training import train_state

from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules
from plugins.sample.mllm_sampler import Sampler, get_model
from plugins.training.rl.remax.module import ReMaxPolicyGradientModule


class ReMaxTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params: Any | None = None


def get_remax_state(
    mesh: Any,
    *,
    training_steps: int,
    grad_accum_steps: int,
    model_path: str,
    beta: float = 0.0,
    gamma: float = 1.0,
    returns_style: str = "official",
    gradient_checkpointing: bool = True,
    create_sampler: bool = True,
    tx: Any | None = None,
) -> tuple[ReMaxTrainState, Any | None, Any]:
    del training_steps

    model, params, tokenizer = get_model(mesh, model_path=model_path)
    ref_model = get_model(mesh, model_path=model_path, only_model=True) if float(beta) != 0.0 else None

    module_kwargs = dict(
        model=model,
        pad_token_id=int(tokenizer.pad_token_id),
        ref_model=ref_model,
        kl_coef=float(beta),
        gamma=float(gamma),
        returns_style=str(returns_style),
    )

    if bool(gradient_checkpointing):
        train_module = flax.linen.remat(
            ReMaxPolicyGradientModule,
            policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        )(**module_kwargs)
    else:
        train_module = ReMaxPolicyGradientModule(**module_kwargs)

    if tx is None:
        raise ValueError("get_remax_state requires an explicit Optax `tx` (use plugins.training.core.optim.build_tx).")

    def init_fn(p):
        grad_accum = None
        if int(grad_accum_steps) > 1:
            grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p)
        return ReMaxTrainState.create(
            apply_fn=train_module.apply,
            params=p,
            tx=tx,
            ref_params=copy.deepcopy(p) if float(beta) != 0.0 else None,
            micro_step=0,
            micro_in_mini=int(grad_accum_steps),
            grad_accum=grad_accum,
        )

    state_shapes = jax.eval_shape(init_fn, params)
    train_state_partition = match_partition_rules(get_partition_rules_llama(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)
    state = jax.jit(init_fn, donate_argnums=(0,), out_shardings=train_state_sharding)(params)

    sampler = Sampler(model, tokenizer, mesh=mesh) if create_sampler else None
    return state, sampler, train_state_sharding


__all__ = ["ReMaxTrainState", "get_remax_state"]

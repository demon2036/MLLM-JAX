from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training import train_state
from jax.sharding import NamedSharding

from MLLM_JAX.train_modules import TrainGRPOModule
from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules

from plugins.llm.bundle import build_llm_bundle
from plugins.sample.mllm_sampler import Sampler
from plugins.training.update.optimizer import OptimizerConfig, build_tx


class GRPOTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params: Any | None = None


def get_grpo_state(
    mesh: Any,
    *,
    training_steps: int,
    grad_accum_steps: int,
    model_path: str,
    num_pre_q: int,
    max_lengths: int | None = None,
    beta: float = 0.0,
    create_sampler: bool = True,
    tx: Any | None = None,
    param_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> tuple[GRPOTrainState, Any | None, Any]:
    """Build GRPO training state + optional sampler on a given mesh.

    This replaces legacy `training2.get_state(...)` usage in plugin runners.
    """
    training_steps = int(training_steps)
    if training_steps <= 0:
        raise ValueError("training_steps must be > 0")
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    bundle = build_llm_bundle(
        mesh=mesh,
        model_path=str(model_path),
        param_dtype=str(param_dtype),
        compute_dtype=str(compute_dtype),
        trust_remote_code=bool(trust_remote_code),
        padding_side="right",
        only_model=False,
    )
    if bundle.params is None:
        raise RuntimeError("Expected bundle.params when only_model=False")

    model = bundle.model
    params = bundle.params

    # Reference model is only needed when KL beta != 0.
    model_ref = None
    if float(beta) != 0.0:
        ref_bundle = build_llm_bundle(
            mesh=mesh,
            model_path=str(model_path),
            tokenizer=bundle.tokenizer,
            param_dtype=str(param_dtype),
            compute_dtype=str(compute_dtype),
            trust_remote_code=bool(trust_remote_code),
            padding_side="right",
            only_model=True,
        )
        model_ref = ref_bundle.model

    train_module = flax.linen.remat(TrainGRPOModule, policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)(
        model=model,
        pad_token_id=float(bundle.pad_token_id),
        ref_model=model_ref,
        num_pre_Q=int(num_pre_q),
        beta=float(beta),
        max_lengths=int(max_lengths) if max_lengths is not None else 0,
    )

    tx_impl = tx if tx is not None else build_tx(training_steps=training_steps, cfg=OptimizerConfig())

    def init_fn(p):
        grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p) if grad_accum_steps > 1 else None
        ref_params = p if float(beta) != 0.0 else None
        return GRPOTrainState.create(
            apply_fn=train_module.apply,
            params=p,
            tx=tx_impl,
            ref_params=ref_params,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum,
        )

    state_shapes = jax.eval_shape(init_fn, params)
    partitions = match_partition_rules(get_partition_rules_llama(), state_shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    state = jax.jit(init_fn, out_shardings=shardings)(params)

    sampler = Sampler(model, bundle.tokenizer, mesh=mesh) if create_sampler else None
    return state, sampler, shardings


__all__ = ["GRPOTrainState", "get_grpo_state"]


from __future__ import annotations

import copy
import os
from typing import Any

import flax
import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training import train_state
from transformers import AutoConfig, AutoTokenizer

from MLLM_JAX.language.llama.llama import LlamaJaxConfig
from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
from MLLM_JAX.sample.sample_state_right_padding2 import Sampler, get_params
from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules
from plugins.training.algorithms import AlgoConfig
from plugins.training.ppo.module import PPOActorCriticModule


class PPOTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params: Any | None = None


def _resolve_param_dtype() -> jnp.dtype:
    param_dtype_raw = os.environ.get("MLLM_JAX_PARAM_DTYPE", "float32").strip().lower()
    if param_dtype_raw in {"float32", "f32"}:
        return jnp.float32
    if param_dtype_raw in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if param_dtype_raw in {"float16", "f16"}:
        return jnp.float16
    raise ValueError(
        f"Unsupported MLLM_JAX_PARAM_DTYPE={param_dtype_raw!r} (expected float32/bfloat16/float16)."
    )


def _init_value_head_params(*, hidden_size: int, param_dtype: jnp.dtype) -> dict[str, jnp.ndarray]:
    kernel = jnp.zeros((int(hidden_size), 1), dtype=param_dtype)
    bias = jnp.zeros((1,), dtype=param_dtype)
    return {"kernel": kernel, "bias": bias}


def get_ppo_state(
    mesh: Any,
    *,
    training_steps: int,
    grad_accum_steps: int,
    model_path: str,
    algo_cfg: AlgoConfig,
    beta: float = 0.0,
    create_sampler: bool = True,
    tx: Any | None = None,
) -> tuple[PPOTrainState, Any, PPOActorCriticModule]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config, "model_type", None) not in {"qwen2"}:
        raise ValueError(f"PPO value head currently supports Qwen2 only, got model_type={config.model_type!r}")

    jax_config = LlamaJaxConfig(mesh=mesh)
    module = PPOActorCriticModule(
        config=config,
        jax_config=jax_config,
        epsilon_low=0.2,
        epsilon_high=0.3,
        value_coef=algo_cfg.ppo_value_coef,
        value_clip_range=algo_cfg.ppo_value_clip_range,
        entropy_coef=algo_cfg.ppo_entropy_coef,
    )

    params = get_params(model_path)
    params = flax.core.unfreeze(params)

    param_dtype = _resolve_param_dtype()
    params["value_head"] = _init_value_head_params(hidden_size=int(config.hidden_size), param_dtype=param_dtype)
    params = flax.core.freeze(params)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=param_dtype), params)

    def init_fn(p):
        grad_accum = None
        if int(grad_accum_steps) > 1:
            grad_accum = jax.tree_util.tree_map(jnp.zeros_like, p)
        return PPOTrainState.create(
            apply_fn=module.apply,
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

    sampler = None
    if create_sampler:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        policy_model = Qwen2ForCausalLM(config, jax_config)
        sampler = Sampler(policy_model, tokenizer, mesh=mesh)

    return state, sampler, module


__all__ = ["PPOTrainState", "get_ppo_state"]

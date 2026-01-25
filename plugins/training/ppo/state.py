from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
from chex import ArrayTree
from flax.training import train_state

from MLLM_JAX.language.llama.llama import LlamaJaxConfig
from MLLM_JAX.utils import get_partition_rules_llama, match_partition_rules
from plugins.common.tokenizer import prepare_tokenizer
from plugins.llm.bundle import build_llm_bundle
from plugins.llm.dtypes import parse_dtype
from plugins.sample.mllm_sampler import Sampler
from plugins.training.algorithms import UpdateConfig
from plugins.training.ppo.module import PPOActorCriticModule


class PPOTrainState(train_state.TrainState):
    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None
    ref_params: Any | None = None


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
    update_cfg: UpdateConfig,
    beta: float = 0.0,
    create_sampler: bool = True,
    tx: Any | None = None,
    param_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
) -> tuple[PPOTrainState, Any, PPOActorCriticModule]:
    bundle = build_llm_bundle(
        mesh=mesh,
        model_path=str(model_path),
        param_dtype=str(param_dtype),
        compute_dtype=str(compute_dtype),
        trust_remote_code=True,
        padding_side="right",
        only_model=False,
    )
    config = bundle.hf_config
    if getattr(config, "model_type", None) not in {"qwen2"}:
        raise ValueError(f"PPO value head currently supports Qwen2 only, got model_type={getattr(config, 'model_type', None)!r}")

    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    pdtype = parse_dtype(str(param_dtype))
    cdtype = parse_dtype(str(compute_dtype))
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=cdtype, param_dtype=pdtype)
    module = PPOActorCriticModule(
        config=config,
        jax_config=jax_config,
        epsilon_low=0.2,
        epsilon_high=0.3,
        value_coef=update_cfg.value_coef,
        value_clip_range=update_cfg.value_clip_range,
        entropy_coef=update_cfg.entropy_coef,
    )

    if bundle.params is None:
        raise RuntimeError("Expected bundle.params when only_model=False")
    params = dict(bundle.params)
    params["value_head"] = _init_value_head_params(hidden_size=int(config.hidden_size), param_dtype=pdtype)

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
        tokenizer, _pad_token_id = prepare_tokenizer(bundle.tokenizer, padding_side="right")
        sampler = Sampler(bundle.model, tokenizer, mesh=mesh)

    return state, sampler, module


__all__ = ["PPOTrainState", "get_ppo_state"]

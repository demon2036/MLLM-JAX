from __future__ import annotations

from typing import Any

import flax


def ppo_training_step(state: Any, inputs: Any):
    """PPO training step with optional grad accumulation."""
    import jax
    import jax.numpy as jnp

    def loss_fn(params):
        if getattr(state, "ref_params", None) is None:
            variables = {"params": params}
        else:
            params_mut = flax.core.unfreeze(params)
            params_mut["ref_model"] = state.ref_params
            variables = {"params": flax.core.freeze(params_mut)}
        metrics = state.apply_fn(variables, inputs)
        per_token_logps = metrics.pop("per_token_logps", None)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics | {"per_token_logps": per_token_logps}

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if getattr(state, "grad_accum", None) is None:
        state = state.apply_gradients(grads=grads)
        return state, metrics

    state = state.replace(
        grad_accum=jax.tree_util.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
        micro_step=state.micro_step + 1,
    )

    def update_fn(st):
        grads_mean = jax.tree_util.tree_map(lambda g: g / st.micro_in_mini, st.grad_accum)
        return st.apply_gradients(
            grads=grads_mean,
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, st.grad_accum),
            micro_step=st.micro_step % st.micro_in_mini,
        )

    state = jax.lax.cond(state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state)
    return state, metrics


__all__ = ["ppo_training_step"]

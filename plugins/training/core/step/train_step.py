from __future__ import annotations

from typing import Any


def training_step(state: Any, inputs: Any):
    """Training step with correct gradient accumulation (when state.grad_accum is set).

    This mirrors `training2.training_step` but fixes the grad-accum logic so that
    `grad_accum_steps > 1` behaves as intended.
    """
    import jax
    import jax.numpy as jnp

    def _maybe_update_ema(st: Any) -> Any:
        ema_params = getattr(st, "ema_params", None)
        if ema_params is None:
            return st
        decay = getattr(st, "ema_decay", None)
        if decay is None:
            raise ValueError("ema_params is set but ema_decay is missing on the TrainState")
        # NOTE: this runs inside jitted training loops, so `decay` may be a tracer
        # (e.g. a 0-d array). Avoid `float(decay)` which would trigger a
        # ConcretizationTypeError.
        #
        # Range validation is performed when building the TrainState (EMA-enabled
        # configs validate decay in Python).
        new_ema = jax.tree_util.tree_map(
            lambda ema, p: ema * jnp.asarray(decay, dtype=ema.dtype) + (1.0 - jnp.asarray(decay, dtype=ema.dtype)) * p,
            ema_params,
            st.params,
        )
        return st.replace(ema_params=new_ema)

    def loss_fn(params):
        variables = {"params": {"model": params}}
        if getattr(state, "ref_params", None) is not None:
            variables["params"]["ref_model"] = state.ref_params
        metrics = state.apply_fn(variables, inputs)
        per_token_logps = metrics.pop("per_token_logps", None)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics | {"per_token_logps": per_token_logps}

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if getattr(state, "grad_accum", None) is None:
        state = state.apply_gradients(grads=grads)
        state = _maybe_update_ema(state)
        return state, metrics

    state = state.replace(
        grad_accum=jax.tree_util.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
        micro_step=state.micro_step + 1,
    )

    def update_fn(st):
        grads_mean = jax.tree_util.tree_map(lambda g: g / st.micro_in_mini, st.grad_accum)
        st = st.apply_gradients(
            grads=grads_mean,
            grad_accum=jax.tree_util.tree_map(jnp.zeros_like, st.grad_accum),
            micro_step=st.micro_step % st.micro_in_mini,
        )
        st = _maybe_update_ema(st)
        return st

    state = jax.lax.cond(state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state)
    return state, metrics


__all__ = ["training_step"]

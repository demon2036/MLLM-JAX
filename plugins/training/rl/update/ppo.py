from __future__ import annotations

from typing import Any, Callable, Mapping


def ppo_update(
    *,
    state: Any,
    datas: Mapping[str, Any],
    total_valid_token_count: Any,
    train_step: Callable[[Any, Any], tuple[Any, Mapping[str, Any]]],
    slice_data: Callable[[Any, int, int], Any],
    grad_accum_steps: int,
    ppo_steps: int,
) -> tuple[Any, dict[str, Any], Mapping[str, Any], Any | None]:
    """Run PPO/GRPO updates over the provided (global) batch.

    Contract:
    - First PPO step collects `per_token_logps` to form `old_per_token_logps`.
    - Subsequent PPO steps consume `old_per_token_logps` from `datas`.
    """
    import jax
    import jax.numpy as jnp

    datas_dict = dict(datas)
    last_meta: Mapping[str, Any] = {}
    entropy: Any | None = None

    per_token_logps: list[Any] = []

    for ppo_step_idx in range(int(ppo_steps)):
        for micro_idx in range(int(grad_accum_steps)):
            local_data = jax.tree_util.tree_map(
                lambda x: slice_data(x, int(grad_accum_steps), int(micro_idx)),
                datas_dict,
            )
            local_data["total_valid_token_count"] = total_valid_token_count
            state, last_meta = train_step(state, local_data)
            if ppo_step_idx == 0:
                per_token_logps.append(last_meta["per_token_logps"])

        if ppo_step_idx == 0:
            datas_dict["old_per_token_logps"] = jnp.concatenate(per_token_logps, axis=0)
            entropy = last_meta.get("entropy")

    return state, datas_dict, last_meta, entropy


__all__ = ["ppo_update"]

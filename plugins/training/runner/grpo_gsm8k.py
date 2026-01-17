from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from plugins.training.grpo.advantages import compute_grpo_advantages_by_group_id
from plugins.training.grpo.rewarding import compute_weighted_rewards


@dataclass(frozen=True)
class GRPOGsm8kConfig:
    model_path: str
    steps: int
    batch_size: int
    num_pre_q: int
    global_length: int
    max_length_sample: int
    max_length_total: int
    ppo_epochs: int
    grad_accum_steps: int
    beta: float
    mesh_shape: str

    wandb_project: str
    wandb_name: str
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)


def _ensure_batch_multiple_of_local_devices(local_batch: int, local_device_count: int) -> int:
    if local_batch % local_device_count == 0:
        return local_batch
    return ((local_batch + local_device_count - 1) // local_device_count) * local_device_count


def _maybe_init_wandb(cfg: GRPOGsm8kConfig):
    import jax

    if jax.process_index() != 0:
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.__dict__)
        return wandb
    except Exception as e:
        print(f"wandb disabled due to init error: {e}")
        return None


def run_grpo_gsm8k(cfg: GRPOGsm8kConfig) -> None:
    """End-to-end GRPO training loop (rollout → reward → advantages → update)."""
    import jax
    import jax.numpy as jnp
    from datasets import load_dataset
    from jax.experimental.multihost_utils import process_allgather

    from MLLM_JAX.utils import _form_global_array, get_jax_mesh2
    from prompts.prompts import system_prompt
    from training2 import (
        get_state,
        reward_correct,
        reward_format,
        slice_data,
        tag_count_reward,
        training_step,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        jax.distributed.initialize()
    except Exception:
        pass

    mesh = get_jax_mesh2(cfg.mesh_shape)
    local_device_count = len(mesh.local_devices)

    local_batch = cfg.batch_size * cfg.num_pre_q
    padded_local_batch = _ensure_batch_multiple_of_local_devices(local_batch, local_device_count)
    if padded_local_batch != local_batch:
        if padded_local_batch % cfg.batch_size != 0:
            raise ValueError(
                f"Cannot pad local batch {local_batch} -> {padded_local_batch} "
                f"because batch_size={cfg.batch_size} does not divide it."
            )
        new_num_pre_q = padded_local_batch // cfg.batch_size
        print(
            f"Padding local batch {local_batch} -> {padded_local_batch} by changing "
            f"num_pre_q {cfg.num_pre_q} -> {new_num_pre_q}."
        )
        cfg = GRPOGsm8kConfig(**{**cfg.__dict__, "num_pre_q": new_num_pre_q})
        local_batch = padded_local_batch

    print(f"backend={jax.default_backend()} process={jax.process_index()}/{jax.process_count()}")
    print(f"device_count={jax.device_count()} local_device_count={local_device_count}")
    print(
        "config="
        + str(
            dict(
                model_path=cfg.model_path,
                steps=cfg.steps,
                batch_size=cfg.batch_size,
                num_pre_q=cfg.num_pre_q,
                local_batch=local_batch,
                global_length=cfg.global_length,
                max_length_sample=cfg.max_length_sample,
                max_length_total=cfg.max_length_total,
                ppo_epochs=cfg.ppo_epochs,
                grad_accum_steps=cfg.grad_accum_steps,
                beta=cfg.beta,
                mesh_shape=cfg.mesh_shape,
                wandb_project=cfg.wandb_project,
                wandb_name=cfg.wandb_name,
                reward_weights=cfg.reward_weights,
            )
        )
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(dataset["question"], dataset["answer"])]
    if jax.process_count() > 1:
        qas = qas[jax.process_index() :: jax.process_count()]
    if not qas:
        raise RuntimeError("No GSM8K data after sharding.")

    state, sampler, _state_sharding = get_state(
        mesh,
        training_steps=cfg.steps,
        grad_accum_steps=cfg.grad_accum_steps,
        model_path=cfg.model_path,
        num_pre_q=cfg.num_pre_q,
        max_lengths=cfg.max_length_total,
        beta=cfg.beta,
        create_sampler=True,
    )

    train_fn = jax.jit(training_step, donate_argnums=(0,))
    wandb = _maybe_init_wandb(cfg)

    reward_funcs = [reward_correct, reward_format, tag_count_reward]

    rng = random.Random(0xC0FFEE + jax.process_index())
    for step in range(cfg.steps):
        batch_items = [rng.choice(qas) for _ in range(cfg.batch_size)]
        prompts_base = [item["Q"] for item in batch_items]
        repeated_prompts = [p for p in prompts_base for _ in range(cfg.num_pre_q)]
        repeated_items = [item for item in batch_items for _ in range(cfg.num_pre_q)]
        group_ids = np.repeat(np.arange(cfg.batch_size, dtype=np.int32), cfg.num_pre_q)

        t0 = time.time()

        # --- Rollout (sampling) ---
        # Keep rollout logic local to avoid cross-host coupling; global sync happens at the batch->global array step.
        from plugins.training.grpo.sampling import generate_answers_and_training_batch

        _chat_prompts, answers, datas_np = generate_answers_and_training_batch(
            prompts=repeated_prompts,
            sampler=sampler,
            params=state.params,
            system_prompt=system_prompt,
            global_length=int(cfg.global_length),
            max_length_sample=cfg.max_length_sample,
        )

        # --- Reward ---
        reward_weights = cfg.reward_weights
        rewards_per_func, rewards_np = compute_weighted_rewards(
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            inputs=repeated_items,
            answers=answers,
        )

        rewards_global = np.asarray(process_allgather(rewards_np))
        mean_global = float(rewards_global.mean())
        std_global = float(max(rewards_global.std(), 1e-6))

        # --- Advantages (group_id based) ---
        advantages_np = compute_grpo_advantages_by_group_id(
            rewards=rewards_np,
            group_ids=group_ids,
            eps=1e-4,
        )

        datas_np["rewards"] = rewards_np
        datas_np["advantages"] = advantages_np
        datas_np["group_ids"] = group_ids

        # --- Update ---
        datas = jax.tree_util.tree_map_with_path(
            lambda path, x: _form_global_array(path, x, global_mesh=mesh),
            datas_np,
        )
        total_valid_token_count = datas["labels"][:, 1:].sum()
        datas = {**datas, "total_valid_token_count": total_valid_token_count}

        metrics: dict[str, Any] = {}
        old_per_token_logps = None
        for ppo_epoch in range(cfg.ppo_epochs):
            if ppo_epoch > 0 and old_per_token_logps is not None:
                ppo_inputs = {**datas, "old_per_token_logps": old_per_token_logps}
            else:
                ppo_inputs = datas

            state, metrics = train_fn(state, ppo_inputs)
            jax.block_until_ready(metrics["loss"])
            if ppo_epoch == 0:
                old_per_token_logps = metrics.get("per_token_logps")

        dt = time.time() - t0

        loss_value = float(np.asarray(metrics["loss"]))
        entropy_value = float(np.asarray(jnp.mean(metrics["entropy"])))
        reward_mean = float(rewards_np.mean())

        if jax.process_index() == 0:
            print(
                f"step={step} loss={loss_value:.6f} entropy={entropy_value:.4f} "
                f"reward_mean={reward_mean:.4f} dt={dt:.2f}s"
            )

        if wandb is not None and jax.process_index() == 0:
            wandb.log(
                dict(
                    loss=loss_value,
                    entropy=entropy_value,
                    reward_mean=reward_mean,
                    mean_global=mean_global,
                    std_global=std_global,
                    dt=dt,
                ),
                step=step,
            )

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from plugins.training.grpo.advantages import compute_grpo_advantages_by_group_id
from plugins.training.grpo.batching import infer_rollout_passes, round_up_passes_for_divisibility
from plugins.training.grpo.rewarding import compute_weighted_rewards
from plugins.training.grpo.update import ppo_update


@dataclass(frozen=True)
class GRPORolloutConfig:
    batch_size: int
    num_pre_q: int
    global_length: int
    max_length_sample: int
    # Optional: global prompts per training step (across all processes).
    global_batch_size: int | None = None
    # Optional: prompts per device per rollout pass (forward-only).
    per_device_batch_size: int | None = None
    # Rollout backend selector (swappable generation engine).
    backend: str = "naive"


@dataclass(frozen=True)
class GRPOTrainConfig:
    # Optional: if set, split the rollout batch into smaller micro-batches for update.
    micro_batch_size: int | None
    max_length_total: int
    ppo_epochs: int
    grad_accum_steps: int
    beta: float
    # Optional: sequences per device per micro-step (backward).
    per_device_micro_batch_size: int | None = None


@dataclass(frozen=True)
class GRPOGsm8kConfig:
    model_path: str
    steps: int
    rollout: GRPORolloutConfig
    train: GRPOTrainConfig
    mesh_shape: str

    wandb_project: str
    wandb_name: str
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
    eval_every_steps: int = 0
    eval_batches: int = 1
    eval_split: str = "test"


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

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=asdict(cfg))
        return wandb
    except Exception as e:
        print(f"wandb disabled due to init error: {e}")
        return None


def _as_float(x: Any) -> float:
    return float(np.asarray(x))


def _as_int(x: Any) -> int:
    return int(np.asarray(x))


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _pad_2d_right(x: np.ndarray, target_len: int, pad_value: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 array, got shape={x.shape}")
    cur = int(x.shape[1])
    target = int(target_len)
    if cur == target:
        return x
    if cur > target:
        raise ValueError(f"Cannot pad to a smaller length: {cur} -> {target}")
    return np.pad(x, ((0, 0), (0, target - cur)), constant_values=pad_value)


def run_grpo_gsm8k(cfg: GRPOGsm8kConfig) -> None:
    """End-to-end GRPO training loop (rollout → reward → advantages → update)."""
    import jax
    import jax.numpy as jnp
    from datasets import load_dataset
    from jax.experimental.multihost_utils import process_allgather
    from transformers import AutoTokenizer

    from MLLM_JAX.utils import _form_global_array, get_jax_mesh2
    from prompts.prompts import system_prompt
    from plugins.training.grpo.train_step import training_step
    from training2 import (
        get_state,
        reward_correct,
        reward_format,
        slice_data,
        tag_count_reward,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        jax.distributed.initialize()
    except Exception:
        pass

    mesh = get_jax_mesh2(cfg.mesh_shape)
    local_device_count = len(mesh.local_devices)
    process_count = int(jax.process_count())

    # --- Resolve rollout batch (prompts) ---
    rollout_batch_size = int(cfg.rollout.batch_size)
    if cfg.rollout.per_device_batch_size is not None:
        per_device_rollout = int(cfg.rollout.per_device_batch_size)
        if per_device_rollout <= 0:
            raise ValueError("rollout.per_device_batch_size must be > 0")
        rollout_batch_size = per_device_rollout * local_device_count

    if rollout_batch_size <= 0:
        raise ValueError("rollout.batch_size must be > 0")

    # Ensure per-pass local batch (sequences) can be evenly split across local devices.
    local_batch_per_pass = int(rollout_batch_size) * int(cfg.rollout.num_pre_q)
    padded_local_batch_per_pass = _ensure_batch_multiple_of_local_devices(local_batch_per_pass, local_device_count)
    if padded_local_batch_per_pass != local_batch_per_pass:
        if padded_local_batch_per_pass % rollout_batch_size != 0:
            raise ValueError(
                f"Cannot pad local batch {local_batch_per_pass} -> {padded_local_batch_per_pass} "
                f"because rollout.batch_size={rollout_batch_size} does not divide it."
            )
        new_num_pre_q = padded_local_batch_per_pass // rollout_batch_size
        print(
            f"Padding local batch {local_batch_per_pass} -> {padded_local_batch_per_pass} by changing "
            f"rollout.num_pre_q {cfg.rollout.num_pre_q} -> {new_num_pre_q}."
        )
        cfg = GRPOGsm8kConfig(**{**cfg.__dict__, "rollout": GRPORolloutConfig(**{**cfg.rollout.__dict__, "num_pre_q": new_num_pre_q})})
        local_batch_per_pass = padded_local_batch_per_pass

    base_rollout_passes, _base_effective_global_prompts = infer_rollout_passes(
        global_batch_size=cfg.rollout.global_batch_size,
        batch_size_per_process=rollout_batch_size,
        process_count=process_count,
    )
    rollout_passes = int(base_rollout_passes)

    # --- Resolve train micro-batch & grad accumulation (sequences) ---
    micro_batch_size_per_process = cfg.train.micro_batch_size
    if cfg.train.per_device_micro_batch_size is not None:
        if micro_batch_size_per_process is not None:
            raise ValueError("Specify only one of train.micro_batch_size and train.per_device_micro_batch_size.")
        per_device_micro = int(cfg.train.per_device_micro_batch_size)
        if per_device_micro <= 0:
            raise ValueError("train.per_device_micro_batch_size must be > 0.")
        micro_batch_size_per_process = per_device_micro * local_device_count

    if micro_batch_size_per_process is not None:
        micro_batch_size_per_process = int(micro_batch_size_per_process)
        if micro_batch_size_per_process <= 0:
            raise ValueError("train.micro_batch_size must be > 0.")
        padded_passes = round_up_passes_for_divisibility(
            passes=rollout_passes,
            sequences_per_pass_per_process=local_batch_per_pass,
            micro_batch_size_per_process=micro_batch_size_per_process,
        )
        if padded_passes != rollout_passes:
            print(
                f"Padding rollout passes {rollout_passes} -> {padded_passes} so that "
                f"passes * local_batch_per_pass is divisible by train.micro_batch_size={micro_batch_size_per_process}."
            )
            rollout_passes = int(padded_passes)

    local_batch = int(rollout_passes) * int(local_batch_per_pass)

    grad_accum_steps = int(cfg.train.grad_accum_steps)
    if micro_batch_size_per_process is not None:
        if local_batch % micro_batch_size_per_process != 0:
            raise ValueError(
                f"train.micro_batch_size={micro_batch_size_per_process} must divide local_batch={local_batch} "
                f"(rollout_passes * rollout.batch_size * rollout.num_pre_q)."
            )
        micro_steps = local_batch // micro_batch_size_per_process
        if grad_accum_steps != 1 and grad_accum_steps != micro_steps:
            print(
                f"Overriding train.grad_accum_steps {grad_accum_steps} -> {micro_steps} "
                f"to respect train.micro_batch_size={micro_batch_size_per_process}."
            )
        grad_accum_steps = micro_steps

    if local_batch % grad_accum_steps != 0:
        raise ValueError(
            f"train.grad_accum_steps={grad_accum_steps} must divide local_batch={local_batch} "
            f"(rollout_passes * rollout.batch_size * rollout.num_pre_q)."
        )

    cfg = GRPOGsm8kConfig(
        **{
            **cfg.__dict__,
            "rollout": GRPORolloutConfig(**{**cfg.rollout.__dict__, "batch_size": rollout_batch_size}),
            "train": GRPOTrainConfig(
                **{
                    **cfg.train.__dict__,
                    "micro_batch_size": micro_batch_size_per_process,
                    "grad_accum_steps": grad_accum_steps,
                }
            ),
        }
    )

    print(f"backend={jax.default_backend()} process={jax.process_index()}/{jax.process_count()}")
    print(f"device_count={jax.device_count()} local_device_count={local_device_count}")
    print(
        "config="
        + str(
            dict(
                model_path=cfg.model_path,
                steps=cfg.steps,
                rollout={**asdict(cfg.rollout), "passes_per_step": int(rollout_passes)},
                train=asdict(cfg.train),
                local_batch=local_batch,
                mesh_shape=cfg.mesh_shape,
                wandb_project=cfg.wandb_project,
                wandb_name=cfg.wandb_name,
                reward_weights=cfg.reward_weights,
                eval_every_steps=cfg.eval_every_steps,
                eval_batches=cfg.eval_batches,
                eval_split=cfg.eval_split,
            )
        )
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(dataset["question"], dataset["answer"])]
    if jax.process_count() > 1:
        qas = qas[jax.process_index() :: jax.process_count()]
    if not qas:
        raise RuntimeError("No GSM8K data after sharding.")

    eval_qas: list[dict[str, str]] = []
    if int(cfg.eval_every_steps) > 0:
        eval_dataset = load_dataset("openai/gsm8k", "main", split=str(cfg.eval_split))
        eval_qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(eval_dataset["question"], eval_dataset["answer"])]
        if jax.process_count() > 1:
            eval_qas = eval_qas[jax.process_index() :: jax.process_count()]
        if not eval_qas and jax.process_index() == 0:
            print(f"WARNING: eval enabled but no eval data after sharding (split={cfg.eval_split!r}).")

    from plugins.training.rollout_backends import create_rollout_backend

    rollout_backend_name = str(cfg.rollout.backend).strip().lower()
    use_sglang_jax = rollout_backend_name in {"sglang_jax", "sglang-jax", "sglang"}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    pad_token_id = int(pad_token_id)

    if use_sglang_jax:
        rollout_backend = create_rollout_backend(
            name=rollout_backend_name,
            sampler=None,
            tokenizer=tokenizer,
            model_path=cfg.model_path,
        )
        # Allocate sglang-jax's model + KV cache before creating the training state.
        # This makes the Engine's KV sizing depend on full device memory rather than
        # "whatever is left after FSDP+optimizer", which is critical on v4-8.
        if hasattr(rollout_backend, "initialize"):
            rollout_backend.initialize()  # type: ignore[attr-defined]
        state, sampler, _state_sharding = get_state(
            mesh,
            training_steps=cfg.steps,
            grad_accum_steps=grad_accum_steps,
            model_path=cfg.model_path,
            num_pre_q=cfg.rollout.num_pre_q,
            max_lengths=cfg.train.max_length_total,
            beta=cfg.train.beta,
            create_sampler=False,
        )
        # Eliminate persistent weight duplication as early as possible (before
        # compiling the training step): swap Engine weights to the freshly
        # created training params, then flush any KV/cache allocations.
        if hasattr(rollout_backend, "sync_weights"):
            rollout_backend.sync_weights(state.params)  # type: ignore[attr-defined]
        if hasattr(rollout_backend, "flush_cache"):
            rollout_backend.flush_cache()  # type: ignore[attr-defined]
    else:
        state, sampler, _state_sharding = get_state(
            mesh,
            training_steps=cfg.steps,
            grad_accum_steps=grad_accum_steps,
            model_path=cfg.model_path,
            num_pre_q=cfg.rollout.num_pre_q,
            max_lengths=cfg.train.max_length_total,
            beta=cfg.train.beta,
            create_sampler=True,
        )
        rollout_backend = create_rollout_backend(
            name=rollout_backend_name,
            sampler=sampler,
            tokenizer=None,
            model_path=cfg.model_path,
        )

    train_fn = jax.jit(training_step, donate_argnums=(0,))
    wandb = _maybe_init_wandb(cfg)

    reward_funcs = [reward_correct, reward_format, tag_count_reward]
    reward_func_names = [fn.__name__ for fn in reward_funcs]

    rng = random.Random(0xC0FFEE + jax.process_index())
    for step in range(cfg.steps):
        t_step0 = time.perf_counter()

        # --- Rollout (sampling) ---
        # Keep rollout logic local to avoid cross-host coupling; global sync happens at the batch->global array step.
        reward_weights = cfg.reward_weights
        answers_all: list[str] = []
        datas_np_all: list[dict[str, np.ndarray]] = []
        rewards_per_func_all: list[np.ndarray] = []
        rewards_all: list[np.ndarray] = []
        advantages_all: list[np.ndarray] = []
        group_ids_all: list[np.ndarray] = []

        t_rollout = 0.0
        t_rollout_sync = 0.0
        t_rollout_generate = 0.0
        t_rollout_flush = 0.0
        t_reward = 0.0
        t_adv = 0.0

        for pass_idx in range(int(rollout_passes)):
            batch_items = [rng.choice(qas) for _ in range(cfg.rollout.batch_size)]
            prompts_base = [item["Q"] for item in batch_items]
            repeated_prompts = [p for p in prompts_base for _ in range(cfg.rollout.num_pre_q)]
            repeated_items = [item for item in batch_items for _ in range(cfg.rollout.num_pre_q)]

            group_ids = np.repeat(
                np.arange(cfg.rollout.batch_size, dtype=np.int32) + pass_idx * cfg.rollout.batch_size,
                cfg.rollout.num_pre_q,
            )

            # --- Rollout (sampling) ---
            t_sync0 = time.perf_counter()
            if hasattr(rollout_backend, "sync_weights"):
                rollout_backend.sync_weights(state.params)  # type: ignore[attr-defined]
            t_rollout_sync += time.perf_counter() - t_sync0

            t_rollout0 = time.perf_counter()
            rollout = rollout_backend.rollout(
                prompts=repeated_prompts,
                params=state.params,
                system_prompt=system_prompt,
                global_length=int(cfg.rollout.global_length),
                max_length_sample=cfg.rollout.max_length_sample,
            )
            answers = rollout.answers
            datas_np = rollout.batch
            t_rollout_generate += time.perf_counter() - t_rollout0

            t_flush0 = time.perf_counter()
            if hasattr(rollout_backend, "flush_cache"):
                rollout_backend.flush_cache()  # type: ignore[attr-defined]
            t_rollout_flush += time.perf_counter() - t_flush0

            # --- Reward ---
            t_reward0 = time.perf_counter()
            rewards_per_func, rewards_np = compute_weighted_rewards(
                reward_funcs=reward_funcs,
                reward_weights=reward_weights,
                inputs=repeated_items,
                answers=answers,
            )
            t_reward += time.perf_counter() - t_reward0

            # --- Advantages (group_id based) ---
            t_adv0 = time.perf_counter()
            advantages_np = compute_grpo_advantages_by_group_id(
                rewards=rewards_np,
                group_ids=group_ids,
                eps=1e-4,
            )
            t_adv += time.perf_counter() - t_adv0

            datas_np = dict(datas_np)
            datas_np["rewards"] = rewards_np
            datas_np["advantages"] = advantages_np
            datas_np["group_ids"] = group_ids

            answers_all.extend(answers)
            datas_np_all.append(datas_np)
            rewards_per_func_all.append(rewards_per_func)
            rewards_all.append(rewards_np)
            advantages_all.append(advantages_np)
            group_ids_all.append(group_ids)

        t_rollout = t_rollout_sync + t_rollout_generate + t_rollout_flush
        rewards_np = np.concatenate(rewards_all, axis=0) if rewards_all else np.asarray([], dtype=np.float32)
        advantages_np = np.concatenate(advantages_all, axis=0) if advantages_all else np.asarray([], dtype=np.float32)
        group_ids = np.concatenate(group_ids_all, axis=0) if group_ids_all else np.asarray([], dtype=np.int32)
        rewards_per_func = (
            np.concatenate(rewards_per_func_all, axis=1) if rewards_per_func_all else np.asarray([], dtype=np.float32)
        )

        # Concatenate per-pass batches. If sequence lengths differ, pad up to the global max length.
        datas_np = {}
        if datas_np_all:
            keys = datas_np_all[0].keys()
            for k in keys:
                if k in {"input_ids", "attention_mask", "labels"}:
                    max_len_local = max(int(d[k].shape[1]) for d in datas_np_all)
                    if k == "input_ids":
                        pad_value = pad_token_id
                    else:
                        pad_value = 0
                    parts = [_pad_2d_right(d[k], max_len_local, pad_value) for d in datas_np_all]
                    datas_np[k] = np.concatenate(parts, axis=0)
                else:
                    datas_np[k] = np.concatenate([d[k] for d in datas_np_all], axis=0)

            seq_len_local = int(datas_np["input_ids"].shape[1])
            seq_len_global = int(np.asarray(process_allgather(np.asarray([seq_len_local], dtype=np.int32))).max())
            if seq_len_global != seq_len_local:
                datas_np["input_ids"] = _pad_2d_right(datas_np["input_ids"], seq_len_global, pad_token_id)
                datas_np["attention_mask"] = _pad_2d_right(datas_np["attention_mask"], seq_len_global, 0)
                datas_np["labels"] = _pad_2d_right(datas_np["labels"], seq_len_global, 0)

        rewards_global = np.asarray(process_allgather(rewards_np)).reshape(-1)
        reward_global_stats = _stats_1d(rewards_global)

        # --- Update ---
        t_shard0 = time.perf_counter()
        datas = jax.tree_util.tree_map_with_path(
            lambda path, x: _form_global_array(path, x, global_mesh=mesh),
            datas_np,
        )
        total_valid_token_count = datas["labels"][:, 1:].sum()
        t_shard = time.perf_counter() - t_shard0

        t_update0 = time.perf_counter()
        state, datas, last_meta, entropy = ppo_update(
            state=state,
            datas=datas,
            total_valid_token_count=total_valid_token_count,
            train_step=train_fn,
            slice_data=slice_data,
            grad_accum_steps=grad_accum_steps,
            ppo_steps=cfg.train.ppo_epochs,
        )
        jax.block_until_ready(last_meta["loss"])
        t_update = time.perf_counter() - t_update0

        t_step = time.perf_counter() - t_step0

        loss_value = _as_float(last_meta["loss"])
        if entropy is None:
            entropy_value = _as_float(jnp.mean(last_meta["entropy"]))
        else:
            entropy_value = _as_float(entropy)

        # --- Derived stats (global) ---
        advantages_global = np.asarray(process_allgather(advantages_np)).reshape(-1)
        adv_global_stats = _stats_1d(advantages_global)

        rewards_per_func_global = np.asarray(process_allgather(rewards_per_func))
        # shape [process_count, num_funcs, B_local]
        per_func_means = rewards_per_func_global.mean(axis=(0, 2))

        labels_np = np.asarray(datas_np["labels"])
        attn_np = np.asarray(datas_np["attention_mask"])
        completion_len_local = labels_np.sum(axis=1).astype(np.float32)
        total_len_local = attn_np.sum(axis=1).astype(np.float32)
        prompt_len_local = (total_len_local - completion_len_local).astype(np.float32)

        completion_len_global = np.asarray(process_allgather(completion_len_local)).reshape(-1)
        total_len_global = np.asarray(process_allgather(total_len_local)).reshape(-1)
        prompt_len_global = np.asarray(process_allgather(prompt_len_local)).reshape(-1)

        completion_stats = _stats_1d(completion_len_global)
        prompt_stats = _stats_1d(prompt_len_global)
        total_len_stats = _stats_1d(total_len_global)

        valid_tokens_local = int(labels_np[:, 1:].sum())
        valid_tokens_global = int(np.asarray(process_allgather(np.asarray([valid_tokens_local], dtype=np.int64))).sum())
        global_batch = int(local_batch * jax.process_count())

        train_log: dict[str, Any] = {
            "train/loss": loss_value,
            "train/entropy": entropy_value,
            "train/reward_total_mean": reward_global_stats["mean"],
            "train/reward_total_std": reward_global_stats["std"],
            "train/reward_total_min": reward_global_stats["min"],
            "train/reward_total_max": reward_global_stats["max"],
            "train/adv_mean": adv_global_stats["mean"],
            "train/adv_std": adv_global_stats["std"],
            "train/adv_min": adv_global_stats["min"],
            "train/adv_max": adv_global_stats["max"],
            "train/seq_prompt_len_mean": prompt_stats["mean"],
            "train/seq_prompt_len_max": prompt_stats["max"],
            "train/seq_completion_len_mean": completion_stats["mean"],
            "train/seq_completion_len_max": completion_stats["max"],
            "train/seq_total_len_mean": total_len_stats["mean"],
            "train/seq_total_len_max": total_len_stats["max"],
            "train/batch_global": global_batch,
            "train/batch_local": int(local_batch),
            "train/total_valid_token_count": valid_tokens_global,
            "time/train/rollout_s": float(t_rollout),
            "time/train/rollout_sync_s": float(t_rollout_sync),
            "time/train/rollout_generate_s": float(t_rollout_generate),
            "time/train/rollout_flush_s": float(t_rollout_flush),
            "time/train/reward_s": float(t_reward),
            "time/train/advantages_s": float(t_adv),
            "time/train/shard_s": float(t_shard),
            "time/train/update_s": float(t_update),
            "time/train/step_s": float(t_step),
        }
        for name, mean_value in zip(reward_func_names, per_func_means):
            train_log[f"train/{name}_mean"] = float(mean_value)

        if t_step > 0:
            train_log["throughput/train/valid_tokens_per_s"] = float(valid_tokens_global) / float(t_step)
        if t_update > 0:
            train_log["throughput/train/valid_tokens_per_s_update"] = float(valid_tokens_global) / float(t_update)

        if wandb is not None and jax.process_index() == 0:
            wandb.log(train_log, step=step)

        if jax.process_index() == 0:
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss={loss_value:.6f}",
                        f"entropy={entropy_value:.4f}",
                        f"reward_mean={reward_global_stats['mean']:.4f}",
                        f"dt={t_step:.2f}s",
                    ]
                )
            )

        # --- Eval (optional; no updates) ---
        if eval_qas and int(cfg.eval_every_steps) > 0 and ((step + 1) % int(cfg.eval_every_steps) == 0):
            eval_logs: dict[str, Any] = {}
            eval_rollout_s = 0.0
            eval_rollout_sync_s = 0.0
            eval_rollout_generate_s = 0.0
            eval_rollout_flush_s = 0.0
            eval_reward_s = 0.0
            eval_step0 = time.perf_counter()
            eval_rewards_all: list[np.ndarray] = []
            eval_rewards_per_func_all: list[np.ndarray] = []

            for eval_batch_idx in range(int(cfg.eval_batches)):
                start = (step * int(cfg.eval_batches) + eval_batch_idx) * cfg.rollout.batch_size
                eval_items = [eval_qas[(start + i) % len(eval_qas)] for i in range(cfg.rollout.batch_size)]
                eval_prompts_base = [item["Q"] for item in eval_items]
                eval_repeated_prompts = [p for p in eval_prompts_base for _ in range(cfg.rollout.num_pre_q)]
                eval_repeated_items = [item for item in eval_items for _ in range(cfg.rollout.num_pre_q)]

                t_eval_sync0 = time.perf_counter()
                if hasattr(rollout_backend, "sync_weights"):
                    rollout_backend.sync_weights(state.params)  # type: ignore[attr-defined]
                eval_rollout_sync_s += time.perf_counter() - t_eval_sync0

                t_eval_rollout0 = time.perf_counter()
                eval_rollout = rollout_backend.rollout(
                    prompts=eval_repeated_prompts,
                    params=state.params,
                    system_prompt=system_prompt,
                    global_length=int(cfg.rollout.global_length),
                    max_length_sample=cfg.rollout.max_length_sample,
                )
                eval_answers = eval_rollout.answers
                eval_rollout_generate_s += time.perf_counter() - t_eval_rollout0

                t_eval_flush0 = time.perf_counter()
                if hasattr(rollout_backend, "flush_cache"):
                    rollout_backend.flush_cache()  # type: ignore[attr-defined]
                eval_rollout_flush_s += time.perf_counter() - t_eval_flush0

                t_eval_reward0 = time.perf_counter()
                eval_rewards_per_func, eval_rewards_np = compute_weighted_rewards(
                    reward_funcs=reward_funcs,
                    reward_weights=reward_weights,
                    inputs=eval_repeated_items,
                    answers=eval_answers,
                )
                eval_reward_s += time.perf_counter() - t_eval_reward0

                eval_rewards_global = np.asarray(process_allgather(eval_rewards_np)).reshape(-1)
                eval_rewards_all.append(eval_rewards_global)

                eval_per_func_global = np.asarray(process_allgather(eval_rewards_per_func))
                # shape [process_count, num_funcs, B_local] -> [num_funcs, process_count * B_local]
                eval_per_func_flat = eval_per_func_global.transpose(1, 0, 2).reshape(len(reward_funcs), -1)
                eval_rewards_per_func_all.append(eval_per_func_flat)

            eval_step_s = time.perf_counter() - eval_step0
            eval_rollout_s = eval_rollout_sync_s + eval_rollout_generate_s + eval_rollout_flush_s
            eval_rewards_concat = np.concatenate(eval_rewards_all, axis=0) if eval_rewards_all else np.asarray([], dtype=np.float32)
            eval_reward_stats = _stats_1d(eval_rewards_concat)

            eval_logs["eval/reward_total_mean"] = float(eval_reward_stats["mean"])
            eval_logs["eval/reward_total_std"] = float(eval_reward_stats["std"])
            eval_logs["eval/reward_total_min"] = float(eval_reward_stats["min"])
            eval_logs["eval/reward_total_max"] = float(eval_reward_stats["max"])

            if eval_rewards_per_func_all:
                eval_per_func_concat = np.concatenate(eval_rewards_per_func_all, axis=1)
                eval_per_func_means = eval_per_func_concat.mean(axis=1)
                for name, mean_value in zip(reward_func_names, eval_per_func_means):
                    eval_logs[f"eval/{name}_mean"] = float(mean_value)

            eval_logs["time/eval/rollout_s"] = float(eval_rollout_s)
            eval_logs["time/eval/rollout_sync_s"] = float(eval_rollout_sync_s)
            eval_logs["time/eval/rollout_generate_s"] = float(eval_rollout_generate_s)
            eval_logs["time/eval/rollout_flush_s"] = float(eval_rollout_flush_s)
            eval_logs["time/eval/reward_s"] = float(eval_reward_s)
            eval_logs["time/eval/step_s"] = float(eval_step_s)

            if wandb is not None and jax.process_index() == 0:
                wandb.log(eval_logs, step=step)

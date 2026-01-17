import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from jax.experimental.multihost_utils import process_allgather
from transformers import PreTrainedTokenizerBase

from MLLM_JAX.utils import _form_global_array, get_jax_mesh2
from prompts.prompts import system_prompt
from training2 import (
    get_advantages,
    get_state,
    reward_correct,
    reward_format,
    tag_count_reward,
    training_step,
)


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _apply_chat_template(tokenizer: PreTrainedTokenizerBase, user_text: str) -> str:
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)


def _load_gsm8k_qas(dataset_name: str, split: str) -> list[dict[str, str]]:
    dataset = load_dataset(dataset_name, "main", split=split)
    return [
        {"Q": q, "A": a.split("####")[-1].strip()}
        for q, a in zip(dataset["question"], dataset["answer"])
    ]


def _calculate_rewards(items: list[dict[str, str]], answers: list[str]) -> np.ndarray:
    rewards = np.zeros((len(answers),), dtype=np.float32)
    for i, (item, answer) in enumerate(zip(items, answers)):
        rewards[i] += float(reward_correct(item, answer)) * 1.0
        rewards[i] += float(reward_format(item, answer)) * 0.5
        rewards[i] += float(tag_count_reward(item, answer)) * 0.5
    return rewards


@dataclass(frozen=True)
class SmokeConfig:
    model_path: str
    steps: int
    batch_size: int
    num_pre_q: int
    max_length_sample: int
    max_length_total: int
    ppo_epochs: int
    grad_accum_steps: int
    beta: float
    mesh_shape: str


def _make_config() -> SmokeConfig:
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    steps = _get_env_int("STEPS", 3)
    batch_size = _get_env_int("BATCH_SIZE", 1)
    num_pre_q = _get_env_int("NUM_PRE_Q", 8)
    max_length_sample = _get_env_int("MAX_LENGTH_SAMPLE", 64)
    max_length_total = _get_env_int("MAX_LENGTH_TOTAL", max_length_sample + 128)
    ppo_epochs = _get_env_int("PPO_EPOCHS", 1)
    grad_accum_steps = _get_env_int("GRAD_ACCUM_STEPS", 1)
    beta = _get_env_float("BETA", 0.0)
    mesh_shape = os.environ.get("MESH_SHAPE_FSDP", "1,-1,1")
    return SmokeConfig(
        model_path=model_path,
        steps=steps,
        batch_size=batch_size,
        num_pre_q=num_pre_q,
        max_length_sample=max_length_sample,
        max_length_total=max_length_total,
        ppo_epochs=ppo_epochs,
        grad_accum_steps=grad_accum_steps,
        beta=beta,
        mesh_shape=mesh_shape,
    )


def _ensure_batch_multiple_of_local_devices(local_batch: int, local_device_count: int) -> int:
    if local_batch % local_device_count == 0:
        return local_batch
    return ((local_batch + local_device_count - 1) // local_device_count) * local_device_count


def _run_generation(
    prompts: list[str],
    sampler: Any,
    tokenizer: PreTrainedTokenizerBase,
    params: Any,
    max_length_sample: int,
) -> tuple[list[str], dict[str, np.ndarray]]:
    inputs = tokenizer(prompts, return_tensors="jax", padding=True, padding_side="right")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1, position_ids)

    global_prompt_len = int(process_allgather(input_ids.shape[1]).max())
    prefill_length = int(sampler.find_ceil(global_prompt_len))
    pad_width = max(0, prefill_length - int(input_ids.shape[1]))
    if pad_width > 0:
        input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=tokenizer.eos_token_id)
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_width)), constant_values=0)
        position_ids = jnp.pad(position_ids, ((0, 0), (0, pad_width)), constant_values=1)

    true_length_prompts = np.asarray(inputs["attention_mask"].sum(axis=1))
    outputs = sampler.generate(
        input_ids_pad=input_ids,
        pad_attention=attention_mask,
        position_ids=position_ids,
        prefill_length=prefill_length,
        max_length=max_length_sample,
        params=params,
    )

    buffer_len = int(outputs["local_token_buffer"].shape[1])
    train_input_ids = np.full((len(prompts), buffer_len), fill_value=tokenizer.pad_token_id, dtype=np.int32)
    train_attention_mask = np.zeros_like(train_input_ids, dtype=np.int32)
    train_completions_mask = np.zeros_like(train_input_ids, dtype=np.int32)

    generated_answers: list[str] = []
    for i, (true_len, gen_step) in enumerate(zip(true_length_prompts, outputs["local_sample_step"])):
        start_idx = prefill_length
        end_idx = min(buffer_len, start_idx + int(gen_step) + 1)
        generated_tokens = outputs["local_token_buffer"][i, start_idx:end_idx]
        generated_answers.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

        true_len_int = int(true_len)
        prompt_tokens = np.asarray(inputs["input_ids"][i, :true_len_int])
        train_input_ids[i, :true_len_int] = prompt_tokens
        train_attention_mask[i, :true_len_int] = 1

        gen_len = end_idx - start_idx
        prompt_end_idx = min(buffer_len, true_len_int + gen_len)
        actual_gen_len = prompt_end_idx - true_len_int
        if actual_gen_len > 0:
            train_input_ids[i, true_len_int:prompt_end_idx] = np.asarray(generated_tokens[:actual_gen_len])
            train_attention_mask[i, true_len_int:prompt_end_idx] = 1
            train_completions_mask[i, true_len_int:prompt_end_idx] = 1

    return generated_answers, {
        "input_ids": train_input_ids,
        "attention_mask": train_attention_mask,
        "labels": train_completions_mask,
    }


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        jax.distributed.initialize()
    except Exception:
        pass

    cfg = _make_config()
    mesh = get_jax_mesh2(cfg.mesh_shape)
    local_device_count = len(mesh.local_devices)

    local_batch = cfg.batch_size * cfg.num_pre_q
    padded_local_batch = _ensure_batch_multiple_of_local_devices(local_batch, local_device_count)
    if padded_local_batch != local_batch:
        if padded_local_batch % cfg.batch_size != 0:
            raise ValueError(
                f"Cannot pad local batch {local_batch} -> {padded_local_batch} "
                f"because BATCH_SIZE={cfg.batch_size} does not divide it. "
                "Adjust BATCH_SIZE or NUM_PRE_Q."
            )
        new_num_pre_q = padded_local_batch // cfg.batch_size
        print(
            f"Padding local batch {local_batch} -> {padded_local_batch} by changing "
            f"NUM_PRE_Q {cfg.num_pre_q} -> {new_num_pre_q}."
        )
        cfg = SmokeConfig(**{**cfg.__dict__, "num_pre_q": new_num_pre_q})
        local_batch = padded_local_batch

    print(f"backend={jax.default_backend()} process_count={jax.process_count()} process_index={jax.process_index()}")
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
                max_length_sample=cfg.max_length_sample,
                max_length_total=cfg.max_length_total,
                ppo_epochs=cfg.ppo_epochs,
                grad_accum_steps=cfg.grad_accum_steps,
                beta=cfg.beta,
                mesh_shape=cfg.mesh_shape,
            )
        )
    )

    qas = _load_gsm8k_qas("openai/gsm8k", "train")
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

    rng = random.Random(0xC0FFEE + jax.process_index())
    for step in range(cfg.steps):
        batch_items = [rng.choice(qas) for _ in range(cfg.batch_size)]
        prompts_base = [_apply_chat_template(sampler.tokenizer, item["Q"]) for item in batch_items]
        repeated_prompts = [p for p in prompts_base for _ in range(cfg.num_pre_q)]
        repeated_items = [item for item in batch_items for _ in range(cfg.num_pre_q)]

        t0 = time.time()
        answers, datas_np = _run_generation(
            repeated_prompts,
            sampler=sampler,
            tokenizer=sampler.tokenizer,
            params=state.params,
            max_length_sample=cfg.max_length_sample,
        )

        rewards_np = _calculate_rewards(repeated_items, answers)
        rewards_global = np.asarray(process_allgather(rewards_np))
        mean_global = float(rewards_global.mean())
        std_global = float(max(rewards_global.std(), 1e-6))

        advantages = get_advantages(
            rewards=jnp.asarray(rewards_np),
            groups=cfg.num_pre_q,
            advantage_estimator="grpo",
            mean_global=mean_global,
            std_global=std_global,
        )
        datas_np["advantages"] = np.asarray(advantages, dtype=np.float32)

        datas_np["rewards"] = rewards_np
        datas = jax.tree_util.tree_map_with_path(
            lambda path, x: _form_global_array(path, x, global_mesh=mesh),
            datas_np,
        )

        total_valid_token_count = datas["labels"][:, 1:].sum()
        datas = {**datas, "total_valid_token_count": total_valid_token_count}

        metrics = {}
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


if __name__ == "__main__":
    main()

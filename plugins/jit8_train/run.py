from __future__ import annotations

import argparse
from typing import Any

import yaml

from plugins.jit8_train.config import DEFAULT_CONFIG, load_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO/GSM8K training (jit8).")
    parser.add_argument(
        "--config",
        default="plugins/jit8_train/configs/gsm8k_default.yaml",
        help="YAML config path (optional).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config entries (repeatable), e.g. --set training_steps=10",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the merged config and exit.",
    )
    return parser.parse_args(argv)


def _get_dtype(cfg: dict[str, Any]):
    import jax.numpy as jnp

    name = str(cfg.get("params_dtype") or "").strip().lower()
    if name in {"", "none", "null"}:
        return None
    return {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float16": jnp.float16,
    }[name]


def _validate_cfg(cfg: dict[str, Any]) -> None:
    weights = cfg.get("reward_funcs_weights")
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("cfg.reward_funcs_weights must be a list of 3 floats")

    try:
        _ = int(cfg["training_steps"])
        _ = int(cfg["batch_size"])
        _ = int(cfg["grad_accum_steps"])
        _ = int(cfg["num_pre_q"])
        _ = int(cfg["ppo_steps"])
        _ = int(cfg["max_length_sample"])
        _ = int(cfg["max_length_extra"])
        _ = int(cfg["global_length"])
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e.args[0]}") from e


def run_training(cfg: dict[str, Any]) -> None:
    import random
    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    from datasets import load_dataset
    from jax.experimental.multihost_utils import process_allgather

    from MLLM_JAX.utils import (
        _form_global_array,
        get_jax_mesh2,
        get_partition_rules_llama,
        match_partition_rules,
    )
    from prompts.prompts import system_prompt
    from training2 import (
        get_advantages,
        get_state,
        repeat,
        reward_correct,
        reward_format,
        slice_data,
        tag_count_reward,
        training_step,
    )

    _validate_cfg(cfg)

    cache_dir = cfg.get("jax_compilation_cache_dir")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))

    jax.distributed.initialize()

    reward_funcs = [reward_correct, reward_format, tag_count_reward]
    reward_funcs_weights = cfg["reward_funcs_weights"]

    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config"], split=cfg["dataset_split"])
    dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())
    qas = [{"Q": q, "A": a.split("####")[-1].strip()} for q, a in zip(dataset["question"], dataset["answer"])]

    mesh_dp = get_jax_mesh2(cfg["mesh_dp"])
    mesh_fsdp = get_jax_mesh2(cfg["mesh_fsdp"])

    max_length_sample = int(cfg["max_length_sample"])
    max_length_total = max_length_sample + int(cfg["max_length_extra"])

    state, sampler, _train_state_sharding = get_state(
        mesh_fsdp,
        int(cfg["training_steps"]),
        model_path=cfg["model_path"],
        grad_accum_steps=int(cfg["grad_accum_steps"]),
        num_pre_q=int(cfg["num_pre_q"]),
        max_lengths=max_length_total,
    )

    params_shapes = jax.eval_shape(lambda x: x, state.params)
    params_partition = match_partition_rules(get_partition_rules_llama(), params_shapes)
    params_sharding_dp = jax.tree_util.tree_map(
        lambda spec: jax.sharding.NamedSharding(mesh_dp, spec), params_partition
    )
    params_to_dp = jax.jit(lambda x: x, out_shardings=params_sharding_dp)

    dtype = _get_dtype(cfg)

    def to_dp_params(params):
        if dtype is None:
            return params_to_dp(params)
        return params_to_dp(jax.tree_util.tree_map(lambda x: x.astype(dtype), params))

    get_advantages_jit = jax.jit(get_advantages, static_argnums=(1,))
    train_step = jax.jit(training_step, donate_argnums=(0,))

    if cfg.get("wandb_enabled") and jax.process_index() == 0:
        import wandb

        wandb.init(name=cfg.get("wandb_name"), project=cfg.get("wandb_project"), config=cfg)
    else:
        wandb = None

    def gen_answers_jax(prompts: list[str], dp_params):
        tokenizer = sampler.tokenizer
        chat_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for x in prompts
        ]

        inputs = tokenizer(chat_prompts, return_tensors="jax", padding=True, padding_side="right")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids = jnp.where(attention_mask == 0, 1, position_ids)

        prefill_length = sampler.find_ceil(int(cfg["global_length"]))
        true_length_prompts = attention_mask.sum(axis=1)

        input_ids_pad = jnp.pad(
            input_ids,
            ((0, 0), (0, prefill_length - input_ids.shape[1])),
            constant_values=tokenizer.eos_token_id,
        )
        pad_attention = jnp.pad(attention_mask, ((0, 0), (0, prefill_length - input_ids.shape[1])))
        pad_position_ids = jnp.pad(position_ids, ((0, 0), (0, prefill_length - input_ids.shape[1])))

        outputs = sampler.generate(
            input_ids_pad,
            pad_attention,
            pad_position_ids,
            prefill_length,
            max_length=max_length_sample,
            params=dp_params,
        )

        train_input_ids = np.full_like(outputs["local_token_buffer"], fill_value=tokenizer.pad_token_id)
        train_attention_mask = np.full_like(outputs["local_attention_mask"], fill_value=0)
        train_completions_mask = np.full_like(outputs["local_attention_mask"], fill_value=0)
        answers: list[str] = []

        for i, (true_length_prompt, step) in enumerate(
            zip(true_length_prompts, outputs["local_sample_step"])
        ):
            decoded = tokenizer.batch_decode(
                outputs["local_token_buffer"][i, prefill_length : prefill_length + step + 1].reshape(1, -1),
                skip_special_tokens=True,
            )
            answers.extend(decoded)

            train_input_ids[i, :true_length_prompt] = outputs["local_token_buffer"][i, :true_length_prompt]
            train_input_ids[i, true_length_prompt : true_length_prompt + step + 1] = outputs["local_token_buffer"][
                i, prefill_length : prefill_length + step + 1
            ]

            train_attention_mask[i, :true_length_prompt] = outputs["local_attention_mask"][i, :true_length_prompt]
            train_attention_mask[
                i, true_length_prompt : true_length_prompt + step + 1
            ] = outputs["local_attention_mask"][i, prefill_length : prefill_length + step + 1]

            train_completions_mask[i, true_length_prompt : true_length_prompt + step + 1] = outputs[
                "local_attention_mask"
            ][i, prefill_length : prefill_length + step + 1]

        print(answers[-2:])
        print("\n\n", flush=True)
        return (
            chat_prompts,
            answers,
            {
                "input_ids": train_input_ids,
                "attention_mask": train_attention_mask,
                "labels": train_completions_mask,
            },
        )

    ema_decay = float(cfg.get("ema_decay", 0.9))
    mean_correct_length = float(max_length_sample)

    for step in range(int(cfg["training_steps"])):
        dp_params = to_dp_params(state.params)
        inputs = random.sample(qas, int(cfg["batch_size"]))
        repeated_inputs = repeat(inputs, int(cfg["num_pre_q"]))
        prompts = [x["Q"] for x in repeated_inputs]

        _chat_prompts, answers, datas = gen_answers_jax(prompts, dp_params)

        rewards_per_func = np.zeros((len(reward_funcs), len(answers)))
        for i, (reward_weight, reward_func) in enumerate(zip(reward_funcs_weights, reward_funcs)):
            for j, (inp, a) in enumerate(zip(repeated_inputs, answers)):
                try:
                    rewards_per_func[i, j] = reward_weight * reward_func(inp, a)
                except Exception as e:
                    print(e)
                    rewards_per_func[i, j] = -1

        rewards = rewards_per_func.sum(axis=0)
        reward_corrects = rewards_per_func[0, :]
        datas["rewards"] = rewards

        reward_corrects_global = process_allgather(reward_corrects)
        completion_ids_global = process_allgather(datas["labels"])
        correct_mask = reward_corrects_global == 1.0
        completion_ids_global_correct = completion_ids_global[correct_mask]
        completion_ids_global_incorrect = completion_ids_global[~correct_mask]

        correct_lengths = completion_ids_global_correct.sum(axis=1)
        incorrect_lengths = completion_ids_global_incorrect.sum(axis=1)
        if correct_lengths.size:
            mean_correct_length = ema_decay * mean_correct_length + (1 - ema_decay) * correct_lengths.max()

        metrics: dict[str, Any] = {}
        if jax.process_index() == 0:
            metrics["completion_ids_correct_mean"] = correct_lengths.mean() if correct_lengths.size else float("nan")
            metrics["completion_ids_correct_max"] = correct_lengths.max() if correct_lengths.size else float("nan")
            metrics["completion_ids_global_incorrect_mean"] = (
                incorrect_lengths.mean() if incorrect_lengths.size else float("nan")
            )
            metrics["completion_ids_global_incorrect_max"] = (
                incorrect_lengths.max() if incorrect_lengths.size else float("nan")
            )
            metrics["mean_correct_length_ema"] = mean_correct_length

        mean_global = process_allgather(datas["rewards"]).mean()
        std_global = process_allgather(datas["rewards"]).std()
        print(f"{step=}", datas["rewards"], np.mean(datas["rewards"]), mean_global, answers[-2:])

        metrics["mean_global"] = mean_global
        metrics["std_global"] = std_global

        advantages = get_advantages_jit(datas["rewards"], int(cfg["num_pre_q"]), mean_global=mean_global, std_global=std_global)
        datas["advantages"] = advantages

        metrics["advantages_max"] = float(advantages.max())
        metrics["advantages_min"] = float(advantages.min())

        rewards_per_func_jnp = jnp.array(rewards_per_func)
        for i, reward_func in enumerate(reward_funcs):
            reward_name = reward_func.__name__
            reward_local = rewards_per_func_jnp[i]
            metrics[reward_name] = float(process_allgather(reward_local).mean())

        datas = jax.tree_util.tree_map_with_path(partial(_form_global_array, global_mesh=mesh_dp), datas)
        total_valid_token_count = datas["labels"][:, 1:].sum()

        per_token_logps = []
        for ppo_step in range(int(cfg["ppo_steps"])):
            for j in range(int(cfg["grad_accum_steps"])):
                local_data = jax.tree_util.tree_map(lambda x: slice_data(x, int(cfg["grad_accum_steps"]), j), datas)
                local_data["total_valid_token_count"] = total_valid_token_count
                state, meta_data = train_step(state, local_data)
                if ppo_step == 0:
                    per_token_logps.append(meta_data["per_token_logps"])

            if ppo_step == 0:
                datas["old_per_token_logps"] = jnp.concat(per_token_logps)
                metrics["entropy"] = float(meta_data["entropy"])

        if wandb is not None and jax.process_index() == 0:
            wandb.log(metrics, step=step)


def cli_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg = {**DEFAULT_CONFIG, **cfg}
    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    run_training(cfg)


__all__ = ["cli_main", "run_training"]

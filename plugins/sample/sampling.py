from __future__ import annotations

from typing import Any

import numpy as np


def build_chat_prompts(tokenizer: Any, prompts: list[str], system_prompt: str) -> list[str]:
    return [
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


def generate_answers_and_training_batch(
    *,
    prompts: list[str],
    sampler: Any,
    params: Any,
    system_prompt: str,
    global_length: int,
    max_length_sample: int,
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    import jax.numpy as jnp

    tokenizer = sampler.tokenizer
    chat_prompts = build_chat_prompts(tokenizer, prompts, system_prompt)

    inputs = tokenizer(chat_prompts, return_tensors="np", padding=True, padding_side="right")
    input_ids = jnp.asarray(inputs["input_ids"], dtype=jnp.int32)
    attention_mask = jnp.asarray(inputs["attention_mask"], dtype=jnp.int32)
    position_ids = attention_mask.cumsum(-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1, position_ids)

    desired_length = max(int(global_length), int(input_ids.shape[1]))
    prefill_length = sampler.find_ceil(desired_length)
    if prefill_length is None:
        raise ValueError(f"No prefill bucket found for desired_length={desired_length}")

    pad_width = prefill_length - int(input_ids.shape[1])
    if pad_width < 0:
        raise ValueError(
            f"Prompt length {int(input_ids.shape[1])} exceeds prefill_length={prefill_length}; "
            f"increase global_length or adjust sampler buckets"
        )

    input_ids_pad = jnp.pad(
        input_ids,
        ((0, 0), (0, pad_width)),
        constant_values=tokenizer.eos_token_id,
    )
    pad_attention = jnp.pad(attention_mask, ((0, 0), (0, pad_width)))
    pad_position_ids = jnp.pad(position_ids, ((0, 0), (0, pad_width)))

    outputs = sampler.generate(
        input_ids_pad,
        pad_attention,
        pad_position_ids,
        prefill_length,
        max_length=max_length_sample,
        params=params,
    )

    train_input_ids = np.full_like(outputs["local_token_buffer"], fill_value=tokenizer.pad_token_id)
    train_attention_mask = np.full_like(outputs["local_attention_mask"], fill_value=0)
    train_completions_mask = np.full_like(outputs["local_attention_mask"], fill_value=0)
    answers: list[str] = []

    true_length_prompts = np.asarray(attention_mask.sum(axis=1))

    for i, (true_length_prompt, step) in enumerate(zip(true_length_prompts, outputs["local_sample_step"])):
        true_length_prompt_i = int(true_length_prompt)
        step_i = int(step)

        decoded = tokenizer.batch_decode(
            outputs["local_token_buffer"][i, prefill_length : prefill_length + step_i + 1].reshape(1, -1),
            skip_special_tokens=True,
        )
        answers.extend(decoded)

        train_input_ids[i, :true_length_prompt_i] = outputs["local_token_buffer"][i, :true_length_prompt_i]
        train_input_ids[
            i, true_length_prompt_i : true_length_prompt_i + step_i + 1
        ] = outputs["local_token_buffer"][i, prefill_length : prefill_length + step_i + 1]

        train_attention_mask[i, :true_length_prompt_i] = outputs["local_attention_mask"][i, :true_length_prompt_i]
        train_attention_mask[
            i, true_length_prompt_i : true_length_prompt_i + step_i + 1
        ] = outputs["local_attention_mask"][i, prefill_length : prefill_length + step_i + 1]

        train_completions_mask[
            i, true_length_prompt_i : true_length_prompt_i + step_i + 1
        ] = outputs["local_attention_mask"][i, prefill_length : prefill_length + step_i + 1]

    return (
        chat_prompts,
        answers,
        {
            "input_ids": train_input_ids,
            "attention_mask": train_attention_mask,
            "labels": train_completions_mask,
        },
    )


__all__ = ["build_chat_prompts", "generate_answers_and_training_batch"]

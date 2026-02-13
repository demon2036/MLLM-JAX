from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from plugins.api.training import RolloutResult
from plugins.sample.workflows.grpo_sync import generate_answers_and_training_batch


def _create_greedy_sampler_from_sampler(sampler: Any) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as PS

    from plugins.sample.backends.mllm_jax_sampler import Sampler as SamplerImpl

    mesh = getattr(sampler, "mesh", None)
    if mesh is None:
        raise ValueError("ReMax mixed rollout requires a sampler with `mesh` (NamedSharding mesh).")

    greedy = SamplerImpl(sampler.model, sampler.tokenizer, mesh=mesh)

    def _greedy_sample(rng, logits):
        rngs = jax.random.split(rng, jax.device_count())

        def sample_inner(_rng, logits_local):
            del _rng
            return jnp.argmax(logits_local, axis=-1)

        sample_fn = shard_map(
            sample_inner,
            mesh=mesh,
            in_specs=(PS(["dp", "fsdp"]), PS(["dp", "fsdp"], "tp")),
            out_specs=PS(["dp", "fsdp"]),
            check_rep=False,
        )
        return sample_fn(rngs, logits)

    greedy.sample_fn = jax.jit(_greedy_sample)
    greedy.jit_infer_step = jax.jit(greedy.infer, donate_argnums=(0,))
    return greedy


@dataclass
class ReMaxMixedNaiveRolloutBackend:
    """Naive sampler backend that injects a greedy baseline completion per prompt group.

    Contract
    --------
    - Callers provide prompts that are already repeated into prompt-groups of size `group_size`.
    - For each group, the completion at `baseline_position` is generated with greedy decoding.
    - All other completions are generated with the provided (stochastic) sampler.
    - Baseline rows have `labels==0` so they do not contribute gradients.
    """

    sampler: Any
    group_size: int
    baseline_position: int = 0

    def __post_init__(self) -> None:
        if int(self.group_size) <= 1:
            raise ValueError("ReMax mixed rollout requires group_size >= 2 (1 baseline + >=1 sampled).")
        baseline_pos = int(self.baseline_position)
        if baseline_pos < 0 or baseline_pos >= int(self.group_size):
            raise ValueError("baseline_position must be in [0, group_size).")
        self.greedy_sampler = _create_greedy_sampler_from_sampler(self.sampler)

    def rollout(
        self,
        *,
        prompts: Sequence[str],
        params: Any,
        system_prompt: str,
        global_length: int,
        max_length_sample: int,
    ) -> RolloutResult:
        prompts_list = list(prompts)
        batch_size = int(len(prompts_list))
        group_size = int(self.group_size)
        if batch_size % group_size != 0:
            raise ValueError(f"Expected prompts length divisible by group_size={group_size}, got {batch_size}.")

        num_groups = batch_size // group_size
        baseline_pos = int(self.baseline_position)

        baseline_prompts: list[str] = []
        sample_prompts: list[str] = []
        baseline_indices: list[int] = []
        sample_indices: list[int] = []

        for group_idx in range(num_groups):
            for j in range(group_size):
                global_idx = group_idx * group_size + j
                if j == baseline_pos:
                    baseline_prompts.append(prompts_list[global_idx])
                    baseline_indices.append(global_idx)
                else:
                    sample_prompts.append(prompts_list[global_idx])
                    sample_indices.append(global_idx)

        baseline_chat_prompts, baseline_answers, baseline_batch = generate_answers_and_training_batch(
            prompts=baseline_prompts,
            sampler=self.greedy_sampler,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )
        sample_chat_prompts, sample_answers, sample_batch = generate_answers_and_training_batch(
            prompts=sample_prompts,
            sampler=self.sampler,
            params=params,
            system_prompt=system_prompt,
            global_length=int(global_length),
            max_length_sample=int(max_length_sample),
        )

        output_chat_prompts: list[str] = [""] * batch_size
        output_answers: list[str] = [""] * batch_size

        b_ptr = 0
        for idx in baseline_indices:
            output_chat_prompts[int(idx)] = baseline_chat_prompts[b_ptr]
            output_answers[int(idx)] = baseline_answers[b_ptr]
            b_ptr += 1

        s_ptr = 0
        for idx in sample_indices:
            output_chat_prompts[int(idx)] = sample_chat_prompts[s_ptr]
            output_answers[int(idx)] = sample_answers[s_ptr]
            s_ptr += 1

        batch_out: dict[str, np.ndarray] = {}
        for key in ("input_ids", "attention_mask", "labels"):
            if key not in baseline_batch or key not in sample_batch:
                raise ValueError(f"Expected rollout batches to include {key!r}.")
            baseline_arr = np.asarray(baseline_batch[key])
            sample_arr = np.asarray(sample_batch[key])
            if baseline_arr.ndim != 2 or sample_arr.ndim != 2:
                raise ValueError(f"Expected {key} arrays to be rank-2, got {baseline_arr.shape} and {sample_arr.shape}.")
            if baseline_arr.shape[1] != sample_arr.shape[1]:
                raise ValueError(
                    f"Mismatched sequence length for {key}: {baseline_arr.shape[1]} vs {sample_arr.shape[1]}. "
                    "Ensure greedy and sampled rollouts share the same max_length_sample/global_length."
                )
            batch_out[key] = np.empty((batch_size, baseline_arr.shape[1]), dtype=baseline_arr.dtype)

        b_ptr = 0
        s_ptr = 0
        for group_idx in range(num_groups):
            for j in range(group_size):
                global_idx = group_idx * group_size + j
                if j == baseline_pos:
                    for key in ("input_ids", "attention_mask", "labels"):
                        batch_out[key][global_idx] = baseline_batch[key][b_ptr]
                    # Baseline rows are used only for rewards/baselines (no gradients).
                    batch_out["labels"][global_idx] = 0
                    b_ptr += 1
                else:
                    for key in ("input_ids", "attention_mask", "labels"):
                        batch_out[key][global_idx] = sample_batch[key][s_ptr]
                    s_ptr += 1

        batch_out["is_baseline"] = np.zeros((batch_size,), dtype=np.int32)
        batch_out["is_baseline"][np.asarray(baseline_indices, dtype=np.int32)] = 1

        return RolloutResult(chat_prompts=output_chat_prompts, answers=output_answers, batch=batch_out)


__all__ = ["ReMaxMixedNaiveRolloutBackend"]

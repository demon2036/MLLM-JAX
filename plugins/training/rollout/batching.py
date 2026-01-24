from __future__ import annotations

import math


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("b must be > 0")
    return (int(a) + int(b) - 1) // int(b)


def infer_rollout_passes(
    *,
    global_batch_size: int | None,
    batch_size_per_process: int,
    process_count: int,
) -> tuple[int, int]:
    """Infer rollout passes from a global prompt target and a per-process prompt batch.

    Returns:
      (passes, effective_global_batch_size)

    Notes:
      - `global_batch_size` is interpreted as global prompts per *training step*.
      - `batch_size_per_process` is prompts per rollout pass on each process.
      - We round up to full passes to keep shapes equal across processes.
    """
    if process_count <= 0:
        raise ValueError("process_count must be > 0")
    if batch_size_per_process <= 0:
        raise ValueError("batch_size_per_process must be > 0")

    per_pass_global = int(batch_size_per_process) * int(process_count)
    if global_batch_size is None:
        return 1, per_pass_global
    if int(global_batch_size) <= 0:
        raise ValueError("global_batch_size must be > 0")

    passes = ceil_div(int(global_batch_size), per_pass_global)
    return passes, passes * per_pass_global


def round_up_passes_for_divisibility(
    *,
    passes: int,
    sequences_per_pass_per_process: int,
    micro_batch_size_per_process: int,
) -> int:
    """Round `passes` up so that total sequences is divisible by micro-batch size.

    Constraint:
      total_sequences_per_process = passes * sequences_per_pass_per_process
      total_sequences_per_process % micro_batch_size_per_process == 0

    This keeps pjit-style shapes consistent across processes and makes it possible
    to slice the global batch into equal micro-batches.
    """
    if passes <= 0:
        raise ValueError("passes must be > 0")
    if sequences_per_pass_per_process <= 0:
        raise ValueError("sequences_per_pass_per_process must be > 0")
    if micro_batch_size_per_process <= 0:
        raise ValueError("micro_batch_size_per_process must be > 0")

    g = math.gcd(int(sequences_per_pass_per_process), int(micro_batch_size_per_process))
    required_multiple = int(micro_batch_size_per_process) // int(g)
    return ceil_div(int(passes), required_multiple) * required_multiple


def required_prompts_multiple(*, n: int, local_device_count: int) -> int:
    """Return the smallest `m` such that (m * n) % local_device_count == 0."""

    if int(n) <= 0:
        raise ValueError("n must be > 0")
    if int(local_device_count) <= 0:
        raise ValueError("local_device_count must be > 0")
    return int(local_device_count) // math.gcd(int(n), int(local_device_count))


def resolve_rollout_prompt_batching(
    *,
    requested_global_prompts_per_step: int,
    process_count: int,
    n: int,
    local_device_count: int,
    max_prompts_per_pass_per_process: int | None,
) -> tuple[int, int, int]:
    """Resolve per-pass prompt batch + pass count for a GRPO rollout step.

    Returns:
      (prompts_per_pass_per_process, rollout_passes, effective_global_prompts_per_step)

    Notes:
      - The runner treats `requested_global_prompts_per_step` as the user-facing
        knob (`rollout.batch_size`): global prompts per *training step*.
      - Each pass uses the same `prompts_per_pass_per_process` on every process
        so shapes match across hosts.
      - We enforce `(prompts_per_pass_per_process * n) % local_device_count == 0`
        so the per-pass sequence batch can be evenly sharded across local devices.
      - If `max_prompts_per_pass_per_process` is set, we split into multiple
        passes (and may pad the effective global prompts per step).
    """

    if int(requested_global_prompts_per_step) <= 0:
        raise ValueError("requested_global_prompts_per_step must be > 0")
    if int(process_count) <= 0:
        raise ValueError("process_count must be > 0")

    required_multiple = required_prompts_multiple(n=int(n), local_device_count=int(local_device_count))
    if int(required_multiple) <= 0:  # pragma: no cover
        raise ValueError("required_multiple must be > 0")

    prompts_per_process = ceil_div(int(requested_global_prompts_per_step), int(process_count))
    prompts_per_process = max(1, int(prompts_per_process))

    prompts_per_pass_target = ceil_div(int(prompts_per_process), int(required_multiple)) * int(required_multiple)

    if max_prompts_per_pass_per_process is None:
        prompts_per_pass = int(prompts_per_pass_target)
        passes = 1
        effective_global = int(prompts_per_pass) * int(process_count)
        return prompts_per_pass, passes, effective_global

    max_prompts_per_pass_per_process = int(max_prompts_per_pass_per_process)
    if max_prompts_per_pass_per_process <= 0:
        raise ValueError("max_prompts_per_pass_per_process must be > 0")
    if max_prompts_per_pass_per_process < int(required_multiple):
        raise ValueError(
            "max_prompts_per_pass_per_process must be >= required_multiple "
            f"({int(required_multiple)}) so that (prompts_per_pass_per_process * n) is divisible by "
            f"local_device_count={int(local_device_count)} (n={int(n)})."
        )

    max_multiple = (int(max_prompts_per_pass_per_process) // int(required_multiple)) * int(required_multiple)
    if max_multiple < int(required_multiple):  # pragma: no cover
        raise ValueError("max_prompts_per_pass_per_process too small after applying required_multiple")

    prompts_per_pass = int(min(int(prompts_per_pass_target), int(max_multiple)))
    prompts_per_pass = max(int(required_multiple), int(prompts_per_pass))

    per_pass_global = int(prompts_per_pass) * int(process_count)
    passes = ceil_div(int(requested_global_prompts_per_step), int(per_pass_global))
    passes = max(1, int(passes))
    effective_global = int(passes) * int(per_pass_global)
    return prompts_per_pass, passes, effective_global


__all__ = [
    "ceil_div",
    "infer_rollout_passes",
    "round_up_passes_for_divisibility",
    "required_prompts_multiple",
    "resolve_rollout_prompt_batching",
]

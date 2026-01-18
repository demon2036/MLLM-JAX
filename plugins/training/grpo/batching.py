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


__all__ = [
    "ceil_div",
    "infer_rollout_passes",
    "round_up_passes_for_divisibility",
]


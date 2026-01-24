from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.rollout.batching import ceil_div, infer_rollout_passes, round_up_passes_for_divisibility
from plugins.training.rollout.batching import resolve_rollout_prompt_batching


class TestGrpoBatchingInference(unittest.TestCase):
    def test_ceil_div(self) -> None:
        self.assertEqual(ceil_div(0, 3), 0)
        self.assertEqual(ceil_div(1, 3), 1)
        self.assertEqual(ceil_div(3, 3), 1)
        self.assertEqual(ceil_div(4, 3), 2)

    def test_infer_rollout_passes_default(self) -> None:
        passes, effective = infer_rollout_passes(global_batch_size=None, batch_size_per_process=4, process_count=2)
        self.assertEqual(passes, 1)
        self.assertEqual(effective, 8)

    def test_infer_rollout_passes_global_target(self) -> None:
        # 2 processes, batch_size_per_process=4 -> per-pass global prompts = 8.
        passes, effective = infer_rollout_passes(global_batch_size=128, batch_size_per_process=4, process_count=2)
        self.assertEqual(passes, 16)
        self.assertEqual(effective, 128)

    def test_round_up_passes_for_divisibility(self) -> None:
        # Need passes * 32 divisible by 6 -> passes multiple of 3.
        self.assertEqual(
            round_up_passes_for_divisibility(passes=1, sequences_per_pass_per_process=32, micro_batch_size_per_process=6),
            3,
        )
        self.assertEqual(
            round_up_passes_for_divisibility(passes=3, sequences_per_pass_per_process=32, micro_batch_size_per_process=6),
            3,
        )
        self.assertEqual(
            round_up_passes_for_divisibility(passes=4, sequences_per_pass_per_process=32, micro_batch_size_per_process=6),
            6,
        )

    def test_resolve_rollout_prompt_batching_capped(self) -> None:
        # v6e-8-like: single-host, n=8, local_device_count=8.
        prompts_per_pass, passes, effective = resolve_rollout_prompt_batching(
            requested_global_prompts_per_step=128,
            process_count=1,
            n=8,
            local_device_count=8,
            max_prompts_per_pass_per_process=16,
        )
        self.assertEqual(prompts_per_pass, 16)
        self.assertEqual(passes, 8)
        self.assertEqual(effective, 128)

        # v6e-16-like: multi-host, 4 processes, n=8, local_device_count=4.
        prompts_per_pass, passes, effective = resolve_rollout_prompt_batching(
            requested_global_prompts_per_step=128,
            process_count=4,
            n=8,
            local_device_count=4,
            max_prompts_per_pass_per_process=16,
        )
        self.assertEqual(prompts_per_pass, 16)
        self.assertEqual(passes, 2)
        self.assertEqual(effective, 128)

    def test_resolve_rollout_prompt_batching_required_multiple_padding(self) -> None:
        # When (prompts_per_pass * n) must be divisible by local_device_count,
        # prompts_per_pass may need to round up and/or split into multiple passes.
        prompts_per_pass, passes, effective = resolve_rollout_prompt_batching(
            requested_global_prompts_per_step=10,
            process_count=1,
            n=3,
            local_device_count=8,
            max_prompts_per_pass_per_process=None,
        )
        self.assertEqual(prompts_per_pass, 16)
        self.assertEqual(passes, 1)
        self.assertEqual(effective, 16)

        prompts_per_pass, passes, effective = resolve_rollout_prompt_batching(
            requested_global_prompts_per_step=10,
            process_count=1,
            n=3,
            local_device_count=8,
            max_prompts_per_pass_per_process=10,
        )
        self.assertEqual(prompts_per_pass, 8)
        self.assertEqual(passes, 2)
        self.assertEqual(effective, 16)


if __name__ == "__main__":
    unittest.main()

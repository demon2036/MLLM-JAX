from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.grpo.batching import ceil_div, infer_rollout_passes, round_up_passes_for_divisibility


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


if __name__ == "__main__":
    unittest.main()

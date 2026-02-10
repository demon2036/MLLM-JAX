from __future__ import annotations

import unittest

import numpy as np

from plugins.training.rl.rollout.dynamic_sampling import (
    build_dynamic_sampling_summary,
    homogeneous_group_flags,
    select_completion_indices,
)


class TestRolloutDynamicSampling(unittest.TestCase):
    def test_homogeneous_flags_acc_all_equal(self) -> None:
        rewards = np.asarray([1, 1, 1, 0, 1, 0], dtype=np.float32)
        acc = np.asarray([1, 1, 1, 0, 1, 0], dtype=np.float32)
        flags = homogeneous_group_flags(
            rewards=rewards,
            metric_values=acc,
            n=3,
            metric="acc",
            homogeneity_threshold=1.0,
            min_unique_reward_values=2,
        )
        # group0 all 1 -> homogeneous, group1 has 0/1 mix -> diverse
        np.testing.assert_array_equal(flags, np.asarray([True, False]))

    def test_homogeneous_flags_seq_reward_threshold(self) -> None:
        rewards = np.asarray([0.1, 0.1, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
        flags = homogeneous_group_flags(
            rewards=rewards,
            metric_values=None,
            n=3,
            metric="seq_reward",
            homogeneity_threshold=0.95,
            min_unique_reward_values=2,
        )
        np.testing.assert_array_equal(flags, np.asarray([True, False]))

    def test_select_completion_indices(self) -> None:
        keep_groups = np.asarray([True, False, True], dtype=bool)
        out = select_completion_indices(keep_groups_mask=~keep_groups, n=2)
        np.testing.assert_array_equal(out, np.asarray([2, 3], dtype=np.int32))

    def test_summary_budget(self) -> None:
        summary = build_dynamic_sampling_summary(
            initial_homogeneous=np.asarray([True, True, False]),
            final_homogeneous=np.asarray([True, False, False]),
            extra_rounds=10,
            max_extra_roll_rounds=10,
            dropped_groups=1,
        )
        self.assertEqual(summary.total_groups, 3)
        self.assertEqual(summary.initially_homogeneous_groups, 2)
        self.assertEqual(summary.remaining_homogeneous_groups, 1)
        self.assertTrue(summary.hit_budget)
        self.assertEqual(summary.dropped_groups, 1)


if __name__ == "__main__":
    unittest.main()


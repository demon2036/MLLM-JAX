from __future__ import annotations

import unittest

import numpy as np

from plugins.training.advantage.estimators import (
    compute_dapo_advantages_by_group_id,
    compute_global_normalized_advantages,
    compute_reinforce_plus_plus_advantages_by_group_id,
    compute_rloo_advantages_by_group_id,
)
from plugins.training.advantage.grpo import compute_grpo_advantages_by_group_id


class TestAdvantageEstimators(unittest.TestCase):
    def test_global_normalized_advantages_matches_numpy(self) -> None:
        rewards = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        eps = 1e-4
        out = compute_global_normalized_advantages(rewards=rewards, eps=eps)
        expected = (rewards - rewards.mean()) / (rewards.std() + eps)
        np.testing.assert_allclose(out, expected.astype(np.float32), rtol=0, atol=1e-6)

    def test_global_normalized_advantages_clip(self) -> None:
        rewards = np.asarray([-100.0, 0.0, 100.0], dtype=np.float32)
        out = compute_global_normalized_advantages(rewards=rewards, eps=1e-6, clip_range=0.5)
        self.assertTrue(np.all(out <= 0.5))
        self.assertTrue(np.all(out >= -0.5))

    def test_rloo_advantages_leave_one_out_baseline(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        out = compute_rloo_advantages_by_group_id(rewards=rewards, group_ids=group_ids, whiten=False, eps=1e-6)
        expected = np.asarray([-1.5, 0.0, 1.5, -10.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)

    def test_rloo_advantages_whitened(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        out = compute_rloo_advantages_by_group_id(rewards=rewards, group_ids=group_ids, whiten=True, eps=1e-6)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertGreater(float(out.std()), 0.5)

    def test_dapo_matches_grpo_plus_global_mix(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        eps = 1e-4
        alpha = 0.5

        grpo = compute_grpo_advantages_by_group_id(rewards=rewards, group_ids=group_ids, eps=eps)
        global_adv = compute_global_normalized_advantages(rewards=rewards, eps=eps)
        expected = grpo + alpha * global_adv

        out = compute_dapo_advantages_by_group_id(rewards=rewards, group_ids=group_ids, eps=eps, alpha=alpha)
        np.testing.assert_allclose(out, expected.astype(np.float32), rtol=0, atol=1e-6)

    def test_reinforce_plus_plus_is_rloo_whitened(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        group_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        eps = 1e-4

        out = compute_reinforce_plus_plus_advantages_by_group_id(rewards=rewards, group_ids=group_ids, eps=eps)
        expected = compute_rloo_advantages_by_group_id(rewards=rewards, group_ids=group_ids, eps=eps, whiten=True)
        np.testing.assert_allclose(out, expected.astype(np.float32), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


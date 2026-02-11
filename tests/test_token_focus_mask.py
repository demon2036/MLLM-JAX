from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from plugins.training.rl.token_focus import build_token_focus_mask


class TestTokenFocusMask(unittest.TestCase):
    def test_selects_first_k_low_prob_tokens_left_to_right(self) -> None:
        probs = jnp.asarray([[0.9, 0.2, 0.1, 0.4, 0.05, 0.25]], dtype=jnp.float32)
        logps = jnp.log(probs)
        valid = jnp.ones_like(logps)

        focus_mask, selected = build_token_focus_mask(
            logps,
            valid,
            prob_threshold=0.3,
            max_tokens_per_sequence=2,
        )

        expected = np.asarray([[0, 1, 1, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(focus_mask), expected, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(selected), np.asarray([2], dtype=np.int32))

    def test_respects_valid_mask_and_does_not_pad_to_k(self) -> None:
        probs = jnp.asarray([[0.05, 0.9, 0.1, 0.2, 0.9, 0.9]], dtype=jnp.float32)
        logps = jnp.log(probs)
        valid = jnp.asarray([[1, 1, 0, 1, 1, 1]], dtype=jnp.float32)

        focus_mask, selected = build_token_focus_mask(
            logps,
            valid,
            prob_threshold=0.3,
            max_tokens_per_sequence=10,
        )

        # eligible (<0.3) tokens are positions 0 and 3; position 2 is invalid.
        expected = np.asarray([[1, 0, 0, 1, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(focus_mask), expected, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(selected), np.asarray([2], dtype=np.int32))

    def test_threshold_validation(self) -> None:
        logps = jnp.zeros((1, 1), dtype=jnp.float32)
        valid = jnp.ones((1, 1), dtype=jnp.float32)
        with self.assertRaises(ValueError):
            build_token_focus_mask(logps, valid, prob_threshold=0.0, max_tokens_per_sequence=1)
        with self.assertRaises(ValueError):
            build_token_focus_mask(logps, valid, prob_threshold=1.0, max_tokens_per_sequence=1)

    def test_k_validation(self) -> None:
        logps = jnp.asarray([[math.log(0.1)]], dtype=jnp.float32)
        valid = jnp.ones_like(logps)
        with self.assertRaises(ValueError):
            build_token_focus_mask(logps, valid, prob_threshold=0.3, max_tokens_per_sequence=0)


if __name__ == "__main__":
    unittest.main()

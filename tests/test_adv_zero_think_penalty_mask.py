from __future__ import annotations

import unittest

import numpy as np

from plugins.training.rl.adv_zero_think_penalty import build_adv_zero_think_window_mask


class TestAdvZeroThinkPenaltyMask(unittest.TestCase):
    def test_finds_tag_in_completion_and_selects_window_after_tag(self) -> None:
        input_ids = np.asarray([[1, 2, 3, 4, 99, 100, 5, 6, 7, 0]], dtype=np.int32)
        labels = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], dtype=np.int32)

        window_mask, tag_found = build_adv_zero_think_window_mask(
            input_ids=input_ids,
            labels=labels,
            tag_token_ids=[99, 100],
            window_tokens=3,
            start_after_tag=True,
            no_think_policy="first_tokens",
        )

        expected = np.asarray([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(window_mask, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(tag_found, np.asarray([True]))

    def test_can_include_tag_tokens_when_start_after_tag_false(self) -> None:
        input_ids = np.asarray([[1, 2, 3, 4, 99, 100, 5, 6, 7, 0]], dtype=np.int32)
        labels = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 1, 0]], dtype=np.int32)

        window_mask, tag_found = build_adv_zero_think_window_mask(
            input_ids=input_ids,
            labels=labels,
            tag_token_ids=[99, 100],
            window_tokens=3,
            start_after_tag=False,
            no_think_policy="first_tokens",
        )

        expected = np.asarray([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(window_mask, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(tag_found, np.asarray([True]))

    def test_fallback_to_first_tokens_when_no_tag_found(self) -> None:
        input_ids = np.asarray([[1, 2, 3, 4, 10, 11, 12, 13, 0, 0]], dtype=np.int32)
        labels = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=np.int32)

        window_mask, tag_found = build_adv_zero_think_window_mask(
            input_ids=input_ids,
            labels=labels,
            tag_token_ids=[99, 100],
            window_tokens=2,
            start_after_tag=True,
            no_think_policy="first_tokens",
        )

        expected = np.asarray([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(window_mask, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(tag_found, np.asarray([False]))

    def test_mask_all_when_no_tag_found_and_policy_mask_all(self) -> None:
        input_ids = np.asarray([[1, 2, 3, 4, 10, 11, 12, 13, 0, 0]], dtype=np.int32)
        labels = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=np.int32)

        window_mask, tag_found = build_adv_zero_think_window_mask(
            input_ids=input_ids,
            labels=labels,
            tag_token_ids=[99, 100],
            window_tokens=2,
            start_after_tag=True,
            no_think_policy="mask_all",
        )

        expected = np.zeros_like(window_mask, dtype=np.float32)
        np.testing.assert_allclose(window_mask, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(tag_found, np.asarray([False]))

    def test_sequence_mask_limits_which_rows_are_processed(self) -> None:
        input_ids = np.asarray(
            [
                [1, 2, 3, 4, 99, 100, 5, 6, 7, 0],
                [1, 2, 3, 4, 99, 100, 5, 6, 7, 0],
            ],
            dtype=np.int32,
        )
        labels = np.asarray(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            ],
            dtype=np.int32,
        )
        window_mask, tag_found = build_adv_zero_think_window_mask(
            input_ids=input_ids,
            labels=labels,
            tag_token_ids=[99, 100],
            window_tokens=2,
            start_after_tag=True,
            no_think_policy="first_tokens",
            sequence_mask=np.asarray([0, 1], dtype=np.int32),
        )

        expected = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(window_mask, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(tag_found, np.asarray([False, True]))


if __name__ == "__main__":
    unittest.main()


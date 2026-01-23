from __future__ import annotations

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.api import BatchSchemaError, validate_grpo_batch


class TestGrpoBatchSchemaValidator(unittest.TestCase):
    def test_schema_validator_happy_path(self) -> None:
        batch_size = 4
        seq_len = 8

        input_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        labels = np.zeros((batch_size, seq_len), dtype=np.int32)
        labels[:, seq_len // 2 :] = 1

        rewards = np.zeros((batch_size,), dtype=np.float32)
        advantages = np.ones((batch_size,), dtype=np.float32)
        old_per_token_logps = np.zeros((batch_size, seq_len - 1), dtype=np.float32)
        group_ids = np.arange(batch_size, dtype=np.int32)
        total_valid_token_count = int(labels[:, 1:].sum())

        validate_grpo_batch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "group_ids": group_ids,
            },
            stage="rollout",
        )
        validate_grpo_batch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rewards": rewards,
                "group_ids": group_ids,
            },
            stage="rewarded",
        )
        validate_grpo_batch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rewards": rewards,
                "advantages": advantages,
                "group_ids": group_ids,
            },
            stage="advantaged",
        )
        validate_grpo_batch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rewards": rewards,
                "advantages": advantages,
                "group_ids": group_ids,
                "total_valid_token_count": total_valid_token_count,
            },
            stage="train_step",
        )
        validate_grpo_batch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rewards": rewards,
                "advantages": advantages,
                "group_ids": group_ids,
                "old_per_token_logps": old_per_token_logps,
                "total_valid_token_count": total_valid_token_count,
            },
            stage="train_ready",
        )

    def test_schema_validator_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(BatchSchemaError):
            validate_grpo_batch(
                {
                    "input_ids": np.zeros((2, 4), dtype=np.int32),
                    "attention_mask": np.ones((2, 5), dtype=np.int32),
                    "labels": np.zeros((2, 4), dtype=np.int32),
                },
                stage="rollout",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import yaml

from plugins2.grpo_observability.config import load_plugins2_config
from plugins2.grpo_observability.observability import (
    build_token_rows,
    compute_token_observability_tensors,
    summarize_rewards,
)


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return f"tok_{int(token_ids[0])}"


class TestPlugins2Observability(unittest.TestCase):
    def test_compute_token_observability_tensors(self) -> None:
        labels = np.asarray(
            [
                [0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        advantages = np.asarray([0.6, -0.3], dtype=np.float32)
        per_token_logps = np.asarray(
            [
                [-1.0, -0.2, -0.3, -2.0],
                [-0.4, -1.2, -0.9, -0.7],
            ],
            dtype=np.float32,
        )

        token_grad, token_loss, token_probs = compute_token_observability_tensors(
            labels=labels,
            advantages=advantages,
            per_token_logps=per_token_logps,
            total_valid_token_count=3.0,
        )

        expected_grad = np.asarray(
            [
                [0.0, -0.2, -0.2, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(token_grad, expected_grad, atol=1e-6)

        self.assertAlmostEqual(float(token_probs[0, 1]), float(np.exp(-0.2)), places=6)
        self.assertAlmostEqual(float(token_probs[0, 2]), float(np.exp(-0.3)), places=6)
        self.assertEqual(float(token_probs[0, 0]), 0.0)
        self.assertEqual(float(token_probs[1, 1]), 0.0)

        self.assertAlmostEqual(float(token_loss[0, 1]), -(-0.2 * 0.6 / 3.0), places=6)
        self.assertAlmostEqual(float(token_loss[1, 0]), -(-0.4 * -0.3 / 3.0), places=6)

    def test_build_token_rows(self) -> None:
        input_ids = np.asarray(
            [
                [101, 11, 12, 13, 14],
                [101, 21, 22, 23, 24],
            ],
            dtype=np.int32,
        )
        labels = np.asarray(
            [
                [0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        per_token_logps = np.asarray(
            [
                [-1.0, -0.2, -0.3, -2.0],
                [-0.4, -1.2, -0.9, -0.7],
            ],
            dtype=np.float32,
        )
        token_grad = np.asarray(
            [
                [0.0, -0.2, -0.2, 0.0],
                [0.1, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        token_loss = np.asarray(
            [
                [0.0, 0.04, 0.06, 0.0],
                [0.02, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        token_probs = np.exp(per_token_logps) * labels[:, 1:]

        rows = build_token_rows(
            input_ids=input_ids,
            labels=labels,
            per_token_logps=per_token_logps,
            token_grad_logprob=token_grad,
            token_loss_contrib=token_loss,
            token_probs=token_probs,
            tokenizer=_DummyTokenizer(),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(len(rows[1]), 1)

        self.assertEqual(rows[0][0]["token_id"], 12)
        self.assertEqual(rows[0][0]["token_text"], "tok_12")
        self.assertEqual(rows[1][0]["token_id"], 21)
        self.assertEqual(rows[1][0]["position"], 0)

    def test_summarize_rewards(self) -> None:
        rewards = np.asarray([1.0, 0.5, -0.5], dtype=np.float32)
        rewards_per_func = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, -0.5],
            ],
            dtype=np.float32,
        )
        names = ["reward_a", "reward_b"]

        stats, per_sample = summarize_rewards(
            rewards=rewards,
            rewards_per_func=rewards_per_func,
            reward_names=names,
        )

        self.assertAlmostEqual(stats["mean"], float(rewards.mean()), places=6)
        self.assertEqual(len(per_sample), 3)
        self.assertAlmostEqual(per_sample[1]["reward_b"], 0.5, places=6)
        self.assertAlmostEqual(per_sample[2]["total"], -0.5, places=6)

    def test_load_plugins2_config(self) -> None:
        payload = {
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "mesh_shape": "auto",
            "reward_weights": [1.0, 0.5, 0.5],
            "rollout": {"k": 4, "global_length": 256, "max_length_sample": 32},
            "train": {
                "training_steps": 8,
                "grad_accum_steps": 1,
                "beta": 0.0,
                "optimizer": {
                    "name": "lion",
                    "clip_norm": 1.0,
                    "weight_decay": 1e-8,
                    "lr_schedule": {
                        "type": "warmup_cosine",
                        "init_value": 0.0,
                        "peak_value": 1e-6,
                        "end_value": 0.0,
                        "warmup_ratio": 0.1,
                    },
                },
            },
            "wandb": {"project": "plugins2-test", "mode": "disabled", "name": "unit"},
            "web": {"host": "127.0.0.1", "port": 9999},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(payload, tmp)
            config_path = tmp.name

        cfg = load_plugins2_config(config_path)
        self.assertEqual(cfg.model_path, payload["model_path"])
        self.assertEqual(cfg.rollout.k, 4)
        self.assertEqual(cfg.rollout.max_length_sample, 32)
        self.assertEqual(cfg.train.training_steps, 8)
        self.assertEqual(cfg.wandb.mode, "disabled")
        self.assertEqual(cfg.web.port, 9999)


if __name__ == "__main__":
    unittest.main()

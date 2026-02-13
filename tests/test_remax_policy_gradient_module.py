from __future__ import annotations

import math
import unittest

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from plugins.training.rl.remax.module import ReMaxPolicyGradientModule


class _ConstantLogitsModel(nn.Module):
    logits: jnp.ndarray

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        b, l = input_ids.shape
        vocab = int(self.logits.shape[0])
        logits = jnp.broadcast_to(self.logits.reshape((1, 1, vocab)), (b, l, vocab))
        return logits, None


class TestReMaxPolicyGradientModule(unittest.TestCase):
    def test_terminal_discounting_matches_remaining_token_count(self) -> None:
        # One sequence, 2 completion tokens (labels at positions 1 and 2).
        input_ids = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1]], dtype=jnp.int32)
        labels = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
        advantages = jnp.asarray([2.0], dtype=jnp.float32)

        model = _ConstantLogitsModel(logits=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        module = ReMaxPolicyGradientModule(
            model=model,
            pad_token_id=0,
            ref_model=None,
            kl_coef=0.0,
            gamma=0.5,
            returns_style="official",
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }
        variables = module.init(jax.random.PRNGKey(0), inputs)
        out = module.apply(variables, inputs)

        # logp for chosen token under uniform logits over 2 tokens.
        logp = -math.log(2.0)
        # remaining completion counts are [2, 1], so returns are [2*0.5^2, 2*0.5^1] = [0.5, 1.0]
        expected_returns = np.asarray([0.5, 1.0], dtype=np.float32)
        expected_loss = -float((expected_returns * logp).sum()) / 2.0

        self.assertAlmostEqual(float(out["return_mean"]), float(expected_returns.mean()), places=5)
        self.assertAlmostEqual(float(out["loss"]), float(expected_loss), places=5)

    def test_baseline_rows_with_empty_completion_mask_produce_zero_loss(self) -> None:
        input_ids = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1]], dtype=jnp.int32)
        labels = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        advantages = jnp.asarray([10.0], dtype=jnp.float32)

        model = _ConstantLogitsModel(logits=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        module = ReMaxPolicyGradientModule(
            model=model,
            pad_token_id=0,
            ref_model=None,
            kl_coef=0.0,
            gamma=1.0,
            returns_style="official",
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }
        variables = module.init(jax.random.PRNGKey(0), inputs)
        out = module.apply(variables, inputs)
        self.assertAlmostEqual(float(out["loss"]), 0.0, places=6)
        self.assertAlmostEqual(float(out["return_mean"]), 0.0, places=6)

    def test_token_local_kl_shaping_adds_without_discount(self) -> None:
        input_ids = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1]], dtype=jnp.int32)
        labels = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
        advantages = jnp.asarray([0.0], dtype=jnp.float32)

        policy = _ConstantLogitsModel(logits=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        ref = _ConstantLogitsModel(logits=jnp.asarray([2.0, 0.0], dtype=jnp.float32))

        module = ReMaxPolicyGradientModule(
            model=policy,
            pad_token_id=0,
            ref_model=ref,
            kl_coef=1.0,
            gamma=0.5,
            returns_style="official",
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }
        variables = module.init(jax.random.PRNGKey(0), inputs)
        out = module.apply(variables, inputs)

        # policy logp for token 0 under uniform logits.
        logp = -math.log(2.0)
        # ref logp for token 0 under logits [2,0].
        ref_logp = 2.0 - math.log(math.exp(2.0) + 1.0)
        kl_log_ratio = logp - ref_logp
        shaping = -kl_log_ratio  # kl_coef=1

        # advantages=0 so terminal term is 0, returns are just token-local shaping (not discounted).
        expected_returns = np.asarray([shaping, shaping], dtype=np.float32)
        expected_loss = -float((expected_returns * logp).sum()) / 2.0

        self.assertAlmostEqual(float(out["kl_log_ratio_mean"]), float(kl_log_ratio), places=5)
        self.assertAlmostEqual(float(out["loss"]), float(expected_loss), places=5)

    def test_verl_style_reverse_cumsum_spreads_terminal_reward_without_discount(self) -> None:
        input_ids = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1]], dtype=jnp.int32)
        labels = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
        advantages = jnp.asarray([2.0], dtype=jnp.float32)

        model = _ConstantLogitsModel(logits=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        module = ReMaxPolicyGradientModule(
            model=model,
            pad_token_id=0,
            ref_model=None,
            kl_coef=0.0,
            gamma=0.5,  # ignored for verl style
            returns_style="verl",
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }
        variables = module.init(jax.random.PRNGKey(0), inputs)
        out = module.apply(variables, inputs)

        logp = -math.log(2.0)
        expected_returns = np.asarray([2.0, 2.0], dtype=np.float32)
        expected_loss = -float((expected_returns * logp).sum()) / 2.0

        self.assertAlmostEqual(float(out["return_mean"]), float(expected_returns.mean()), places=5)
        self.assertAlmostEqual(float(out["loss"]), float(expected_loss), places=5)

    def test_verl_style_reverse_cumsum_accumulates_kl_penalty(self) -> None:
        input_ids = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1]], dtype=jnp.int32)
        labels = jnp.asarray([[0, 1, 1]], dtype=jnp.int32)
        advantages = jnp.asarray([0.0], dtype=jnp.float32)

        policy = _ConstantLogitsModel(logits=jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        ref = _ConstantLogitsModel(logits=jnp.asarray([2.0, 0.0], dtype=jnp.float32))

        module = ReMaxPolicyGradientModule(
            model=policy,
            pad_token_id=0,
            ref_model=ref,
            kl_coef=1.0,
            gamma=1.0,
            returns_style="verl",
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }
        variables = module.init(jax.random.PRNGKey(0), inputs)
        out = module.apply(variables, inputs)

        logp = -math.log(2.0)
        ref_logp = 2.0 - math.log(math.exp(2.0) + 1.0)
        kl_log_ratio = logp - ref_logp
        shaping = -kl_log_ratio

        # token_level_rewards is [shaping, shaping], returns are reverse-cumsum: [2*shaping, shaping]
        expected_returns = np.asarray([2.0 * shaping, shaping], dtype=np.float32)
        expected_loss = -float((expected_returns * logp).sum()) / 2.0

        self.assertAlmostEqual(float(out["kl_log_ratio_mean"]), float(kl_log_ratio), places=5)
        self.assertAlmostEqual(float(out["loss"]), float(expected_loss), places=5)


if __name__ == "__main__":
    unittest.main()

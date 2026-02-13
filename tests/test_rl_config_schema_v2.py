from __future__ import annotations

import unittest

from plugins.training.rl.algorithms import (
    AlgoConfig,
    PluginConfig,
    normalize_algo_config,
)
from projects.gsm8k_grpo.config_schema import GRPODynamicSamplingConfig


class TestRlConfigSchemaV2(unittest.TestCase):
    def test_gae_requires_ppo_update(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="gae", kwargs={}),
                    update=PluginConfig(name="policy_gradient", kwargs={}),
                )
            )

    def test_policy_gradient_rejects_ppo_kwargs(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={}),
                    update=PluginConfig(name="policy_gradient", kwargs={"value_coef": 0.5}),
                )
            )

    def test_policy_gradient_loss_level_defaults_to_token(self) -> None:
        normalized, _algo_name, estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="grpo", kwargs={}),
                update=PluginConfig(name="policy_gradient", kwargs={}),
            )
        )
        self.assertEqual(estimator_name, "grpo")
        self.assertEqual(update_name, "policy_gradient")
        self.assertEqual(normalized.update.kwargs["loss_level"], "token")

    def test_policy_gradient_loss_level_sequence_is_accepted(self) -> None:
        normalized, _algo_name, _estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="grpo", kwargs={}),
                update=PluginConfig(name="policy_gradient", kwargs={"loss_level": "sequence"}),
            )
        )
        self.assertEqual(update_name, "policy_gradient")
        self.assertEqual(normalized.update.kwargs["loss_level"], "sequence")

    def test_policy_gradient_loss_level_invalid_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={}),
                    update=PluginConfig(name="policy_gradient", kwargs={"loss_level": "banana"}),
                )
            )

    def test_ppo_defaults_filled(self) -> None:
        normalized, _algo_name, estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="gae", kwargs={}),
                update=PluginConfig(name="ppo", kwargs={}),
            )
        )
        self.assertEqual(estimator_name, "gae")
        self.assertEqual(update_name, "ppo")
        self.assertEqual(float(normalized.update.kwargs["value_coef"]), 0.5)
        self.assertEqual(float(normalized.update.kwargs["value_clip_range"]), 0.2)
        self.assertEqual(float(normalized.update.kwargs["entropy_coef"]), 0.0)

    def test_rloo_defaults_filled(self) -> None:
        normalized, _algo_name, estimator_name, _update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="rloo", kwargs={}),
                update=PluginConfig(name="policy_gradient", kwargs={}),
            )
        )
        self.assertEqual(estimator_name, "rloo")
        self.assertTrue(bool(normalized.estimator.kwargs["whiten"]))

    def test_dynamic_sampling_defaults(self) -> None:
        cfg = GRPODynamicSamplingConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.trigger, "homogeneous_group")
        self.assertEqual(cfg.metric, "acc")
        self.assertEqual(float(cfg.homogeneity_threshold), 1.0)
        self.assertEqual(int(cfg.min_unique_reward_values), 2)
        self.assertEqual(int(cfg.max_extra_roll_rounds), 10)
        self.assertIsNone(cfg.target_valid_groups)
        self.assertEqual(cfg.fallback_policy, "keep_last")


if __name__ == "__main__":
    unittest.main()

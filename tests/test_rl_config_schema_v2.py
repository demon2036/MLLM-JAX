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

    def test_policy_gradient_accepts_token_focus(self) -> None:
        normalized, _algo_name, _estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="grpo", kwargs={}),
                update=PluginConfig(
                    name="policy_gradient",
                    kwargs={
                        "token_focus": {
                            "enabled": True,
                            "prob_threshold": 0.3,
                            "max_tokens_per_sequence": 10,
                        }
                    },
                ),
            )
        )
        self.assertEqual(update_name, "policy_gradient")
        token_focus = normalized.update.kwargs["token_focus"]
        self.assertTrue(bool(token_focus["enabled"]))
        self.assertEqual(float(token_focus["prob_threshold"]), 0.3)
        self.assertEqual(int(token_focus["max_tokens_per_sequence"]), 10)
        self.assertTrue(bool(token_focus["use_old_logps"]))

    def test_policy_gradient_accepts_adv_zero_think_penalty(self) -> None:
        normalized, _algo_name, _estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="grpo", kwargs={}),
                update=PluginConfig(
                    name="policy_gradient",
                    kwargs={
                        "adv_zero_think_penalty": {
                            "enabled": True,
                            "tag": "<think>",
                            "window_tokens": 20,
                            "penalty": -0.5,
                        }
                    },
                ),
            )
        )
        self.assertEqual(update_name, "policy_gradient")
        adv0 = normalized.update.kwargs["adv_zero_think_penalty"]
        self.assertTrue(bool(adv0["enabled"]))
        self.assertEqual(str(adv0["tag"]), "<think>")
        self.assertTrue(bool(adv0["start_after_tag"]))
        self.assertEqual(int(adv0["window_tokens"]), 20)
        self.assertLess(float(adv0["penalty"]), 0.0)
        self.assertEqual(str(adv0["no_think_policy"]), "first_tokens")
        self.assertEqual(str(adv0["normalize"]), "per_sequence")
        self.assertEqual(str(adv0["scale_mode"]), "fixed")

    def test_adv_zero_think_penalty_accepts_reward_gap_scale_mode(self) -> None:
        normalized, _algo_name, _estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="grpo", kwargs={}),
                update=PluginConfig(
                    name="policy_gradient",
                    kwargs={
                        "adv_zero_think_penalty": {
                            "enabled": True,
                            "penalty": -0.1,
                            "scale_mode": "reward_gap",
                        }
                    },
                ),
            )
        )
        self.assertEqual(update_name, "policy_gradient")
        adv0 = normalized.update.kwargs["adv_zero_think_penalty"]
        self.assertEqual(str(adv0["scale_mode"]), "reward_gap")

    def test_adv_zero_think_penalty_rejects_non_negative_penalty(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={}),
                    update=PluginConfig(
                        name="policy_gradient",
                        kwargs={
                            "adv_zero_think_penalty": {
                                "enabled": True,
                                "penalty": 0.0,
                            }
                        },
                    ),
                )
            )

    def test_adv_zero_think_penalty_rejects_invalid_scale_mode(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={}),
                    update=PluginConfig(
                        name="policy_gradient",
                        kwargs={
                            "adv_zero_think_penalty": {
                                "enabled": True,
                                "penalty": -0.1,
                                "scale_mode": "unknown",
                            }
                        },
                    ),
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

    def test_ppo_accepts_token_focus_bool(self) -> None:
        normalized, _algo_name, estimator_name, update_name = normalize_algo_config(
            AlgoConfig(
                estimator=PluginConfig(name="gae", kwargs={}),
                update=PluginConfig(name="ppo", kwargs={"token_focus": True}),
            )
        )
        self.assertEqual(estimator_name, "gae")
        self.assertEqual(update_name, "ppo")
        token_focus = normalized.update.kwargs["token_focus"]
        self.assertTrue(bool(token_focus["enabled"]))
        self.assertEqual(float(token_focus["prob_threshold"]), 0.3)
        self.assertEqual(int(token_focus["max_tokens_per_sequence"]), 10)

    def test_token_focus_prob_threshold_bounds(self) -> None:
        with self.assertRaises(ValueError):
            normalize_algo_config(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={}),
                    update=PluginConfig(
                        name="policy_gradient",
                        kwargs={
                            "token_focus": {
                                "enabled": True,
                                "prob_threshold": 1.0,
                                "max_tokens_per_sequence": 10,
                            }
                        },
                    ),
                )
            )

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

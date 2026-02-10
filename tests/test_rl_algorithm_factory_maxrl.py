from __future__ import annotations

import unittest

from plugins.training.rl.algorithms import (
    AlgoConfig,
    PluginConfig,
    create_algorithm,
)


class TestRlAlgorithmFactoryMaxRl(unittest.TestCase):
    def test_create_algorithm_maxrl_alias(self) -> None:
        algo = create_algorithm(
            AlgoConfig(
                name="max-rl",
                estimator=PluginConfig(name="max-rl", kwargs={"eps": 1e-6, "clip_range": None}),
                update=PluginConfig(name="policy_gradient", kwargs={}),
            ),
            reward_funcs=(lambda _input, _answer: 0.0,),
            reward_weights=(1.0,),
        )
        self.assertEqual(algo.name, "maxrl")
        self.assertEqual(algo.estimator_name, "maxrl")
        self.assertEqual(algo.update_name, "policy_gradient")
        self.assertFalse(algo.requires_value_head)
        self.assertEqual(type(algo.advantage_module).__name__, "MaxRLAdvantageModule")

    def test_policy_gradient_rejects_ppo_block(self) -> None:
        with self.assertRaises(ValueError):
            create_algorithm(
                AlgoConfig(
                    estimator=PluginConfig(name="grpo", kwargs={"eps": 1e-4, "clip_range": None}),
                    update=PluginConfig(
                        name="policy_gradient",
                        # invalid: value-head params under policy_gradient
                        kwargs={"value_coef": 0.5},
                    ),
                ),
                reward_funcs=(lambda _input, _answer: 0.0,),
                reward_weights=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()

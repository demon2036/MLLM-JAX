from __future__ import annotations

import unittest

from plugins.training.rl.algorithms import AlgoConfig, PluginConfig, create_algorithm


class TestRlAlgorithmFactoryReMax(unittest.TestCase):
    def test_create_algorithm_remax(self) -> None:
        algo = create_algorithm(
            AlgoConfig(
                name="remax",
                estimator=PluginConfig(name="remax", kwargs={"baseline_position": 0}),
                update=PluginConfig(name="remax", kwargs={"gamma": 1.0}),
            ),
            reward_funcs=(lambda _input, _answer: 0.0,),
            reward_weights=(1.0,),
        )
        self.assertEqual(algo.name, "remax")
        self.assertEqual(algo.estimator_name, "remax")
        self.assertEqual(algo.update_name, "remax")
        self.assertFalse(algo.requires_value_head)
        self.assertEqual(type(algo.advantage_module).__name__, "ReMaxGreedyBaselineAdvantageModule")

    def test_remax_requires_matching_estimator_and_update(self) -> None:
        with self.assertRaises(ValueError):
            create_algorithm(
                AlgoConfig(
                    name="remax",
                    estimator=PluginConfig(name="grpo", kwargs={"eps": 1e-4, "clip_range": None}),
                    update=PluginConfig(name="remax", kwargs={"gamma": 1.0}),
                ),
                reward_funcs=(lambda _input, _answer: 0.0,),
                reward_weights=(1.0,),
            )


if __name__ == "__main__":
    unittest.main()

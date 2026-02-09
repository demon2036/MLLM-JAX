from __future__ import annotations

import unittest

from plugins.training.rl.algorithms import AlgoConfig, create_algorithm


class TestRlAlgorithmFactoryMaxRl(unittest.TestCase):
    def test_create_algorithm_maxrl_alias(self) -> None:
        algo = create_algorithm(
            AlgoConfig(name="max-rl"),
            reward_funcs=(lambda _input, _answer: 0.0,),
            reward_weights=(1.0,),
        )
        self.assertEqual(algo.name, "maxrl")
        self.assertEqual(algo.estimator_name, "maxrl")
        self.assertEqual(algo.update_name, "policy_gradient")
        self.assertFalse(algo.requires_value_head)
        self.assertEqual(type(algo.advantage_module).__name__, "MaxRLAdvantageModule")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from typing import Any

from plugins.training.rl.reward.modules import WeightedRewardModule
from training2 import reward_correct, reward_format, tag_count_reward


GSM8K_REWARD_NAMES = ["reward_correct", "reward_format", "tag_count_reward"]



def build_gsm8k_reward_module(weights: tuple[float, float, float]) -> tuple[WeightedRewardModule, list[str]]:
    funcs = [reward_correct, reward_format, tag_count_reward]
    module = WeightedRewardModule(reward_funcs=funcs, reward_weights=tuple(float(x) for x in weights))
    return module, list(GSM8K_REWARD_NAMES)



def build_gsm8k_reward_inputs(*, label: str, batch_size: int) -> list[dict[str, Any]]:
    return [{"A": str(label)} for _ in range(int(batch_size))]


__all__ = ["GSM8K_REWARD_NAMES", "build_gsm8k_reward_inputs", "build_gsm8k_reward_module"]

"""ReMax algorithm building blocks (JAX).

This package provides the ReMax update module with a configurable return shaping style:

- "official": discounted terminal reward propagated across completion tokens + token-local KL shaping.
- "verl": reverse-cumsum returns with KL-in-reward accumulation (VERL-aligned semantics).

Both variants assume the input `advantages` are already greedy-baseline-subtracted:
  advantage = r(sample) - r(greedy)
"""

from plugins.training.rl.remax.module import ReMaxPolicyGradientModule
from plugins.training.rl.remax.state import ReMaxTrainState, get_remax_state

__all__ = [
    "ReMaxPolicyGradientModule",
    "ReMaxTrainState",
    "get_remax_state",
]

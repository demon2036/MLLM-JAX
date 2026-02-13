"""ReMax algorithm building blocks (JAX).

This package implements the core ReMax semantics used in the official reference:
- greedy baseline advantage: r(sample) - r(greedy)
- token-local KL shaping: -kl_coef * (logp - logp_ref)
- discounted terminal reward propagated across completion tokens
"""

from plugins.training.rl.remax.module import ReMaxPolicyGradientModule
from plugins.training.rl.remax.state import ReMaxTrainState, get_remax_state

__all__ = [
    "ReMaxPolicyGradientModule",
    "ReMaxTrainState",
    "get_remax_state",
]

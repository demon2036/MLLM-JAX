from __future__ import annotations

"""Compatibility shim for MiniOneRec SID SFT runner (plugin-owned path)."""

from projects.minionerec.sft.runner import (
    SidSftConfig,
    SidSftDataConfig,
    SidSftEvalConfig,
    SidSftJaxConfig,
    SidSftTasksConfig,
    SidSftTrainConfig,
    SidSftWandbConfig,
    run_sid_sft,
)

__all__ = [
    "SidSftConfig",
    "SidSftDataConfig",
    "SidSftEvalConfig",
    "SidSftJaxConfig",
    "SidSftTasksConfig",
    "SidSftTrainConfig",
    "SidSftWandbConfig",
    "run_sid_sft",
]

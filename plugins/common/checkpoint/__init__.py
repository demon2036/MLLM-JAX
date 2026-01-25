"""Checkpointing utilities shared across SFT + RL runners.

This layer is intentionally minimal. It provides:
- A small backend interface (serialize/deserialize).
- A simple manager for step-based saving and retention.

For large TPU runs, prefer an Orbax backend (future work); msgpack is kept for
small/CPU-friendly checkpoints and unit tests.
"""

from plugins.common.checkpoint.backend import CheckpointBackend
from plugins.common.checkpoint.manager import CheckpointManager, CheckpointManagerConfig
from plugins.common.checkpoint.msgpack_backend import MsgpackCheckpointBackend

__all__ = ["CheckpointBackend", "CheckpointManager", "CheckpointManagerConfig", "MsgpackCheckpointBackend"]


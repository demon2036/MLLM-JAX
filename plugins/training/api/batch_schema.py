from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class BatchSchemaError(ValueError):
    """Raised when a rollout/train batch does not match the expected schema."""


@dataclass(frozen=True)
class _ArrayInfo:
    key: str
    shape: tuple[int, ...]
    dtype: str


def _is_array_like(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _array_info(key: str, x: Any) -> _ArrayInfo:
    if not _is_array_like(x):
        raise BatchSchemaError(f"{key} must be array-like (has .shape/.dtype), got: {type(x).__name__}")
    try:
        shape = tuple(int(s) for s in x.shape)
    except Exception as e:
        raise BatchSchemaError(f"{key}.shape is not readable: {e}") from e
    return _ArrayInfo(key=key, shape=shape, dtype=str(x.dtype))


def _require_keys(batch: Mapping[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise BatchSchemaError(f"Missing required keys: {missing}. Present keys: {sorted(batch.keys())}")


def _require_rank(info: _ArrayInfo, rank: int) -> None:
    if len(info.shape) != rank:
        raise BatchSchemaError(f"{info.key} must be rank-{rank} but got shape={info.shape} dtype={info.dtype}")


def _require_same_shape(a: _ArrayInfo, b: _ArrayInfo) -> None:
    if a.shape != b.shape:
        raise BatchSchemaError(
            f"{a.key} and {b.key} must have the same shape; got {a.key}={a.shape} {b.key}={b.shape}"
        )


def _require_first_dim(info: _ArrayInfo, expected: int) -> None:
    if not info.shape:
        raise BatchSchemaError(f"{info.key} must be at least rank-1, got shape={info.shape}")
    if info.shape[0] != expected:
        raise BatchSchemaError(
            f"{info.key} first dim must be {expected} to match batch size, got shape={info.shape}"
        )


def validate_grpo_batch(batch: Mapping[str, Any], *, stage: str) -> None:
    """Validate the GRPO/GSM8K training batch schema used by `test_jit8.py`.

    This validator is intentionally minimal and focused on *structural* invariants
    (keys + ranks + basic shape alignment). It does not enforce exact dtypes
    (e.g., int32 vs int64) because host/device conversions may differ.

    Parameters
    ----------
    batch:
        A mapping containing rollout/train tensors (NumPy arrays, JAX arrays, etc).
    stage:
        One of:
        - `rollout`: expects `input_ids`, `attention_mask`, `labels`
        - `rewarded`: + expects `rewards`
        - `advantaged`: + expects `advantages`
        - `train_step`: + expects `total_valid_token_count` (and validates `old_per_token_logps` if present)
        - `train_ready`: + expects `old_per_token_logps`, `total_valid_token_count`
    """
    stage = str(stage).strip().lower()
    if stage not in {"rollout", "rewarded", "advantaged", "train_step", "train_ready"}:
        raise ValueError(f"Unknown stage={stage!r}")

    _require_keys(batch, ["input_ids", "attention_mask", "labels"])
    input_ids = _array_info("input_ids", batch["input_ids"])
    attention_mask = _array_info("attention_mask", batch["attention_mask"])
    labels = _array_info("labels", batch["labels"])

    _require_rank(input_ids, 2)
    _require_rank(attention_mask, 2)
    _require_rank(labels, 2)
    _require_same_shape(input_ids, attention_mask)
    _require_same_shape(input_ids, labels)

    bsz = input_ids.shape[0]
    if bsz <= 0:
        raise BatchSchemaError(f"Batch size must be > 0, got {bsz}")
    seq_len = input_ids.shape[1]
    if seq_len <= 1:
        raise BatchSchemaError(f"Sequence length must be > 1 (needs T-1), got {seq_len}")

    if stage in {"rewarded", "advantaged", "train_step", "train_ready"}:
        _require_keys(batch, ["rewards"])
        rewards = _array_info("rewards", batch["rewards"])
        _require_rank(rewards, 1)
        _require_first_dim(rewards, bsz)

    if stage in {"advantaged", "train_step", "train_ready"}:
        _require_keys(batch, ["advantages"])
        advantages = _array_info("advantages", batch["advantages"])
        _require_rank(advantages, 1)
        _require_first_dim(advantages, bsz)

    if stage in {"train_step", "train_ready"}:
        _require_keys(batch, ["total_valid_token_count"])

        tvtc = batch["total_valid_token_count"]
        if not _is_array_like(tvtc):
            # Allow Python ints as well.
            if not isinstance(tvtc, (int, float)):
                raise BatchSchemaError(
                    "total_valid_token_count must be a scalar array or Python number, "
                    f"got {type(tvtc).__name__}"
                )
        else:
            tvtc_info = _array_info("total_valid_token_count", tvtc)
            if tvtc_info.shape != ():
                raise BatchSchemaError(
                    f"total_valid_token_count must be scalar (shape=()), got {tvtc_info.shape}"
                )

        if stage == "train_ready":
            _require_keys(batch, ["old_per_token_logps"])

        if "old_per_token_logps" in batch:
            old_logps = _array_info("old_per_token_logps", batch["old_per_token_logps"])
            _require_rank(old_logps, 2)
            _require_first_dim(old_logps, bsz)
            if old_logps.shape[1] != seq_len - 1:
                raise BatchSchemaError(
                    "old_per_token_logps second dim must be T-1 to match input_ids[:,1:]; "
                    f"got old_per_token_logps={old_logps.shape} input_ids={input_ids.shape}"
                )

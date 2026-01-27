from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray  # int32 [B, L]
    attention_mask: np.ndarray  # int32 [B, L]
    labels: np.ndarray  # int32 [B, L]

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask, "labels": self.labels}


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 0:
        return x
    r = x % multiple
    return x if r == 0 else x + (multiple - r)


def collate_sft_batch(
    examples: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    label_pad_id: int = -100,
    pad_to_multiple_of: int = 8,
    pad_to_length: int | None = None,
    padding_side: str = "right",
) -> Batch:
    if not examples:
        raise ValueError("Empty batch")

    max_len = int(pad_to_length) if pad_to_length is not None and int(pad_to_length) > 0 else max(len(x["input_ids"]) for x in examples)
    max_len = _round_up(int(max_len), int(pad_to_multiple_of))

    input_ids = np.full((len(examples), max_len), int(pad_token_id), dtype=np.int32)
    attention_mask = np.zeros((len(examples), max_len), dtype=np.int32)
    labels = np.full((len(examples), max_len), int(label_pad_id), dtype=np.int32)

    for i, ex in enumerate(examples):
        ids = np.asarray(ex["input_ids"], dtype=np.int32)
        mask = np.asarray(ex.get("attention_mask") or [1] * len(ids), dtype=np.int32)
        lab = np.asarray(ex.get("labels") or [label_pad_id] * len(ids), dtype=np.int32)
        if ids.shape[0] != mask.shape[0] or ids.shape[0] != lab.shape[0]:
            raise ValueError("input_ids/attention_mask/labels length mismatch")

        length = int(min(int(ids.shape[0]), int(max_len)))
        if str(padding_side).lower() == "left":
            input_ids[i, max_len - length : max_len] = ids[-length:]
            attention_mask[i, max_len - length : max_len] = mask[-length:]
            labels[i, max_len - length : max_len] = lab[-length:]
        else:
            input_ids[i, :length] = ids[:length]
            attention_mask[i, :length] = mask[:length]
            labels[i, :length] = lab[:length]

    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def iter_indices(*, n: int, seed: int, shuffle: bool) -> list[int]:
    idx = list(range(int(n)))
    if not shuffle:
        return idx
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    return idx


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = ["Batch", "collate_sft_batch", "iter_indices", "batched"]

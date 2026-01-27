from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharDataset:
    train_ids: np.ndarray
    val_ids: np.ndarray
    itos: list[str]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def stoi(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.itos)}

    def decode(self, ids: np.ndarray | list[int]) -> str:
        if isinstance(ids, np.ndarray):
            ids_list = [int(x) for x in ids.reshape(-1).tolist()]
        else:
            ids_list = [int(x) for x in ids]
        return "".join(self.itos[i] for i in ids_list)


def _download_text(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    with open(dst_path, "wb") as f:
        f.write(data)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def prepare_tinyshakespeare_char(
    *,
    cache_dir: str,
    url: str,
    train_ratio: float = 0.9,
) -> CharDataset:
    """Prepare a char-level Tiny Shakespeare dataset with simple numpy caching.

    Artifacts (under `cache_dir/tinyshakespeare_char/`):
    - `input.txt` (downloaded raw text)
    - `train.npy`, `val.npy` (token ids, int32)
    - `meta.json` (vocab `itos`)
    """
    ds_dir = os.path.join(cache_dir, "tinyshakespeare_char")
    os.makedirs(ds_dir, exist_ok=True)

    input_txt = os.path.join(ds_dir, "input.txt")
    train_npy = os.path.join(ds_dir, "train.npy")
    val_npy = os.path.join(ds_dir, "val.npy")
    meta_json = os.path.join(ds_dir, "meta.json")

    if os.path.isfile(train_npy) and os.path.isfile(val_npy) and os.path.isfile(meta_json):
        meta = _load_json(meta_json)
        itos = list(meta["itos"])
        return CharDataset(
            train_ids=np.load(train_npy, allow_pickle=False),
            val_ids=np.load(val_npy, allow_pickle=False),
            itos=itos,
        )

    if not os.path.isfile(input_txt):
        _download_text(url, input_txt)

    text = open(input_txt, "r", encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}

    ids = np.fromiter((stoi[ch] for ch in text), count=len(text), dtype=np.int32)
    split_idx = int(len(ids) * float(train_ratio))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    np.save(train_npy, train_ids, allow_pickle=False)
    np.save(val_npy, val_ids, allow_pickle=False)
    _save_json(meta_json, {"itos": vocab})

    return CharDataset(train_ids=train_ids, val_ids=val_ids, itos=vocab)


def sample_batch(
    *,
    rng: np.random.Generator,
    ids: np.ndarray,
    batch_size: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got shape={ids.shape}")
    if len(ids) <= block_size + 1:
        raise ValueError(f"dataset too small for block_size={block_size}: len={len(ids)}")

    starts = rng.integers(0, len(ids) - block_size - 1, size=(batch_size,))
    x = np.stack([ids[i : i + block_size] for i in starts], axis=0).astype(np.int32, copy=False)
    y = np.stack([ids[i + 1 : i + block_size + 1] for i in starts], axis=0).astype(np.int32, copy=False)
    return x, y


__all__ = ["CharDataset", "prepare_tinyshakespeare_char", "sample_batch"]


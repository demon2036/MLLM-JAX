from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sid_from_info_line(line: str) -> str:
    parts = line.split("\t")
    if not parts:
        return ""
    return parts[0].strip()


def _read_sids_from_info_file(info_file: str) -> list[str]:
    lines = Path(info_file).read_text(encoding="utf-8").splitlines()
    sids: list[str] = []
    for line in lines:
        sid = _sid_from_info_line(line)
        if sid:
            sids.append(sid)
    return sids


def build_valid_sids_from_info(info_file: str) -> set[str]:
    return set(_read_sids_from_info_file(info_file))


def get_official_prefix_index(base_model: str) -> int:
    model_name = str(base_model or "").lower()
    if "gpt2" in model_name:
        return 4
    return 3


def hash_token_path(token_ids: Sequence[int]) -> str:
    return "-".join(str(int(token_id)) for token_id in token_ids)


@dataclass(frozen=True)
class OfficialPrefixConstraintMap:
    prefix_index: int
    eos_token_id: int
    transition_map: dict[str, np.ndarray]

    def allowed(self, token_ids: Sequence[int]) -> np.ndarray:
        return self.transition_map.get(hash_token_path(token_ids), np.asarray([], dtype=np.int32))


def _tokenize_constraint_path(*, tokenizer: Any, text: str, base_model: str) -> list[int]:
    ids = [int(token_id) for token_id in tokenizer(text).input_ids]
    if "llama" in str(base_model or "").lower() and ids:
        ids = ids[1:]
    return ids


def _resolve_eos_token_id(tokenizer: Any, eos_token_id: int | None) -> int:
    if eos_token_id is not None:
        return int(eos_token_id)
    token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        raise ValueError("eos_token_id is required when tokenizer.eos_token_id is None")
    return int(token_id)


def build_official_prefix_constraint_map_from_info(
    *,
    info_file: str,
    tokenizer: Any,
    base_model: str,
    eos_token_id: int | None = None,
    prefix_index: int | None = None,
) -> OfficialPrefixConstraintMap:
    sids = _read_sids_from_info_file(info_file)
    eos_id = _resolve_eos_token_id(tokenizer, eos_token_id)
    prefix_idx = int(get_official_prefix_index(base_model) if prefix_index is None else prefix_index)

    transitions: dict[str, set[int]] = {}

    for sid in sids:
        text = f"### Response:\n{sid}\n"
        token_ids = _tokenize_constraint_path(tokenizer=tokenizer, text=text, base_model=base_model)
        path_ids = token_ids + [eos_id]

        if len(path_ids) <= prefix_idx:
            raise ValueError(
                f"Tokenized path for SID {sid!r} is too short for prefix_index={prefix_idx}: {path_ids}"
            )

        for i in range(prefix_idx, len(path_ids)):
            if i == prefix_idx:
                key = hash_token_path(path_ids[:i])
            else:
                key = hash_token_path(path_ids[prefix_idx:i])
            transitions.setdefault(key, set()).add(int(path_ids[i]))

    transition_map: dict[str, np.ndarray] = {}
    for key in sorted(transitions.keys()):
        transition_map[key] = np.asarray(sorted(transitions[key]), dtype=np.int32)

    return OfficialPrefixConstraintMap(prefix_index=prefix_idx, eos_token_id=eos_id, transition_map=transition_map)


@dataclass(frozen=True)
class SidTrie:
    pad_id: int
    first_ids: np.ndarray
    second_keys: np.ndarray
    second_table: np.ndarray
    third_table: np.ndarray


def _token_to_id(tokenizer: Any, token: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0:
            raise ValueError(f"Token {token!r} maps to invalid id {token_id}")
        return token_id

    encoded = tokenizer.encode(token, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in encoded]
    if len(token_ids) != 1:
        raise ValueError(f"Token {token!r} encoded to {token_ids}, expected exactly one token id")
    return int(token_ids[0])


def build_sid_trie_from_index(
    *,
    tokenizer: Any,
    sid_index_path: str,
    pad_id: int = -1,
) -> SidTrie:
    raw = json.loads(Path(sid_index_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("sid_index_path must contain a JSON object mapping item_id -> 3-token SID list")

    first: set[int] = set()
    second: dict[int, set[int]] = {}
    third: dict[tuple[int, int], set[int]] = {}

    for item_id, sid_tokens in raw.items():
        if not isinstance(sid_tokens, list) or len(sid_tokens) != 3:
            raise ValueError(
                f"sid_index item {item_id!r} must be a list of exactly 3 SID tokens, got: {sid_tokens!r}"
            )

        t1 = _token_to_id(tokenizer, str(sid_tokens[0]))
        t2 = _token_to_id(tokenizer, str(sid_tokens[1]))
        t3 = _token_to_id(tokenizer, str(sid_tokens[2]))

        first.add(t1)
        second.setdefault(t1, set()).add(t2)
        third.setdefault((t1, t2), set()).add(t3)

    first_ids = np.asarray(sorted(first), dtype=np.int32)
    if first_ids.size == 0:
        raise ValueError("sid_index_path contains no valid 3-level SID tokens")

    second_keys = first_ids
    max_second = max(len(second.get(int(key), set())) for key in second_keys)
    second_table = np.full((len(second_keys), int(max_second)), int(pad_id), dtype=np.int32)

    for row_index, token_1 in enumerate(second_keys):
        token_2_values = sorted(second.get(int(token_1), set()))
        if token_2_values:
            second_table[row_index, : len(token_2_values)] = np.asarray(token_2_values, dtype=np.int32)

    max_third = 0
    for row_index, token_1 in enumerate(second_keys):
        for column_index in range(int(max_second)):
            token_2 = int(second_table[row_index, column_index])
            if token_2 == int(pad_id):
                continue
            max_third = max(max_third, len(third.get((int(token_1), token_2), set())))

    if max_third <= 0:
        raise ValueError("sid_index_path produced empty third-level expansions")

    third_table = np.full((len(second_keys), int(max_second), int(max_third)), int(pad_id), dtype=np.int32)
    for row_index, token_1 in enumerate(second_keys):
        for column_index in range(int(max_second)):
            token_2 = int(second_table[row_index, column_index])
            if token_2 == int(pad_id):
                continue
            token_3_values = sorted(third.get((int(token_1), token_2), set()))
            if token_3_values:
                third_table[row_index, column_index, : len(token_3_values)] = np.asarray(token_3_values, dtype=np.int32)

    return SidTrie(
        pad_id=int(pad_id),
        first_ids=first_ids,
        second_keys=second_keys,
        second_table=second_table,
        third_table=third_table,
    )


__all__ = [
    "OfficialPrefixConstraintMap",
    "SidTrie",
    "build_official_prefix_constraint_map_from_info",
    "build_sid_trie_from_index",
    "build_valid_sids_from_info",
    "get_official_prefix_index",
    "hash_token_path",
]

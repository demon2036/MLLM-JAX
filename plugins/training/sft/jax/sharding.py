from __future__ import annotations

"""Sharding helpers for SFT (plugin-owned)."""

import re
from typing import Any, Iterable

import jax
import numpy as np
from jax.sharding import PartitionSpec as PS


def _tree_path_to_string(path: Iterable[Any], sep: str) -> str:
    keys: list[str] = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    return sep.join(keys)


def _named_tree_map(f, tree, *rest, sep: str):
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(_tree_path_to_string(path, sep=sep), x, *r),
        tree,
        *rest,
    )


def match_partition_rules(rules, params):
    def get_partition_spec(name: str, leaf: Any):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            return PS()
        for rule, spec in rules:
            if re.search(rule, name) is not None:
                return spec
        raise ValueError(f"Partition rule not found for param: {name}")

    return _named_tree_map(get_partition_spec, params, sep="/")


def get_partition_rules_llama():
    return (
        (".*/self_attn/q_proj/kernel", PS("fsdp", "tp")),
        (".*/self_attn/k_proj/kernel", PS("fsdp", "tp")),
        (".*/self_attn/v_proj/kernel", PS("fsdp", "tp")),
        (".*/self_attn/o_proj/kernel", PS("tp", "fsdp")),
        (".*/mlp/gate_proj/kernel", PS("fsdp", "tp")),
        (".*/mlp/up_proj/kernel", PS("fsdp", "tp")),
        (".*/mlp/down_proj/kernel", PS("tp", "fsdp")),
        ("embed_tokens/embedding", PS("fsdp", "tp")),
        ("lm_head/kernel", PS("fsdp", "tp")),
        ("scale", PS(None)),
        (".*", PS(None)),
    )


__all__ = ["get_partition_rules_llama", "match_partition_rules"]

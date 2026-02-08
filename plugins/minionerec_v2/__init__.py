"""MiniOneRec v2 reusable primitives."""

from .beam_decode import BeamSearchOutput, constrained_beam_search_sid3, constrained_beam_search_sid3_prefill

from .constraints import (
    OfficialPrefixConstraintMap,
    SidTrie,
    build_official_prefix_constraint_map_from_info,
    build_sid_trie_from_index,
    build_valid_sids_from_info,
    get_official_prefix_index,
    hash_token_path,
)

__all__ = [
    "BeamSearchOutput",
    "constrained_beam_search_sid3",
    "constrained_beam_search_sid3_prefill",
    "OfficialPrefixConstraintMap",
    "SidTrie",
    "build_official_prefix_constraint_map_from_info",
    "build_sid_trie_from_index",
    "build_valid_sids_from_info",
    "get_official_prefix_index",
    "hash_token_path",
]

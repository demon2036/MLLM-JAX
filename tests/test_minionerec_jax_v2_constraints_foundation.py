from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_constraints = _load_module("_minionerec_v2_constraints_under_test", "plugins/minionerec_v2/constraints.py")

build_sid_trie_from_index = _constraints.build_sid_trie_from_index
build_valid_sids_from_info = _constraints.build_valid_sids_from_info


class _TrieTokenizer:
    def __init__(self, mapping: dict[str, int]):
        self._mapping = dict(mapping)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._mapping.get(token, -1))


def test_build_valid_sids_from_info_reads_unique_first_column(tmp_path) -> None:
    info_path = tmp_path / "item-info.txt"
    info_path.write_text(
        "sid-A\tTitle A\t100\n"
        "sid-B\tTitle B\t101\n"
        "sid-A\tTitle A duplicate\t102\n"
        "\n"
        "\tmissing-sid\t103\n",
        encoding="utf-8",
    )

    valid_sids = build_valid_sids_from_info(str(info_path))

    assert valid_sids == {"sid-A", "sid-B"}


def test_build_sid_trie_from_index_builds_expected_three_level_tables(tmp_path) -> None:
    sid_index_path = tmp_path / "sid-index.json"
    sid_index_path.write_text(
        json.dumps(
            {
                "item-1": ["<a_1>", "<b_1>", "<c_1>"],
                "item-2": ["<a_1>", "<b_2>", "<c_2>"],
                "item-3": ["<a_2>", "<b_3>", "<c_3>"],
                "item-4": ["<a_2>", "<b_3>", "<c_4>"],
            }
        ),
        encoding="utf-8",
    )

    tokenizer = _TrieTokenizer(
        {
            "<a_1>": 11,
            "<a_2>": 12,
            "<b_1>": 21,
            "<b_2>": 22,
            "<b_3>": 23,
            "<c_1>": 31,
            "<c_2>": 32,
            "<c_3>": 33,
            "<c_4>": 34,
        }
    )

    trie = build_sid_trie_from_index(tokenizer=tokenizer, sid_index_path=str(sid_index_path), pad_id=-1)

    assert trie.pad_id == -1
    np.testing.assert_array_equal(trie.first_ids, np.asarray([11, 12], dtype=np.int32))
    np.testing.assert_array_equal(trie.second_keys, np.asarray([11, 12], dtype=np.int32))
    np.testing.assert_array_equal(
        trie.second_table,
        np.asarray(
            [
                [21, 22],
                [23, -1],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        trie.third_table,
        np.asarray(
            [
                [[31, -1], [32, -1]],
                [[33, 34], [-1, -1]],
            ],
            dtype=np.int32,
        ),
    )

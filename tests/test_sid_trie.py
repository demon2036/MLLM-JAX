import json

import numpy as np

from plugins.sample.constraints.sid_trie import build_sid_trie_from_index


class DummyTokenizer:
    def __init__(self, token_to_id: dict[str, int], *, vocab_size: int = 128):
        self._token_to_id = dict(token_to_id)
        self._vocab_size = int(vocab_size)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._token_to_id[token])

    def __len__(self) -> int:
        return self._vocab_size


def test_build_sid_trie_from_index_builds_tables(tmp_path):
    token_to_id = {
        "<a>": 10,
        "<b>": 11,
        "<c>": 12,
        "<d>": 13,
        "<e>": 14,
        "<f>": 15,
    }
    sid_index = {
        "item1": ["<a>", "<b>", "<c>"],
        "item2": ["<a>", "<d>", "<e>"],
        "item3": ["<f>", "<b>", "<c>"],
    }
    path = tmp_path / "sid_index.json"
    path.write_text(json.dumps(sid_index), encoding="utf-8")

    trie = build_sid_trie_from_index(
        tokenizer=DummyTokenizer(token_to_id, vocab_size=32),
        sid_index_path=str(path),
        eos_token_id=0,
        pad_id=-1,
    )

    assert trie.pad_id == -1
    assert trie.eos_token_id == 0
    assert trie.vocab_size == 32

    assert np.asarray(trie.first_ids).tolist() == [10, 15]
    assert np.asarray(trie.second_keys).tolist() == [10, 15]

    assert np.asarray(trie.second_table).shape == (2, 2)
    assert np.asarray(trie.second_table)[0].tolist() == [11, 13]
    assert np.asarray(trie.second_table)[1].tolist() == [11, -1]

    assert np.asarray(trie.third_table).shape == (2, 2, 1)
    assert np.asarray(trie.third_table)[0, 0].tolist() == [12]
    assert np.asarray(trie.third_table)[0, 1].tolist() == [14]
    assert np.asarray(trie.third_table)[1, 0].tolist() == [12]
    assert np.asarray(trie.third_table)[1, 1].tolist() == [-1]


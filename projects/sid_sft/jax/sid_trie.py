"""Back-compat shim for SID trie utilities.

Canonical location: `plugins/sample/constraints/sid_trie.py`.
"""

from plugins.sample.constraints.sid_trie import SidTrie, build_sid_trie_from_index

__all__ = ["SidTrie", "build_sid_trie_from_index"]

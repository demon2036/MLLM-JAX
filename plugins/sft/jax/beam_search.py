"""Back-compat shim for constrained decoding.

Canonical location: `plugins/sample/decoding/sid3_constrained_beam_search.py`.
"""

from plugins.sample.decoding.sid3_constrained_beam_search import BeamSearchOutput, constrained_beam_search_sid3

__all__ = ["BeamSearchOutput", "constrained_beam_search_sid3"]

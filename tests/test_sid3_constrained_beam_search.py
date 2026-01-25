import numpy as np

import jax.numpy as jnp

import plugins.sample.decoding.sid3_constrained_beam_search as beam_mod
from plugins.sample.constraints.sid_trie import SidTrie


class DummyModel:
    def __init__(self, *, vocab_size: int):
        self.vocab_size = int(vocab_size)
        self.config = object()

    def apply(self, _variables, *, input_ids, position_ids, attention_mask, cache):
        del _variables, position_ids, attention_mask
        input_np = np.asarray(input_ids)
        batch, seq_len = input_np.shape
        vocab = int(self.vocab_size)

        logits = np.full((batch, seq_len, vocab), -20.0, dtype=np.float32)
        if seq_len > 1:
            # Prefill: pick tok1=2 slightly over tok1=1.
            logits[:, -1, 1] = -1.0
            logits[:, -1, 2] = 0.0
            return jnp.asarray(logits), cache

        for i, tok in enumerate(input_np[:, -1]):
            tok_i = int(tok)
            if tok_i == 1:
                # Step1 for tok1=1: make tok2=4 nearly deterministic.
                logits[i, -1, 3] = 0.0
                logits[i, -1, 4] = 10.0
            elif tok_i == 2:
                # Step1 for tok1=2: uniform-ish distribution (penalize this path).
                logits[i, -1, :] = 0.0
                logits[i, -1, 6] = 0.1
            elif tok_i == 4:
                # Step2 for tok2=4: make tok3=9 nearly deterministic.
                logits[i, -1, 9] = 10.0
                logits[i, -1, 10] = 0.0
            elif tok_i == 6:
                # Step2 for tok2=6: uniform-ish distribution.
                logits[i, -1, :] = 0.0
                logits[i, -1, 13] = 0.1
            else:
                # Step3+: EOS score only; keep it constant across beams.
                logits[i, -1, 0] = 0.0

        return jnp.asarray(logits), cache


def test_constrained_beam_search_sid3_picks_best_sequence(monkeypatch):
    def dummy_init_cache(_config, batch_size: int, *, max_cache_length: int, dtype):
        del _config, max_cache_length, dtype
        return jnp.zeros((int(batch_size), 1), dtype=jnp.int32)

    def dummy_pad_cache(cache, prompt_len: int, max_cache_length: int, input_len: int):
        del prompt_len, max_cache_length, input_len
        return cache

    monkeypatch.setattr(beam_mod, "init_cache", dummy_init_cache)
    monkeypatch.setattr(beam_mod, "pad_cache", dummy_pad_cache)

    trie = SidTrie(
        pad_id=-1,
        eos_token_id=0,
        vocab_size=32,
        first_ids=np.asarray([1, 2], dtype=np.int32),
        second_keys=np.asarray([1, 2], dtype=np.int32),
        second_table=np.asarray([[3, 4], [5, 6]], dtype=np.int32),
        third_table=np.asarray(
            [
                [[7, 8], [9, 10]],
                [[11, 12], [13, 14]],
            ],
            dtype=np.int32,
        ),
    )

    prompt_input_ids = jnp.asarray([[101, 102]], dtype=jnp.int32)
    out = beam_mod.constrained_beam_search_sid3(
        model=DummyModel(vocab_size=int(trie.vocab_size)),
        params=None,
        prompt_input_ids=prompt_input_ids,
        trie=trie,
        num_beams=2,
        max_cache_length=16,
        suffix_token_ids=None,
    )

    token_ids = np.asarray(out.token_ids)
    assert token_ids.shape == (1, 2, 3)
    assert token_ids[0, 0].tolist() == [1, 4, 9]
    assert token_ids[0, 1].tolist() == [2, 6, 13]


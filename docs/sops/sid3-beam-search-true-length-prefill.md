# SOP: SID3 beam search mixed-length true_len (prefill buckets)

- **Title**: Use sampler-style prefill buckets to support mixed prompt true lengths in SID3 constrained beam search
  **Prereqs**: `jax` installed (TPU or CPU backend); SID trie available; eval dataset provides per-sample `input_ids`
  **Environment (verified)**: Windows / Python 3.13 (note: `jax` not installed here, so JAX-dependent tests were skipped)

## Why

- The attention cache write index is effectively batch-shared (`end_index = cache["end_index"][0]` in `MLLM_JAX/language/llama/llama.py`), so you cannot safely set different `end_index` / decode starts per sample inside the same batch.
- The training sampler works around this by fixing `prefill_len` per batch and encoding per-sample true lengths via `attention_mask` + per-sample `position_ids`.

## What to use

- **Mixed true lengths in one batch**: `plugins/sample/decoding/sid3_constrained_beam_search.py` → `constrained_beam_search_sid3_prefill`
- **Single true length shared in batch** (legacy): `constrained_beam_search_sid3` (rejects vector `prompt_true_len`)

## Steps

1) Bucket prompts by prefill length (sampler-style, e.g. 128/256/512/...) and right-pad to the bucket length.
2) Pass per-sample `prompt_true_len` (vector) to `constrained_beam_search_sid3_prefill`.
3) Ensure `max_cache_length` leaves headroom for decoding (`max_cache_length > prefill_len + 2 + len(suffix)`; suffix includes EOS appended internally).
4) Optional (avoid many compilations): use a **single fixed** `prefill_len` for the entire run (no multi-bucket), right-pad all prompts to it, and you will only compile once for that shape.
5) Local sanity:
   - `python -m pytest -q`

## Expected result

- Evaluator/runner can batch prompts with different true lengths without per-length bucketing.
- Fewer compile shapes than per-prompt-length bucketing (bounded by the prefill bucket set).
- If using a single fixed `prefill_len`: exactly one compilation for the prompt shape (assuming batch size and other static args stay constant).

## Troubleshooting

- `max_cache_length too small`: increase `jax.max_cache_length` so it is larger than the chosen `prefill_len` plus decode headroom.
- If `--print-config` fails with missing ML deps: ensure CLI paths that only print config do not import training runtimes.

## References

- `plugins/sample/decoding/sid3_constrained_beam_search.py`
- `projects/sid_sft/jax/evaluator.py`
- `plugins/sample/sampling.py`
- `MLLM_JAX/language/llama/llama.py`

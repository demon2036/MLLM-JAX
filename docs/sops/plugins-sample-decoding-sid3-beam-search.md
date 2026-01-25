# SOP: Extract SID3 constrained beam search into `plugins/sample/`

- **Title**: SOP: Extract SID3 constrained beam search + SidTrie into shared `plugins/sample/` packages
  **Prereqs**: Repo checkout; Python + JAX; `pytest`; W&B credentials for online smoke
  **Environment (verified)**: Ubuntu Linux (repo path `/home/john/workdir/minionerec`)

## Goal

- Move SFT eval constrained decoding (SID3 beam search) out of `plugins/sft/jax/` into `plugins/sample/` so it can be reused by other pipelines without merging folders.
- Keep backward compatibility by leaving thin re-export shims under `plugins/sft/jax/`.

## Steps (commands actually used)

### 1) Create new shared packages

- `mkdir -p plugins/sample/decoding plugins/sample/constraints`

### 2) Run tests

- `python -m pytest -q`

### 3) End-to-end smoke (W&B online)

- `./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

## Expected result

- Canonical modules:
  - `plugins/sample/constraints/sid_trie.py` (SidTrie builder for constrained decoding)
  - `plugins/sample/decoding/sid3_constrained_beam_search.py` (SID3 constrained beam search)
- Back-compat shims:
  - `plugins/sft/jax/sid_trie.py`
  - `plugins/sft/jax/beam_search.py`
- Callers use canonical imports:
  - `plugins/sft/jax/evaluator.py`
- Unit tests exist and pass:
  - `tests/test_sid_trie.py`
  - `tests/test_sid3_constrained_beam_search.py`
- `python -m pytest -q` exits `0`.
- The W&B-online SFT smoke run exits `0` and prints a run URL.

## Troubleshooting

- `max_cache_length too small` during eval: increase `jax.max_cache_length` in the YAML config so it exceeds `prompt_len + 2 + len(suffix)`.
- W&B auth failure: ensure `WANDB_API_KEY` is set and `wandb_mode=online` is used in the config.

## References

- `plugins/sample/constraints/sid_trie.py`
- `plugins/sample/decoding/sid3_constrained_beam_search.py`


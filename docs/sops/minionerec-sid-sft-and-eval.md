# SOP: Run MiniOneRec SID SFT + HR/NDCG eval (projects/sid_sft, JAX)

- **Title**: SOP: Run MiniOneRec SID SFT + constrained-decoding HR@K/NDCG@K eval via `projects/sid_sft/` (JAX backend)
- **Prereqs**: Ubuntu Linux; Python `3.12.2`; network access (to download HF models for the provided smoke config)
- **Environment (verified)**: Ubuntu Linux; Python `3.12.2`

## Steps (commands actually run)

- Run local unit tests:
  - `python -m pytest -q`

- Run a local end-to-end smoke (JAX backend; tiny llama; small sample; constrained decoding enabled):
  - `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_jax_smoke_tiny_llama.yaml --run-mode train_eval`

- Cross-check HR/NDCG against upstream `calc.py` (same `eval_predictions.json`):
  - `python workdir/MiniOneRec/calc.py --path runs/sid_sft_jax_smoke_tiny_llama/eval_predictions.json --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt`

## Expected Result

- `python -m pytest -q` exits 0.
- The smoke run exits 0 and writes:
  - `runs/sid_sft_jax_smoke_tiny_llama/run_summary.json`
  - `runs/sid_sft_jax_smoke_tiny_llama/eval_predictions.json`
  - `runs/sid_sft_jax_smoke_tiny_llama/eval_predictions.metrics.json` (HR/NDCG + invalid count)
- `calc.py` prints the same HR/NDCG as the plugin metrics JSON.

## W&B (online)

- The JAX backend supports W&B; for TPU runs, use: `docs/sops/minionerec-sid-sft-jax-tpu.md`.

## References

- Upstream reference implementation: `workdir/MiniOneRec/sft.py`, `workdir/MiniOneRec/evaluate.py`, `workdir/MiniOneRec/calc.py`
- Project entrypoints: `scripts/run_sid_sft.py`, `projects/sid_sft/runner/sid_sft.py`

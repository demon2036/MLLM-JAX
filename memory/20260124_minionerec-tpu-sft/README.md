# Memory: MiniOneRec SFT on TPU (JAX implementation)

## Goal

- Replace the HF/Trainer-based `plugins/sft` SFT pipeline with a TPU-runnable JAX implementation.
- Keep upstream `workdir/MiniOneRec/` untouched; all overrides live in `plugins/`.
- Ensure evaluation HR@K / NDCG@K matches MiniOneRec `workdir/MiniOneRec/calc.py` semantics.
- Provide deterministic, config-driven entrypoints + SOP for TPU VM runs.

## Completion Criteria

- `python -m pytest -q` exits 0.
- Local CPU smoke run completes end-to-end (train → eval → metrics) with exit 0.
- TPU VM smoke run completes end-to-end with exit 0 (when TPU is available).
- Outputs include a machine-readable summary JSON + eval metrics JSON.
- SOP updated with commands *actually run*.

## Evidence (append as executed)

- Commands + exit codes:
  - `python -m pytest -q` (exit=?)
  - `./scripts/run_sid_sft.sh --config <tpu-yaml> --run-mode train_eval` (exit=?)
- Key artifacts:
  - `<output_dir>/run_summary.json`
  - `<output_dir>/eval_predictions.json`
  - `<output_dir>/eval_predictions.metrics.json`
- Notes:
  - W&B online requires `WANDB_API_KEY` and may be blocked in this environment.


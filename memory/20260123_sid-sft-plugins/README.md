# Memory: MiniOneRec SID SFT (plugins-first)

## Goal

- Implement MiniOneRec-style SID SFT as a non-invasive plugin under `plugins/sft/`.
- Provide deterministic training + evaluation entrypoints driven by committed YAML configs.
- Implement HR@K / NDCG@K evaluation with constrained decoding (valid SID outputs).
- Add W&B logging support with `wandb_mode=online` (requires `WANDB_API_KEY` provided externally via `.env` or env var; no secrets committed).

## Completion Criteria

- Code lives under `plugins/sft/` and does not modify `workdir/MiniOneRec/`.
- `python -m pytest -q` exits 0.
- A local smoke run (tiny model + small sample) completes end-to-end (train → eval → metrics) with exit 0.
- SOP captured with *actual* commands run and expected outputs.

## Evidence (to be appended)

- Commands run + exit codes:
  - `python -m pytest -q` (exit=0)
  - `./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_smoke_tiny.yaml --run-mode train_eval` (exit=0; wrote `runs/sid_sft_smoke_tiny/run_summary.json`)
- Environment fixes (so `transformers.Trainer` works cleanly on NumPy2):
  - `python -m pip install -U wandb`
  - `python -m pip install -U "scipy>=1.14.1" "scikit-learn>=1.5.2"`
  - `python -m pip install -U numexpr bottleneck`
  - `python -m pip install -U "accelerate>=0.26.0"`
- Files changed:
  - `plugins/sft/` (new)
  - `scripts/run_sid_sft.py`
  - `scripts/run_sid_sft.sh`
  - `plugins/sft/configs/*.yaml`
  - `tests/test_sid_sft_metrics.py`
  - `tests/test_sid_sft_smoke_run.py`
- Metrics evidence:
  - Smoke run writes `runs/sid_sft_smoke_tiny/eval_predictions.metrics.json` (HR/NDCG + invalid count).
- W&B online evidence:
  - Blocked in this environment because `WANDB_API_KEY` is not configured and W&B now requires login for `mode=online`.
  - To run online safely (no secrets committed):
    - Create `.env` from `.env.example` and fill `WANDB_API_KEY` locally.
    - Run `./scripts/run_sid_sft.sh --config plugins/sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`
    - The run URL prints to stdout and appears under project `minionerec-sid-sft`.

# SOP: Make `plugins/sft/` JAX-only (no torch imports)

- **Title**: SOP: Remove Torch dependencies from `plugins/sft/` and validate JAX-only SFT + eval
- **Prereqs**: `rg` (ripgrep); a valid `WANDB_API_KEY` in `.env` for the W&B-online smoke
- **Environment (verified)**: Linux `6.14.0-37-generic`; Python `3.12.2`; JAX `0.8.2`; ripgrep `15.1.0`

## Steps (commands actually run)

- Confirm `plugins/sft/` has no `torch` references:
  - `rg -n "\\btorch\\b" plugins/sft || true`

- Run unit tests:
  - `python -m pytest -q`

- Run an end-to-end JAX smoke with W&B online:
  - `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

## Expected Result

- `rg` prints no matches.
- `pytest` exits 0.
- The smoke run exits 0, creates `runs/sid_sft_smoke_tiny_wandb_online/run_summary.json`, and prints a W&B run URL.

## Troubleshooting

- Missing safetensors weights:
  - The JAX runner loads HF weights via safetensors only. Ensure the base model provides `model.safetensors` (or an index + shards). If the repo only has `pytorch_model.bin`, either convert to safetensors, set `train.train_from_scratch=true`, or provide `train.resume_from_checkpoint`.
- W&B auth errors:
  - Ensure `.env` contains `WANDB_API_KEY=...` (not committed).

## References

- Task record: `memory/20260124_scan_sft_jax_only/README.md`


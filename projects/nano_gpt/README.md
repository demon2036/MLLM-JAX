# nanoGPT (JAX/TPU baseline)

This is a minimal nanoGPT-style baseline that runs on TPU via JAX/Flax.

- Reusable model core: `plugins/nano_gpt/`
- Project code (config/data/runner + entrypoint): `projects/nano_gpt/`

## Run (TPU v4-8 smoke)

From repo root on the TPU VM:

- `bash projects/nano_gpt/scripts/run_train.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v4_8_smoke.yaml`

## Run (TPU v6e-8 standard, W&B online)

Prereq: set `WANDB_API_KEY` in `/root/.env` on the TPU VM.

From repo root on the TPU VM:

- `bash projects/nano_gpt/scripts/run_train.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard.yaml`

If you cannot use W&B (no key / auth issues), run the no-W&B variant:

- `bash projects/nano_gpt/scripts/run_train.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v6e_8_standard_no_wandb.yaml`

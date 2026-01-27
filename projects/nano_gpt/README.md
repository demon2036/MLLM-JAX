# nanoGPT (JAX/TPU baseline)

This is a minimal nanoGPT-style baseline that runs on TPU via JAX/Flax.

- Reusable implementation: `plugins/nano_gpt/`
- Project wiring (configs + entrypoint): `projects/nano_gpt/`

## Run (TPU v4-8 smoke)

From repo root on the TPU VM:

- `bash projects/nano_gpt/scripts/run_train.sh --config projects/nano_gpt/configs/tinyshakespeare_char_v4_8_smoke.yaml`


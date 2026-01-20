# GRPO/GSM8K runner: `batch_size` semantics + local validation

- **Title**: SOP: Validate `scripts/run_grpo_gsm8k_training.py` YAML configs and understand rollout batch size fields
  **Prereqs**: Windows PowerShell; Python installed; repo checkout at `MLLM-jax-sglang`; no JAX needed for `--print-config`
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9`
  - pytest: `8.4.2`
  - Repo HEAD: `d489ac0`

## Batch size semantics (runner YAML)

In `plugins/training/configs/*.yaml` (used by `scripts/run_grpo_gsm8k_training.py`):

- `rollout.batch_size`: **global sequences per training step** (sum across all JAX processes / TPU hosts). Each "sequence" is one `(prompt, completion)` sample.
  - If `null`, runner does **exactly one rollout pass** and the effective global batch size is derived from `rollout.prompt_batch_size * rollout.num_pre_q * process_count`.
- `rollout.prompt_batch_size`: prompts per rollout pass **per process** (legacy `BATCH_SIZE` semantics).
- `rollout.per_device_batch_size`: prompts per rollout pass **per device**; runner derives `prompt_batch_size = per_device_batch_size * local_device_count`.
- `rollout.num_pre_q`: samples per prompt (GRPO group size, `K`).

Notes:
- YAML configs in `plugins/training/configs/` intentionally omit `train.grad_accum_steps` because when `train.micro_batch_size` is set, the runner infers the required accumulation steps from the effective per-process batch.
- YAML configs also omit `train.max_length_total` because it is currently not used to truncate training sequences in the GRPO runner; the effective training sequence length is determined by the sampler outputs and global padding.

## Steps (commands actually used)

From repo root:

- Print merged default config (no JAX required):
  - `python scripts/run_grpo_gsm8k_training.py --print-config`
- Print merged bs128 config (no JAX required):
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`
- Run local test suite:
  - `pytest -q`

## Expected Result

- Both `--print-config` commands exit with code `0` and print YAML.
- `pytest -q` exits with code `0`.

## Troubleshooting

- `TypeError: ... unexpected keyword argument ...`: update the runner/config schema together (see references).
- `pytest` fails on Windows with path/encoding issues: run `pytest -q` from repo root and ensure `sys.executable` points to the intended Python.

## References

- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/config.py`
- `plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml`

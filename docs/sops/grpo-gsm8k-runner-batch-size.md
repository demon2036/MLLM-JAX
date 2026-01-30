# GRPO/GSM8K runner: `batch_size` semantics + local validation

- **Title**: SOP: Validate `projects/gsm8k_grpo/scripts/run_train.py` YAML configs and understand rollout batch size fields
  **Prereqs**: Windows PowerShell; Python installed; repo checkout at `MLLM-jax-sglang`; no JAX needed for `--print-config`
  **Environment (verified)**:
  - OS: Windows `11` (`Windows-11-10.0.22631-SP0`)
  - Python: `3.13.9`
  - pytest: `8.4.2`
  - Repo HEAD: `d489ac0`

## Batch size semantics (runner YAML)

In `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (used by `projects/gsm8k_grpo/scripts/run_train.py`):

- `rollout.batch_size`: **global prompts per training step** (sum across all JAX processes / TPU hosts).
  - Each prompt expands to `rollout.n` completions, so global sequences per step = `rollout.batch_size * rollout.n`.
- `rollout.n`: samples per prompt (GRPO group size, `K` / `num_pre_q`).
  - Alias: `rollout.num_pre_q`.
- Runner derives the per-process prompt batch from the global target and pads it so
  `(prompts_per_process * rollout.n)` is divisible by `local_device_count`.
- Legacy keys like `rollout.global_prompt_batch_size`, `rollout.prompt_batch_size`,
  and `rollout.per_device_batch_size` are rejected; use only `rollout.batch_size` + `rollout.n`.

Notes:
- The default YAML config intentionally omits `train.grad_accum_steps` because when `train.micro_batch_size` is set, the runner infers the required accumulation steps from the effective per-process batch.
- The default YAML config also omits `train.max_length_total` because it is currently not used to truncate training sequences in the GRPO runner; the effective training sequence length is determined by the sampler outputs and global padding.
- The default config targets `128 sequences/step` by setting `rollout.batch_size=16` and `rollout.n=8` (i.e., 16 prompts * 8 = 128 sequences).

## Steps (commands actually used)

From repo root:

- Print merged default config (no JAX required):
  - `python projects/gsm8k_grpo/scripts/run_train.py --print-config`
- Print merged config with an explicit `--config` (recommended):
  - `python projects/gsm8k_grpo/scripts/run_train.py --print-config --config projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Run local test suite:
  - `pytest -q`

## Expected Result

- Both `--print-config` commands exit with code `0` and print YAML.
- `pytest -q` exits with code `0`.

## Troubleshooting

- `TypeError: ... unexpected keyword argument ...`: update the runner/config schema together (see references).
- `pytest` fails on Windows with path/encoding issues: run `pytest -q` from repo root and ensure `sys.executable` points to the intended Python.

## References

- `projects/gsm8k_grpo/scripts/run_train.py`
- `projects/gsm8k_grpo/jax/train.py`
- `plugins/training/rl/config.py`
- `projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`

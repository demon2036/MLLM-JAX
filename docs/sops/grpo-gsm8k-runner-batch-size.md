# GRPO/GSM8K runner: `batch_size` semantics + local validation

- **Title**: SOP: Validate `scripts/run_grpo_gsm8k_training.py` YAML configs and understand rollout batch size fields
  **Prereqs**: Bash/PowerShell; Python installed; repo checkout; no JAX needed for `--print-config`
  **Environment (verified)**:
  - OS: Linux (Ubuntu kernel `6.14`)
  - Python: `3.12.2`
  - pytest: `7.4.4`

## Batch size semantics (runner YAML)

In `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` (used by `scripts/run_grpo_gsm8k_training.py`):

- `rollout.batch_size`: **global prompts per training step** (sum across all JAX processes / TPU hosts).
  - Each prompt expands to `rollout.n` completions, so global sequences per step = `rollout.batch_size * rollout.n`.
- `rollout.n`: samples per prompt (GRPO group size, `K` / `num_pre_q`).
  - Alias: `rollout.num_pre_q`.
- `rollout.max_prompts_per_pass_per_process`: optional cap to keep `sampler.generate()` shapes manageable.
  - When set, the runner splits one training step into multiple rollout passes.
  - Each pass uses the same `prompts_per_pass_per_process` on every JAX process so shapes match across hosts.
  - Runner enforces `(prompts_per_pass_per_process * rollout.n) % local_device_count == 0`.
  - Runner may pad the effective global prompts/step to satisfy shape/divisibility constraints.
- Runner derives the per-process prompt batch from the global target and pads it so
  `(prompts_per_process * rollout.n)` is divisible by `local_device_count`.
- Legacy keys like `rollout.global_prompt_batch_size`, `rollout.prompt_batch_size`,
  and `rollout.per_device_batch_size` are rejected; use only `rollout.batch_size` + `rollout.n`.

Notes:
- The default YAML config intentionally omits `train.grad_accum_steps` because when `train.micro_batch_size` is set, the runner infers the required accumulation steps from the effective per-process batch.
- The default YAML config also omits `train.max_length_total` because it is currently not used to truncate training sequences in the GRPO runner; the effective training sequence length is determined by the sampler outputs and global padding.
- The default config targets `128 sequences/step` by setting `rollout.batch_size=16` and `rollout.n=8` (i.e., 16 prompts * 8 = 128 sequences).
- For prompt batch size experiments, configs use a `pbs<NUM>` prefix (e.g. `pbs128` means `rollout.batch_size: 128` prompts/step).

## Steps (commands actually used)

From repo root:

- Print merged default config (no JAX required):
  - `python scripts/run_grpo_gsm8k_training.py --print-config`
- Print merged config with an explicit `--config` (recommended):
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Print merged config for prompt-batch benchmarks:
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_pbs128_steps12_v6e8_bench.yaml`
  - `python scripts/run_grpo_gsm8k_training.py --print-config --config plugins/training/configs/grpo_gsm8k_qwen25_3b_pbs128_steps12_v6e16_bench.yaml`
- Run local test suite:
  - `pytest -q`

## Expected Result

- Both `--print-config` commands exit with code `0` and print YAML.
- The `pbs128` `--print-config` commands exit with code `0` and print YAML.
- `pytest -q` exits with code `0`.

## Troubleshooting

- `TypeError: ... unexpected keyword argument ...`: update the runner/config schema together (see references).
- `pytest` fails on Windows with path/encoding issues: run `pytest -q` from repo root and ensure `sys.executable` points to the intended Python.

## References

- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `plugins/training/config.py`
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`

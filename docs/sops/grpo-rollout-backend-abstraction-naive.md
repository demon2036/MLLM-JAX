# SOP: Introduce a rollout backend abstraction (naive sampler) and validate locally

- **Title**: SOP: Introduce a rollout backend abstraction (naive sampler) and validate locally
  **Prereqs**: Ubuntu Linux; Python `3.12.x`; `pytest`

## Steps (commands actually used)

### 1) Compile the changed Python modules

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `python -m py_compile plugins/training/config.py scripts/run_grpo_gsm8k_training.py plugins/training/runner/grpo_gsm8k.py plugins/training/rollout_backends/__init__.py plugins/training/rollout_backends/base.py plugins/training/rollout_backends/naive_sampler.py plugins/training/rollout_backends/factory.py tests/test_rollout_backend_factory.py`

### 2) Verify the CLI config surface shows `rollout.backend`

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `python scripts/run_grpo_gsm8k_training.py --print-config | grep -n "backend" || true`

### 3) Run local unit tests

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `python -m pytest -q`

### 4) Compileall smoke check (optional but fast)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `python -m compileall -q plugins scripts`

## Expected Result

- The printed config includes `rollout.backend: naive`.
- `pytest` exits with code `0`.

## Troubleshooting

- `TypeError: non-default argument ... follows default argument`:
  - Ensure dataclass fields in `GRPORolloutConfig` keep non-default fields before default fields.

## References

- `answers/rollout-backend-abstraction.md`
- `plugins/training/rollout_backends/factory.py`
- `plugins/training/rollout_backends/naive_sampler.py`
- `plugins/training/runner/grpo_gsm8k.py`
- `scripts/run_grpo_gsm8k_training.py`


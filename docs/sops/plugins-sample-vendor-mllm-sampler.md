# SOP: Vendor MLLM_JAX sampler into `plugins/sample/`

- **Title**: SOP: Copy/vendoring `MLLM_JAX/sample/sample_state_right_padding2.py` into `plugins/sample/` to remove `MLLM_JAX.sample.*` imports
  **Prereqs**: Repo checkout; Python; `pytest`; (optional) W&B credentials for online smoke
  **Environment (verified)**: Ubuntu Linux; repo `/home/john/workdir/minionerec`

## Goal

- Stop importing sampler code from `MLLM_JAX/sample/*` in repo entrypoints.
- Keep sampler implementation under `plugins/sample/` so it can evolve independently.
- Avoid modifying upstream `MLLM_JAX/` code (non-invasive development).

## Steps (commands actually used)

### 1) Locate all sampler imports

- `rg -n "MLLM_JAX\\.sample" -S`

### 2) Run tests

- `python -m pytest -q`

### 3) End-to-end smoke (W&B online)

- `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

## Expected result

- New canonical sampler module exists: `plugins/sample/mllm_sampler.py`.
- Repo code no longer imports `MLLM_JAX.sample.*` (only docs may reference it).
- `python -m pytest -q` exits `0`.
- The W&B-online SFT smoke run exits `0` and prints a run URL (process 0).


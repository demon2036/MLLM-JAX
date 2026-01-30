# SOP: Add `plugins/sample/` shared sampling package (no folder merge)

- **Title**: SOP: Introduce `plugins/sample/` for sampler/rollout/sampling utilities shared across pipelines
  **Prereqs**: Repo checkout; Python; `pytest`; (optional) W&B credentials for online smoke
  **Environment (verified)**: Ubuntu Linux; repo `/home/john/workdir/minionerec`

## Goal

- Create a new folder under `plugins/` to host “sampling” (prompt -> completion) utilities.
- Move sampling-related code out of `plugins/training/rollout/*` into `plugins/sample/*`.

## Steps (commands actually used)

### 1) Create the new package folders

- `mkdir -p plugins/sample/optimizations`

### 2) Sanity-check imports (fast)

- `python -c "from plugins.sample import sampling, batching; from plugins.sample.optimizations import patch_sampler_generate_fast; from plugins.training.rl.rollout.sampling import build_chat_prompts; from plugins.training.rl.rollout.batching import ceil_div; from plugins.training.rl.rollout.optimizations import patch_qwen2_attention_decode_fast; print('ok')"`

### 3) Run tests

- `python -m pytest -q`

### 4) End-to-end smoke (W&B online)

- `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

## Expected result

- New canonical modules exist under `plugins/sample/`:
  - `plugins/sample/sampling.py` (prompt formatting + sampler.generate wiring)
  - `plugins/sample/batching.py` (rollout-pass inference helpers)
  - `plugins/sample/optimizations/*` (sampler/attention patches)
- RL rollout wrappers re-export the sampling helpers:
  - `plugins/training/rl/rollout/sampling.py`
- `python -m pytest -q` exits `0`.
- The W&B-online SFT smoke run exits `0` and prints a run URL (process 0).

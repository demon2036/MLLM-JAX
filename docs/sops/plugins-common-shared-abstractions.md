# SOP: Add `plugins/common/` shared abstractions (no folder merge)

- **Title**: SOP: Introduce `plugins/common/` as the shared abstraction layer for `projects/sid_sft/` + `plugins/training/`
  **Prereqs**: Repo checkout; Python; `pytest`
  **Environment (verified)**: Ubuntu Linux; repo `/home/john/workdir/minionerec`

## Goal

- Create a new folder under `plugins/` to host shared utilities (instead of merging `projects/sid_sft` and `plugins/training`).
- Refactor duplicated utilities (config loading, W&B init, tokenizer prep, dotenv, hf safetensors loader, batch sharding helpers) to use the shared layer.

## Steps (commands actually used)

### 1) Create the shared package

- `mkdir -p plugins/common`

### 2) Sanity-check imports (fast)

- `python -m py_compile plugins/common/__init__.py plugins/common/config_loader.py plugins/common/wandb_utils.py plugins/common/tokenizer.py plugins/common/env.py plugins/common/hf_safetensors.py plugins/common/data/__init__.py plugins/common/data/padding.py plugins/common/sharding/__init__.py plugins/common/sharding/batch.py projects/sid_sft/config.py plugins/training/config.py plugins/training/runner/grpo_gsm8k.py projects/sid_sft/runner/sid_sft.py projects/sid_sft/jax/state.py projects/sid_sft/jax/train.py scripts/run_sid_sft.py scripts/run_grpo_gsm8k_training.py`

### 3) Run tests

- `python -m pytest -q`

### 4) End-to-end smoke (W&B online)

- `./scripts/run_sid_sft.sh --config projects/sid_sft/configs/sid_sft_smoke_tiny_wandb_online.yaml --run-mode train_eval`

## Expected result

- `plugins/common/` exists and is importable.
- `projects/sid_sft/config.py` and `plugins/training/config.py` delegate to `plugins/common/config_loader.py`.
- GRPO/SFT runners use shared `plugins/common/wandb_utils.py` + `plugins/common/tokenizer.py`.
- SFT uses `plugins/training/update/optimizer.build_tx` (constant LR schedule) for the JAX backend.
- CLI scripts use shared `plugins/common/env.load_dotenv_if_present`.
- `python -m pytest -q` exits `0`.
- The SFT smoke run exits `0` and shows a W&B run URL (process 0).

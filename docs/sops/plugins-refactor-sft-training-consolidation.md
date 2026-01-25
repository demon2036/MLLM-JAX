# SOP: Refactor plan to consolidate `plugins/sft` and `plugins/training`

- **Title**: SOP: Audit + propose a consolidation plan for sampler/model/data/sharding/optimizer across `plugins/sft/` and `plugins/training/`
  **Prereqs**: Repo checkout; `rg`; no JAX runtime required for inspection (tests are separate)
  **Environment (verified)**: Ubuntu Linux (repo path `/home/john/workdir/minionerec`)

## Goal

- Produce a concrete refactor plan for `plugins/`:
  - Identify merge candidates (sampler/model/data loading, sharding, optimizer, config/utils).
  - Propose a target package layout + file-level mapping.
  - Define migration phases with validation gates.

## Steps (commands actually used)

### 1) Inventory plugin surface

- `ls -la plugins/sft && ls -la plugins/training`
- `find plugins -maxdepth 3 -type f -print | sort`
- `find plugins -maxdepth 2 -type d -print | sort`

### 2) Sampler + generation path (GRPO vs SFT eval)

- `rg -n "\\bclass\\s+Sampler\\b" -S`
- `rg -n "sample_state_right_padding2" -S`
- `nl -ba plugins/training/rollout/sampling.py | sed -n '1,320p'`
- `nl -ba plugins/training/rollout/backends/naive_sampler.py | sed -n '1,320p'`
- `nl -ba MLLM_JAX/sample/sample_state_right_padding2.py | sed -n '1,320p'`
- `nl -ba plugins/sft/jax/evaluator.py | sed -n '1,320p'`

### 3) Model loading path (safetensors vs torch)

- `nl -ba plugins/sft/hf_safetensors.py | sed -n '1,220p'`
- `nl -ba plugins/sft/runner/sid_sft.py | sed -n '1,280p'`
- `nl -ba training2.py | sed -n '1,260p'`
- `nl -ba plugins/training/ppo/state.py | sed -n '1,260p'`
- `rg -n "load_hf_safetensors_state_dict|AutoModelForCausalLM|convert_torch_to_flax_llama" -S`

### 4) Data loading + batching (SFT datasets vs GSM8K)

- `nl -ba plugins/sft/datasets/csv_utils.py | sed -n '1,260p'`
- `nl -ba plugins/sft/datasets/sid_next_item.py | sed -n '1,260p'`
- `rg -n "\\bqas\\b|eval_qas|load_dataset\\(" plugins/training/runner/grpo_gsm8k.py | head -n 120`
- `nl -ba plugins/training/runner/grpo_gsm8k.py | sed -n '620,820p'`

### 5) Mesh/sharding + optimizer duplication scan

- `nl -ba plugins/training/mesh.py | sed -n '1,260p'`
- `rg -n "NamedSharding|PartitionSpec" plugins/sft/jax plugins/training training2.py | head -n 160`
- `rg -n "_form_training_global_array|make_array_from_single_device_arrays" plugins/training/runner/grpo_gsm8k.py`
- `nl -ba plugins/training/update/optimizer.py | sed -n '1,320p'`
- `nl -ba plugins/training/update/train_step.py | sed -n '1,320p'`
- `nl -ba plugins/sft/jax/state.py | sed -n '1,320p'`

### 6) Run tests (verification gate)

- `python -m pytest -q`

## Proposed target layout

Recommended: keep `plugins/sft/` and `plugins/training/` separate, but extract shared concerns into a small shared package.

### Option A (recommended): add `plugins/common/`

- `plugins/common/config_loader.py`: shared YAML loader (env expansion + overrides).
- `plugins/common/wandb_utils.py`: shared `maybe_init_wandb`.
- `plugins/common/tokenizer.py`: shared tokenizer preparation (`pad_token_id`, `padding_side`).
- `plugins/common/hf_safetensors.py`: shared safetensors loader (move/re-export from `plugins/sft/hf_safetensors.py`).
- `plugins/common/sharding/params.py`: shared “partition rules -> NamedSharding -> device_put”.
- `plugins/common/sharding/batch.py`: shared “numpy batch -> global jax.Array” builder (dp/fsdp-preferred + fallback).
- `plugins/common/data/padding.py`: shared right-padding helpers (optional early win).

### Option A2 (also recommended): add `plugins/sample/`

- `plugins/sample/sampling.py`: shared prompt -> completion -> batch wiring (used by RL rollout; reusable elsewhere).
- `plugins/sample/batching.py`: shared rollout-pass inference helpers.
- `plugins/sample/optimizations/*`: sampler/attention performance patches (fast decode, etc).

### Option B (bigger move): merge SFT under training

- Move SFT JAX modules to `plugins/training/sft/` and keep `plugins/sft/` as re-exports for 1–2 iterations.

## File-level mapping (Option A)

- **Config**: `plugins/sft/config.py` + `plugins/training/config.py` keep `DEFAULT_CONFIG`, import shared loader helpers.
- **W&B**: replace GRPO runner `_maybe_init_wandb` with shared helper; optionally re-export `plugins/sft/wandb_utils.py`.
- **Tokenizer prep**: unify “pad token fallback + padding_side=right” into `plugins/common/tokenizer.py`.
- **Model weights**: move `plugins/sft/hf_safetensors.py` -> `plugins/common/hf_safetensors.py` (keep old path as re-export).
- **Unified loader (new)**: add `plugins/llm/loader.py` (or `plugins/common/model_loader.py`) and migrate:
  - `training2.get_state` (remove `MLLM_JAX.sample.*` imports; use `plugins/sample/mllm_sampler.py`)
  - `plugins/training/ppo/state.get_ppo_state`
  - optionally `plugins/sft/runner/sid_sft.py` to reduce inline duplication
- **Batch sharding**: extract `_form_training_global_array` and padding helpers from `plugins/training/runner/grpo_gsm8k.py`.
- **Optimizer**: make SFT JAX state use `plugins/training/update/optimizer.build_tx` and retire SFT-local `_build_optimizer`.
- **Sampling**: move rollout sampling utils (`plugins/training/rollout/*`) -> `plugins/sample/*` and keep `plugins/training/rollout/*` as thin shims.

## Migration phases (validation gates)

1) Extract pure utils (`config_loader`, `wandb_utils`, `tokenizer`); keep compatibility wrappers.
2) Centralize safetensors loader; introduce unified model loader (safetensors-first; torch fallback only when needed).
3) Migrate RL loaders (`training2`, PPO) off Torch-based `get_model/get_params`.
4) Extract batch sharding + padding helpers from GRPO runner; optionally adopt in SFT for multi-host correctness.
5) Extract sampling utilities into `plugins/sample/` (keep `plugins/training/rollout` as shims).
6) Unify optimizer surface (SFT uses `OptimizerConfig/build_tx`).
7) Optional: folder merge (`plugins/training/sft/`) once imports are stable.

## Expected result

- A staged refactor plan that reduces duplication while keeping the training pipeline runnable at each phase.
- Shared modules are isolated under `plugins/` (non-invasive to upstream code).
- `python -m pytest -q` exits `0`.

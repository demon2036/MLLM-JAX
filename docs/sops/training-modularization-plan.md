# Training Modularization Plan (non-invasive via plugins)

- **Title**: SOP: Plan and stage a modular refactor of the training pipeline (keep upstream code untouched)
  **Prereqs**: Repo checkout; basic familiarity with JAX/Flax + `Mesh`/sharding; (optional) TPU VM access if you will run smoke tests
  **Environment (verified)**: Ubuntu Linux; Python `3.12.2`

## Scope / Goal

This SOP captures a deterministic way to:

- Identify the current “training path” in this repo (entrypoints → state → train_step).
- Design a modular training architecture *without invasive edits* (put new code under `plugins/`).
- Stage the refactor in small milestones with validation gates (CPU first, then TPU).

This is a planning SOP: it records the repo touchpoints and the rollout strategy. It is intentionally not an implementation guide for the final refactor.

## Steps (commands actually used in this repo)

- List repo root and key directories:
  - `ls -la`
  - `ls -la docs docs/sops 2>/dev/null || true`

- Confirm the jit8 entrypoint is plugin-based (so modularization work stays under `plugins/`):
  - `cat test_jit8.py`

- Inspect the current jit8 plugin surface:
  - `sed -n '1,260p' plugins/jit8_train/run.py`
  - `sed -n '1,260p' plugins/jit8_train/config.py`
  - `sed -n '1,260p' plugins/jit8_train/sampling.py`

- Inspect the reusable contracts layer (schema validation):
  - `sed -n '1,260p' plugins/api/training/schemas/grpo_batch.py`
  - `sed -n '1,260p' plugins/api/training/rl.py`

- Inspect the GSM8K/GRPO project runner:
  - `sed -n '1,260p' projects/gsm8k_grpo/jax/train.py`

- Smoke-check the CLI config path (no JAX required for print-config):
  - `python test_jit8.py --print-config`
  - `python test_jit8.py --print-config --set validate_schema=true`
  - `python projects/gsm8k_grpo/scripts/run_train.py --print-config`

- Run the lightweight local regression test (no JAX required):
  - `python tests/test_jit8_schema_and_cli.py`

## Expected Result

- You have a precise map of the current training surface area:
  - Smoke train entrypoint: `scripts/run_smoke_train_qwen25_7b.py`
  - Core primitives: `training2.py` (`get_state`, `training_step`, reward/advantage helpers)
  - Loss module: `MLLM_JAX/train_modules/__init__.py` (`TrainGRPOModule`)
  - Model + sampler: `MLLM_JAX/sample/sample_state_right_padding2.py` (`get_model`, `Sampler`)
  - Mesh/sharding helpers: `MLLM_JAX/utils.py` (`get_jax_mesh2`, partition rules, checkpoint helpers)
  - jit8 entrypoint: `test_jit8.py` → `plugins/jit8_train/run.py`
  - Training contracts (schema): `plugins/api/training/schemas/grpo_batch.py`
  - GSM8K/GRPO entrypoint: `projects/gsm8k_grpo/scripts/run_train.py` → `projects/gsm8k_grpo/jax/train.py`

- You can draft a staged modularization plan that keeps new code in `plugins/` and introduces new scripts under `scripts/` (instead of editing upstream logic directly).

## Troubleshooting

- If `rg` (ripgrep) is not available:
  - Use `grep -RIn --include='*.py' ...` for repository search.

## References

- `docs/sops/repo-setup.md`
- `docs/sops/grpo-gsm8k-jit8-yaml-config.md`

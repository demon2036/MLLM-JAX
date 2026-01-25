# Tiled cross-entropy Pallas kernel (fix + validation)

## Goal

- Fix correctness of the TPU Pallas tiled cross-entropy/logprob kernel (forward + backward).
- Keep everything non-invasive under `plugins/`.
- Prepare for TPU gradcheck on `mllm-jax-v6e-8-spot-260124132428` with `wandb_mode=online`.

## Key finding (root cause)

- **BlockSpec `index_map` semantics were wrong**: for `pl.BlockSpec((..., BLOCK), index_map)`, the `index_map`
  should return **block indices** (e.g. `(b, t, k)`), not element start offsets (e.g. `(b, t * BLOCK_T, k * BLOCK)`).
- The previous code multiplied by block sizes twice, leading to **out-of-bounds reads** under strict TPU interpret
  mode and silent wrong numerics under the loose `interpret=True` path.

## Reference comparison (MaxText)

- Cloned: `workdir/maxtext` @ `b646a53` (`https://github.com/google/maxtext.git`)
- File: `workdir/maxtext/src/MaxText/vocabulary_tiling.py`
- MaxText’s “vocab tiling” here is **token-chunking** + `jax.lax.scan` + `custom_vjp` (recompute logits in bwd),
  not a Pallas fused kernel. It’s still a good baseline pattern for “no full-logits residency”.

## Evidence (commands actually run)

### Local unit tests

- `python -m pytest -q`
  - Result: `33 passed, 1 warning` (exit `0`)

## Files changed

- `plugins/training/kernels/tiled_cross_entropy_pallas.py`
  - Fix BlockSpecs to use `(b, t, k)` block indices.
  - Pass through `interpret` (allow `pltpu.InterpretParams`) instead of forcing `bool`.
- `plugins/training/kernels/grpo_loss_pallas.py`
  - Same BlockSpec fix for the GRPO kernel (it used the same pattern).
  - Pass through `interpret`.
- `tests/test_tiled_cross_entropy_pallas_kernel.py`
  - Use `pltpu.InterpretParams(out_of_bounds_reads='raise')` for strict checking.
- `tests/test_grpo_pallas_kernel.py`
  - Same strict interpret params; also parametrize vocab to cover `blocks > 2`.

## Next: TPU validation (to fill)

### TPU validation (completed)

- TPU VM: `mllm-jax-v6e-8-spot-260124132428` (zone `us-east1-d`)
- Repo checkout: `cef0a7e`

#### Commands actually run (local → TPU)

- Sync repo via Git:
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d \"$REPO_DIR/.git\" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout cef0a7e; git status -sb; git rev-parse --short HEAD'`
- Sync secrets (`WANDB_API_KEY`) to TPU:
  - `scripts/sync_env_to_tpu_vm.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --worker all`
- TPU Python env bootstrap (PEP 668 → venv):
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; apt-get update -y; apt-get install -y python3.12-venv'`
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; rm -rf \"$VENV\"; python3 -m venv \"$VENV\"; \"$VENV/bin/python\" -V; \"$VENV/bin/pip\" --version'`
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U pip; \"$VENV/bin/pip\" install -U \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html'`
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U flax optax chex transformers tqdm wandb pyyaml einops \"huggingface_hub[hf_transfer]\" safetensors'`
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U torch'`
  - `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --command 'set -euo pipefail; VENV=/root/venvs/mllm-jax; \"$VENV/bin/pip\" install -U webdataset datasets math_verify'`

#### Gradcheck command (W&B online)

- `scripts/ssh_tpu_vm_root.sh --name mllm-jax-v6e-8-spot-260124132428 --zone us-east1-d --env-file /root/.env --command 'set -euo pipefail; cd /root/MLLM-JAX; /root/venvs/mllm-jax/bin/python -u scripts/cross_entropy_kernel_gradcheck.py --config plugins/training/configs/cross_entropy_kernel_gradcheck_qwen25_1p5b.yaml'`

#### Result (exit 0 + tight diffs)

- W&B run: `https://wandb.ai/johntitordemon2036/mllm-jax-ce-kernel/runs/1815sclb`
- Key metrics printed by the script:
  - `abs_diff_loss`: `0.0`
  - `fwd/logp_max_abs`: `9.5367431640625e-07`
  - `dlogits_max_abs`: `4.76837158203125e-07`
  - `dlogits_max_rel`: `0.006723258178681135`
  - `kernel/block_size`: `2048`
  - `kernel/time_block`: `8`

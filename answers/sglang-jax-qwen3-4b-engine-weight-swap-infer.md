# sglang-jax (JAX/TPU): Qwen3-4B Engine weight-swap inference

## Goal

On a TPU `v4-8`, validate that we can:

1) Initialize a `sglang-jax` `Engine` with `load_format="dummy"` (fast shape/compile setup).
2) Load real `Qwen/Qwen3-4B` weights (HF safetensors) into the in-memory model.
3) Replace the `Engine` runtime parameters (`model_state_leaves`) in-place.
4) Run a single inference with prompt `你是谁`.

This is a **non-invasive** approach: we do not modify upstream `sglang-jax`; we only add code under `plugins/` and a runnable test script under `tests/`.

## What was already in this repo (baseline)

- Upstream `sglang-jax` is cloned under `workdir/sglang-jax` (gitignored local scratch) at commit:
  - `bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
- A TPU SOP exists that describes the same validation:
  - `docs/sops/tpu-sglang-jax-qwen3-4b-engine-weight-swap-infer.md`
- The core implementation pieces already existed:
  - `plugins/sglang_jax_inference/engine_weight_swap.py`
  - `tests/run_sglang_jax_qwen3_4b_param_swap.py`

## How sglang-jax enables runtime weight swapping

### 1) Engine composition exposes the in-process scheduler

- File: `workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
- `Engine` keeps an in-process reference `self.scheduler_info` when started with `enable_single_process=True`.
- That makes it possible (inside the same Python process) to reach:
  - `engine.scheduler_info["scheduler"].tp_worker.worker.model_runner`

### 2) ModelRunner exports a “leaf list” that is fed into the jitted function

- File: `workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
- In `ModelRunner.initialize_jit()`:
  - `model_state` is extracted via `nnx.split(self.model)`
  - `model_state` is flattened via `jax.tree_util.tree_flatten(model_state)`
  - the flattened list is stored as `self.model_state_leaves`
- The jitted forward path reconstructs the model state each call using:
  - `jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)`

**Key implication**: if we replace `self.model_state_leaves` with a same-structure list (same tree-def + leaf count/shapes), the compiled function will execute with the new parameters, without recompiling.

### 3) Qwen3 weight loading is defined by explicit HF→internal mappings

- File: `workdir/sglang-jax/python/sgl_jax/srt/models/qwen3.py`
- `QWen3ForCausalLMModel.load_weights()` builds a `WeightLoader(...)` and calls:
  - `loader.load_weights_from_safetensors(weight_mappings)`
- The mapping is produced in `_create_qwen3_weight_mappings()` and layer helpers.

## What we run on TPU (the actual validation)

### 1) Runtime swap helper (plugin)

- File: `plugins/sglang_jax_inference/engine_weight_swap.py`
- Responsibilities:
  1) Download a HF snapshot directory (`snapshot_download`) when given a model id.
  2) Locate the `model_runner` from an `Engine` (requires `enable_single_process=True`).
  3) Temporarily update `model_config.model_path` to the snapshot directory.
  4) Call `model_runner.model.load_weights(model_config)` (sglang-jax’s official loader).
  5) Rebuild and flatten the nnx state; overwrite `model_runner.model_state_leaves`.
  6) Sanity-check: leaf count must remain identical (same tree structure).

### 2) End-to-end param-swap inference script

- File: `tests/run_sglang_jax_qwen3_4b_param_swap.py`
- Pipeline:
  1) Create `Engine(... load_format="dummy", enable_single_process=True, device="tpu")`
  2) Print JSON `engine_ready_dummy` with config + `num_model_state_leaves`
  3) `swap_engine_weights_from_hf(engine, "Qwen/Qwen3-4B", cache_dir=...)`
  4) Print JSON `weights_swapped`
  5) `engine.generate(prompt="你是谁", ...)`
  6) Print JSON `generate_result`

#### Improvements made in this turn

In `tests/run_sglang_jax_qwen3_4b_param_swap.py`:

- Added `--prompt` CLI override (default: `你是谁`).
- Switched HF cache/download dirs to be repo-relative (`<repo>/workdir/...`) instead of hardcoding `/root/MLLM-JAX/...`.
- Added extra metadata fields to the `engine_ready_dummy` JSON:
  - `model_id`, `tp_size`, `dp_size`, `dtype`, `load_format`, `download_dir`, `hf_cache_dir`, `sgl_jax_version`
- Made JSON logs more robust for streaming/grepping:
  - `flush=True` on all phase prints.
- Added an explicit JSON error phase `weights_swap_error` with a retry hint when HF snapshot download fails.

## TPU execution record (to be filled after running)

This section will be updated with:

- TPU name, zone, runtime image
- Python/JAX versions
- The **exact** JSON output lines for:
  - `engine_ready_dummy`
  - `weights_swapped`
  - `generate_result` (including the model’s answer text)

## Conclusion (to be filled after running)

We will conclude whether Engine parameter replacement is feasible for `Qwen/Qwen3-4B` on TPU by checking:

- Does `weights_swapped` succeed without changing leaf structure?
- Does `generate_result.text` return a non-empty answer for `你是谁`?


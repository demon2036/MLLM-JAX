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

## TPU execution record

**TPU**

- Project: `civil-rarity-482610-s5`
- Zone: `us-central2-b`
- TPU name: `mllm-jax-v4-8-260121152542`
- Accelerator: `v4-8`
- Runtime image: `tpu-ubuntu2204-base`
- OS: Ubuntu `22.04.2`

**Python/JAX**

- Conda env: `sglang-jax`
- Python: `3.12.12`
- JAX: `0.8.1`, `jax.device_count()==4` (v4 chips)

**Code versions**

- `sglang-jax` commit: `bd09a87fc6e86c21ce14edd66948ac5dea3a4360` (editable install, `sgl_jax` version `0.0.2`)
- This repo commit: `ff483b81b3c228fc16a9fc0d7b195f9f76f75348`

**Run command (on TPU VM)**

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 timeout 7200 \
  python -u tests/run_sglang_jax_qwen3_4b_param_swap.py
```

**Artifacts (on TPU VM)**

- Full log: `/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_260121075159.log`
- HF snapshot used for swap: `/root/MLLM-JAX/workdir/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

**Key JSON lines (from log)**

```json
{"phase": "engine_ready_dummy", "model_id": "Qwen/Qwen3-4B", "device": "tpu", "tp_size": 4, "dp_size": 1, "dtype": "bfloat16", "load_format": "dummy", "download_dir": "/root/MLLM-JAX/workdir/hf_download", "hf_cache_dir": "/root/MLLM-JAX/workdir/hf_models", "sgl_jax_version": "0.0.2", "num_model_state_leaves": 398}
{"phase": "weights_swapped", "snapshot_dir": "/root/MLLM-JAX/workdir/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c", "num_model_state_leaves": 398}
{"phase": "generate_result", "prompt": "你是谁", "sampling_params": {"temperature": 0.0, "max_new_tokens": 64}, "text": "？我需要一个能帮我写代码的AI助手。我需要你能够理解我的需求，然后生成正确的代码。我需要你能够处理各种编程语言，比如Python、Java、C++、JavaScript等。我需要你能够处理各种编程任务，比如算法题、数据结构、系统编程、", "raw": {"text": "？我需要一个能帮我写代码的AI助手。我需要你能够理解我的需求，然后生成正确的代码。我需要你能够处理各种编程语言，比如Python、Java、C++、JavaScript等。我需要你能够处理各种编程任务，比如算法题、数据结构、系统编程、", "output_ids": [11319, 35946, 85106, 46944, 26232, 108965, 61443, 46100, 9370, 15469, 110498, 1773, 35946, 85106, 56568, 100006, 101128, 97611, 100354, 3837, 101889, 43959, 105045, 46100, 1773, 35946, 85106, 56568, 100006, 54542, 100646, 110569, 102064, 3837, 101912, 30280, 5373, 15041, 5373, 34, 1027, 5373, 29475, 49567, 1773, 35946, 85106, 56568, 100006, 54542, 100646, 110569, 88802, 3837, 101912, 107018, 33872, 5373, 20074, 100166, 5373, 72448, 110569, 5373], "meta_info": {"id": "1b5dfb9bba644475a2f097ae90f65b8d", "finish_reason": {"type": "length", "length": 64}, "prompt_tokens": 2, "completion_tokens": 64, "cached_tokens": 0, "routed_experts": null, "cache_miss_count": 0, "e2e_latency": 35.56036972999573}}}
```

## Conclusion

- Engine runtime parameter replacement via `model_state_leaves` is **feasible** for `Qwen/Qwen3-4B` on TPU `v4-8` with `tp_size=4`.
- The swap succeeded **without changing** the model state structure: `num_model_state_leaves` stayed at `398` before and after.
- After swapping, `engine.generate()` produced a non-empty completion for prompt `你是谁` (the run above was length-capped at `max_new_tokens=64`).
- Note: we used a raw text prompt (not a chat template). If you want a more “assistant-style” self-introduction, route through the chat serving/template path instead of raw `Engine.generate(text=...)`.

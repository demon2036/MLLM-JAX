# SOP: Clone and inspect Tunix → sglang-jax rollout integration

- **Title**: SOP: Clone and inspect Tunix → sglang-jax rollout integration
  **Prereqs**: Ubuntu Linux; `git`; outbound network access; Python `3.12.2`

## Steps (commands actually used)

### 1) Clone into repo-local `workdir/` (gitignored)

- `cd /home/john/works/MLLM-JAX-mllm-jax-sglang`
- `mkdir -p workdir`
- `git clone --depth 1 https://github.com/sgl-project/sglang-jax.git workdir/sglang-jax`
- `git clone --depth 1 https://github.com/google/tunix.git workdir/tunix`

Record the inspected revisions:

- `git -C workdir/sglang-jax rev-parse HEAD`
- `git -C workdir/tunix rev-parse HEAD`

### 2) Locate Tunix rollout entrypoints and engine wiring

- `sed -n '1020,1220p' workdir/tunix/scripts/grpo_demo_llama3_qwen2.py`
- `sed -n '760,920p' workdir/tunix/tunix/rl/rl_cluster.py`
- `sed -n '1,260p' workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
- `sed -n '1,320p' workdir/tunix/tunix/generate/sglang_jax_sampler.py`
- `sed -n '150,220p' workdir/tunix/tunix/generate/sglang_jax_sampler.py`  # mesh -> tp_size/device_indexes + load_format logic

### 3) Locate sglang-jax Engine internals relevant to rollout + weight hot-swap

- `sed -n '1,260p' workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
- `sed -n '120,260p' workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
- `sed -n '1,260p' workdir/sglang-jax/python/sgl_jax/srt/server_args.py`
- `sed -n '220,360p' workdir/sglang-jax/python/sgl_jax/srt/managers/tokenizer_manager.py`
- `sed -n '200,320p' workdir/sglang-jax/python/sgl_jax/srt/managers/scheduler.py`
- `sed -n '1470,1560p' workdir/sglang-jax/python/sgl_jax/srt/managers/scheduler.py`  # thread mode returns scheduler object
- `sed -n '1,240p' workdir/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py`  # ModelWorker.model_runner

## Expected Result

- You can point to the full chain:
  - Tunix RLCluster generation path: `workdir/tunix/tunix/rl/rl_cluster.py`
  - sglang-jax rollout wrapper: `workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
  - Engine callsite: `workdir/tunix/tunix/generate/sglang_jax_sampler.py`
  - Engine architecture + single-process mode: `workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
  - Weight hot-swap mechanism (`model_state_leaves`): `workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`

## Troubleshooting

- If `workdir/` already exists from prior experiments:
  - Remove and re-clone: `rm -rf workdir/sglang-jax workdir/tunix`

## References

- Tunix: https://github.com/google/tunix
- sglang-jax: https://github.com/sgl-project/sglang-jax
- Answer write-up in this repo: `answers/tunix-sglang-jax-rollout-integration.md`

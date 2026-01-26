# GRPO Pallas kernel: multi-device (shard_map) + TPU 100-step A/B train

## Goal

- Upgrade the existing logits-level GRPO Pallas kernel so it:
  - runs correctly on **single device** (baseline correctness),
  - runs correctly and efficiently on **multi-device** (TPU v6e-8) via `jax.experimental.shard_map`,
  - is faster and uses less peak memory than the pure-JAX baseline (`log_softmax`/`softmax` paths),
  - can be enabled/disabled via **YAML config** (W&B reproducible).
- Validate end-to-end on TPU `mllm-jax-v6e-8-spot-260124132428` with:
  - GRPO kernel gradcheck (W&B online, exit 0)
  - GRPO/GSM8K training 100 steps A/B (baseline vs kernel), W&B online

Update (2026-01-26):
- The user-provided TPU VM name was not found in the current GCP project; the
  end-to-end validation below is executed on a fresh single-host `v6e-8` TPU VM:
  - TPU: `mllm-jax-v6e-8-grpo-kernel-260127045556` (zone `europe-west4-a`)

## Completion criteria

- TPU kernel gradcheck exits `0` with diffs within tolerances; W&B run URL recorded.
- TPU baseline 100-step run exits `0`; W&B run URL recorded.
- TPU kernel 100-step run exits `0`; W&B run URL recorded.
- Metrics are consistent between baseline and kernel runs (within expected noise), and kernel run shows improved:
  - `time/train/step_avg_last10_s` (lower is better)
  - peak memory (as available via logs/JAX profile/TPU metrics)
- SOPs updated with **exact commands actually run**.

## Work log (evidence)

### Reference repos (gitignored clones under `workdir/`)

- `workdir/jax` @ `a0aedf1` (`https://github.com/jax-ml/jax.git`)
- `workdir/maxtext` @ `b646a53` (`https://github.com/google/maxtext.git`)
- `workdir/liger-kernel` @ `9eb9a1e` (`https://github.com/linkedin/Liger-Kernel.git`)
- `workdir/unsloth` @ `4cb7229` (`https://github.com/unslothai/unsloth.git`)

### Notes to carry into implementation

- Current gradcheck scripts force `device0` because Mosaic kernels are not SPMD auto-partitionable.
- Target solution: wrap kernels with `jax.experimental.shard_map` (MaxText/Splash pattern) so multi-device works.
- Liger GRPO Triton kernel patterns to mirror:
  - streaming logsumexp over vocab tiles
  - unclipped-vs-clipped branch for gradients
  - optional KL term (beta)
- MaxText SplashAttention multi-device pattern to mirror:
  - keep the heavy Pallas kernel single-device and wrap with `jax.shard_map`
  - use a kernel object’s `manual_sharding_spec(...)` to derive `shard_map` partition specs
  - keep `check_vma=False` by default for perf (`MaxText/layers/attention_op.py`)
- Unsloth selective log-softmax pattern to mirror:
  - stream over vocab tiles (`BLOCK_N`) and keep `(m_i, l_i)` accumulators (Flash-style)
  - avoid materializing full `log_softmax` for per-token logp
  - (GPU-only) use `tl.load(LOGITS + ids)` for chosen logit (TPU Pallas may need onehot trick)

### Concrete reference pointers (cloned under this repo’s `workdir/`)

- JAX SplashAttention kernel:
  - `workdir/jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py`
  - key ops inside kernel use `lax.dot_general(..., preferred_element_type=float32)`
- MaxText shard_map calling pattern:
  - `workdir/maxtext/src/MaxText/layers/attention_op.py` (`wrap_flash_attention` uses `jax.shard_map`)
  - `workdir/maxtext/src/MaxText/kernels/splash_attention_kernel.py` (`SplashAttentionKernel.manual_sharding_spec`)
- Liger GRPO loss kernels:
  - `workdir/liger-kernel/src/liger_kernel/ops/grpo_loss.py`
- Unsloth RL selective log-softmax:
  - `workdir/unsloth/unsloth/models/rl.py` (injects replacement code; actual kernels in `unsloth_zoo`)

### GRPO training call chain (repo)

- Entry: `scripts/run_grpo_gsm8k_training.py` → `plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k()`
- State construction: `plugins/training/runner/grpo_gsm8k.py` imports `training2.get_state()`
- Baseline loss module: `training2.py:get_state()` hardcodes `MLLM_JAX.train_modules.TrainGRPOModule` (pure JAX ops)
- Update step: `plugins/training/update/train_step.py:training_step()` calls `state.apply_fn(...)`
- Implication: to A/B compare kernel vs baseline, we must add a **config-driven switch** in the state/module wiring:
  - baseline: `TrainGRPOModule` (existing behavior)
  - kernel: `plugins.training.grpo.TrainGRPOModulePallas` (multi-device shard_map wrapper around Pallas kernel)

## Suspected bottlenecks (to validate on TPU)

1) **Backward `pallas_call` barrier forces `dlogits` materialization**
- The current GRPO kernel defines a custom VJP whose backward path is another `pl.pallas_call(...)` that returns `dlogits`.
- On TPU, this makes the gradient w.r.t logits a “custom call output” which XLA cannot fuse into the downstream matmuls (e.g. LM head / upstream layers).
- Hypothesis: this increases peak HBM and is the reason the current kernel needs `micro_batch_size_per_device=2` on `v6e-8`, while baseline can run with larger micro-batches.

2) **Entropy metric path materializes `softmax(logits)`**
- Both baseline and `TrainGRPOModulePallas` compute:
  - `probs = softmax(logits / temperature)` → `token_entropy = -sum(probs * log(probs))`
- This is `[B,T,V]` work and can dominate memory/time in the loss module, especially when `V` is large.
- Hypothesis: replacing this with a streaming entropy/logsumexp computation (no `probs` tensor) is required for a clear win.

### Commands / runs (to fill as executed)

- Local:
  - `python -m pytest -q`
- TPU (repo: `/root/MLLM-JAX`, conda env: `mllm-jax`, W&B: `wandb_mode=online`):
  - Gradcheck (exit `0`):
    - `python -u scripts/grpo_kernel_gradcheck.py --config plugins/training/configs/grpo_kernel_gradcheck_qwen25_1p5b.yaml`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-kernel/runs/y836ygee`
  - Baseline GRPO/GSM8K 100-step (exit `0`):
    - `python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/aovd31pm`
    - `wandb-summary.json` highlights:
      - `time/train/step_avg_last10_s=11.578276114599998`
      - `time/train/update_s=3.932779269999628` (last step)
      - `time/train/rollout_generate_s=8.76871432400003` (last step)
      - `eval/reward/total/mean=1.796875` (final eval snapshot)
  - Kernel GRPO/GSM8K 100-step (attempt 1, exit `1`):
    - `python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas_kernel.yaml`
    - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/9lcvmn1d`
    - Error: `AssertionError` from `jax._src.pallas.pallas_call._pallas_call_jvp_rule` triggered by `entropy_pallas`.
  - Fix + restart:
    - Commit: `5e8ff7e` (`fix: make entropy pallas jvp-safe`)
    - Re-run kernel GRPO/GSM8K 100-step (attempt 2, exit `0`):
      - W&B: `https://wandb.ai/johntitordemon2036/mllm-jax-grpo-gsm8k/runs/ucesd0oc`
      - `wandb-summary.json` highlights:
        - `time/train/step_avg_last10_s=13.63050850999998`
        - `time/train/update_s=4.2935676090000925` (last step)
        - `throughput/train/valid_tokens_per_s_update=8186.432170375367` (last step)
        - `train-other/total_valid_token_count=35149` (last step; longer completions)
        - `eval/reward/total/mean=1.828125` (final eval snapshot)

  - Peak-memory probe (single-device synthetic logits, `jax.devices()[0].memory_stats()`):
    - Baseline (JAX ops: `log_softmax` + `softmax` entropy), B=8 T=1024 V=16384:
      - `peak_bytes_in_use=1883052032` (≈1.75 GiB)
    - Kernel (Pallas loss + Pallas entropy, `bwd_impl=jax`), B=8 T=1024 V=16384:
      - `peak_bytes_in_use=2956557824` (≈2.75 GiB)

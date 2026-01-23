# SOP: TPU v4-8 GRPO JAX recompile check (sglang backend)

- **Title**: SOP: Check JAX recompiles in GRPO runner (TPU v4-8, sglang)
- **Prereqs**: `gcloud` authenticated; TPU VM reachable; repo pushed to GitHub; conda env `sglang-jax` present on TPU VM
- **Environment (verified)**:
  - Project: `civil-rarity-482610-s5`
  - Zone: `us-central2-b`
  - TPU VM: `mllm-jax-v4-8-260122100610`
  - Branch: `mllm-jax-sglang`
  - Commit: `584af84`
  - Python: `3.12.12`
  - JAX: not checked in this run
  - SSH hostkey (plink): `SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`
  - Log: `/root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_20260123_040740.log`

## Steps (commands actually used)

### 1) (Local) TPU SSH reachability check

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command "whoami; hostname; date"
```

### 2) (TPU) Git sync to the target commit

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; git fetch --all --prune; git checkout mllm-jax-sglang; git reset --hard origin/mllm-jax-sglang; git clean -fd; git rev-parse --short HEAD'
```

### 3) (TPU) Activate conda env

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; python -V'
```

### 4) (TPU) Clear libtpu lockfile

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'rm -f /tmp/libtpu_lockfile || true; ls -la /tmp/libtpu_lockfile || true'
```

### 5) (TPU) Create log directory

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'mkdir -p /root/MLLM-JAX/workdir/logs/jax_recompile; ls -la /root/MLLM-JAX/workdir/logs/jax_recompile'
```

### 6) (TPU) Run GRPO with JAX compile logging

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate sglang-jax; cd /root/MLLM-JAX; export JAX_LOG_COMPILES=1; export JAX_EXPLAIN_CACHE_MISSES=1; export PRINT_JAX_COMPILE_FLAGS=1; export PRINT_JAX_JIT_FNS=1; export WANDB_MODE=disabled; export TOKENIZERS_PARALLELISM=false; TS=$(date -u +%Y%m%d_%H%M%S); LOG=/root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_${TS}.log; set +e; python -u scripts/run_grpo_gsm8k_training.py --config plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml --set steps=3 --set rollout.backend=sglang --set model_path=Qwen/Qwen2.5-3B-Instruct 2>&1 | tee "$LOG"; status=${PIPESTATUS[0]}; set -e; echo "log_file=$LOG"; echo "exit_status=${status}"; exit ${status}'
```

### 7) (TPU) Inspect log for compile and cache-miss lines

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'grep -n Compiling /root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_20260123_040740.log | head -n 5'

gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'grep -n TRACING /root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_20260123_040740.log | head -n 5'

gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'grep -n Compiling /root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_20260123_040740.log | tail -n 1'

gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag="-batch" \
  --ssh-flag="-hostkey" \
  --ssh-flag="SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s" \
  --command 'grep -n "^step=" /root/MLLM-JAX/workdir/logs/jax_recompile/grpo_sglang_recompile_20260123_040740.log | tail -n 1'
```

## Expected Result

- The log prints `jax_log_compiles=1` and `jax_explain_cache_misses=1`, plus `train_fn=` and `slice_data=` lines.
- The run emits `step=0`, `step=1`, `step=2` and ends with `exit_status=0`.
- The last `Compiling` log line appears before the first `step=` line, indicating no per-step recompilation during steps 1–2.

## Troubleshooting

- Hostkey prompt from `gcloud` on Windows: add `--ssh-flag="-hostkey"` + `--ssh-flag="SHA256:..."` as shown above.
- If `step=` lines never appear, check `log_file=...` output and inspect the log directly on TPU.
- If `exit_status` is non-zero, re-run with fewer steps and confirm model download succeeds.

## References

- `plugins/training/runner/grpo_gsm8k.py`
- `docs/sops/tpu-vm-repo-sync.md`

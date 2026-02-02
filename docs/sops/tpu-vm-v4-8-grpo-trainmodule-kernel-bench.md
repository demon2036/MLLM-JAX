# SOP: TPU v4-8 TrainGRPOModule baseline vs fused GRPO kernel microbench（B=1, L=4096, V=151936；含 forward + value_and_grad）

- **Title**: SOP: Run TPU microbenchmark comparing legacy TrainGRPOModule-style math vs fused `grpo_loss_logp_entropy` kernel (batch=1, seq_len=4096, vocab=151936)
  **Prereqs**:
  - `gcloud` installed + authenticated; TPU API enabled
  - `scripts/create_tpu_vm.sh` available
  - This run was executed **without** `WANDB_API_KEY`, so `WANDB_MODE=disabled` is used throughout
  **Environment (verified in this run)**:
  - Zone: `us-central2-b`
  - TPU name (v4-8): `grpo-trainmodule-kernel-v4-8-260203033924`
  - Repo commit on TPU: `a9f6046`
  - Conda env: `/root/miniconda3/envs/mllm-jax` (python=3.12)

## Steps（commands actually used）

### 0) Try creating v6e-8 first (FAILED: quota limit 0)

- v6e spot create (FAILED):
  - cmd:

```bash
scripts/create_tpu_vm.sh --type v6e-8 --zone us-central2-b --name grpo-trainmodule-kernel-v6e8-260203033702
```

  - error (excerpt):

```text
Error: Quota limit 'TPUV6EPreemptiblePerProjectPerZoneForTPUAPI' ... Limit: 0
```

- v6e on-demand create (FAILED):
  - cmd:

```bash
scripts/create_tpu_vm.sh --type v6e-8 --zone us-central2-b --name grpo-trainmodule-kernel-v6e8-ondemand-260203033801 --on-demand
```

  - error (excerpt):

```text
Error: Quota limit 'TPUV6EPerProjectPerZoneForTPUAPI' ... Limit: 0
```

### 1) Create a v4-8 spot TPU (SUCCESS)

- cmd:

```bash
scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name grpo-trainmodule-kernel-v4-8-260203033924
```

### 2) Sync repo to TPU VM (Git) and checkout commit

- Repo path on TPU: `/root/MLLM-JAX`
- Branch: `grpo-fused-kernel`
- Commit: `a9f6046`
- Note: clone/checkout commands were executed on the TPU VM; not captured inline here. See:
  - `docs/sops/tpu-vm-repo-sync.md`

### 3) Bootstrap env + install deps (TPU VM)

- Conda env: `/root/miniconda3/envs/mllm-jax` (python=3.12)
- W&B: `WANDB_MODE=disabled`
- cmd (from inside the conda env):

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U -r requirements-tpu.txt
```

### 4) Confirm TPU devices (TPU VM)

- This run printed `jax.devices()` before benchmarking (output not captured in this SOP).
  For a reproducible check, see: `docs/sops/tpu-alive-check.md`.

### 5) Run microbench (baseline TrainGRPOModule math vs fused kernel)

- cmd:

```bash
PJRT_DEVICE=TPU WANDB_MODE=disabled python tests/grpo_fused_kernel/bench_grpo_trainmodule_kernel.py --mode both --batch 1 --seq-len 4096 --vocab 151936 --iters 10
```

- output (key snippets):

```text
forward timing: baseline 5.322 ms/iter, fused 9.744 ms/iter
value_and_grad timing: baseline 18.839 ms/iter, fused 15.923 ms/iter
baseline forward temp_size_in_bytes: 1244917760
fused forward temp_size_in_bytes: 1244885504
baseline value_and_grad temp_size_in_bytes: 1245192704
fused value_and_grad temp_size_in_bytes: 1244788736
```

## Conclusion（达标点）

- forward：fused temp memory 基本持平，但时间更慢（baseline 5.322 ms/iter；fused 9.744 ms/iter）。
- training 关键路径（`value_and_grad`）：fused **更快**，且 temp memory 更低（baseline 18.839 ms/iter；fused 15.923 ms/iter；temp_size_in_bytes: 1245192704 → 1244788736）。

## Troubleshooting

- v6e-8 quota limit 0（spot/on-demand）：需要为 `tpu.googleapis.com` 申请对应 quota，或切换到 v6e quota 非 0 的 zone；否则先用 v4-8。
- Quota 通过后：复用本 SOP，改用 `--type v6e-8` 创建 TPU，再在 v6e-8 上重跑第 5 步 microbench。

## References

- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-bootstrap.md`
- `docs/sops/tpu-alive-check.md`

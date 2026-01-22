# SOP: TPU 上 drop/rebuild sglang-jax KV cache（Qwen2.5-3B）并验证 HBM 释放

- **Title**: SOP: Drop & rebuild sglang-jax KV cache buffers on TPU (Qwen2.5-3B)
- **Goal**: rollout 推理后释放 KV cache 的 HBM，占用训练/其他任务完成后再重建 KV cache 并继续推理

## Prereqs

- 本地已安装并登录 `gcloud`，且具备目标 project/zone 权限
- 已有可用 TPU VM（本 SOP 复用现有 `v4-8`，不销毁）
- TPU VM 上已安装 Miniconda，并已有可用 conda env：`sglang-jax`
- TPU VM 可访问 Hugging Face 下载权重

## Env（本次实测）

- Project: `civil-rarity-482610-s5`
- Zone: `us-central2-b`
- TPU VM: `mllm-jax-v4-8-260122100610`
- JAX: `0.8.1`
- sglang-jax commit（固定）: `bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
- 本仓库分支: `mllm-jax-sglang`
- 本仓库 commit: `eb9c774`
- SSH hostkey（plink）: `SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`
- Log: `/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_kv_drop_rebuild_20260122_083116.log`

## Steps（本次实际执行命令）

### 1)（本地）SSH 连到 TPU VM

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag='-batch' \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s' \
  --command 'whoami; hostname; date'
```

### 2)（TPU VM）Git 同步到最新 commit

```bash
cd /root/MLLM-JAX
git fetch --all --prune
git checkout mllm-jax-sglang
git reset --hard origin/mllm-jax-sglang
git clean -fd
git rev-parse --short HEAD
```

### 3)（TPU VM）运行 KV drop/rebuild 实验脚本

说明：
- `--max-total-tokens 32768`：把 KV cache 预分配做大，便于观察“drop 释放”的 HBM 差值
- `--kv-drop-rebuild`：在一次 `generate()` 后 drop KV buffers，再额外分配第二套 params，然后 rebuild KV，再次 `generate()`

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX

LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_kv_drop_rebuild_$(date -u +%Y%m%d_%H%M%S).log
python -u tests/run_sglang_jax_qwen25_3b_mllm_param_swap.py \
  --prompt "Hello" \
  --max-total-tokens 32768 \
  --kv-drop-rebuild \
  --verify-param-sharing \
  2>&1 | tee "$LOG"
```

## Expected Result（本次实测结果）

### 1) KV cache drop 确实释放了 HBM（bytes_in_use_sum 下降）

本次 `max_total_tokens=32768` 时，sglang-jax log 显示：
- `JAX Fused KV Cache allocated... Fused KV size: 2.25 GB`

对应的 TPU 侧 `bytes_in_use_sum`（4 devices 求和）在关键 phase 的变化：
- drop 前（`flush_cache_before_kv_drop`）：`8957558784` bytes（≈ 8.34 GiB）
- drop 后（`kv_cache_dropped`）：`6541565952` bytes（≈ 6.09 GiB）
- 差值：`2415992832` bytes（≈ 2.25 GiB）

并且 drop 输出确认 `.delete()` 生效：
- `deleted_buffers=36`, `missing_delete_method=0`

### 2) drop 后可额外分配第二套 params（2 params without KV）

在 KV drop 后再 build 第二份 `param_dict2`：
- `sglang_param_dict2_ready`: `bytes_in_use_sum=12790538240` bytes（≈ 11.91 GiB）

### 3) rebuild KV 后可继续推理（且没有再次“首轮编译”级别的卡顿）

rebuild 后：
- `kv_cache_rebuilt`: `bytes_in_use_sum=15206531072` bytes（≈ 14.16 GiB）
- 与 `sglang_param_dict2_ready` 的差值同样为 `≈ 2.25 GiB`（等于 KV cache 大小）

第二次推理：
- `generate_result_after_kv_rebuild` 成功返回，且 `e2e_latency≈0.117s`（明显快于首次 `generate()` 的 `≈25s`，说明没有重新走“首轮编译”级别的开销）

## Troubleshooting

- 如果 `kv_cache_dropped` 后 `bytes_in_use_sum` 没明显下降：
  - 检查 drop 是否走到了 `.delete()`（`missing_delete_method` 是否为 0）
  - TPU allocator 可能会做复用；但一般 `bytes_in_use` 仍应下降
  - 尝试加 `--kv-drop-clear-jax-caches`（会更可能触发后续 recompile）

## References

- KV cache sizing/预分配原理：`docs/sops/sglang-jax-kv-cache-sizing.md`


# SOP: TPU v4-8 Qwen2.5-3B big KV cache + drop/rebuild + W&B memory/text logging

- **Title**: SOP: Run Qwen2.5-3B (MLLM-JAX params) on sglang-jax with a large preallocated KV cache, then drop KV, allocate a second param set, rebuild KV, and log per-phase memory + sample text to W&B.
- **Prereqs**:
  - `gcloud` installed + authenticated (Windows default may use plink; pin hostkey).
  - A READY TPU VM.
  - TPU VM has Miniconda + conda env `sglang-jax`.
  - TPU VM can access Hugging Face model weights.
  - W&B API key available (must NOT be committed to Git).

## Environment (verified)

- Project: `civil-rarity-482610-s5`
- Zone: `us-central2-b`
- TPU VM: `mllm-jax-v4-8-260122100610`
- SSH hostkey (plink): `SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`
- JAX: `0.8.1` (backend `tpu`, `jax.device_count()==4`)
- sglang-jax commit: `bd09a87`
- Repo branch/commit (W&B run): `mllm-jax-sglang@ebaf414`
- Repo bugfix note: `mllm-jax-sglang@78a7e76` fixes `kv_cache_dropped.drop_info.fused_bytes_total` (was 0 when `dtype` is a class).
- Log: `/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_kv_big_wandb_det0_same.log`
- W&B run: `https://wandb.ai/johntitordemon2036/sglang-jax-qwen25-3b-kv-drop-rebuild-bigkv/runs/kja6s32i`

## Steps (commands actually used)

### 1) Check TPU is READY (local)

```bash
gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --format='value(state,acceleratorType)'
```

### 2) SSH to TPU with hostkey pin (local)

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag='-batch' \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s' \
  --command 'whoami; hostname; date -u'
```

### 3) Sync repo on TPU via Git (TPU VM)

```bash
cd /root/MLLM-JAX
git fetch --all --prune
git checkout mllm-jax-sglang
git reset --hard origin/mllm-jax-sglang
git clean -fd
git rev-parse --short HEAD
```

### 4) Verify JAX runtime (TPU VM)

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
python -c 'import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.default_backend(), jax.device_count())'
```

### 5) Set W&B secrets (TPU VM; do NOT commit)

```bash
umask 077
printf '%s\n' \
  "WANDB_API_KEY=<YOUR_WANDB_API_KEY>" \
  "WANDB_MODE=online" \
  > /root/.env
chmod 600 /root/.env
```

### 6) Create a “complex code” prompt file (TPU VM)

This prompt is used to generate a longer response (up to 4096 tokens).

```bash
mkdir -p /root/MLLM-JAX/workdir/prompts
echo WW91IGFyZSBhIHNlbmlvciBzb2Z0d2FyZSBlbmdpbmVlci4KV3JpdGUgYSBzaW5nbGUtZmlsZSBQeXRob24gMy4xMSBwcm9ncmFtLgpHb2FsOiBpbXBsZW1lbnQgYW4gaW4tbWVtb3J5IGtleS12YWx1ZSBzdG9yZSB3aXRoIFRUTCBhbmQgTFJVIGV2aWN0aW9uLgpFeHBvc2UgYW4gYXN5bmNpbyBUQ1Agc2VydmVyIHdpdGggYSB0aW55IHRleHQgcHJvdG9jb2w6IFNFVCBrZXkgdHRsX3NlY29uZHMgdmFsdWUgfCBHRVQga2V5IHwgREVMIGtleSB8IFNUQVRTIHwgUVVJVC4KTWFrZSBpdCBjb25jdXJyZW5jeS1zYWZlIGFuZCBpbmNsdWRlIGEgYmFja2dyb3VuZCBjbGVhbnVwIHRhc2suCkluY2x1ZGUgYSBzbWFsbCBDTEkgKGFyZ3BhcnNlKSB0byBzZXQgaG9zdC9wb3J0L21heF9pdGVtcy4KSW5jbHVkZSB1bml0IHRlc3RzICh1bml0dGVzdCkgaW4gdGhlIHNhbWUgZmlsZS4KT3V0cHV0IE9OTFkgb25lIE1hcmtkb3duIGNvZGUgYmxvY2sgY29udGFpbmluZyB0aGUgZnVsbCBwcm9ncmFtLg== | base64 -d > /root/MLLM-JAX/workdir/prompts/complex_code_prompt_4096.txt
wc -l /root/MLLM-JAX/workdir/prompts/complex_code_prompt_4096.txt
head -n 3 /root/MLLM-JAX/workdir/prompts/complex_code_prompt_4096.txt
```

### 7) Run big-KV drop/rebuild driver with W&B (TPU VM)

Key knobs:
- `--max-total-tokens 131072` => large preallocated KV cache (~9.00 GB in this run).
- `--max-new-tokens 4096` => allow long generation.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
set -a
source /root/.env
set +a
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_kv_big_wandb_det0_same.log
timeout 7200 python -u tests/run_sglang_jax_qwen25_3b_mllm_param_swap.py \
  --wandb \
  --wandb-project sglang-jax-qwen25-3b-kv-drop-rebuild-bigkv \
  --wandb-name qwen25_3b_kv_big_131072_det0_same \
  --prompt-file /root/MLLM-JAX/workdir/prompts/complex_code_prompt_4096.txt \
  --max-total-tokens 131072 \
  --max-new-tokens 4096 \
  --temperature 0.0 \
  --kv-drop-rebuild \
  --verify-param-sharing \
  --assert-same-output \
  2>&1 | tee "$LOG"
echo log_path=$LOG
```

## Expected Result

- Log contains `wandb_init` JSON with a non-null `url`, and W&B UI shows:
  - Per-phase memory metrics (`mem/jax/*`, `mem/process/*`).
  - Logged text media / table (`sample_texts`, `sample_output_text_*`) and Files (`workdir/wandb_samples/<run_id>/...`).
- Log contains phase JSON for:
  - `engine_ready_dummy`, `weights_swapped_from_mllm`, `generate_result`
  - `flush_cache_before_kv_drop`, `kv_cache_dropped`
  - `sglang_param_dict2_ready`, `kv_cache_rebuilt`, `generate_result_after_kv_rebuild`
- Process exits with code `0`.

## Measured Results (this run)

- `max_total_tokens=131072` allocated KV:
  - sglang-jax log: `Fused KV size: 9.00 GB`
  - `kv_cache_rebuilt.rebuild_info.fused_bytes_total=9663750144` bytes (~9.00 GiB)
- Key memory checkpoints (`bytes_in_use_sum`, summed across 4 devices):
  - `engine_ready_dummy`: `15912880128` bytes (~14.82 GiB)
  - `flush_cache_before_kv_drop`: `16244695040` bytes (~15.13 GiB)
  - `kv_cache_dropped`: `6580944896` bytes (~6.13 GiB)
    - delta: `9663750144` bytes (~9.00 GiB) == KV `fused_bytes_total`
  - `sglang_param_dict2_ready` (two param sets, KV dropped): `12829917184` bytes (~11.95 GiB)
  - `kv_cache_rebuilt` (two param sets + KV): `22493880320` bytes (~20.95 GiB)
  - `after_drop_param_dict2`: `16244908032` bytes (~15.13 GiB)
- Determinism (temperature=0, same sampling params):
  - `determinism/same_output=1`
  - `output_1_sha256 == output_2_sha256 == 9e9f6b55176c580c914189b3e40c6f80c58a5d2fa3caf41cab0b4b2cdb36b714`

## References

- Driver: `tests/run_sglang_jax_qwen25_3b_mllm_param_swap.py`
- KV lifecycle helpers: `plugins/sglang_jax_inference/kv_cache_lifecycle.py`
- KV sizing notes: `docs/sops/sglang-jax-kv-cache-sizing.md`

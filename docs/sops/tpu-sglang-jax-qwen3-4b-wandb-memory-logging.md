# TPU：sglang-jax Qwen3-4B 权重热替换 + W&B 分阶段内存打点（实测：offline + online）

- **标题**：SOP：在 TPU 上运行 `Qwen/Qwen3-4B` 的 sglang-jax param-swap 脚本，并在各阶段打印/上报（W&B）内存占用
  **前置条件**：
  - 本地已安装并登录 `gcloud`，且已设置默认 project。
  - 目标 zone 有可用 TPU VM。
  - TPU VM runtime 镜像：`tpu-ubuntu2204-base`。
  - 本仓库已 push 到 GitHub（TPU 侧通过 Git 同步拉代码；不要用 scp 拷代码）。
  - **W&B online**：需要在 TPU 上设置 `WANDB_API_KEY`（不要写入 repo）。
  **环境（本次实测）**：
  - **实测 A（v4-8 spot，W&B offline）**
    - Project：`civil-rarity-482610-s5`
    - Zone：`us-central2-b`
    - TPU 名称：`mllm-jax-v4-8-260121193102`（完成两次 run 后被 maintenance `PREEMPTED`）
    - Conda env：`sglang-jax`（Python `3.12.12`）
    - JAX：`0.8.1`（`jax[tpu]==0.8.1`）
    - W&B：`wandb==0.24.0`
    - sglang-jax commit（固定）：`bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
    - 本仓库 commit：`2ac9ec1`
    - TPU log：
      - `/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_wandb_offline_260121120625.log`
    - W&B offline run dir：`/root/MLLM-JAX/workdir/wandb/wandb/offline-run-20260121_120631-hm401xup`
  - **实测 B（v6e-8，W&B online）**
    - Project：`civil-rarity-482610-s5`
    - Zone：`us-east1-d`
    - TPU 名称：`mllm-jax-v6e-8-260120021350`（`jax.device_count()==8`）
    - Conda env：`mllm-jax`（Python `3.12.12`）
    - JAX：`0.8.2`
    - W&B：`wandb==0.24.0`
    - 本仓库 commit：`8f47868`
    - W&B project：`sglang-jax-qwen3-4b-weight-swap-memory-online`
    - W&B run url：`https://wandb.ai/johntitordemon2036/sglang-jax-qwen3-4b-weight-swap-memory-online/runs/15ukva9g`
    - TPU log：
      - `/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_wandb_online_260121145251.log`

## 背景与目标

我们希望判断 “dummy 初始化 + 权重替换” 流程的内存行为是否会在 swap 期间同时驻留两套 params，以及把各阶段内存以**可复现**方式上报到 W&B（online 时可实时观察）。

本 SOP 通过在 `tests/run_sglang_jax_qwen3_4b_param_swap.py` 的 phase JSON 里记录：

- `process_rss_bytes`（host RSS）
- `jax.devices()[i].memory_stats()`（`bytes_in_use / peak_bytes_in_use / bytes_limit / largest_free_block_bytes` 等）

并用 `_wandb_log_phase(...)` 把同一份分阶段内存结构 flatten 后写到 W&B：

- `phase/name`、`phase/idx`
- `mem/process/rss_bytes`
- `mem/jax/*` 与 `mem/jax/device{i}/*`

## 步骤（本次实际执行的命令）

### 1) 创建 TPU VM（v4-8 spot 示例，本地）

```bash
gcloud alpha compute tpus tpu-vm create mllm-jax-v4-8-260121193102 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --version=tpu-ubuntu2204-base \
  --spot \
  --quiet
gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260121193102 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --format='value(state,acceleratorType)'
```

### 2) 获取并固定 hostkey（Windows/plink 示例，本地）

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260121193102 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --quiet \
  --ssh-flag='-batch' \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:RalpU1z30DMIxdugVxiK76JdZifDqBfkDnJMeNCFB+M' \
  --command 'whoami; hostname; head -n 5 /etc/os-release'
```

### 3) 安装 Miniconda + 创建 conda env（TPU VM 上）

```bash
curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /root/miniconda.sh -b -p /root/miniconda3
rm -f /root/miniconda.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -y -n sglang-jax python=3.12
conda activate sglang-jax
pip install -U pip
python -V
```

### 4) 安装 JAX TPU + HF 依赖（TPU VM 上）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax

pip install -U jax[tpu]==0.8.1 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip show jax

pip install -U huggingface-hub==0.34.3 hf_transfer
python -m pip show huggingface-hub
python -m pip show hf_transfer
```

### 5) Git clone 本仓库（TPU VM 上）

```bash
git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX
cd /root/MLLM-JAX
git fetch --all --prune
git checkout mllm-jax-sglang
echo HEAD=$(git rev-parse --short HEAD)
mkdir -p /root/MLLM-JAX/workdir /root/MLLM-JAX/workdir/hf_download /root/MLLM-JAX/workdir/hf_models /root/MLLM-JAX/workdir/wandb
```

### 6) Git clone sglang-jax 并安装 editable（TPU VM 上）

```bash
git clone https://github.com/sgl-project/sglang-jax.git /root/MLLM-JAX/workdir/sglang-jax
cd /root/MLLM-JAX/workdir/sglang-jax
git fetch --all --prune
git checkout bd09a87fc6e86c21ce14edd66948ac5dea3a4360

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
pip install -e /root/MLLM-JAX/workdir/sglang-jax/python
python -m pip show sglang-jax
```

### 7) 安装 wandb（TPU VM 上）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
pip install -U wandb
python -m pip show wandb
```

### 8) 运行脚本（TPU VM 上，W&B offline）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
TS=$(date +%y%m%d%H%M%S)
LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_wandb_offline_${TS}.log
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py \
  --wandb \
  --wandb-project sglang-jax-qwen3-4b-weight-swap-memory \
  --wandb-name qwen3_4b_mem_${TS} \
  2>&1 | tee $LOG
echo log_path=$LOG
```

### 9) 运行脚本（TPU VM 上，W&B online）

本次实际执行的命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
set -a
source /root/.env_wandb_sglang_jax
set +a
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
TS=$(date +%y%m%d%H%M%S)
LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_wandb_online_${TS}.log
timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py \
  --wandb \
  --wandb-project sglang-jax-qwen3-4b-weight-swap-memory-online \
  --wandb-name qwen3_4b_v6e8_${TS} \
  2>&1 | tee $LOG
echo log_path=$LOG
```

## 预期结果

- 控制台（以及 `$LOG`）会包含以下 phase JSON：
  - `wandb_init`：online 模式包含 `url`，offline 模式 `url=null`
  - `jax_ready`
  - `engine_ready_dummy`
  - `weights_swapped`
  - `generate_result`
- `memory.jax_device_memory_stats` 的数组长度应等于 `jax.device_count()`。
- `generate_result.text` 非空（提示词默认 `你是谁`）。
- 进程退出码为 `0`。

## 实测内存关键数值（摘自本次日志）

- **实测 A（v4-8 spot，4 device 合计）**
  - `weights_swapped`：`bytes_in_use_sum=8652005376`（≈ `8.06 GiB`），`peak_bytes_in_use_max≈3.90 GiB/device`
  - `generate_result`：`bytes_in_use_sum=9074287616`（≈ `8.45 GiB`），增量 ≈ `0.39 GiB`
- **实测 B（v6e-8，8 device 合计）**
  - `engine_ready_dummy`：`bytes_in_use_sum=8652901376`（≈ `8.06 GiB`，≈ `1.01 GiB/device`）
  - `weights_swapped`：`bytes_in_use_sum=8655358976`（≈ `8.06 GiB`），`peak_bytes_in_use_max=2094266368`（≈ `1.95 GiB/device`）
  - `generate_result`：`bytes_in_use_sum=9164978176`（≈ `8.54 GiB`），增量 ≈ `0.47 GiB`

## 备注与故障排查

- **W&B online**
  - 需要 `WANDB_API_KEY`；本次在 TPU 上使用 root-only 文件 `/root/.env_wandb_sglang_jax`（`chmod 600`）保存，不写入 repo。
  - 建议仍按仓库 SOP：本地 `.env`（gitignore）→ `scripts/sync_env_to_tpu_vm.sh` 分发到 worker=all。
- **TPU 被其它进程占用**
  - 现象：`The TPU is already in use by process with pid ...` 或 `open(/dev/vfio/0): Device or resource busy`。
  - 本次在 `v6e-8` 上实际使用的处理方式（会 kill 掉后台训练进程；注意确认这是你自己的 TPU）：
    - `pkill -9 -f '[r]un_grpo_gsm8k_training.py' || true`
- **spot TPU 频繁 PREEMPTED**
  - v4-8 spot 可能在跑完后被 maintenance `PREEMPTED`（terminal state，无法再次 SSH/复用）。
- **wandb 退出时出现 atexit BrokenPipe traceback**
  - 曾在更早一次 run 中遇到 `Exception ignored in atexit callback ... BrokenPipeError`（wandb service teardown）。
  - runner 脚本已增加显式 `wandb_service.teardown(0)`，避免影响退出码。


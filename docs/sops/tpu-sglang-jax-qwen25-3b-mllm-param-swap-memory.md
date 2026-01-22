# TPU：sglang-jax Qwen2.5-3B MLLM params 热替换 + 共享 params 内存验证

- **标题**：SOP：在 TPU 上用 MLLM-JAX 侧的 `Qwen/Qwen2.5-3B-Instruct` params 注入 sglang-jax Engine，并验证运行时内存处于“单套 params”量级
  **前置条件**：
  - 本地已安装并登录 `gcloud`，且已设置默认 project。
  - 目标 project/zone 有可用 TPU VM（本次复用现有 `v4-8`）。
  - TPU VM 可 root SSH（Windows 默认可能走 plink，需要 hostkey pin）。
  - TPU VM 上已安装 Miniconda（本次复用已有 `/root/miniconda3`）。
  - 本仓库已 push 到 GitHub（TPU 侧通过 Git 同步拉代码）。
  - TPU VM 环境能访问 Hugging Face 下载权重。
  **环境（本次实测）**：
  - Project：`civil-rarity-482610-s5`
  - Zone：`us-central2-b`
  - TPU：`v4-8`（本机 JAX 以 **4 devices** 形式暴露：`jax.device_count()==4`，疑似 v4 megacore 行为）
  - TPU 名称：`mllm-jax-v4-8-260122100610`
  - SSH hostkey（plink）：`SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`
  - TPU VM OS：Ubuntu `22.04.2`
  - Conda env：`sglang-jax`
  - JAX：`0.8.1`（TPU backend）
  - sglang-jax commit（固定）：`bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
  - 本仓库分支/commit：`mllm-jax-sglang@f6a3ed1`
  - TPU log：`/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_mllm_param_swap_260122065028.log`

## 背景与目标

我们希望验证以下链路可用且内存不出现“双套 params 常驻”的情况：

1) 用 MLLM-JAX 的权重转换（torch state_dict -> flax params tree）拿到 Qwen2.5-3B 的 params（CPU）。
2) 适配成 sglang-jax Qwen2/Qwen2.5 nnx param path -> sharded `jax.Array`（device）。
3) 运行时替换 Engine 内部 nnx Params，并刷新 `model_state_leaves`。
4) `generate()` 时即使 `param_dict` 仍然存活（用于验证共享），TPU `bytes_in_use_sum` 仍应处于单套 params 量级。

本次使用脚本：
- `tests/run_sglang_jax_qwen25_3b_mllm_param_swap.py`（带 `--keep-param-dict --verify-param-sharing`）

## 步骤（本次实际执行的命令）

### 1)（本地）找到可用 TPU VM 并确认 READY

```bash
gcloud alpha compute tpus tpu-vm list --project civil-rarity-482610-s5 --zone us-central2-b
gcloud alpha compute tpus tpu-vm describe mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --format='value(state,acceleratorType)'
```

### 2)（本地）获取并固定 hostkey（plink batch 模式）

第一次用 `-batch` 连接会输出 fingerprint（并失败），从输出中摘出 `SHA256:...`：

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag='-batch' \
  --command 'whoami'
```

本次得到：
- `SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s`

后续 SSH 统一带上：
- `--ssh-flag='-hostkey' --ssh-flag='SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s'`

### 3)（本地）root SSH 到 TPU VM 并检查 OS

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260122100610 \
  --project civil-rarity-482610-s5 \
  --zone us-central2-b \
  --worker 0 \
  --quiet \
  --ssh-flag='-batch' \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:Xy1NoS+m4LpYQNWiLlkc3Co5iIhoPzAFC39CNLmSN3s' \
  --command 'whoami; hostname; head -n 5 /etc/os-release'
```

### 4)（TPU VM）确认 conda env 存在

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda env list
```

### 5)（TPU VM）Git 同步本仓库（checkout 到目标 commit）

```bash
REPO_URL=https://github.com/demon2036/MLLM-JAX.git
REPO_DIR=/root/MLLM-JAX
if [ ! -d "$REPO_DIR/.git" ]; then rm -rf "$REPO_DIR"; git clone "$REPO_URL" "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch --all --prune
git checkout mllm-jax-sglang
git reset --hard origin/mllm-jax-sglang
git clean -fd
git rev-parse --short HEAD
```

### 6)（TPU VM）安装依赖（并修复与 sglang-jax 的版本冲突）

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX

# 基础依赖（会升级到最新版，可能与 sglang-jax pin 冲突）
python -m pip install -U -r requirements-tpu.txt

# sglang-jax 需要：
python -m pip install -U huggingface-hub==0.34.3 typing-extensions==4.14.1

# 为了与 typing-extensions==4.14.1 兼容，将 chex 回退到 0.1.90
python -m pip install -U chex==0.1.90
```

### 7)（TPU VM）准备并安装 sglang-jax（固定 commit，editable）

```bash
SG_DIR=/root/MLLM-JAX/workdir/sglang-jax
mkdir -p /root/MLLM-JAX/workdir
if [ ! -d "$SG_DIR/.git" ]; then rm -rf "$SG_DIR"; git clone https://github.com/sgl-project/sglang-jax.git "$SG_DIR"; fi
cd "$SG_DIR"
git fetch --all --prune
git checkout bd09a87fc6e86c21ce14edd66948ac5dea3a4360

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
python -m pip install -e /root/MLLM-JAX/workdir/sglang-jax/python
python -c 'from sgl_jax.version import __version__; print(__version__)'
```

### 8)（TPU VM）验证 JAX 设备数

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
python -c 'import jax; print(jax.__version__); print(jax.default_backend()); print(jax.device_count()); print(jax.local_device_count()); print(jax.process_index())'
```

本次输出：`0.8.1 / tpu / 4 / 4 / 0`

### 9)（TPU VM）运行 Qwen2.5-3B MLLM params -> Engine 热替换验证

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
mkdir -p /root/MLLM-JAX/workdir /root/MLLM-JAX/workdir/hf_download /root/MLLM-JAX/workdir/hf_models
rm -f /tmp/libtpu_lockfile || true
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
TS=$(date +%y%m%d%H%M%S)
LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen25_3b_mllm_param_swap_${TS}.log
timeout 7200 python -u tests/run_sglang_jax_qwen25_3b_mllm_param_swap.py \
  --keep-param-dict \
  --verify-param-sharing \
  2>&1 | tee "$LOG"
echo "log_path=$LOG"
```

## 预期结果

- 脚本 phase JSON 至少包含：
  - `engine_ready_dummy`
  - `mllm_params_cpu_ready`
  - `sglang_param_dict_ready`
  - `weights_swapped_from_mllm`
  - `param_sharing_check`（sample paths 的 `same_object=true`）
  - `before_generate_param_dict_kept`
  - `generate_result`（`text` 非空）
  - `after_drop_param_dict`
- 进程退出码：`0`

## 本次实测关键内存数值（来自 log 的 `memory.jax_device_memory_summary.bytes_in_use_sum`）

> 说明：本次 v4-8 上 `jax.device_count()==4`，因此这里的 `bytes_in_use_sum` 是 4 个 device 的总和。

- `engine_ready_dummy`：`6551193600` bytes（`6.10 GiB`）
- `sglang_param_dict_ready`：`12800165888` bytes（`11.92 GiB`，dummy + param_dict 同时存活的阶段）
- `weights_swapped_from_mllm`：`6551193600` bytes（`6.10 GiB`，swap 后回到单套 params 量级）
- `before_generate_param_dict_kept`：`6551193600` bytes（`6.10 GiB`，保持 `param_dict` 存活仍不翻倍）
- `generate_result`：`6860432384` bytes（`6.39 GiB`）
- `after_drop_param_dict`：`6860387328` bytes（`6.39 GiB`）

## 故障排查

- **plink host key 未缓存导致 batch 模式失败**
  - 先用 `--ssh-flag='-batch'` 连接一次，从输出中拿到 `SHA256:...` 指纹，再加：
    - `--ssh-flag='-hostkey' --ssh-flag='SHA256:<fingerprint>'`
- **依赖冲突（huggingface-hub / typing-extensions / chex）**
  - 以 sglang-jax 依赖为准（本次 `sglang-jax==0.0.2`）：`huggingface-hub==0.34.3` + `typing-extensions==4.14.1`
  - `chex` 与 `typing-extensions` 冲突时，将 `chex` 回退到 `0.1.90`
- **TPU “already in use by pid …” / lockfile**
  - `rm -f /tmp/libtpu_lockfile || true`
  - 确认没有残留占用进程

## 参考

- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-sglang-jax-qwen3-4b-engine-weight-swap-infer.md`
- `docs/sops/tpu-sglang-jax-qwen3-4b-wandb-memory-logging.md`


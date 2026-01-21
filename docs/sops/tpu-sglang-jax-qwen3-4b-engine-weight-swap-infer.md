# TPU：sglang-jax Qwen3-4B 权重热替换推理验证（Engine 参数替换）

- **标题**：SOP：在 TPU 上用 sglang-jax 跑 `Qwen/Qwen3-4B`，然后把权重注入 dummy Engine 并生成“你是谁”
  **前置条件**：
  - 本地机器已完成 `gcloud` 登录认证，且已设置默认 project。
  - 目标 zone 具备 TPU VM `v4-8` 配额。
  - TPU VM runtime 镜像：`tpu-ubuntu2204-base`。
  - 本仓库已 push 到 GitHub（TPU 侧通过 Git 同步拉代码）。
  - **Windows 备注**：`gcloud ... tpu-vm ssh` 可能使用 plink，并需要做 hostkey 处理。
  **环境（本次实测）**：
  - Project：`civil-rarity-482610-s5`
  - Zone：`us-central2-b`
  - TPU：`v4-8`（spot）
  - TPU 名称：`mllm-jax-v4-8-260121193102`（完成两次 run 后被 maintenance PREEMPTED）
  - TPU VM OS：Ubuntu `22.04.2`
  - Conda env：`sglang-jax`（Python `3.12.12`）
  - JAX：`0.8.1`（TPU backend）
  - sglang-jax commit（固定）：`bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
  - 本仓库 commit（runner 脚本）：`2ac9ec1`
  - TPU log（runner 输出）：`/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_nwandb_260121120224.log`

## 步骤

### 1) 创建 v4-8 TPU VM（本地）

本次实际执行的命令：

```bash
gcloud auth list --format='table(account,status)'
gcloud config get-value project
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

### 2) 以 root 身份 SSH 到 TPU VM（本地）

本次实际执行的命令（Windows/plink hostkey pin，针对该 TPU 实例）：

```bash
gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v4-8-260121193102 \
  --project=civil-rarity-482610-s5 \
  --zone=us-central2-b \
  --quiet \
  --ssh-flag='-batch' \
  --ssh-flag='-hostkey' \
  --ssh-flag='SHA256:RalpU1z30DMIxdugVxiK76JdZifDqBfkDnJMeNCFB+M' \
  --command 'whoami; hostname; cat /etc/os-release | head -n 5'
```

### 3) 安装 Python 3.12 + JAX TPU（TPU VM 上）

本次实际执行的命令：

```bash
# 安装 Miniconda
curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /root/miniconda.sh -b -p /root/miniconda3
rm -f /root/miniconda.sh

# 创建 env
source /root/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda create -y -n sglang-jax python=3.12
conda activate sglang-jax
pip install -U pip

# 安装 JAX TPU
pip install -U jax[tpu]==0.8.1 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -c 'import jax; print(jax.__version__); print(jax.device_count()); print(jax.local_device_count())'

# 可选：启用更快的 HF 下载
pip install -U huggingface-hub==0.34.3
pip install -U hf_transfer
```

### 4) 通过 Git 在 TPU 上拉取仓库（不要用 SCP）

本次实际执行的命令：

```bash
# Clone 本仓库
git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX
cd /root/MLLM-JAX
git fetch --all --prune
git checkout mllm-jax-sglang
echo HEAD=$(git rev-parse --short HEAD)
mkdir -p /root/MLLM-JAX/workdir /root/MLLM-JAX/workdir/hf_download /root/MLLM-JAX/workdir/hf_models

# Clone sglang-jax 到本地 scratch
git clone https://github.com/sgl-project/sglang-jax.git /root/MLLM-JAX/workdir/sglang-jax
cd /root/MLLM-JAX/workdir/sglang-jax
git fetch --all --prune
git checkout bd09a87fc6e86c21ce14edd66948ac5dea3a4360
```

### 5) 在 TPU 上安装 sglang-jax（editable）

本次实际执行的命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
pip install -e /root/MLLM-JAX/workdir/sglang-jax/python
python -c 'import sgl_jax; from sgl_jax.version import __version__; print(__version__)'
```

### 6) 运行 Engine 权重热替换推理脚本（TPU VM 上）

本次实际执行的命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
TS=$(date +%y%m%d%H%M%S)
LOG=/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_nwandb_${TS}.log
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py 2>&1 | tee "$LOG"
echo "log_path=$LOG"
```

## 预期结果

- 脚本会依次打印：
  - `{"phase": "engine_ready_dummy", ...}`（dummy 权重 Engine 初始化完成）
  - `{"phase": "weights_swapped", ...}`（从 `Qwen/Qwen3-4B` safetensors 加载权重并注入 `model_runner.model_state_leaves`）
  - `{"phase": "generate_result", "prompt": "你是谁", ...}` 且 `text` 非空
- 进程退出码为 `0`

## sglang-jax 是怎么做到的（最小架构笔记）

- Engine 组合结构（见 sglang-jax `python/sgl_jax/srt/entrypoints/engine.py`）：
  - `Engine` 启动 `TokenizerManager` + `Scheduler` + `DetokenizerManager`。
- 权重加载链路：
  - `ModelRunner.load_model()` 调用 `model_loader.load_model()`，再通过 `WeightLoader.load_weights_from_safetensors()` 进入 `Qwen3ForCausalLM.load_weights()`。
  - Qwen3 的 HF key 映射在 `python/sgl_jax/srt/models/qwen3.py`（`_create_qwen3_weight_mappings`）。
- 为什么运行时替换可行：
  - `ModelRunner.initialize_jit()` 会导出 `model_state_leaves`（展平后的 nnx state）。编译后的函数每次调用都用这些 leaves 重新构造模型状态。
  - 用同结构的 leaf list 替换 `model_state_leaves`，本质上就是在不改变编译 shape 的前提下做参数热替换。

## 故障排查

- **TPU “already in use by pid …”**
  - 找到并 kill 卡住的进程：
    - `ps -fp <pid>`
    - `kill -9 <pid>`
- **Windows 上 `gcloud tpu-vm ssh` host key 不匹配（plink 弹窗）**
  - 先重连一次看到新的 `SHA256:...` 指纹，然后传入：
    - `--ssh-flag='-hostkey' --ssh-flag='SHA256:<fingerprint>'`
  - 或者删除本地缓存的 key 后重新连接。

## 参考

- `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
- `docs/sops/tpu-vm-repo-sync.md`
- sglang-jax repo: https://github.com/sgl-project/sglang-jax

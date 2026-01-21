# sglang-jax（JAX/TPU）：Qwen3-4B Engine 权重热替换推理验证

## 目标

在 TPU `v4-8` 上验证：

1) 用 `load_format="dummy"` 初始化 `sglang-jax` 的 `Engine`（快速完成形状准备/编译初始化）。
2) 从 Hugging Face safetensors 加载真实 `Qwen/Qwen3-4B` 权重到内存模型。
3) 原地替换 `Engine` 运行时参数（`model_state_leaves`）。
4) 用提示词 `你是谁` 完成一次推理。

这是一个**非侵入式**方案：不修改上游 `sglang-jax`；只在本仓库 `plugins/` 下新增辅助代码，并在 `tests/` 下提供可直接运行的验证脚本。

## 仓库基线（已有内容）

- 上游 `sglang-jax` 克隆在 `workdir/sglang-jax`（gitignore 的本地 scratch），固定 commit：
  - `bd09a87fc6e86c21ce14edd66948ac5dea3a4360`
- TPU SOP（描述同一次验证流程）：
  - `docs/sops/tpu-sglang-jax-qwen3-4b-engine-weight-swap-infer.md`
- 核心实现：
  - `plugins/sglang_jax_inference/engine_weight_swap.py`
  - `tests/run_sglang_jax_qwen3_4b_param_swap.py`

## sglang-jax 如何支持运行时权重替换

### 1) `Engine` 组合结构暴露进程内 scheduler

- 文件：`workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
- 以 `enable_single_process=True` 启动时，`Engine` 会保留进程内引用 `self.scheduler_info`。
- 因此在同一 Python 进程里，可以定位到：
  - `engine.scheduler_info["scheduler"].tp_worker.worker.model_runner`

### 2) `ModelRunner` 导出喂给 jitted 函数的 “leaf list”

- 文件：`workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
- `ModelRunner.initialize_jit()` 内部：
  - 通过 `nnx.split(self.model)` 抽取 `model_state`
  - 通过 `jax.tree_util.tree_flatten(model_state)` 展平
  - 展平后的列表保存为 `self.model_state_leaves`
- jitted 前向每次调用都用以下方式重建模型状态：
  - `jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)`

**关键含义**：只要把 `self.model_state_leaves` 替换成“同结构”的 leaf 列表（同 tree-def，leaf 数量与 shape 对齐），编译好的函数就会在**不重新编译**的情况下使用新参数运行。

### 3) Qwen3 权重加载依赖显式的 HF → 内部映射

- 文件：`workdir/sglang-jax/python/sgl_jax/srt/models/qwen3.py`
- `QWen3ForCausalLMModel.load_weights()` 会构造 `WeightLoader(...)` 并调用：
  - `loader.load_weights_from_safetensors(weight_mappings)`
- 映射由 `_create_qwen3_weight_mappings()` 与各 layer helper 构造。

## TPU 上实际跑的验证流程

### 1) 运行时替换辅助函数（plugin）

- 文件：`plugins/sglang_jax_inference/engine_weight_swap.py`
- 职责：
  1) 给定 model id 时下载 HF snapshot（`snapshot_download`）。
  2) 从 `Engine` 定位 `model_runner`（要求 `enable_single_process=True`）。
  3) 临时把 `model_config.model_path` 指向 snapshot 目录。
  4) 调用 `model_runner.model.load_weights(model_config)`（sglang-jax 官方加载逻辑）。
  5) 重新 split/flatten nnx state，并覆写 `model_runner.model_state_leaves`。
  6) 一致性校验：leaf 数量必须完全一致（tree 结构不变）。

### 2) 端到端参数替换推理脚本

- 文件：`tests/run_sglang_jax_qwen3_4b_param_swap.py`
- 流程：
  1) 创建 `Engine(... load_format="dummy", enable_single_process=True, device="tpu")`
  2) 打印 JSON phase `engine_ready_dummy`（包含 config + `num_model_state_leaves`）
  3) `swap_engine_weights_from_hf(engine, "Qwen/Qwen3-4B", cache_dir=...)`
  4) 打印 JSON phase `weights_swapped`
  5) `engine.generate(prompt="你是谁", ...)`
  6) 打印 JSON phase `generate_result`

#### 本轮做的增量改动

在 `tests/run_sglang_jax_qwen3_4b_param_swap.py` 中：

- 增加 `--prompt` CLI 覆盖（默认 `你是谁`）。
- HF cache/download 目录改为 repo 相对路径（`<repo>/workdir/...`），避免硬编码 `/root/MLLM-JAX/...`。
- `engine_ready_dummy` JSON 增加元信息字段：
  - `model_id`, `tp_size`, `dp_size`, `dtype`, `load_format`, `download_dir`, `hf_cache_dir`, `sgl_jax_version`
- 所有 phase JSON 输出加 `flush=True`，便于 `tee`/`grep` 实时抓取。
- HF snapshot 下载失败时输出 `weights_swap_error` JSON，并给出重试提示。

## TPU 执行记录

**TPU**

- 项目：`civil-rarity-482610-s5`
- Zone：`us-central2-b`
- TPU 名称：`mllm-jax-v4-8-260121193102`（`v4-8 spot`，完成两次 run 后被 maintenance PREEMPTED）
- 加速器：`v4-8`（spot）
- Runtime 镜像：`tpu-ubuntu2204-base`
- OS：Ubuntu `22.04.2`

**Python/JAX**

- Conda env：`sglang-jax`
- Python：`3.12.12`
- JAX：`0.8.1`，`jax.device_count()==4`（v4 芯片）

**代码版本**

- `sglang-jax` commit：`bd09a87fc6e86c21ce14edd66948ac5dea3a4360`（editable install，`sgl_jax` version `0.0.2`）
- 本仓库 commit：`2ac9ec1`（含 W&B 分阶段内存打点 + `wandb_service.teardown(0)` 修复）

**运行命令（TPU VM 上）**

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
cd /root/MLLM-JAX
# 1) baseline：禁用 wandb
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py

# 2) W&B offline：记录各 phase 内存（本次 run 的 offline-run dir 会打印在 stdout）
export WANDB_MODE=offline
timeout 7200 python -u tests/run_sglang_jax_qwen3_4b_param_swap.py \
  --wandb \
  --wandb-project sglang-jax-qwen3-4b-weight-swap-memory \
  --wandb-name qwen3_4b_mem_$(date +%y%m%d%H%M%S)
```

**产物（TPU VM 上）**

- 完整日志：
  - `/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_nwandb_260121120224.log`
  - `/root/MLLM-JAX/workdir/sglang_jax_qwen3_4b_param_swap_wandb_offline_260121120625.log`
- W&B offline run dir：`/root/MLLM-JAX/workdir/wandb/wandb/offline-run-20260121_120631-hm401xup`
- 备注：若要实时同步到云端，需要在 TPU 环境里配置 `WANDB_API_KEY` 并使用 `WANDB_MODE=online`；本次仅完成 offline 验证（退出码=0，且无 traceback）。
- 用于 swap 的 HF snapshot：`/root/MLLM-JAX/workdir/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

**关键 JSON 行（从日志摘录）**

```json
{"phase": "engine_ready_dummy", "model_id": "Qwen/Qwen3-4B", "device": "tpu", "tp_size": 4, "dp_size": 1, "dtype": "bfloat16", "load_format": "dummy", "download_dir": "/root/MLLM-JAX/workdir/hf_download", "hf_cache_dir": "/root/MLLM-JAX/workdir/hf_models", "sgl_jax_version": "0.0.2", "num_model_state_leaves": 398}
{"phase": "weights_swapped", "snapshot_dir": "/root/MLLM-JAX/workdir/hf_models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c", "num_model_state_leaves": 398}
{"phase": "generate_result", "prompt": "你是谁", "sampling_params": {"temperature": 0.0, "max_new_tokens": 64}, "text": "？我需要一个能帮我写代码的AI助手。我需要你能够理解我的需求，然后生成正确的代码。我需要你能够处理各种编程语言，比如Python、Java、C++、JavaScript等。我需要你能够处理各种编程任务，比如算法题、数据结构、系统编程、", "raw": {"text": "？我需要一个能帮我写代码的AI助手。我需要你能够理解我的需求，然后生成正确的代码。我需要你能够处理各种编程语言，比如Python、Java、C++、JavaScript等。我需要你能够处理各种编程任务，比如算法题、数据结构、系统编程、", "output_ids": [11319, 35946, 85106, 46944, 26232, 108965, 61443, 46100, 9370, 15469, 110498, 1773, 35946, 85106, 56568, 100006, 101128, 97611, 100354, 3837, 101889, 43959, 105045, 46100, 1773, 35946, 85106, 56568, 100006, 54542, 100646, 110569, 102064, 3837, 101912, 30280, 5373, 15041, 5373, 34, 1027, 5373, 29475, 49567, 1773, 35946, 85106, 56568, 100006, 54542, 100646, 110569, 88802, 3837, 101912, 107018, 33872, 5373, 20074, 100166, 5373, 72448, 110569, 5373], "meta_info": {"id": "1b5dfb9bba644475a2f097ae90f65b8d", "finish_reason": {"type": "length", "length": 64}, "prompt_tokens": 2, "completion_tokens": 64, "cached_tokens": 0, "routed_experts": null, "cache_miss_count": 0, "e2e_latency": 35.56036972999573}}}
```

## 分阶段内存结论（TPU HBM）

以下为脚本 `tests/run_sglang_jax_qwen3_4b_param_swap.py` 在各 phase 打印的 `jax_device_memory_summary`（4 device 合计）：

- `engine_ready_dummy`：`bytes_in_use_sum=8650770432`（≈ `8.06 GiB`）
- `weights_swapped`：`bytes_in_use_sum=8652005376`（≈ `8.06 GiB`）
- `generate_result`：`bytes_in_use_sum=9074183168`（≈ `8.45 GiB`，相比 `weights_swapped` 增量 ≈ `0.393 GiB`；W&B offline run 为 `9074287616`）

同时 `weights_swapped` 阶段的 `peak_bytes_in_use_max≈3.90 GiB/device`，而稳态 `bytes_in_use≈2.01 GiB/device`，说明 swap 过程中存在**短暂的额外峰值分配**（并非两套 params 长期常驻）。

## 结论

- 通过替换 `model_state_leaves` 的方式对 `Engine` 做运行时参数替换，对 `Qwen/Qwen3-4B` 在 TPU `v4-8`（`tp_size=4`）上是**可行**的。
- 替换成功且**未改变**模型状态结构：替换前后 `num_model_state_leaves` 都是 `398`。
- 替换后 `engine.generate()` 对提示词 `你是谁` 返回非空输出（本次运行设置 `max_new_tokens=64`，因此被长度截断）。
- 备注：本次使用的是纯文本 prompt（非 chat template）。若想更符合“助手自我介绍”的聊天效果，建议走 chat serving/template 路径，而不是直接 `Engine.generate(text=...)`。

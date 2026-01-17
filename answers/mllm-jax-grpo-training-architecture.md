# MLLM-JAX：GRPO/GSM8K 模块化训练链路（plugins-first）

> 目标：把训练过程拆成可组合、可验证、可扩展的阶段化模块：`rollout → reward → advantages → update`，并确保可在 **2-host TPU v4-16** 上端到端跑通（W&B 在线打点）。
>
> 参考（设计思想）：`answers/areal-rl-organization-and-mllm-jax-modularization.md`（AReaL 的分层：API/contracts → backend → customization → entrypoint）。

## 0) TL;DR（你从哪里开始看）

- 直接跑训练入口：`scripts/run_grpo_gsm8k_training.py`
- 端到端 runner（编排）：`plugins/training/runner/grpo_gsm8k.py`
- 四阶段可复用模块（核心实现）：
  - rollout：`plugins/training/grpo/sampling.py`
  - reward：`plugins/training/grpo/rewarding.py`
  - advantages（group_id）：`plugins/training/grpo/advantages.py`
  - update（PPO/GRPO loop）：`plugins/training/grpo/update.py`
- API/契约与 schema 校验：`plugins/training/api/interfaces.py`、`plugins/training/api/batch_schema.py`
- 同步模块封装（把“四阶段”对象化）：`plugins/training/modules/grpo_sync.py`
- TPU v4-16 + `.env` + W&B 的**可复现命令**：
  - 20 steps：`docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`
  - 100 steps（含 eval）：`docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`
- 我们建议打哪些 log（以及当前已实现哪些）：`answers/mllm-jax-logging-metrics-plan.md`

## 1) 设计目标与硬约束

### 1.1 目标

1. **阶段化**：把“脚本里揉在一起”的训练拆成明确阶段（rollout/reward/adv/update），每个阶段返回结构化结果。
2. **契约优先**：先把 batch/trajectory 的 *shape + key* 不变量写死（schema），再迭代实现，避免重构时静默破坏训练。
3. **多机可跑**：同一份 entrypoint 在 `jax.process_count()>1` 的 TPU VM 上可运行；W&B 只由 process 0 打点。
4. **非侵入**：不修改 `MLLM_JAX/`、不改上游训练模块；所有新增逻辑放在 `plugins/` + `scripts/` + `docs/`。

### 1.2 约束

- **2-host TPU VM**：用 `jax.distributed.initialize()`，并确保每个 worker 同时启动（否则会 hang）。
- **batch 对齐本地 device 数**：`MLLM_JAX.utils._form_global_array()` 会把本地 batch 切到 `len(mesh.local_devices)` 份；因此本地 batch 必须可整除 local device 数，否则必崩。
- **W&B secret 不进 Git**：`WANDB_API_KEY` 必须在 `.env`（gitignored）并通过脚本同步到 TPU。

## 2) 总体架构（从 entrypoint 到参数更新）

用 AReaL 的语言对齐一下（在本仓库的“插件化、同步 rollout”版本）：

- **Entrypoint layer（入口装配）**：`scripts/run_grpo_gsm8k_training.py`
  - 只做：加载 `.env` → 读 env vars → 组装 config → 调用 runner。
- **Runner layer（编排/生命周期）**：`plugins/training/runner/grpo_gsm8k.py`
  - 只做：初始化分布式/mesh → dataset → 循环 step → 调用四阶段函数 → 日志。
- **Customization layer（算法与 workflow）**：`plugins/training/grpo/*`
  - 只做：rollout/reward/advantages/update 的纯逻辑，可复用到其他 runner。
- **API layer（契约+校验）**：`plugins/training/api/*`
  - 只做：Protocol/dataclass + schema validator（不塞实现）。
- **Backend（JAX/TPU sharding 等）**：仍由现有上游能力提供
  - `training2.get_state()` / `training2.training_step()`（模型、优化器、loss）
  - `MLLM_JAX.utils.get_jax_mesh2()` / `_form_global_array()`（mesh 与 global array 组装）
  - `MLLM_JAX.sample.* Sampler`（采样）

## 3) 数据契约：GRPO batch/trajectory schema（必读）

Schema 校验入口：`plugins/training/api/batch_schema.py` 的 `validate_grpo_batch(batch, stage=...)`。

### 3.1 核心张量（最小闭环）

batch 内关键字段（shape 以 `B=样本数`, `T=序列长度` 表示）：

- `input_ids`: `[B, T]`
- `attention_mask`: `[B, T]`（与 `input_ids` 同 shape）
- `labels`: `[B, T]`（这里是 **completion mask**；训练模块用它筛选有效 token）
- `rewards`: `[B]`（reward 阶段产物）
- `group_ids`: `[B]`（**显式分组 id**；用于按组算 advantages）
- `advantages`: `[B]`（advantages 阶段产物）
- `old_per_token_logps`: `[B, T-1]`（PPO 多 epoch 时需要）
- `total_valid_token_count`: `()`（scalar；通常来自 `labels[:,1:].sum()`）

### 3.2 阶段化校验（为什么要做）

用 `stage` 把训练步分成几段分别断言（rollout → rewarded → advantaged → train_step/train_ready），好处是：

- 重构时能快速定位“哪一个阶段破坏了隐含约定”
- 对齐 AReaL 的 `check_trajectory_format` 思路：先写死 schema，再做异步/多机等复杂功能

## 4) Rollout：prompt → completion → 训练 batch

实现：`plugins/training/grpo/sampling.py`

### 4.1 输入/输出

- 输入：`prompts: list[str]`、`sampler`、`params`、`system_prompt`、`global_length`、`max_length_sample`
- 输出：
  - `answers: list[str]`
  - `batch: {input_ids, attention_mask, labels}`（都在 host 侧 NumPy 上，后续再转 global array）

### 4.2 关键实现点

- `build_chat_prompts()` 用 `tokenizer.apply_chat_template(...)` 统一格式（system + user + generation prompt）。
- tokenization 走 `return_tensors="jax"`（因为 sampler 需要 JAX 张量输入）。
- 通过 `sampler.find_ceil(desired_length)` 找预填充 bucket，pad 到 `prefill_length`，再调用 `sampler.generate(...)`。
- 训练 batch 的构造策略：
  - 先用 sampler 的 `local_token_buffer/local_attention_mask/local_sample_step` 取出真实 completion tokens
  - 再把 prompt 与 completion “拼回去”形成 `train_input_ids/train_attention_mask`
  - `labels` 是 completion mask（只对 completion token 置 1）

## 5) Reward：多 reward function 的加权组合

实现：`plugins/training/grpo/rewarding.py`

- `compute_weighted_rewards(reward_funcs, reward_weights, inputs, answers)`
  - 输出 `rewards_per_func: [num_funcs, B]` 与 `rewards: [B]`
  - 单样本 reward func 抛异常时：**只惩罚该样本**（reward = -1），不中断整步训练（避免“一个坏样本拖垮整次 job”）。

在 `plugins/training/runner/grpo_gsm8k.py` 中，reward functions 直接复用 `training2.py`：

- `reward_correct` / `reward_format` / `tag_count_reward`

## 6) Advantages：从“依赖 group_size reshape”到“显式 group_ids”

实现：`plugins/training/grpo/advantages.py` 的 `compute_grpo_advantages_by_group_id(...)`

### 6.1 为什么必须改：旧的 reshape 方式非常脆弱

上游历史代码（见 `training2.py` 的 `get_advantages(..., groups=...)`）典型写法是：

- `rewards.reshape(-1, groups).mean/std` → `repeat(groups)`

问题在于：

- `groups` 一旦配置错，可能 **不报错但分组错**（静默把不同 prompt 的 completion 混到一起），导致训练信号被污染，bug 难排。
- 不支持 **变长 group**（未来做“过滤无效 completion”“不同 prompt 采样数不同”时必然会遇到）。

### 6.2 新设计：显式 `group_ids: [B]`

在 runner 里（示例）：

- 先采 `batch_size` 个 prompt（每个 prompt 复制 `num_pre_q` 次 rollout）
- `group_ids = np.repeat(np.arange(batch_size), num_pre_q)`

优势计算：

1. `np.unique(group_ids, return_inverse=True)` 把任意 id 映射到连续 `[0..G)`。
2. 用 `np.bincount` 计算每组的 `sum/count` → `mean`。
3. 用 `centered = rewards - mean[group]` 再 `bincount(centered^2)` 得 `var/std`。
4. `advantages = centered / (std[group] + eps)`

这套实现：

- 不要求 group_ids 连续/有序
- 支持 group size 不同
- 只要 `group_ids` 正确，分组永远不会因为 reshape 配置错而静默失真

## 7) Update：梯度计算与参数更新（PPO/GRPO loop）

实现：`plugins/training/grpo/update.py` 的 `ppo_update(...)`；对象封装见 `plugins/training/modules/grpo_sync.py` 的 `PPOUpdateModule`。

核心 contract（和 AReaL 的“阶段化”一致）：

- 第 0 个 PPO step：从 `train_step` 产出 `per_token_logps`，拼成 `old_per_token_logps`
- 后续 PPO step：把 `old_per_token_logps` 放回 batch，再次调用 `train_step`
- 支持 `grad_accum_steps`：通过 `training2.slice_data(x, grad_accum_steps, micro_idx)` 做 micro-batch slicing

上游 loss/optimizer 来自：

- `training2.get_state(...)`：创建 `TrainState`、optimizer（含 grad_accum 初始化）
- `training2.training_step(state, inputs)`：前向/反向与指标（`loss/entropy/per_token_logps/...`）

## 8) 多主机 TPU 关键点（2-host v4-16）

端到端 runner：`plugins/training/runner/grpo_gsm8k.py`

### 8.1 分布式初始化

- `jax.distributed.initialize()`：runner 做 best-effort（重复 init 不致命）。
- dataset 按 process 分片：
  - `qas = qas[jax.process_index() :: jax.process_count()]`

### 8.2 本地 batch 对齐 local devices

`MLLM_JAX.utils._form_global_array()` 会在当前 host 上把数据切到 `len(mesh.local_devices)` 份：

- runner 先计算 `local_batch = batch_size * num_pre_q`
- 若 `local_batch % local_device_count != 0`，会把 `num_pre_q` **向上 padding**到可整除（同时保持 `batch_size` 能整除）

### 8.3 把 host NumPy batch 放到 global mesh

`datas = jax.tree_util.tree_map_with_path(lambda path,x: _form_global_array(path, x, global_mesh=mesh), datas_np)`

- 先在 host 构造 `datas_np`（NumPy）
- 再统一转成 global array（跨 host/device 的 sharded array）

### 8.4 跨 host 统计（用于监控/归一化）

runner 使用 `jax.experimental.multihost_utils.process_allgather`：

- 把各 host 的 `rewards` gather 到每个 host，算 `mean_global/std_global`（用于 log；也便于扩展到“全局 baseline”变体）

## 9) W&B + secret 管理（`.env` + TPU worker 全同步）

### 9.1 本地与 TPU 的 `.env` 读取

入口脚本：`scripts/run_grpo_gsm8k_training.py`

- 会尝试加载：
  - `<repo>/.env`
  - `/root/.env`
- 读取到 `WANDB_API_KEY` 且未显式设置 `WANDB_MODE` 时，默认切到 `WANDB_MODE=online`。

### 9.2 同步 `.env` 到 2-host TPU VM（worker=all）

脚本：`scripts/sync_env_to_tpu_vm.sh`

- 用 `gcloud alpha compute tpus tpu-vm scp --worker=all` 分发 `.env` 到每个 worker 的 `/root/.env`
- 并设 `chmod 600 /root/.env`（降低泄露风险）

### 9.3 只让 process 0 写 W&B

runner：`plugins/training/runner/grpo_gsm8k.py`

- `jax.process_index() == 0` 才 `wandb.init(...)` / `wandb.log(...)`，避免重复 run。

## 10) 已验证的端到端运行（TPU v4-16 + W&B 在线）

可复现步骤与命令：

- 20 steps：`docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-20steps.md`
- 100 steps（含 eval，每 10 steps 跑 1 个 batch）：`docs/sops/tpu-vm-v4-16-grpo-gsm8k-wandb-100steps.md`

验证点（该 SOP 已记录）：

- `process_count=2`（2-host）
- `device_count=8`（v4-16 megacore）
- 训练输出 `step=0..N`（`N=19/99`）并正常退出
- W&B 项目 `mllm-jax-grpo-gsm8k` 有对应 run 与 metrics/timings（只由 process 0 打点）

## 11) 演进路线（对齐 AReaL，但保持最小侵入）

在当前“同步 rollout”基础上，下一步可以自然演进为：

1. **Workflow 对象化**（把 prompt 模板 + sampling + reward 组合成 `Workflow.collect(...)`，runner 只认接口）
2. **异步 rollout**：引入 AReaL 风格的 `submit/wait/prepare_batch` + version/staleness（先把 `version` 写进 schema）
3. **算法变体配置化**：把 advantage 归一化策略、clipping/kl/entropy 等做成 config toggles（避免复制 trainer）

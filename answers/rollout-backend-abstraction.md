# Rollout Backend 抽象（先做 naive sampler，预留 sglang-jax 接入点）

## 你问的点：现在这里没有实现和训练逻辑解耦吗？

结论：**“概念上有分层，但实现上 rollout 仍然强耦合；现在通过 `rollout.backend` 把 rollout 变成可替换 backend 了。”**

### 之前的状态（耦合点）

- 训练 loop 的 phase 分段是清楚的（`rollout → reward → advantages → update`），在 `plugins/training/runner/grpo_gsm8k.py` 里能直接看到（`run_grpo_gsm8k`）。
- 但 **rollout 实现被硬编码**：runner 在 step loop 内直接 import 并调用 `plugins/training/grpo/sampling.generate_answers_and_training_batch`，这意味着：
  - 想换 rollout 引擎（例如 `sglang-jax` / `tunix` 的方式）必须改 runner 内部代码。
  - batch 的构造逻辑依赖当前 sampler 的私有返回字段（例如 `local_token_buffer` 等），进一步加深耦合（见 `plugins/training/grpo/sampling.py:62` 开始）。

### 现在的状态（解耦点）

新增了一层 **rollout backend 抽象**：

- 配置层新增 `rollout.backend`，默认 `naive`：
  - `plugins/training/config.py:20`
  - `plugins/training/configs/grpo_gsm8k_default.yaml:12`
  - CLI 入口支持 env override：`scripts/run_grpo_gsm8k_training.py:82`（`ROLLOUT_BACKEND`）
- runner 不再直接调用具体 sampling 函数，而是通过 backend：
  - backend 创建：`plugins/training/runner/grpo_gsm8k.py:289`
  - rollout 调用：`plugins/training/runner/grpo_gsm8k.py:330`
- 当前只实现 **naive backend**（仍然复用现有 sampler），但 **runner 的依赖点已变成 backend 接口**：
  - 工厂：`plugins/training/rollout_backends/factory.py:13`
  - naive backend：`plugins/training/rollout_backends/naive_sampler.py:10`

这就是“实现与训练解耦”的关键变化：**训练 loop 只依赖 `rollout_backend.rollout(...) -> RolloutResult`，而不再绑定某个 sampler/engine 的具体调用方式。**

---

## 新架构（最小可用版）

### 模块边界

- **Runner（训练编排）**：`plugins/training/runner/grpo_gsm8k.py`
  - 负责：数据采样（挑选 GSM8K 问题）、repeat prompts（K 次采样）、group_ids、reward/adv/update、全局同步、日志。
  - 不负责：具体“怎么生成 token / 怎么组 batch”。
- **Rollout Backend（可替换引擎）**：`plugins/training/rollout_backends/*`
  - 负责：给定 prompts + params，输出 `RolloutResult(chat_prompts, answers, batch)`。
  - `batch` 至少包含：`input_ids / attention_mask / labels`（与 `plugins/training/api/batch_schema.py` 一致）。
- **Sampling（当前实现细节）**：`plugins/training/grpo/sampling.py`
  - 现阶段仍由 naive backend 调用（之后可以被 sglang-jax backend 替换掉）。

### 数据契约（contract）

- `RolloutResult` 定义在 `plugins/training/api/interfaces.py:44`：
  - `chat_prompts: list[str]`
  - `answers: list[str]`
  - `batch: dict[str, Any]`（至少含 `input_ids/attention_mask/labels`）
- batch schema validator：`plugins/training/api/batch_schema.py:51`

---

## 完整链路（从入口到 sampler.generate）

下面按 **“你运行一个训练命令”** 的真实链路列出来（路径 + 关键点）：

1) CLI 入口（读 YAML / env / overrides，构造 dataclass config）

- `scripts/run_grpo_gsm8k_training.py`
  - 读 config：`plugins/training/config.load_config`（`DEFAULT_CONFIG` 已包含 `rollout.backend`）
  - env override：`ROLLOUT_BACKEND`（`scripts/run_grpo_gsm8k_training.py:82`）
  - 构造 `GRPORolloutConfig(backend=...)`：`scripts/run_grpo_gsm8k_training.py:211`

2) Runner 启动（初始化 mesh / state / sampler）

- `plugins/training/runner/grpo_gsm8k.py:94` `run_grpo_gsm8k(cfg)`
  - 调 `training2.get_state(...)` 返回 `state, sampler, _`（仍然是当前 repo 的 in-process 模式）

3) Backend 创建（关键解耦点）

- `plugins/training/runner/grpo_gsm8k.py:289`
  - `rollout_backend = create_rollout_backend(name=cfg.rollout.backend, sampler=sampler)`
- 工厂映射表：`plugins/training/rollout_backends/factory.py:10`
  - 现在支持：`("naive",)`

4) 每个 step 的 rollout（调用 backend，不再直接调用 sampling）

- `plugins/training/runner/grpo_gsm8k.py:330`
  - `rollout = rollout_backend.rollout(prompts=..., params=state.params, ...)`

5) naive backend 内部（复用现有 sampling 逻辑）

- `plugins/training/rollout_backends/naive_sampler.py:25`
  - 调 `plugins/training/grpo/sampling.generate_answers_and_training_batch(...)`

6) sampling 内部（tokenize → pad 到 bucket → sampler.generate → 组 batch）

- `plugins/training/grpo/sampling.py:22`
  - `build_chat_prompts(...)`：把 `system_prompt` + user prompt 用 tokenizer 的 chat template 拼成模型输入
  - tokenize：`tokenizer(chat_prompts, return_tensors="jax", padding=True, padding_side="right")`
  - bucket/prefill：`prefill_length = sampler.find_ceil(desired_length)`（`desired_length=max(global_length, prompt_len)`）
  - **实际生成**：`outputs = sampler.generate(..., params=params)`（`plugins/training/grpo/sampling.py:62`）
  - 从 `outputs["local_token_buffer"/"local_attention_mask"/"local_sample_step"]` 组出：
    - `answers`（decode 生成段）
    - `batch["input_ids"/"attention_mask"/"labels"]`

> 到这里，训练侧拿到的就是 backend-neutral 的 `RolloutResult`，后面的 reward/adv/update 与 rollout 引擎无关。

---

## 如何借鉴 Tunix/sglang-jax（未来接入点长什么样）

你前面让我们研究的 Tunix 链路（已在 `answers/tunix-sglang-jax-rollout-integration.md` 里详细写了）可以直接映射到这里：

- Tunix 里：`SglangJaxRollout` 本质是一个 **Rollout Backend**（把 prompts 丢给 engine，拿 completions）。
- 我们这里：只要实现一个 `SglangJaxRolloutBackend`，遵守同一个 `RolloutBackend.rollout(...) -> RolloutResult` 契约即可。

建议的落地策略（不改上游/保持 plugins-first）：

1) 新增 backend：`plugins/training/rollout_backends/sglang_jax.py`
   - 参考 Tunix 的 Engine 初始化、mesh/device_indexes 的推导、以及权重热更新（如果要做）。
2) 在 `factory.py` 里注册 `sglang_jax`：
   - `SUPPORTED_ROLLOUT_BACKENDS = ("naive", "sglang_jax")`
3) backend 内部产出同样的 `batch`：
   - 你可以沿用 `plugins/training/grpo/sampling.py` 的“组 batch”逻辑，但需要把 sglang-jax 的输出 tokens 对齐到 `input_ids/attention_mask/labels` 的结构。

核心对齐点是：**让 “rollout 引擎” 与 “训练/更新” 之间只通过 `RolloutResult` 交互**。这就是可替换 backend 的关键收益。

---

## 本次改动对应的验证

- 本地单测：`python -m pytest -q`（10 passed）
- Print-config 验证：`python scripts/run_grpo_gsm8k_training.py --print-config | grep -n backend`

对应 SOP：`docs/sops/grpo-rollout-backend-abstraction-naive.md`


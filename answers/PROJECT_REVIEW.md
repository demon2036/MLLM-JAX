# MLLM-JAX 项目审阅（文件作用索引 + GRPO 训练调用路径）

- 仓库路径：`/home/john/github/MLLM-JAX`
- 清理前完整快照：分支 `john`（commit：`6db7d36868943ffad4ce8537b3db2ce7b1cecf6e`）
- 清理后训练分支：`main`（本分支将删除明显与训练无关的文件；清单见文末）
- 文档目的：方便你“逐文件严格审阅”，并把 **GRPO 训练从入口到 loss 的完整调用链**写清楚（含关键数据结构）。

---

## 0. 你应该从哪里开始读

如果你的目标是“先跑起来/先看训练”，建议按这个顺序看：

1. `test_jit9.py`：当前最完整的 **GSM8K +（GRPO 风格 advantage）+ PPO 更新**训练脚本（含采样→奖励→优势→PPO→日志）。
2. `training2.py`：被 `test_jit9.py` 复用的训练组件（`get_state / training_step / reward_* / get_advantages`），**本文件没有 main**。
3. `MLLM_JAX/train_modules/__init__.py`：`TrainGRPOModule`（PPO loss 的核心实现，按 token 计算 ratio+clip）。
4. `MLLM_JAX/sample/sample_state_right_padding2.py`：Qwen2 的 **JAX 采样器**（prefill + decode loop）。
5. `MLLM_JAX/language/qwen2/*` + `MLLM_JAX/language/llama/llama.py`：Qwen2 模型本体与权重转换/分片相关。
6. `MLLM_JAX/utils.py`：mesh/sharding/partition rules、把 numpy 分发到多设备等工具。

---

## 1. GRPO 训练：完整调用路径（主路径：`test_jit9.py`）

这里把“从启动命令 → Python main → 采样 → reward/advantage → PPO 更新 → loss 计算”的路径按层级展开。

### 1.1 外部入口（Shell 脚本）

- `grpo_test.sh`
  - 行为：循环遍历传入参数 `$@`，每轮都会清理进程与 TPU lock，然后 **固定执行** `python -u test_jit9.py`。
  - 注意：如果你直接运行 `bash grpo_test.sh` 且不传参数，`for script in "$@"` 不会进入循环，脚本将“什么也不做”。要么：
    - 直接运行：`python -u test_jit9.py`
    - 或者传一个占位参数：`bash grpo_test.sh dummy`
  - TPU 优化：脚本里设置了 `LIBTPU_INIT_ARGS=...`（一长串 XLA/TPU fusion 参数）。

### 1.2 Python 顶层入口（`test_jit9.py:main()`）

`test_jit9.py` 的整体结构可以概括为：

1. 初始化分布式：`jax.distributed.initialize()`（失败则退化为单进程模式）。
2. 训练配置：`TrainingConfig()`（模型、batch、num_pre_q、ppo_epochs、mesh 形状、wandb 等）。
3. 奖励函数：`reward_setup()` -> `[reward_correct, reward_format, tag_count_reward]` 与权重。
4. 初始化训练环境：`setup_jax(config)`（**创建 mesh / state / sampler / jitted 函数**）。
5. 加载数据：`load_data(config)`（默认 `openai/gsm8k`）。
6. 训练循环 `for step in range(training_steps)`：
   - 构造 prompts（system prompt + user question；可选 replay buffer continuation）
   - 采样生成（`Sampler.generate(...)`）
   - 计算 rewards（多 reward 加权求和）
   - 计算 advantages（`get_advantages` 的不同 estimator：`grpo_clip2` / `reinforce`）
   - PPO 更新（多 epoch，可选 grad_accum；首个 epoch 记录 old_logps）
   - 记录 entropy/长度/reward 等到 wandb

### 1.3 `setup_jax()`：state/sampler/函数 JIT 的构造链

`test_jit9.py:setup_jax(config)` 的核心调用链如下（箭头表示调用方向）：

1. `setup_jax`
   -> `MLLM_JAX.utils.get_jax_mesh2(...)` 创建两套 mesh：
   - `mesh_dp`：用于数据并行/采样参数分布
   - `mesh_fsdp`：用于 FSDP/参数切分（初始化 state 用）
2. `setup_jax`
   -> `training2.get_state(mesh_fsdp, ...)`
   - 内部会：
     1. `MLLM_JAX.sample.sample_state_right_padding2.get_model(mesh, model_path=...)`
        - 从 HF `AutoModelForCausalLM` 拉取 torch 权重并 `state_dict()`
        - `convert_torch_to_flax_llama(...)` 转为 Flax params tree
        - `match_partition_rules(get_partition_rules_llama(), ...)` 做参数分片规则匹配
        - 将 params `device_put` 到 mesh 对应设备上
        - 返回：`model(Flax)`, `params`, `tokenizer`
     2. 构造参考模型：`get_model(..., only_model=True)`（只返回结构，不加载 params）
     3. 构造 `TrainGRPOModule`：
        - `train_module = flax.linen.remat(TrainGRPOModule, ...)(model=model, ref_model=model_ref, ...)`
        - 这里的 `TrainGRPOModule` 定义在 `MLLM_JAX/train_modules/__init__.py`
     4. `TrainState.create(...)`：
        - optimizer：`optax.lion(...)` + `optax.clip_by_global_norm(1.0)`
        - schedule：`optax.warmup_cosine_decay_schedule(...)`
        - `state.apply_fn = train_module.apply`
        - `state.ref_params = deepcopy(params)`（当 `beta != 0` 时）
     5. 返回 `state, sampler, train_state_sharding`
3. `setup_jax`
   -> JIT/分片工具：
   - `params_to_dp = jax.jit(init_fn, out_shardings=params_sharding_dp)`
   - `get_advantages_jitted_funcs = { 'grpo_clip2': jax.jit(partial(get_advantages,...)), 'reinforce': ... }`
   - `train_fn_jit = jax.jit(training_step, donate_argnums=(0,))`

### 1.4 训练循环：数据与张量流（关键字段约定）

在 `test_jit9.py` 的训练循环中，`datas`（后续会变成 JAX array）包含这些关键键：

- `datas['input_ids']`（`np.int32`）：
  - 形状：`[B, T]`
  - 内容：prompt tokens + sampled completion tokens（右侧 padding）
- `datas['attention_mask']`（`np.int32`）：
  - 形状：`[B, T]`
  - 1 表示有效 token
- `datas['labels']`（`np.int32`）：
  - 形状：`[B, T]`
  - 这里用于 **completion mask**：仅 completion 区间为 1（prompt 区间为 0）
- `datas['rewards']`（`np.float32`/`np.float64`，由 numpy 产生）：
  - 形状：`[B]`
  - 多 reward 函数加权求和的总 reward
- `datas['advantages']`（`jax.numpy` array）：
  - 形状：`[B]`
  - 来自 `training2.get_advantages(...)`（按 `groups=num_pre_q` 做分组归一化/混合）
- `datas['old_per_token_logps']`（仅在 `ppo_epoch>0` 时存在）：
  - 形状：`[B, T-1]`
  - 来自第 0 个 PPO epoch 的 `per_token_logps` 拼接，用于后续 epoch 计算 ratio

**GRPO 的“组”概念在这里的体现：**

- `num_pre_q`（config 中默认 16）表示每个原始问题重复采样 N 次；
- rewards/advantages 都按 `groups=num_pre_q` 进行分组统计（例如 `grpo_clip2`）。

### 1.5 PPO 更新与 loss 计算链（最重要的一段）

从 `test_jit9.py` 到真正 loss 的调用链是：

1. `test_jit9.perform_ppo_update(...)`
   - 把 numpy 数据通过 `MLLM_JAX.utils._form_global_array(...)` 放到 `mesh_dp` 对应设备
   - 循环 `ppo_epochs`
   - 每个 epoch 内部调用 `train_fn_jit(state, local_data)`
2. `training2.training_step(state, inputs)`
   - `jax.value_and_grad(loss_fn, has_aux=True)(state.params)`
   - `loss_fn` 内部调用：
3. `state.apply_fn({'params': {'model': params, 'ref_model': state.ref_params}}, inputs)`
   -> 实际执行的是：
4. `MLLM_JAX.train_modules.TrainGRPOModule.__call__(inputs)`
   - `self.model(...)` 前向得到 `logits`
   - （可选）`self.ref_model(...)` 得到 `ref_logits`（**当前实现里计算了 ref_logps，但未把 KL penalty 真正加到 loss**）
   - 计算 `per_token_logps`
   - 读取/生成 `old_per_token_logps`
   - `ratio = exp(per_token_logps - old_per_token_logps)`
   - `clipped_ratio = clip(ratio, 1-ε_low, 1+ε_high)`
   - `advantages` 从 `inputs['advantages']` broadcast 成 `[B, 1]`
   - `per_token_ppo_loss = min(ratio*A, clipped_ratio*A)`
   - `loss = -sum(per_token_ppo_loss * completion_mask) / total_valid_token_count`
   - 返回 `{'loss', 'per_token_logps', 'entropy', 'entropy_loss', ...}`

---

## 2. 文件作用索引（按目录）

说明：

- “入口脚本”一般带 `if __name__ == "__main__":`。
- 本分支 `main` 已删除：`app*/`、`deprecated_*`、`MLLM_JAX/deprecated/`、`.idea/`、`__pycache__/`、`*.pyc` 等非训练主体文件；完整历史可切到 `john` 查看。

### 2.1 顶层（root）

- `answers/PROJECT_REVIEW.md`：项目审阅文档（本文件）。
- `clean.sh`：清理脚本（循环参数；kill python、删除 `/tmp/libtpu_lockfile`、activate conda）。
- `grpo_test.sh`：GRPO/训练启动脚本（见上文，实际运行 `test_jit9.py`）。
- `setup.sh`：环境安装脚本（miniconda、jax[tpu]、flax/optax、fastapi/uvicorn、math_verify 等）。
- `test_jit8.py`：较早的 JAX 训练/采样实验脚本（包含奖励与优势计算雏形）。
- `test_jit9.py`：**当前主 GRPO/PPO 训练脚本**（GSM8K；动态 advantage estimator；replay buffer）。
- `test_jit10.py`：`test_jit9.py` 的变体（参数/逻辑略有不同，用于对比实验）。
- `test_jit11.py`：`test_jit9.py` 的变体（参数/逻辑略有不同，用于对比实验）。
- `training2.py`：**训练组件模块**（`get_state/training_step/reward_* / get_advantages`；被 `test_jit9/10/11` 复用）。

### 2.3 `prompts/`（提示词模板）

- `prompts/prompts.py`：训练/生成用 system prompt（要求输出 `<think>...</think>\\n<answer>...</answer>`），与 `reward_format/tag_count_reward` 配套。

### 2.6 `MLLM_JAX/`（核心库代码）

> 该包整体目标：提供一套基于 JAX/Flax 的 LLM/MLLM 模型实现、采样器、分片规则、以及训练模块（含 GRPO/PPO loss）。

#### 2.6.1 顶层模块

- `MLLM_JAX/__init__.py`：包初始化（当前为空文件）。
- `MLLM_JAX/activations.py`：激活函数表 `ACT2FN`（含 `quick_gelu`）。
- `MLLM_JAX/efficient.py`：Pallas/TPU 上的 attention/flash_attention 实验实现（kernel + wrapper）。
- `MLLM_JAX/efficient2.py`：attention kernel 的另一版实验实现（分块计算/不同 grid）。
- `MLLM_JAX/efficient3.py`：attention kernel 的另一版实验实现（不同并行维度语义）。
- `MLLM_JAX/mask.py`：Gemma 风格 attention mask 工具（causal + bidirectional image tokens）。
- `MLLM_JAX/multinomial_sample.py`：Gemma3 多模态采样/推理与 torch→flax 参数转换实验（定义 `SampleState`/`Sampler` 等）。
- `MLLM_JAX/utils.py`：通用工具（mesh 构建、PartitionSpec 匹配、checkpoint 保存、数据收集等）。

#### 2.6.2 `MLLM_JAX/train_modules/`

- `MLLM_JAX/train_modules/__init__.py`：
  - `TrainGRPOModule(nn.Module)`：PPO loss（ratio+clip）实现；输入依赖 `inputs['advantages']` 与 completion mask。
  - `get_advantages(...)`：一个简化版 advantage 计算（当前 `TrainGRPOModule` 内部并未调用，主训练脚本用的是 `training2.get_advantages`）。

#### 2.6.3 `MLLM_JAX/sample/`（采样器与生成状态）

- `MLLM_JAX/sample/__init__.py`：包初始化。
- `MLLM_JAX/sample/sanple_utils.py`：采样工具函数（top-k、temperature、nucleus 等；含 `_top_k_sampling_batched`）。
- `MLLM_JAX/sample/sample_state_left_padding.py`：采样 state/生成逻辑（left padding 版本，实验用）。
- `MLLM_JAX/sample/sample_state_right_padding.py`：采样 state/生成逻辑（right padding 版本，旧）。
- `MLLM_JAX/sample/sample_state_right_padding2.py`：采样 state/生成逻辑（right padding 版本，当前 `training2.get_state` 在用）。
- `MLLM_JAX/sample/sample_state_right_padding3.py`：采样 state/生成逻辑的另一实验版本。

#### 2.6.4 `MLLM_JAX/language/`（语言模型实现）

- `MLLM_JAX/language/__init__.py`：子包初始化。
- `MLLM_JAX/language/gemma/__init__.py`：Gemma 子包初始化。
- `MLLM_JAX/language/gemma/layers.py`：Gemma 层定义（attention/MLP 等）。
- `MLLM_JAX/language/gemma/modules.py`：Gemma 模块组装（block/embedding 等）。
- `MLLM_JAX/language/gemma/params.py`：Gemma 参数/权重相关工具。
- `MLLM_JAX/language/gemma/positional_embeddings.py`：Gemma 位置编码实现。
- `MLLM_JAX/language/gemma/sampler.py`：Gemma 采样/生成相关实现。
- `MLLM_JAX/language/gemma/transformer.py`：Gemma Transformer 主体实现（含 `TransformerConfig`、`Transformer(nn.Module)` 等）。
- `MLLM_JAX/language/gemma3/__init__.py`：Gemma3 子包初始化。
- `MLLM_JAX/language/gemma3/configuration_gemma3.py`：Gemma3 配置。
- `MLLM_JAX/language/gemma3/modeling_gemma3.py`：Gemma3 模型实现（JAX/Flax；含 `Gemma3TextModel`、`GemmaForCausalLM` 等）。
- `MLLM_JAX/language/gemma3/model_torch_gemma3.py`：Gemma3 torch 侧模型/对齐辅助。
- `MLLM_JAX/language/llama/__init__.py`：LLaMA 子包初始化。
- `MLLM_JAX/language/llama/configuration_llama.py`：LLaMA 配置类。
- `MLLM_JAX/language/llama/llama.py`：torch→flax 参数转换 + `LlamaJaxConfig` 等（被采样器/模型加载复用）。
- `MLLM_JAX/language/llama/modeling_llama.py`：LLaMA 模型实现。
- `MLLM_JAX/language/llama/ref.py`：参考实现/对齐辅助（按文件内容审阅）。
- `MLLM_JAX/language/llama/test.py`：LLaMA 测试脚本。
- `MLLM_JAX/language/qwen2/__init__.py`：Qwen2 子包初始化。
- `MLLM_JAX/language/qwen2/configuration_qwen2.py`：Qwen2 配置与 KV cache 工具（`Qwen2Config`；`init_cache/pad_cache/pad_cache_right` 等）。
- `MLLM_JAX/language/qwen2/modular_qwen2.py`：Qwen2 模型实现（`Qwen2Model`、`Qwen2ForCausalLM` 等；被采样器调用）。
- `MLLM_JAX/language/qwen3/__init__.py`：Qwen3 子包初始化。
- `MLLM_JAX/language/qwen3/configuration_qwen3.py`：Qwen3 配置。
- `MLLM_JAX/language/qwen3/modular_qwen3.py`：Qwen3 模型实现（`Qwen3ForCausalLM` 等）。
- `MLLM_JAX/language/qwen3_moe/__init__.py`：Qwen3 MoE 子包初始化。
- `MLLM_JAX/language/qwen3_moe/configuration_qwen3.py`：Qwen3 MoE 配置。
- `MLLM_JAX/language/qwen3_moe/modular_qwen3.py`：Qwen3 MoE 模型实现（稀疏 MoE block；`Qwen3MoeForCausalLM` 等）。
- `MLLM_JAX/language/qwen3_moe/qwen3_torch.py`：Qwen3 MoE torch 侧实现/对齐辅助。

#### 2.6.5 `MLLM_JAX/vision/`（视觉编码器）

- `MLLM_JAX/vision/__init__.py`：子包初始化。
- `MLLM_JAX/vision/clip/`：CLIP 相关实现目录（本仓库未提供 `__init__.py`，通过 namespace package 方式导入）。
- `MLLM_JAX/vision/clip/clip.py`：CLIP 视觉侧实现（`CLIPVisionEmbeddings/CLIPVisionTransformer/CLIPVisionModel` 等）。
- `MLLM_JAX/vision/clip/configuration_clip.py`：CLIP 配置。
- `MLLM_JAX/vision/clip/ref.py`：参考实现/对齐辅助。
- `MLLM_JAX/vision/clip/vit.py`：ViT backbone（供 CLIP 使用）。
- `MLLM_JAX/vision/siglip/`：SigLIP 相关实现目录（本仓库未提供 `__init__.py`，通过 namespace package 方式导入）。
- `MLLM_JAX/vision/siglip/configuration_siglip.py`：SigLIP 配置。
- `MLLM_JAX/vision/siglip/embedding.py`：SigLIP embedding/patch 等组件。
- `MLLM_JAX/vision/siglip/modeling_siglip.py`：SigLIP 视觉侧实现（`SiglipVisionEmbeddings/SiglipVisionTransformer/SiglipVisionModel` 等）。
- `MLLM_JAX/vision/siglip/modeling_siglip_torch.py`：SigLIP torch 侧实现/对齐辅助。
- `MLLM_JAX/vision/siglip/siglipmlp.py`：SigLIP MLP/模块实现。
- `MLLM_JAX/vision/siglip/vit_pali.py`：面向 Pali/Gemma 系列的 ViT 变体/适配。

#### 2.6.6 `MLLM_JAX/mutinomial/`（多模态/LLava/Gemma3 相关）

> 注意：目录名为 `mutinomial`（疑似 “multinomial” 的拼写变体），但以仓库实际路径为准。

- `MLLM_JAX/mutinomial/__init__.py`：子包初始化。
- `MLLM_JAX/mutinomial/gemma3/__init__.py`：Gemma3 多模态子包初始化。
- `MLLM_JAX/mutinomial/gemma3/configuration_gemma3.py`：Gemma3 多模态配置。
- `MLLM_JAX/mutinomial/gemma3/modeling_gemma3.py`：Gemma3 多模态模型实现（`Gemma3MultiModalProjector`、`Gemma3ForConditionalGeneration` 等；含 torch→flax conversion 入口）。
- `MLLM_JAX/mutinomial/llava/__init__.py`：LLaVA 子包初始化。
- `MLLM_JAX/mutinomial/llava/configuration_llava.py`：LLaVA 配置。
- `MLLM_JAX/mutinomial/llava/modeling_llava.py`：LLaVA 模型实现（`LlavaMultiModalProjector`、`LlavaForConditionalGeneration` 等）。

#### 2.6.7 `MLLM_JAX/kernels/megablox/`（自定义 kernel/算子）

- `MLLM_JAX/kernels/__init__.py`：子包初始化。
- `MLLM_JAX/kernels/megablox/__init__.py`：megablox 子包初始化。
- `MLLM_JAX/kernels/megablox/common.py`：TPU 类型/能力检测与 dtype 选择工具。
- `MLLM_JAX/kernels/megablox/gmm.py`：GMM/grouped matmul 相关实现。
- `MLLM_JAX/kernels/megablox/ops.py`：megablox 算子封装。

---

## 3. main 分支已删除内容（完整历史在 `john`）

以下内容与“训练主路径（`test_jit9.py` + `training2.py` + `MLLM_JAX/*`）”无直接依赖，已从 `main` 移除：

- 推理/服务端与代理：`app/`、`app2.py`、`app3.py`、`app.sh`、`tes_server4.py`、`prompt.py`、`safe_decode.py`、`tes_server.py`
- 历史实验：`deprecated_grpo/`、`deprecated_huan/`、`MLLM_JAX/deprecated/`
- 缓存/IDE：`.idea/`、`__pycache__/`、`**/__pycache__/`、`*.pyc`
- 额外推理测试：`test_qwen.py`、`MLLM_JAX/test_qwen.py`

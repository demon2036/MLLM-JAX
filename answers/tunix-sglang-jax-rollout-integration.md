# Tunix 如何调用 sglang-jax 做 RL Rollout（以及迁移到本仓库的借鉴方案）

> 目的：
>
> 1) 在 `workdir/` 里 clone `sglang-jax` 与 `tunix`，基于源码梳理 **tunix → sglang-jax rollout** 的完整调用链路与架构；  
> 2) 结合本仓库当前的 GRPO/GSM8K 训练链路，给出 **非侵入（plugins-first）** 的可落地借鉴方案（包含路径/模块/执行链路）。

---

## 0) 本次分析的代码版本（commit 固化）

本仓库使用 repo-local `workdir/`（已在 `.gitignore` 忽略）：

- Tunix：`workdir/tunix`（origin: `https://github.com/google/tunix.git`）
  - 本次检视 commit：`f71f2f76c7f303b6a63cb208ff5bc7adb0d9250d`
- sglang-jax：`workdir/sglang-jax`（origin: `https://github.com/sgl-project/sglang-jax.git`）
  - 本次检视 commit：`a516381961e2cf8dd15bfc6c05c72b3702c6d6c6`

---

## 1) 结论先行：tunix 的 sglang-jax rollout 是“库级嵌入式 Engine”，不是 HTTP client

在 Tunix 里，“sglang-jax rollout”不是通过 HTTP 调一个外部服务，而是 **直接 import sglang-jax 的 Python Engine 并在进程内启动其 runtime**：

- 关键 import：`workdir/tunix/tunix/generate/sglang_jax_sampler.py`
  - `from sgl_jax.srt.entrypoints.engine import Engine`
- 关键调用：`SglangJaxSampler.__call__` 内部直接 `self.engine.generate(...)`

这意味着：

1) rollout 线程/子进程、KV cache、调度器全部在当前训练进程里启动；  
2) 权重同步可以“就地修改 Engine 内部 model_state”（tunix 正是这么做的）；  
3) 但也带来 **内存复制/资源竞争** 的现实问题（见第 6 节迁移建议）。

---

## 2) Tunix 侧的整体架构：RLCluster + 可插拔 Rollout Engine

### 2.1 关键文件（路径索引）

Tunix 里与 rollout engine 选择相关的主路径：

- Demo/入口脚本（选择 rollout_engine）：`workdir/tunix/scripts/grpo_demo_llama3_qwen2.py`
- 集群抽象与训练/推理角色：`workdir/tunix/tunix/rl/rl_cluster.py`
- rollout 抽象接口与配置：`workdir/tunix/tunix/rl/rollout/base_rollout.py`
- sglang-jax rollout 实现：`workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
- sglang-jax sampler（真正对接 Engine 的地方）：`workdir/tunix/tunix/generate/sglang_jax_sampler.py`

### 2.2 组件关系（文本架构图）

Tunix 把 RL 训练拆成一个“cluster”（多角色、可 disaggregate 的 mesh），其中 rollout 只是一个可替换的后端：

```text
scripts/grpo_demo_llama3_qwen2.py
  └─ rl_cluster = RLCluster(actor=..., reference=..., tokenizer=..., cluster_config=...)
       ├─ Role.ACTOR     -> Trainer (backprop)
       ├─ Role.REFERENCE -> InferenceWorker (ref logps)
       └─ Role.ROLLOUT   -> BaseRollout (pluggable)
            ├─ VanillaRollout  (纯 JAX decode)
            ├─ VllmRollout     (vLLM TPU backend)
            └─ SglangJaxRollout (sglang-jax Engine)
```

### 2.3 从脚本选择 sglang-jax rollout

Demo 脚本里通过 `args.rollout_engine` 切换（支持 `"vanilla"`, `"vllm"`, `"sglang_jax"`）：

- 选择与构造 rollout 配置：`workdir/tunix/scripts/grpo_demo_llama3_qwen2.py`
  - `def get_rollout_config(engine: str) -> base_rollout.RolloutConfig`
  - engine 为 `"sglang_jax"` 时，会填充 `RolloutConfig` 的 sglang-jax 专属字段：
    - `rollout_sglang_jax_model_version`
    - `rollout_sglang_jax_mem_fraction_static`
    - `rollout_sglang_jax_disable_radix_cache`
    - `rollout_sglang_jax_precompile_*`
    - LoRA 静态配置（可选）

然后把它塞进 `ClusterConfig` 并创建 `RLCluster`：

```python
# workdir/tunix/scripts/grpo_demo_llama3_qwen2.py
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={Role.ACTOR: ..., Role.REFERENCE: ..., Role.ROLLOUT: ...},
    rollout_engine=args.rollout_engine,
    rollout_config=get_rollout_config(args.rollout_engine),
)
rl_cluster = rl_cluster_lib.RLCluster(actor=training_model, reference=ref_model, tokenizer=model_tokenizer, cluster_config=cluster_config)
```

---

## 3) Tunix → sglang-jax 的 rollout 调用链路（完整链路）

### 3.1 “生成”主链路：RLCluster.generate → SglangJaxRollout.generate → Engine.generate

1) 业务侧（GRPO learner）调用：

- `workdir/tunix/tunix/rl/grpo/grpo_learner.py`
  - `rollout_output = self.rl_cluster.generate(prompts=..., ...)`

2) `RLCluster.generate(...)` 对 prompts 做 chat template（可选），然后 micro-batch chunk，并调用 `self.rollout.generate(...)`：

- `workdir/tunix/tunix/rl/rl_cluster.py`
  - `def generate(...) -> base_rollout.RolloutOutput`
  - 核心：`self.rollout.generate(string_prompts[s], rollout_config)`

3) 当 rollout_engine=`"sglang_jax"` 时，`RLCluster` 在初始化阶段选择：

- `workdir/tunix/tunix/rl/rl_cluster.py`
  - `elif self.cluster_config.rollout_engine == "sglang_jax":`
  - `self._rollout = sglang_jax_rollout.SglangJaxRollout(...)`

4) `SglangJaxRollout.generate(...)` 只是一个薄封装，最终调用 sampler：

- `workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
  - `self.output = self._sampler(input_strings=prompts, ...)`

5) `SglangJaxSampler.__call__(...)` 直接调用 sglang-jax Engine：

- `workdir/tunix/tunix/generate/sglang_jax_sampler.py`
  - `self.engine = Engine(**self.args)`
  - `outputs = self.engine.generate(input_ids=[...], sampling_params=sampling_params)`

到这里已经完成了 “tunix 如何调用 sglang-jax 做 rollout”的核心链路：**tunix 把 sglang-jax 当作一个进程内可调用的 inference runtime。**

---

## 4) sglang-jax Engine 的内部架构（为什么 tunix 能“进程内”启动它）

sglang-jax Engine 的自解释注释非常清晰：

- `workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py`
  - `class Engine(EngineBase):`
  - 注释描述 3 个核心组件：
    1. `TokenizerManager`
    2. `Scheduler`（默认子进程；或 thread 模式）
    3. `DetokenizerManager`（默认子进程；或 thread 模式）
  - 进程间通信使用 ZMQ（Engine 内部会分配 IPC ports：`PortArgs.init_new(server_args)`）

Tunix 对接 rollout 时非常关键的一点是：

- Tunix 配置里默认会把 `enable_single_process=True`（`workdir/tunix/tunix/generate/sglang_jax_sampler.py:SglangJaxConfig.enable_single_process`）
- Engine 会据此改为线程模式：
  - `workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py:_launch_subprocesses_or_threads`
  - `if server_args.enable_single_process: return _launch_threads(...)`

原因：**只有在 single-process/thread 模式下，tunix 才能在 Python 侧直接拿到 scheduler 对象，从而做“权重热更新”（见下一节）。**

### 4.1 Tunix mesh 如何变成 sglang-jax 的 `tp_size/device_indexes`

tunix 并不会把 `jax.sharding.Mesh` 直接传给 sglang-jax（Engine 的参数里没有 mesh），而是把 mesh “翻译”为两组关键参数：

- `workdir/tunix/tunix/generate/sglang_jax_sampler.py:SglangJaxSampler._sglang_jax_config`
  - `args["tp_size"] = math.prod(mesh.shape.values())`
  - `args["device_indexes"] = mesh.device_ids.flatten().tolist()`

然后在 sglang-jax 内部，Scheduler 会用这两个参数重新创建它自己的 device mesh：

- `workdir/sglang-jax/python/sgl_jax/srt/managers/scheduler.py:Scheduler.__init__`
  - `self.mesh = create_device_mesh(..., ici_parallelism=[-1, self.tp_size], device_indexes=server_args.device_indexes)`

这也解释了 tunix 里 `_find_tp_size` 的注释：**sglang-jax 目前不支持 DP**，所以 tunix 侧 mesh 的各个维度会被“拍扁”成一个 `tp_size`（`math.prod(...)`）。

### 4.2 Engine.generate 的请求流（从 Python 调用到 ModelRunner）

从 tunix `self.engine.generate(...)` 进入 sglang-jax 后，内部的“请求 → 调度 → forward → 解码”链路大致是：

1) `workdir/sglang-jax/python/sgl_jax/srt/entrypoints/engine.py:Engine.generate`
   - 构造 `GenerateReqInput`，调用 `TokenizerManager.generate_request(...)`
2) `workdir/sglang-jax/python/sgl_jax/srt/managers/tokenizer_manager.py:TokenizerManager.generate_request`
   - tokenization / 参数校验，随后把请求发给 Scheduler（ZMQ IPC）
3) `workdir/sglang-jax/python/sgl_jax/srt/managers/scheduler.py:Scheduler.event_loop_*`
   - continuous batching 调度，并调用 `self.tp_worker.forward_batch_generation(...)`
4) `workdir/sglang-jax/python/sgl_jax/srt/managers/tp_worker.py:ModelWorker`
   - 持有 `self.model_runner = ModelRunner(...)`，最终走到
5) `workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py:ModelRunner.jitted_run_model`
   - 从 `model_state_leaves` unflatten 出 `model_state` 并 forward（也是 tunix 能热更新权重的关键点）

---

## 5) 权重同步（rollout on-policy）的实现：tunix 直接改 sglang-jax ModelRunner 的 model_state_leaves

> 这部分决定了“sglang-jax 是否能用于 RL rollout（随训练更新权重）”。

### 5.1 权重同步入口：RLCluster.sync_weights()

- `workdir/tunix/tunix/rl/rl_cluster.py`
  - `def sync_weights(self):`
  - 取 actor trainer 的 params（LoRA/Full 取决于 filter_types），然后：
    - `self.rollout.update_params(src_filtered_params, filter_types)`

### 5.2 sglang-jax rollout 的 update_params：transfer + reshard + 写回 model_state_leaves

在 sglang-jax sampler 里：

- `workdir/tunix/tunix/generate/sglang_jax_sampler.py`
  - `def update_params(self, updated_weights, filter_types=None):`
  - 核心步骤：
    1. `dst_state = self.transformer_state`（从 Engine 的 model_runner 拿 nnx state）
    2. `new_state = utils.transfer_state_with_mappings(...)`
    3. `new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)`
    4. `self._model_runner.model_state_leaves = new_model_state_leaves`

其中 `_model_runner` 的拿法是：

```python
# workdir/tunix/tunix/generate/sglang_jax_sampler.py
return self.engine.scheduler_info["scheduler"].tp_worker.worker.model_runner
```

只有 `enable_single_process=True` 时，`scheduler_info` 里才会有这个对象（否则 scheduler 在子进程里）。

### 5.3 为什么改 model_state_leaves 就能“热更新”权重

sglang-jax 的 ModelRunner 在 JIT 初始化时会把 model_state flatten 成 leaves，然后 `jitted_run_model` 每次都从 leaves unflatten：

- `workdir/sglang-jax/python/sgl_jax/srt/model_executor/model_runner.py`
  - `self.model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)`
  - `model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)`

因此，只要替换 `model_state_leaves`，下一次 forward 就会用新的权重。

### 5.4 mappings：tunix 如何把 trainer 权重“翻译”成 sglang-jax 的权重树

tunix 用 `MappingConfig` 解决“训练模型参数树”与“rollout engine 参数树”结构不一致的问题：

- MappingConfig 构建：
  - `workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
    - `mapping_config = mappings.MappingConfig.build(... backend="sglang_jax")`
- sglang-jax 的 mapping table（按模型族拆分）：
  - Llama3：`workdir/tunix/tunix/models/llama3/mapping_sglang_jax.py`
  - Qwen2：`workdir/tunix/tunix/models/qwen2/mapping_sglang_jax.py`
- 应用 mapping 的实现：
  - `workdir/tunix/tunix/generate/utils.py:transfer_state_with_mappings`

此外还有两个“必须知道”的细节：

1) LoRA 的 key 改写：sglang-jax 用 `base_layer` 包裹底层线性层，tunix 会改 mapping target path：
   - `workdir/tunix/tunix/generate/sglang_jax_sampler.py:update_hf_key_mappings_with_lora`
2) `SGLANG_JAX_TP_AXIS_NAME` 默认是 `"tensor"`（用于映射里的 sharding spec）：
   - `workdir/tunix/tunix/utils/env_utils.py`

---

## 6) 把这套方案迁移到本仓库：我们应该借鉴什么？怎么落地？

本仓库当前的 GRPO 训练链路（建议先看现有总结）：

- 训练架构总览：`answers/mllm-jax-grpo-training-architecture.md`
- runner：`plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k`
- rollout（当前实现）：`plugins/training/grpo/sampling.py:generate_answers_and_training_batch`
- 低层采样器：`MLLM_JAX/sample/sample_state_right_padding2.py:Sampler.generate`

### 6.1 现状 vs tunix（对齐点）

两者的共同点：

- 都是同步 pipeline：`rollout → reward → advantages → update`
- 都有“可替换/可重构”的 rollout 边界（本仓库已经拆到 `plugins/training/grpo/sampling.py`）

关键差异：

- 本仓库 rollout 直接复用训练用的 `Sampler.generate`（同一份 params/mesh，几乎不复制权重）
- tunix 的 sglang-jax rollout 是 **另起一个 Engine**，需要把权重同步到 Engine（天然会带来权重复制与 reshard 成本）

因此，“借鉴”时必须正视两个工程现实：

1) **内存**：同一块 TPU 上同时持有 `trainer weights + optimizer states + ref weights + sglang-jax engine weights + KV cache`，很容易炸。  
2) **sharding 转换**：本仓库训练常用 `mesh_shape="1,-1,1"`（纯 FSDP），而 sglang-jax Engine 主要按 `"tensor"` 轴做 TP；两者不匹配时，同步权重会触发跨设备重分片（很重）。

### 6.2 最推荐的借鉴路径：先把“rollout engine 抽象”做出来，再决定是否上 sglang-jax

我们不需要一开始就把 sglang-jax 接进来；先把“接口/边界”确定，未来换后端就轻松很多（这也是 tunix 的核心价值之一）。

建议在本仓库（plugins-first）新增一个 rollout engine 抽象层，参考 tunix 的 `BaseRollout`：

```text
plugins/
  rollout_engines/
    base.py                # Protocol / dataclass: RolloutConfig, RolloutOutput
    native_sampler.py      # 复用现有 Sampler.generate（当前实现迁移/封装）
    sglang_jax/
      engine.py            # Engine 生命周期（create/shutdown）、sampling params
      weights.py           # trainer params -> sglang-jax model_state 的映射与同步
      rollout.py           # 统一 generate() 输出成 (text/tokens/prompt_tokens)
```

然后 `plugins/training/grpo/sampling.py` 只做：

- prompt template
- tokenization（或把 tokenization 也放进 engine）
- 把 engine.generate 的结果标准化为训练 batch（`input_ids/attention_mask/labels`）

这样，“rollout 后端是否是 sglang-jax”就变成配置层的问题（YAML/CLI），而不是侵入式改训练逻辑。

### 6.3 如果确定要用 sglang-jax 做 rollout：两种可行部署形态

#### A) Colocated（同一份 TPU slice 上串行使用）：最简单，但最容易 OOM

特点：

- rollout 和 training 复用同一组 device；
- 每个 step 先 rollout，再 train（串行，不并发）；
- 但需要同时驻留两套模型权重（训练模型 + sglang-jax Engine 模型）。

适用场景：

- 模型足够小，且优化器/参考模型开销可控；
- 或能做到“trainer weights 与 engine weights 共享/复用”（tunix 也在 TODO 里指出它还没优化到这一步：`workdir/tunix/tunix/generate/sglang_jax_sampler.py`）。

#### B) Disaggregated（训练与 rollout 用不同设备池）：更符合 tunix 设计初衷

特点：

- training 使用 device pool A，rollout Engine 使用 device pool B；
- 需要在 step 边界做权重同步（reshard + copy）；
- 资源隔离更清晰、rollout 不会挤爆训练侧 HBM。

适用场景：

- 你愿意为 rollout 额外申请 TPU（例如 v4-32 / v4-64）；
- 你确实被 rollout 性能卡住（推理吞吐成为 RL 训练瓶颈）。

> 本仓库目前的 TPU SOP 以 v4-16 为主（2-host）。如果要 disaggregate，SOP 也应该新增一份“多 slice 分配与同步策略”的流程（后续可补）。

### 6.4 迁移到本仓库时，“权重同步”要怎么做？（可选路线）

参考 tunix 的三条路线（按工程复杂度排序）：

1) **只做 SFT / 固定 rollout 权重**：不需要权重同步（最简单，但不 on-policy）  
2) **LoRA-only 同步**：训练只更新 LoRA 参数，sglang-jax 开 `enable_static_lora`，每步只同步 LoRA buffers  
3) **Full weights 同步**：每步把完整 params reshard 并写入 Engine（最重，但最通用）

在本仓库，当前训练是 full fine-tune（`training2.py:get_state` 直接优化 `params`），并且还持有 `ref_params`（beta!=0 时 copy 全量权重）。因此若要接 sglang-jax，建议优先评估：

- 能否把训练切到 LoRA（显著降低同步成本与 HBM 压力）；
- 或至少把 reference model 从“全量 copy params”改为更省内存的方案（否则 colocated 基本必炸）。

### 6.5 本仓库里“插入点”在哪里（完整链路对齐）

现有链路（本仓库）：

```text
scripts/run_grpo_gsm8k_training.py
  └─ plugins/training/runner/grpo_gsm8k.py:run_grpo_gsm8k
       ├─ training2.get_state(...) -> (state, sampler, sharding)
       ├─ plugins/training/grpo/sampling.generate_answers_and_training_batch(...)
       │    └─ MLLM_JAX/sample/sample_state_right_padding2.py:Sampler.generate(...)
       └─ plugins/training/grpo/update.ppo_update(...) -> training_step(...)
```

要借鉴 tunix，把 rollout 替换为 sglang-jax，最小侵入的做法是：

- 保持 runner/四阶段结构不动；
- 只替换 rollout 模块 `plugins/training/grpo/sampling.py` 的底层生成实现，让它调用一个“rollout engine”：
  - `NativeSamplerRolloutEngine`（现有 `Sampler.generate`）
  - `SglangJaxRolloutEngine`（新实现，对接 `sgl_jax.srt.entrypoints.engine.Engine`）

并在每个 step（或每 N step）调用：

- `rollout_engine.sync_weights(state.params)`（类比 `RLCluster.sync_weights()`）

---

## 7) 风险点/坑位清单（迁移前必须明确）

1) **HBM 预算**：colocated 下多份权重常常直接 OOM；disaggregated 才是长期解。  
2) **sharding 不匹配导致 reshard 很贵**：本仓库常用 FSDP；sglang-jax 偏 TP。  
3) **权重树命名不一致**：本仓库 params（flax.linen）与 tunix（nnx）/sglang-jax（nnx）命名和形状可能不一致，需要 mapping+transpose（tunix 的 mapping 文件可参考但不可直接照搬）。  
4) **sglang_jax_sampler 里的 load_format 逻辑疑似覆盖 bug**：
   - `workdir/tunix/tunix/generate/sglang_jax_sampler.py` 里先写 `args["load_format"]="dummy"`，后又无条件 `args["load_format"]=config.load_format`。
   - 如果你想“随机初始化 + 纯 in-memory sync”，需要确认 `rollout_sglang_jax_load_format` 是否显式设为 `"dummy"`。  
5) **logprobs / old_logps 需求**：tunix 的 sglang-jax sampler 当前返回 `logits=None, logprobs=None`；如果你在算法里需要这些（多 epoch PPO / importance sampling），要么：
   - rollout 时让 Engine 返回 token logprobs（`GenerateReqInput.return_logprob`），要么
   - 另行用 actor 模型计算 logps（会吃算力）。
6) **`SglangJaxRollout.model()` 返回 `None` 的连带限制**：
   - `workdir/tunix/tunix/generate/sglang_jax_sampler.py` 里 `SglangJaxSampler.transformer` 明确 `return None`（“doesn't expose the underlying model”）。
   - 因此 Tunix 的两类能力在 `sglang_jax` rollout 下默认不可用（或需要你额外改代码）：
     - `cluster_config.offload_to_cpu=True` 时会尝试 `nnx.state(model)`（model 为 None 会直接报错）
     - `algo_config.num_iterations > 1` 时会走 `RLCluster.get_old_per_token_logps(...)`，间接调用 `rollout.get_per_token_logps(self.model(), ...)`（同样会报错）

---

## 8) 下一步建议（按优先级）

1) 先在本仓库落一个最小“rollout engine 抽象层”（plugins-first），把 rollout 后端与训练逻辑解耦。  
2) 明确你要的目标：
   - **提升 rollout 吞吐**（优先考虑 disaggregated + sglang-jax/vLLM）
   - **提升训练正确性/可扩展性**（优先抽象边界、加入 version/staleness，为 async rollout 铺路）
3) 如果要接 sglang-jax：优先做一个 **TPU 上的最小 smoke**（小 batch、小 max_tokens，单步 rollout），确认：
   - Engine 能起、能生成
   - 权重同步不会 hang/爆显存
   - 端到端能接回 `input_ids/attention_mask/labels`

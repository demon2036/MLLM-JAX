# AReaL 强化学习（RL）组织方式调研 & MLLM-JAX 可借鉴的模块化训练方案（plugins-first）

> 本文件输出位置：`answers/areal-rl-organization-and-mllm-jax-modularization.md`

## 0. 一页结论（TL;DR）

AReaL（inclusionAI/AReaL）把“强化学习训练系统”做成**四层结构**，并且把“异步 rollout + staleness（off-policy）控制”做成**独立系统组件**：

- **API Layer（契约）**：只放接口 + dataclass 配置，不塞实现，强约束边界与数据结构。
- **Backend Layer（后端适配）**：训练引擎/推理引擎/分布式/权重同步/launcher 都在这一层处理。
- **Customization Layer（可改动层）**：算法（PPO/GRPO/GSPO/SAPO/M2PO…）与 rollout workflow（单轮/多轮/agentic）在这里实现，且尽量 backend-agnostic。
- **Entrypoint Layer（入口组合）**：examples 只负责装配：config + dataset + workflow + trainer。

对 MLLM-JAX（JAX/TPU）最值得借鉴的不是 AReaL 的 PyTorch 具体实现，而是它的**模块边界/数据契约/训练生命周期的阶段化/以及异步 rollout 的最小原语**。我们可以在本仓库的“非侵入约束（新代码放 `plugins/`）”下，按阶段把训练拆成：`API（schema/contracts）→ backend（mesh/sharding/ckpt）→ workflow（rollout→reward→traj）→ algorithm（adv/loss/update）→ runner（编排+hooks）→ entrypoint（scripts）`，最终具备走向“同步→异步”的演进空间。

---

## 1. 本次调研环境与版本（可复现信息）

- 当前仓库路径：`/home/john/github/MLLM-JAX`
- 运行环境：Ubuntu Linux
- Python：`3.12.2`
- AReaL clone 路径（repo 内 `workdir/`，已在 `.gitignore` 忽略）：`/home/john/github/MLLM-JAX/workdir/areal`
- AReaL 本次阅读版本：`d082767c82571f596963b701f829381a9ca2a3f5`

> 复现实操命令与入口文件定位已整理成 SOP：`docs/sops/areal-rl-organization.md`

---

## 2. AReaL 的总体组织：四层结构（AReaL-lite Design）

参考（AReaL repo）：`/home/john/github/MLLM-JAX/workdir/areal/areal/README.md`

### 2.1 API Layer：接口与数据类（把边界写死）

核心思想：**API 层只定义“你能对系统做什么”，不定义“系统怎么做”**。这样算法/工作流可以被替换，而不牵扯后端细节。

关键入口：

- `areal/api/engine_api.py`：训练侧 `TrainEngine` 与推理侧 `InferenceEngine` 的抽象接口（例如：`prepare_batch`、`update_weights`、`connect_engine`）。
- `areal/api/workflow_api.py`：RolloutWorkflow 的统一契约：`arun_episode(engine, data)->trajectory|None`。
- `areal/api/cli_args.py`：配置 dataclass（如 `PPOConfig/PPOActorConfig/InferenceEngineConfig`），并通过 Hydra/OmegaConf 做 YAML 合并。
- `areal/api/io_struct.py`：`ModelRequest/ModelResponse`、`WeightUpdateMeta/SaveLoadMeta/StepInfo` 等系统级数据结构。

可借鉴点（对 MLLM-JAX）：

- **先定义“最小 batch/trajectory schema”与接口**，再决定实现细节；否则训练逻辑会在脚本中越长越难拆。

### 2.2 Backend Layer：把分布式/推理/训练后端复杂性封起来

关键入口：

- `areal/engine/fsdp_engine.py`、`areal/engine/megatron_engine.py`：不同训练后端（FSDP2/Megatron）实现同一训练接口。
- `areal/core/remote_inf_engine.py`：远端推理引擎的通用封装（HTTP 请求、重试、异步任务执行、权重更新请求）。
- `areal/launcher/*.py`：Ray/Slurm/Local 启动器，负责资源编排与进程/任务启动顺序。
- `areal/api/alloc_mode.py`：`AllocationMode.from_str(...)` 解析字符串表达的资源划分（生成与训练如何分离/共置）。

可借鉴点：

- MLLM-JAX 现有 `get_jax_mesh2`、NamedSharding、partition rules 等属于 backend 能力；应从“实验脚本”中抽出来统一管理，让 runner 只依赖接口。

### 2.3 Customization Layer：算法与 workflow（真正的“可改动区”）

关键入口（AReaL）：

- 算法：`areal/engine/ppo/actor.py`、`areal/engine/ppo/critic.py`
  - 多个 RL 变体（GRPO/GSPO/SAPO/M2PO/DAPO/RLOO …）主要通过 `PPOActorConfig` 的开关控制，而不是复制 trainer。
- 工作流：`areal/workflow/rlvr.py`、`areal/workflow/multi_turn.py`
  - workflow 决定“怎么从 dataset item 组织 prompt、怎么调用推理、怎么算 reward、怎么产出 trajectory”。

可借鉴点：

- 在 MLLM-JAX，`Sampler` + prompt 模板 + reward 计算 + trajectory 拼装应被放到 workflow；算法层只消费统一 schema。

### 2.4 Entrypoint Layer：example 只做装配，不做系统实现

示例：`examples/math/gsm8k_rl.py`（AReaL）

- entrypoint 很短：load config → load tokenizer/dataset → 创建 trainer → `trainer.train(workflow=..., workflow_kwargs=...)`
- 启动通过 launcher：`python -m areal.launcher.ray <entrypoint.py> --config <yaml> ...`

可借鉴点：

- MLLM-JAX 的 `test_jit11.py` 目前承担了“entrypoint + runner + workflow + algo + 日志/缓冲”的多重职责；应拆分，让脚本回到“装配”。

---

## 3. AReaL 的 RL 数据流：从 dataset 到参数更新（一步训练的阶段化）

以 `PPOTrainer`（AReaL）为代表的训练步大致可以抽象成以下阶段（概念上最关键的是“阶段化”和“边界清晰”）：

1) **rollout / data collection**
   - 从 dataloader 取样（dataset item）。
   - 用 workflow 生成 completion、计算 reward、产出 trajectory dict。
   - 异步模式下通过 `prepare_batch` 保持“生成与训练重叠”。
2) **(optional) critic values**
   - PPO 才需要；GRPO 可无 critic。
3) **advantages / returns**
   - `compute_advantages`：根据 `rewards/logprobs/ref_logp/values/...` 计算优势、returns、KL reward 等。
4) **actor update（核心 loss）**
   - 将 batch 切 micro-batches（token-balanced）。
   - `train_batch(loss_fn=...)` 执行 forward/backward/step。
5) **(optional) critic update**
6) **pause inference → update weights → bump version**
   - 在更新推理服务权重时暂停新的 rollout 进入（避免混乱与 off-policy 爆炸）。
7) **save / recover checkpoint / eval / logging**
   - 保存 HF 权重、保存可恢复 checkpoint、eval rollout、提交 metrics。
8) **resume inference**

对 MLLM-JAX 的直接启示：

- “生成/奖励/adv/更新/保存/日志”是不同职责：要通过 runner 的阶段化把它们隔离开；否则后续加异步、加多机，会变成无法维护的脚本。

---

## 4. AReaL 的异步 rollout 子系统（最值得抄“设计”，不抄实现）

关键文件：

- `areal/core/workflow_executor.py`：`WorkflowExecutor` + `BatchTaskDispatcher`
- `areal/core/staleness_manager.py`：`StalenessManager`
- `areal/core/async_task_runner.py`：实际 async 执行队列

### 4.1 最小原语：`submit` / `wait` / `prepare_batch`

AReaL 把 rollout 的异步能力抽象成最小原语：

- `submit(data, workflow, ...) -> task_id`：非阻塞提交
- `wait(count) -> trajectories`：等若干条 accepted trajectory
- `prepare_batch(dataloader, workflow, ...) -> trajectories`：持续提交并等待，直到凑齐一个 batch（支持 `dynamic_bs`）

这组原语的好处：

- 算法层/训练层只需要“我能拿到一个 batch trajectory”，不必知道底层如何调度、线程/进程如何交互。

### 4.2 BatchTaskDispatcher：producer/consumer 双线程 + 队列

`BatchTaskDispatcher` 的结构是：

- producer：检查 capacity → 从 pending_inputs 取任务 → 提交到 AsyncTaskRunner
- consumer：不断从 AsyncTaskRunner 拉完成结果 → 放入 pending_results → 唤醒 waiters

关键能力：

- **fail-fast**：后台线程异常会被传播到前台，避免 silent failure。
- **callback**：支持任务完成时回调（用于权重同步等协作）。

### 4.3 StalenessManager：off-policy 约束的系统化表达

它用一个非常工程化的方式把“staleness 约束”写成 capacity 计算：

- 控制并发：`max_concurrent_rollouts - running`
- 控制 off-policy：根据 `current_version` 与 `max_staleness` 限制可接受的 pending/accepted 数量

对 MLLM-JAX 的启示：

- 如果未来要做“生成与训练并行”，必须有一个类似的容量控制器；否则很容易出现：推理侧产出过快、训练侧吃不完、样本版本差越积越大、最终训练不稳定/显存爆炸/队列爆炸。

### 4.4 trajectory format check + dump（debug 的关键）

`WorkflowExecutor.check_trajectory_format(...)` 会强校验：

- 是否包含 `input_ids/attention_mask` 等必需键
- tensor 的维度是否一致
- keys 是否随训练步漂移（expected_keys tracking）

并支持将 trajectory dump 为 JSONL（按 version 目录组织），方便离线检查 prompt/completion/reward。

对 MLLM-JAX 的启示：

- 我们要先把 batch schema “写死”，并在 runner 中每步校验，否则重构时很难定位哪里破坏了隐含约定。

---

## 5. 算法层：AReaL 如何用一套 PPOActor 支撑多个变体

关键文件：

- `areal/engine/ppo/actor.py`
- `areal/api/cli_args.py`（`PPOActorConfig`）

### 5.1 关键开关（配置驱动变体，而不是复制 trainer）

典型开关（示意）：

- `importance_sampling_level`：`token`（标准 PPO） vs `sequence`（GSPO）
- `use_sapo_loss`：启用 SAPO 替代 PPO clipping
- `m2_threshold`：启用 M2PO 的二阶动量过滤
- `use_decoupled_loss`：启用 decoupled PPO（支持异步/更强 off-policy）
- `prox_logp_method`：proximal logp 的计算方式（recompute / loglinear / metrics）
- `behav_imp_weight_cap`：过滤极端 importance weight 的 token

对 MLLM-JAX 的启示：

- 我们也可以把“算法变体”做成 config toggles（比如不同 advantage 归一化、不同 clipping、不同 entropy/kl 约束），避免出现 N 个几乎一样的训练脚本。

### 5.2 关键设计：把“proximal policy”计算方法抽象成可替换策略

在异步 RL 下，rollout 的行为策略与训练更新时刻可能有版本差。AReaL 把 prox_logp 的计算写成一个可插策略（甚至支持近似）。

对 MLLM-JAX 的启示：

- 如果我们未来做异步，必须考虑“旧策略 logp 的来源与对齐”：是 rollout 缓存、还是训练端重算、还是近似；这应该在算法层被显式建模，而不是散落在脚本。

---

## 6. 系统层：launcher / scheduler / controller / weight update / recover

### 6.1 启动器（launcher）把“编排”固定化

AReaL 的 `areal/launcher/ray.py` 做了：

- 解析 config（Hydra）
- 按 allocation_mode 启动 LLM server（SGLang/vLLM）
- 启动 trainer（并设置 torch 分布式 env vars）
- 统一 wait/stop/recover 流程

对 MLLM-JAX 的启示：

- TPU/多机的启动与环境变量设置最好脚本化、标准化；否则每个实验脚本会自行处理分布式初始化，导致不可复现。

### 6.2 单控制器（single-controller）模式的 controller

在 single-controller 模式下：

- `TrainController` 负责创建 worker、分发数据、收集结果。
- `RolloutController` 负责创建推理 worker、管理队列、回调、staleness。

对 MLLM-JAX 的启示：

- 即使我们暂时不做多进程 controller，也可以先把 runner 的阶段与接口抽出来；未来要做 controller 时，接口不会变。

### 6.3 权重同步与版本化（weight update + version bump）

训练 loop 会在每步更新后做：

- `rollout.pause()`
- `actor.update_weights(weight_update_meta)`
- `actor.set_version(global_step+1)` + `rollout.set_version(global_step+1)`
- `rollout.resume()`

对 MLLM-JAX 的启示：

- “更新推理权重”和“训练更新”必须有明确阶段；否则同步/异步切换时非常容易出现旧权重产出的样本被当作新权重训练数据使用（隐性 off-policy）。

### 6.4 recover、save、stats logger

关键文件（AReaL）：

- `areal/utils/recover.py`：recoverable checkpoint + dataloader state
- `areal/utils/saver.py`：保存 HF 权重
- `areal/utils/stats_logger.py`：wandb/tensorboard 统一提交
- `areal/utils/exp_metadata.py`、`areal/version.py`：记录 commit/dirty/version

对 MLLM-JAX 的启示：

- 我们也应该将“保存/恢复/日志/版本信息”从实验脚本剥离成 runner hooks（先同步，后异步）。

---

## 7. 测试与可复现（为什么 AReaL 的内部可大胆重构）

AReaL 有大量 tests 覆盖系统关键组件：

- `areal/tests/test_staleness_manager.py`
- `areal/tests/test_async_task_runner.py`
- `areal/tests/test_rollout_controller.py`
- `areal/tests/grpo/*`（GRPO 集成测试入口）
- `.github/workflows/test-areal.yml`（CI）

对 MLLM-JAX 的启示：

- 模块化训练必须配“最小 smoke + schema 校验 + 关键数学函数单测”，否则每次重构都得靠手跑大实验验证。

---

## 8. 对照 MLLM-JAX：我们现在的训练模块分布与耦合点

当前（以可跑路径为准）：

- **训练核心（state/init/step + reward/adv 混杂）**：`training2.py:1`
- **算法/loss（更像 customization algo）**：`MLLM_JAX/train_modules/__init__.py:1`（`TrainGRPOModule`）
- **rollout/生成（更像 inference engine + workflow 一部分）**：`MLLM_JAX/sample/sample_state_right_padding2.py:1`
- **mesh/sharding/ckpt 工具（backend 工具层）**：`MLLM_JAX/utils.py:1`
- **最小 smoke 入口（entrypoint）**：`scripts/run_smoke_train_qwen25_7b.py:1`
- **较完整的训练编排雏形（但脚本化、职责混杂）**：`test_jit11.py:1`

耦合点（问题）：

- 实验脚本需要理解太多底层细节（prefill、padding、old_logps、labels mask、adv shape）。
- reward/adv 逻辑分散且缺少统一契约，导致“改 reward”可能连带改训练 step。
- rollout（Sampler）与训练 step（TrainGRPOModule）之间的 batch schema 约定并没有被显式化、可校验化。

---

## 9. 借鉴落地：MLLM-JAX 的 plugins-first 模块化训练蓝图（不侵入上游目录）

> 约束：新实现全部放 `plugins/`；避免直接改 `MLLM_JAX/` 或 `training2.py`，并通过 `scripts/` 入口选择新 runner；TPU 覆盖用脚本 + `PYTHONPATH`。

### 9.1 目标分层（对齐 AReaL-lite 思路，但适配 JAX/TPU）

1) **API/Contracts（薄）**
   - 只定义：batch schema、workflow/algo/runner 接口、config dataclass（或最小 dict 约束）。
2) **Backend（JAX/TPU 系统能力）**
   - mesh/sharding、params/state init、checkpoint/save/load、host 集合通信（如 allgather）。
3) **Customization（workflow + algorithm）**
   - workflow：prompt→generate→reward→trajectory（标准 schema）
   - algorithm：advantages + loss + update_step（只依赖 schema）
4) **Entrypoint（scripts）**
   - 只做装配：选模型、选 workflow、选算法、选 runner 配置并启动。

### 9.2 推荐目录结构（规划）

```
plugins/training/
  api/
    batch_schema.md / batch_schema.py   # schema 说明与校验器（说明可以先写 md）
    interfaces.md / interfaces.py       # workflow/algo/runner 接口（先写 md 也行）
  backend/
    mesh.md
    sharding.md
    checkpointing.md
  workflows/
    single_turn_grpo.md
    grouped_sampling.md
  algorithms/
    grpo.md
    ppo_clip.md
  runner/
    trainer.md
    hooks.md
scripts/
  train_grpo_plugins.md（或 .py 入口；当前先写计划）
tests/
  schema_smoke.md（或最小 python 验证；当前先写计划）
```

> 注：上面列了 md 与 py 两种形式；你当前要求“不要 coding”，所以可以先把契约与目录结构写成 md，等团队确认后再编码。

### 9.3 关键契约：Batch/Trait（先写死，后续才能稳定重构）

建议先固定一个最小 `Trajectory`/`Batch` 契约（与现有实现对齐，不求完美但求稳定）：

- `input_ids`: `[B, T]`（int32）
- `attention_mask`: `[B, T]`（0/1 或 bool）
- `labels`（或 `loss_mask`）: `[B, T]`（0/1，completion 区域为 1）
- `old_per_token_logps`: `[B, T-1]`（float32；与你现在的 `per_token_logps` 对齐）
- `rewards`: `[B]`（float32）
- `advantages`: `[B]`（float32；由 algorithm 计算并广播到 token loss）
- `total_valid_token_count`: 标量 int（用于 loss 归一化，与你当前实现一致）
- 可选预留：
  - `versions`: `[B]` 或 `[B, T]`（int32，用于未来 async/staleness）
  - `reward_breakdown`: dict（host-only，用于 debug/日志）

### 9.4 Runner 生命周期（阶段化 + hooks）

借鉴 AReaL 的 phase 设计，把 runner 固定为（同步版）：

1) `collect_rollout_batch()`：workflow 产出 trajectory（同步）
2) `compute_advantages()`：算法计算优势（明确 group_size 语义）
3) `train_step()`：JAX/Flax 的 update（可复用 `training2.training_step` 或 `TrainGRPOModule`）
4) `checkpoint()`：保存参数（可复用 `save_checkpoint_in_background*`）
5) `eval()`：可选
6) `log_metrics()`：可选（wandb/print）

并在每一阶段提供 hook：

- `on_step_start`
- `on_rollout_end`
- `on_advantages_end`
- `on_update_end`
- `on_ckpt_end`
- `on_eval_end`
- `on_step_end`

> 这样未来要做“异步 rollout”时，只是把 `collect_rollout_batch` 的实现替换为 `submit/prepare_batch`，而 algorithm 与 update_step 不需要改接口。

### 9.5 异步 rollout（可选，后续演进路线）

等同步 runner 稳定后，可按 AReaL 思路增加：

- `submit(item)->task_id`
- `wait(n)->trajectories`
- `prepare_batch(dataloader)->batch`
- `StalenessManager`：cap 并发 + cap 版本差（off-policy）
- `version` 字段贯穿：训练每步 bump version；rollout 产出样本带 version；超过 max_staleness 的样本拒绝/丢弃

### 9.6 TPU 非侵入覆盖策略（原则）

在 TPU 上遵守仓库约束：

- 新实现都在 `plugins/`，通过脚本同步到 TPU VM（或挂载）后设置 `PYTHONPATH` 优先级；
- 不直接修改上游目录（避免 “fork drift”）。

---

## 10. 分阶段迁移计划（M0..M7；每阶段有 DoD）

> 目标：用最小代价把训练“拆开”，而不是一次性重写。

### M0：冻结基线（golden path）

- **目标**：明确“不能坏”的回归入口与指标。
- **基线入口**：`scripts/run_smoke_train_qwen25_7b.py:1`
- **DoD**：
  - 能跑 N step（比如 3 step）；
  - 每步输出 finite loss；
  - shape/keys 稳定。
- **风险**：重构时不知哪里变了。
- **回滚策略**：永远保留 baseline 脚本作为回归对照。

### M1：先建 API 契约（schema + interfaces），不改实现

- **目标**：把“隐含约定”显式化（schema 校验）。
- **产物**：
  - `plugins/training/api/batch_schema.md`（字段/shape/dtype/约束）
  - `plugins/training/api/interfaces.md`（workflow/algo/runner 的最小签名）
- **DoD**：
  - 可以对现有 batch/metrics 做校验（失败时能给出可读错误）。

### M2：Runner 阶段化（先同步，先可读）

- **目标**：把 `test_jit11.py` 的脚本式编排收敛成 runner 的阶段函数。
- **产物**：`plugins/training/runner/trainer.md`（阶段定义、hook 点、状态机）
- **DoD**：
  - runner 逻辑能覆盖“生成→奖励→adv→训练→log”；
  - 任何新实验只需写装配脚本，而不是复制粘贴训练循环。

### M3：Workflow 化（rollout + reward 变成可插拔）

- **目标**：把 prompt 构造/采样/解码/奖励计算从 runner 中拿出去，变成 workflow。
- **产物**：
  - `plugins/training/workflows/single_turn_grpo.md`：定义 episode 输入/输出（trajectory schema）
  - reward 函数接口文档：统一签名（prompt/completion/token ids/extra fields）
- **DoD**：
  - 替换 reward 不需要改训练 loop；
  - workflow 输出的 trajectory 通过 schema 校验。

### M4：Rollout engine 封装（Sampler “后端化”）

- **目标**：把 `Sampler` 相关的 prompt tokenization、prefill bucketing、padding 对齐、old_logps 对齐封装为 rollout engine。
- **DoD**：
  - algorithm/update 不需要知道 prefill_length/padding 等细节；
  - 只消费 schema 中的字段。

### M5：Algorithm 层收敛（adv + loss + update_step）

- **目标**：把 GRPO/PPO 变体做成“配置开关”，而不是 N 份训练脚本。
- **DoD**：
  - 切换 advantage 估计器（grpo/rloo/...）不改 runner；
  - 切换 clipping/entropy/kl 等策略不改 workflow。

### M6：Checkpoint/Eval/Recover/Metadata 体系化（hook 化）

- **目标**：把保存/恢复/评估/日志统一成 runner hooks，并记录版本信息。
- **DoD**：
  - 保存路径固定、恢复可跑；
  - 每次 run 可追溯版本与配置。

### M7（可选）：异步 rollout + staleness（对齐 AReaL 的系统优势）

- **目标**：生成与训练并行，吞吐更高，且 off-policy 可控。
- **DoD**：
  - 有明确 capacity 机制（不会队列爆炸）；
  - 有明确 staleness 策略（不会版本差无限扩大）；
  - 训练稳定性可回归对比同步版本。

---

## 11. 本次调研（实际执行过的）关键命令摘要

> 更完整的命令序列请看：`docs/sops/areal-rl-organization.md`

- 创建 `workdir/` 并 clone AReaL（本仓库已在 `.gitignore` 忽略 `workdir/`）：
  - `mkdir -p workdir`
  - `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
  - `git -C workdir/areal rev-parse HEAD`

---

## 12. 参考入口（便于你继续深入阅读）

- AReaL-lite 设计文档：`/home/john/github/MLLM-JAX/workdir/areal/areal/README.md`
- Async rollout 核心：`/home/john/github/MLLM-JAX/workdir/areal/areal/core/workflow_executor.py`
- Staleness manager：`/home/john/github/MLLM-JAX/workdir/areal/areal/core/staleness_manager.py`
- PPO/GRPO 算法核心：`/home/john/github/MLLM-JAX/workdir/areal/areal/engine/ppo/actor.py`
- 训练 loop（阶段化范例）：`/home/john/github/MLLM-JAX/workdir/areal/areal/experimental/trainer/rl.py`
- 启动器（编排）：`/home/john/github/MLLM-JAX/workdir/areal/areal/launcher/ray.py`

# RL 四阶段接口统一方案（rollout / reward / advantage / update）

> 目标：为后续多算法并行开发“接口先行”。把所有算法的共性固定在接口与数据契约里，把差异收敛到 advantage 与 update。
>
> 本文基于源码调研（repo-local `workdir/`，已在 `.gitignore` 忽略）：
> - Tunix `dd833a9` (`https://github.com/google/tunix.git`)
> - AReaL `b066584` (`https://github.com/inclusionAI/AReaL.git`)
> - VERL `c0975d8` (`https://github.com/volcengine/verl.git`)
> - MaxText `4bcee99` (`https://github.com/google/maxtext.git`)

## 0) 结论先行：你提的 4 个接口是对的，但要把“数据契约”一起钉死

多数 LLM-RL 算法的差异确实主要集中在：
1) **advantage 如何算**（baseline/归一化/组内相对/GAE/leave-one-out/REMAX…）
2) **update 如何做**（PPO-clip / decoupled PPO / GRPO/GSPO 的 loss 形态 / off-policy 修正）

而 **rollout**（prompt→completion/trajectory）与 **reward**（completion→reward）通常能复用，但要注意两个例外：
- **“rollout 与 reward 融合”** 对 agentic/multi-turn 很常见（AReaL workflow 直接返回 `rewards`）。
- **“rollout 与 update 的耦合点”** 主要是 *logprobs/old_logps/ref_logps/versions*：有的引擎拿不到 logprobs，就必须在 update 里补算（Tunix 的 sglang-jax backend 就是典型坑位）。

所以：接口要做成“4 阶段 + 统一 batch/trajectory schema + 可选能力(capabilities)”。

## 1) 参考仓库的共同设计点（我们直接照抄其“边界”，不照抄实现）

### Tunix
- 关键抽象：`BaseRollout.generate(...) -> RolloutOutput` + `common.TrainExample`（把 rollout 后需要的张量都装进一个结构）。
- 关键工程点：colocated vs disaggregated（`Role -> mesh`）是第一公民；权重同步是显式动作（是否共享权重决定是否需要 sync）。
- 插件化：policy_loss/adv_estimator/reward_manager 都用 registry（字符串 key → 函数）。

### AReaL
- 关键抽象：`workflow.arun_episode(engine, data) -> trajectory|None`（workflow 是 rollout 单元，轨迹 dict 是唯一数据载体）。
- 关键工程点：异步 rollout 的 “submit/wait/prepare_batch + staleness” 是系统能力；并且有强 schema check。
- 算法实现（PPOActor）可以把 reward/adv/update 串起来，但 workflow 也可以把 reward 融进 rollout（对 agentic 很实用）。

### VERL
- 关键抽象：`DataProto`（一个 batch 贯穿全链路，阶段性往里加 keys）。
- 关键工程点：rollout worker 明确提供 `resume/update_weights/release`（权重/kv cache 生命周期是接口的一部分）。
- 插件化：adv estimator / policy loss 都是 registry；并且把训练-推理 mismatch（off-policy）当成一级问题（rollout correction）。

### MaxText
- 关键点：MaxText 的 RL 基本“借用 Tunix 的接口”，它额外提供 **模型适配器**（让 MaxText 模型长得像 Tunix 模型）与 **权重映射**（训练树 ↔ rollout 引擎树）。

## 2) 我建议在本仓库采用的统一分层（plugins-first，和现状完全兼容）

你现在仓库里其实已经具备雏形：
- 四阶段结果类型：`plugins/training/api/interfaces.py`（`RolloutResult/RewardResult/AdvantageResult/UpdateResult`）
- rollout engine 抽象：`plugins/training/rollout_backends/base.py`（带 `sync_weights/release_weights/...` 的 duck-typed hooks）
- batch schema 验证：`plugins/training/api/batch_schema.py`（GRPO 的 stage validator）

建议把它“升级成正式架构”：

1) **API / Contracts（钉死数据契约）**：`plugins/training/api/*`
2) **Backends（rollout 引擎可插拔）**：`plugins/training/rollout_backends/*`
3) **Algorithms（只关心 reward/adv/update 的数学与所需字段）**：`plugins/training/algorithms/*`（后续新增）
4) **Runner（编排、并行、日志、评测、同步）**：`plugins/training/runner/*`
5) **Entrypoints（脚本入口）**：`scripts/*`

## 3) 四阶段接口：最小 I/O 与“可选能力”

### 3.1 统一数据载体：一个 `Batch`（张量）+ 一个 `Meta`（非张量）

参考 VERL 的 `DataProto` 思路，但保持我们当前的轻量实现（dict + schema validator）：

- `batch: dict[str, Any]`：只放 array-like（JAX/NumPy）为主
- `meta: dict[str, Any]`：放字符串、采样参数、版本号、uid/group_ids、调试信息、原始文本等

并且**约定“阶段性新增 keys”**，像现在的 `validate_grpo_batch(stage=...)` 那样做 gate。

### 3.2 Rollout（采样/轨迹收集）

**职责**：把 prompts 变成 “可训练的轨迹 batch”（至少 `input_ids/attention_mask/labels`）。

建议契约（与现有代码对齐）：
- 输入：`prompts: list[str]` + `params` + 采样配置（温度/最大长度/分组信息）
- 输出：`RolloutResult(chat_prompts, answers, batch)`，其中 `batch` 至少含：
  - `input_ids: [B, T]`
  - `attention_mask: [B, T]`
  - `labels: [B, T]`（在本仓库语义上是 completion mask；等价于更通用的 `loss_mask`）

**可选能力（用 duck-typing，不强耦合）**：
- `initialize()` / `shutdown()`
- `sync_weights(params)`（disaggregated 需要）
- `release_weights()`（避免训练 donate buffer 被 rollout backend 持有引用）
- `flush_cache()`（外部引擎 KV cache）
- `get_version()/set_version()`（为 AReaL/VERL 式 staleness 预留）

### 3.3 Reward（打分）

**职责**：把 completion 变成 reward 信号（sequence-level 或 token-level）。

建议输出两条线：
- `rewards: [B]`（sequence-level）
- 可选：`token_level_rewards: [B, T-1]`（PPO/GAE 常用）

并允许两种实现形态：
- **独立 reward module**（同步/异步都行）
- **workflow 内联 reward**（AReaL 风格；rollout 直接产出 `rewards`，reward stage 变成 no-op）

### 3.4 Advantage estimate（优势估计）

**职责**：把 reward（可能还有 values/ref_logps/old_logps）变成 `advantages`（以及可选 `returns`）。

关键点：**强制使用显式 group_ids/uid 来做 grouping**，不要依赖 reshape（我们当前已经这么做了）。

### 3.5 Update（梯度/参数更新）

**职责**：把 batch + advantages 喂给 train_step，完成：
- 可能的 “first pass collect old_logps”（PPO-style）
- N 次 PPO epoch / mini-batch / grad-accum
- 返回 updated state + metrics

这里建议把 update 的输入统一成：
- `state`（包含 actor/critic/ref 等必要参数）
- `batch`（含 tokens/masks/rewards/advantages/…）
- `update_config`（ppo_epochs/microbatch/clip/beta…）

避免 runner 把 update 的细节参数（`slice_data/train_step/total_valid_token_count`）散落到各处；这些应收敛进一个 `UpdateContext`（后续落地时再做）。

## 4) “5 大类算法”如何落到这 4 阶段接口上

我建议按“需要哪些额外信号/模型”来分 5 类（比按论文名更稳定）：

1) **Outcome PG（REINFORCE/GRPO/RLOO 类）**
   - reward：sequence-level
   - advantage：基于 reward 的 baseline（group/leave-one-out/全局归一化）
   - update：policy-gradient（可带 PPO-clip / KL-to-ref）

2) **Actor-Critic（PPO/GAE 类）**
   - reward：token-level 或 sequence→token 分配
   - advantage：GAE（需要 values）
   - update：actor+critic（可能还有 value clip）

3) **Decoupled/Async（显式 staleness/off-policy 修正）**
   - rollout：带 `versions`/行为策略标识
   - reward：同上
   - advantage：同上
   - update：需要 IS weights / rejection mask / prox policy（AReaL/VERL 都有现成模式）

4) **Preference-as-Reward（把偏好转成 reward 的 RL 形态）**
   - reward：来自 RM 或偏好模型，仍可输出标量/逐 token
   - 其余同上（核心在 reward module 的可替换性）

5) **Direct Preference Optimization（DPO/IPO/KTO 等，严格说不是 rollout-RL）**
   - rollout：可退化成 “从数据集取 pair/batch”
   - reward/advantage：可以为 no-op
   - update：实现成独立 update module（pairwise loss）
   - 这类如果你要支持，建议 runner 在接口层允许某些阶段为空（no-op module）。

## 5) 对本仓库的落地建议（现在就能开始写代码，不用等大重构）

1) 把 `plugins/training/api/interfaces.py` + `plugins/training/rollout_backends/*` 当作 **唯一稳定边界**（后续新增算法只加新模块）。
2) 为每种算法定义：
   - `RewardModule`（可复用）
   - `AdvantageModule`（差异最大）
   - `UpdateModule`（差异最大）
3) 新增一个 “algo registry / factory”（参考 Tunix/VERL 的 registry），runner 只拿到一个 `algo` 对象，避免 if-else 漫延。
4) 把 schema validator 扩展成 “按阶段 + 按算法” 的 required-keys（参考 AReaL `check_trajectory_format`）。

---

## 6) 我们接下来讨论时最关键的 3 个决策点（会影响接口细节）

1) **rollout 与 reward 是否允许融合**（workflow/agentic 必须允许，否则后面会被迫破接口）。
2) **是否从一开始就支持 disaggregated rollout**（如果是：必须把 `sync_weights/version/staleness` 纳入接口语义）。
3) **Batch 的“规范字段名”**：继续沿用 `input_ids/attention_mask/labels`（现状）还是切到更通用的 `loss_mask/prompts/responses`（需要一轮兼容层）。


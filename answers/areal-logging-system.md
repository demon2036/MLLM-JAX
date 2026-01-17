# AReaL 的日志系统（Logging / Metrics / Tracing）调研笔记（面向 MLLM-JAX 落地）

> 本文聚焦 AReaL 的“log 系统”到底由哪些组件组成、调用链是什么、分布式下如何避免重复打点，以及它的设计点哪些最值得在 MLLM-JAX（JAX/TPU）里复用。
>
> 调研对象：本仓库的本地 clone：`workdir/areal`（gitignored）  
> AReaL 版本：`d082767`（`fix tests (#831)`）

---

## 0) 结论概览：AReaL 把“日志”拆成三条线

AReaL 并不是只有 “wandb.log” 这么简单，它把“日志/观测”拆成三条彼此协作但边界清晰的线：

1) **结构化文本日志（Python logging）**：给人读的 console/file log  
2) **训练指标统计（stats tracking + stats sink）**：给仪表盘/对比用的 metrics（W&B/TensorBoard/SwanLab）  
3) **性能与 session tracing（PerfTracer + SessionTracer）**：给性能分析/rollout 生命周期分析用的 trace（JSONL，Perfetto/Chrome tracing）

这三条线的核心文件分别是：

- 文本日志：`workdir/areal/areal/utils/logging.py`
- 指标统计：`workdir/areal/areal/utils/stats_tracker.py` + `workdir/areal/areal/utils/stats_logger.py`
- 性能 tracing：`workdir/areal/areal/utils/perf_tracer.py`

训练入口如何把它们串起来：

- `workdir/areal/areal/api/cli_args.py`：定义 `StatsLoggerConfig/WandBConfig/TensorBoardConfig/PerfTracerConfig`，并在 `load_expr_config` 时把 `config.yaml` 落盘到 logdir
- `workdir/areal/areal/experimental/trainer/rl.py` / `.../sft.py`：初始化 file logging、StatsLogger，并在每个 step 结束时 commit stats

---

## 1) 文本日志（Python logging）：统一格式 + 颜色 +（可选）落盘

### 1.1 目标与特点

AReaL 的 `areal.utils.logging` 主要解决：

- **统一 log format**（时间戳 + logger 名 + level + message）
- **按组件类别着色**（scheduler/launcher/engine/workflow/stats 等不同颜色）
- **控制器进程可选落盘**：`main.log` + `merged.log`（实时 flush，便于 tail）
- **对抗外部库污染**：每次 `getLogger()` 都 reset root logger（注释中点名 transformer_engine 会改 logging config）

### 1.2 关键 API（读代码就够用）

文件：`workdir/areal/areal/utils/logging.py`

- `logging.getLogger(name, type_=...)`
  - 内部会 `logging.config.dictConfig(log_config)`，并保证没有配置过的 name 会被动态加入 `log_config["loggers"]`
  - `type_` 影响 stdout handler 颜色风格（`plain/colored/system/benchmark`）
- `logging.setup_file_logging(log_dir, filename="main.log")`
  - 会创建两个 file handler：
    - `main.log`：当前进程主日志（带 ANSI 颜色）
    - `merged.log`：固定宽度 prefix（默认 `[main]`）+ ANSI 颜色
  - 设计细节：它不是只挂在 root logger 上，而是挂到“所有通过 getLogger() 创建过的 logger”上（因为 getLogger 会 reset root）

### 1.3 训练入口如何启用 file logging

在 `workdir/areal/areal/experimental/trainer/rl.py`：

- 如果是 single-controller 模式，会在 `PPOTrainer.__init__` 一开始调用：  
  `logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))`

这意味着：AReaL 默认把所有文本日志也归档进同一个 logdir（与 wandb/tb 的落盘目录一致），便于一次性收集实验产物。

---

## 2) 指标系统（Metrics）：StatsTracker（采集/聚合） + StatsLogger（sink/落地）

这部分是 AReaL 最“工程化、可移植”的设计：把 metrics 分成 **采集/聚合** 与 **输出/写入** 两层。

### 2.1 StatsTracker：分布式可聚合的指标采集器

文件：`workdir/areal/areal/utils/stats_tracker.py`

#### 2.1.1 它解决什么问题

- “每个模块都能随手打点，但不会把 wandb/tb 的依赖扩散到全项目”
- “token/seq 级指标要按有效 mask 归一化”，不能简单平均
- “多进程/多 worker 下要能 all-reduce，且 key 结构要能对齐”

#### 2.1.2 核心数据模型（很像 AReaL 的 API layer 思维）

- **层级 scope（路径化 key）**：
  - `stats_tracker.scope("ppo_actor")` → key 变成 `ppo_actor/...`
  - `record_timing("rollout")` 固定落在 `timeperf/rollout`
- **denominator（bool mask）**：
  - `stats_tracker.denominator(n_valid_tokens=mask_bool)`  
  - 然后 `stats_tracker.stat(..., denominator="n_valid_tokens")` 才能算“按 mask 的平均/最小/最大”
- **reduce_type**：
  - 默认 tensor stat：`AVG_MIN_MAX`（产出 `key/avg`, `key/min`, `key/max`）
  - scalar：`SCALAR`（会产出 `key` + `key__count`，count 用于跨 worker 加权平均）

#### 2.1.3 分布式聚合方式

- `export(reduce_group=...)` / `export_all(reduce_group=...)`
  - 先用 `dist.all_gather_object` 对齐“各 rank 的 key 集合”
  - 对 tensor/stat/scalar 分别走 `dist.all_reduce`（SUM/AVG/MIN/MAX）
  - scalar 的 `__count` 是体系内的一等公民（后面 rollout controller 会用它做加权平均）

这就是 AReaL “指标系统可扩展但不乱”的关键：所有模块只依赖 `stats_tracker` 这一个轻 API，而不是直接依赖 wandb。

### 2.2 StatsLogger：把 export 出来的 stats 落到 wandb/swanlab/tensorboard + console

文件：`workdir/areal/areal/utils/stats_logger.py`

#### 2.2.1 配置入口（在 cli_args 定义）

配置结构在：`workdir/areal/areal/api/cli_args.py`

- `StatsLoggerConfig(experiment_name, trial_name, fileroot, wandb, swanlab, tensorboard)`
- `WandBConfig`：
  - `mode`（默认 disabled）、`entity/project/name/group/tags/notes/...`
  - `wandb_base_url`、`wandb_api_key`（支持连接自建 wandb host）
  - `id_suffix`（默认 "train"，也支持 timestamp）
- `BaseExperimentConfig` 会把 `stats_logger`、`perf_tracer` 作为所有实验的公共字段

#### 2.2.2 logdir 规则（所有产物的落点）

`StatsLogger.get_log_path(...)`（`workdir/areal/areal/utils/stats_logger.py`）定义了统一落盘目录：

`{fileroot}/logs/{user}/{experiment_name}/{trial_name}`

它被多个地方复用：

- 文本日志 file logging：`logging.setup_file_logging(StatsLogger.get_log_path(...))`
- wandb.init 的 `dir=...`（wandb 的本地 artifact 会落在这里）
- swanlab.init 的 `logdir=...`
- `load_expr_config` 在训练开始前就把 `config.yaml` 保存到这里（见 2.2.5）

#### 2.2.3 初始化（只在 rank0 做 sink 初始化）

`StatsLogger.init()` 的第一行就做了“多进程防重”：

- 若 `torch.distributed` 已初始化且 `rank != 0`：直接 return，不创建 wandb/tb writer

因此 AReaL 的策略是：

- **全体进程/worker 都可以记录 stats（stats_tracker）**
- **只有 rank0 负责把 stats 落到外部系统（wandb/tb/swanlab）**

#### 2.2.4 wandb / swanlab / tensorboard 的连接与 run 元信息

wandb（`workdir/areal/areal/utils/stats_logger.py`）：

- （可选）支持自建 host：
  - 配置里提供 `wandb_base_url` + `wandb_api_key`
  - init 里会把它们写到 `WANDB_BASE_URL` / `WANDB_API_KEY`
- `wandb.login()`：仅在 `mode != "disabled"` 时调用
- `wandb.init(...)`：
  - `mode`、`entity/project/name/tags/notes/group` 都来自 `WandBConfig`
  - `project` 默认回退到 `experiment_name`
  - `name` 默认回退到 `trial_name`
  - `group` 默认回退到 `"{experiment_name}_{trial_name}"`
  - `id` 会拼接 `experiment/trial/suffix`，并设置 `resume="allow"`（方便恢复训练不断点）
  - `config` 会把“完整实验配置（asdict）+ version_info”存进 wandb，便于溯源

swanlab：

- 逻辑与 wandb 类似；`mode != disabled` 时先 login，再 `swanlab.init(...)`（即使 disabled 也会 init，靠 mode 控制行为）

TensorBoard：

- 用 `tensorboardX.SummaryWriter(log_dir=...)`
- 写入逻辑在 commit 时 `add_scalar(key, val, step)`

#### 2.2.5 commit 语义：保证 step 单调 + 控制输出粒度

`StatsLogger.commit(epoch, step, global_step, data)` 的关键点：

- **只在 rank0 执行**
- `data` 允许是 `dict` 或 `list[dict]`  
  - AReaL 会把 `list[dict]` 视为“一次 commit 多条记录”（例如同一步里多个子阶段统计）
- **step 单调**：  
  - 内部用 `self._last_commit_step` 记录最后写入的 step  
  - `log_step = max(global_step, self._last_commit_step + 1)`，避免 wandb step 回退导致异常
- **过滤 `__count`**：
  - `stats_tracker` 的 scalar 聚合会生成 `key__count`
  - `commit` 会过滤掉所有以 `__count` 结尾的 key（这些 key 用来做聚合，不适合展示）
- **console 可读性**：
  - 用 `tabulate` 做 `fancy_grid` 表格输出（`workdir/areal/areal/utils/printing.py`）

#### 2.2.6 配置落盘（config.yaml）与版本信息（version_info）

AReaL 在 `load_expr_config`（`workdir/areal/areal/api/cli_args.py`）里做了一件很重要的事情：

- 解析 hydra/omegaconf 配置后，在训练开始前就把 `config.yaml` 保存到 logdir：
  - `save_config(cfg, StatsLogger.get_log_path(cfg.stats_logger))`（仅 rank0）

同时，StatsLogger 还会把 git 版本信息写进 wandb config：

- `workdir/areal/areal/version.py:version_info` 会用 git 命令采集：
  - branch、commit、是否 dirty
- `exp_config_dict["version_info"]=...` 会被作为 wandb config 的一部分保存

（另外也有 `workdir/areal/areal/utils/exp_metadata.py`，可把 version 信息保存成 `version.json`；这条路径属于“元信息归档”，与 StatsLogger 并行。）

### 2.3 指标数据从哪里来：谁写入 stats_tracker，谁 export_stats

#### 2.3.1 训练 loop 里的 timing（最稳定、最推荐复用的结构）

RL trainer 会把每个训练步切成阶段并自动统计耗时（与我们的 `rollout→reward→adv→update` 很像）：

- `workdir/areal/areal/experimental/trainer/rl.py`
  - `with stats_tracker.record_timing("rollout"):` …
  - `with stats_tracker.record_timing("compute_advantage"):` …
  - `with stats_tracker.record_timing("train_step"):` …
  - `with stats_tracker.record_timing("update_weights"):` …

最终这些会出现在 stats 里（key 类似）：`timeperf/rollout`, `timeperf/train_step`, …

#### 2.3.2 算法/工作流/引擎的指标

AReaL 的代码里很多地方会直接调用：

- `stats_tracker.scalar(...)`：记一个 float
- `stats_tracker.denominator(...)` + `stats_tracker.stat(...)`：记 token/seq 级张量指标（带 mask 归一化）
- `stats_tracker.get(scope).scalar(...)`：在 workflow 里按 session/episode scope 打点

#### 2.3.3 export_stats 的聚合方式（不同角色不同策略）

训练侧（train engine）：

- 例如 FSDP engine：`workdir/areal/areal/engine/fsdp_engine.py:export_stats`  
  - 直接 `stats_tracker.export_all(reduce_group=self.data_parallel_group)`

Rollout 侧（rollout controller）：

- `workdir/areal/areal/controller/rollout_controller.py:export_stats`
  - 从每个 rollout worker 拉一份 stats（RPC）
  - 用 `__count` 做加权平均（只返回“均值类”的最终结果）

控制器聚合（train controller）：

- `workdir/areal/areal/controller/train_controller.py:export_stats`
  - 本地 `stats_tracker.export_all()` + 下游 engine 自定义 export_stats

### 2.4 Trainer 端的统一 commit（把各角色的 stats 合并）

RL trainer 在每步末尾做：

- `stats = self.actor.export_stats()`
- `stats.update(self.rollout.export_stats())`
- `stats.update(self.eval_rollout.export_stats())`
- `self.stats_logger.commit(epoch, epoch_step, global_step, stats)`

对应代码：`workdir/areal/areal/experimental/trainer/rl.py:_export_and_commit_stats`

SFT trainer 类似但没有 rollout：`workdir/areal/areal/experimental/trainer/sft.py:_export_and_commit_stats`

---

## 3) PerfTracer / SessionTracer：性能与 session 生命周期 trace（JSONL）

文件：`workdir/areal/areal/utils/perf_tracer.py`

### 3.1 AReaL 的定位：这不是 wandb 的替代，而是“可视化 trace”

PerfTracer 输出的是 Chrome Trace 兼容 JSON（更适合：

- 训练步内各阶段耗时的瀑布图
- 通信/同步事件的定位
- rollout session 的生命周期跟踪（SessionTracer）

### 3.2 输出文件与默认文件名

在 `perf_tracer.py` 里定义：

- perf trace：`traces.jsonl`
- session trace：`sessions.jsonl`

并且按 rank/role 输出到子目录（`perf_tracer/` 与 `session_tracer/`）。

### 3.3 使用方式（trainer 里的 trace_scope）

RL trainer 会在关键阶段包一层：

- `perf_tracer.trace_scope("train.rollout", category=Category.COMPUTE, args={...})`

（和 `stats_tracker.record_timing` 类似，但 PerfTracer 更偏“可视化 trace”，stats 更偏“标量指标”。）

### 3.4 与 Engine API 的集成点

Train/Inference engine 的 API 里预留了：

- `config_perf_tracer(...)`
- `save_perf_tracer(...)`

例如 `workdir/areal/areal/engine/fsdp_engine.py` 会在 `config_perf_tracer` 时调用 `perf_tracer.configure(...)`，在 `save_perf_tracer` 时调用 `perf_tracer.save(...)`。

---

## 4) AReaL 的“实验产物目录结构”值得直接照抄

统一 logdir：`{fileroot}/logs/{user}/{experiment}/{trial}`（见 2.2.2）

常见产物（不同模式略有差异）：

- `config.yaml`（训练开始前保存；见 2.2.6）
- `main.log` / `merged.log`（控制器进程；见 1.3）
- wandb 本地文件（`wandb.init(dir=...)`）
- swanlab 本地文件（`swanlab.init(logdir=...)`）
- tensorboard events（如果配置 `tensorboard.path`）
- `perf_tracer/traces.jsonl`（如果启用 perf tracer）
- `session_tracer/sessions.jsonl`（如果启用 session tracer）

---

## 5) 对 MLLM-JAX（JAX/TPU）的落地建议（plugins-first）

结合我们当前的实现（`plugins/training/*` + `scripts/run_grpo_gsm8k_training.py`）以及 AReaL 的设计点，建议按优先级落地：

1) **统一 logdir 与 config 落盘**
   - 像 AReaL 一样把 `config.yaml` 作为强制产物（启动即写），并把 wandb/tb 文件都写进同一个 logdir。
2) **把“指标采集”与“写入后端”解耦**
   - AReaL 的 `stats_tracker` 是关键抽象：各模块只打点，不关心 wandb。
   - 在 JAX/TPU 下可以用 `process_allgather` 或 `jax.lax.psum` 做聚合，保留 `__count` 或显式 denominator 思路。
3) **严格多进程策略：只让 process 0 做外部写入**
   - 我们当前 runner 已经这么做（W&B 只在 `jax.process_index()==0` init/log），这点与 AReaL 一致。
4) **Perf tracing（可选，但很值得）**
   - AReaL 的 PerfTracer/SessionTracer 可作为“未来异步 rollout”排障基础设施；建议先把阶段化 scope 写进我们的 runner（哪怕先不落 JSONL）。

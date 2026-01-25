# SOP: Inspect AReaL logging/metrics/tracing system (StatsLogger/StatsTracker/PerfTracer)

- **Title**: SOP: Inspect AReaL logging system (python logging + metrics + tracing) for design borrowing
  **Prereqs**: Repo checkout; AReaL cloned to `workdir/areal` (see `docs/sops/clone-reference-repos-into-workdir.md`)
  **Environment (verified)**: Ubuntu Linux; AReaL commit `b066584` (inspected 2026-01-25)

## Goal

- Locate AReaL 的三类“日志系统”实现：
  - 文本日志：`areal/utils/logging.py`
  - 指标：`areal/utils/stats_tracker.py` + `areal/utils/stats_logger.py`
  - tracing：`areal/utils/perf_tracer.py`
- 找到 trainer 侧的调用链（哪里 init、哪里 commit）。

## Steps (commands actually used)

### 1) 确认 AReaL clone 与版本

- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- `git -C workdir/areal rev-parse --short HEAD`
- `git -C workdir/areal show -s --oneline HEAD`
- `git -C workdir/areal remote -v`

### 2) 快速定位 logging/metrics/tracing 核心文件

- `cd workdir/areal`
- `find areal -maxdepth 4 -type f \( -name '*log*' -o -name '*logger*' -o -name '*wandb*' -o -name '*tensorboard*' -o -name '*tb*' \) | sort | head -n 120`
- `grep -RIn "wandb" areal | head -n 40`
- `grep -RIn "setup_file_logging" areal | head -n 40`

### 3) 阅读核心模块（建议按顺序）

- 文本日志（颜色/handler/落盘）：`sed -n '1,260p' workdir/areal/areal/utils/logging.py`
- 指标 sink（wandb/swanlab/tensorboard）：`sed -n '1,260p' workdir/areal/areal/utils/stats_logger.py`
- 指标采集与分布式聚合：`sed -n '1,260p' workdir/areal/areal/utils/stats_tracker.py`
- PerfTracer/SessionTracer（trace 输出 JSONL）：`sed -n '1240,1510p' workdir/areal/areal/utils/perf_tracer.py`

### 4) 找到 trainer 调用链（init + commit）

- RL trainer 初始化（file logging + StatsLogger）：`sed -n '1,160p' workdir/areal/areal/experimental/trainer/rl.py`
- RL trainer commit（export_stats + StatsLogger.commit）：`sed -n '620,820p' workdir/areal/areal/experimental/trainer/rl.py`
- SFT trainer commit（对比）：`sed -n '300,460p' workdir/areal/areal/experimental/trainer/sft.py`

### 5) 查看 config（StatsLoggerConfig / WandBConfig / PerfTracerConfig）

- `rg -n "class StatsLoggerConfig|class WandBConfig|class TensorBoardConfig|class SwanlabConfig" workdir/areal/areal/api/cli_args.py -n`
- `sed -n '1530,1625p' workdir/areal/areal/api/cli_args.py`
- `rg -n "class PerfTracerConfig" workdir/areal/areal/api/cli_args.py -n`
- `sed -n '1625,1715p' workdir/areal/areal/api/cli_args.py`

### 6) 查看 export_stats 如何跨 worker 聚合（可选）

- RolloutController 的加权平均（使用 `__count`）：`sed -n '820,910p' workdir/areal/areal/controller/rollout_controller.py`
- TrainController 的 stats 汇总：`sed -n '360,460p' workdir/areal/areal/controller/train_controller.py`
- Train engine 的 export_stats（例：FSDP）：`sed -n '560,660p' workdir/areal/areal/engine/fsdp_engine.py`

## Expected Result

- 能明确指出 AReaL 的三条观测链路：
  - `areal.utils.logging`（文本日志 + 可选落盘）
  - `areal.utils.stats_tracker`（指标采集/聚合）→ `export_stats`（聚合/汇总）→ `areal.utils.stats_logger.StatsLogger.commit`（写入 wandb/tb/swanlab）
  - `areal.utils.perf_tracer`（trace_scope/session tracer 输出 JSONL）

## References

- AReaL repo: https://github.com/inclusionAI/AReaL
- 本仓库借鉴方案：`docs/sops/logging-modularization-borrow-areal.md`

## Windows notes (PowerShell, 2026-01-20)

Commands actually used (Windows, this repo):

- `git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal`
- `git -C workdir/areal rev-parse --short HEAD` (inspected: `37e4f84`)
- `rg -n "class StatsLogger|class DistributedStatsTracker|class PerfTracer" workdir/areal/areal/utils -S`

Key takeaways for W&B metric naming:

- AReaL does **not** put everything under `train/*`. It uses hierarchical scopes such as:
  - `timeperf/<stage>` from `stats_tracker.record_timing(...)`
  - `<scope>/<metric>` from `with stats_tracker.scope("<scope>")`
- `stats_tracker` exports `key__count` for scalar aggregation across workers; `StatsLogger.commit` filters `__count` keys before logging to W&B.

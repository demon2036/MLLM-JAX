# SOP: Inspect AReaL logging/metrics/tracing system (StatsLogger/StatsTracker/PerfTracer)

- **Title**: SOP: Inspect AReaL logging system (python logging + metrics + tracing) for design borrowing
  **Prereqs**: Repo `/home/john/github/MLLM-JAX`; AReaL cloned to `workdir/areal` (see `docs/sops/areal-rl-organization.md`)
  **Environment (verified)**: Ubuntu Linux; local clone commit `d082767`

## Goal

- Locate AReaL 的三类“日志系统”实现：
  - 文本日志：`areal/utils/logging.py`
  - 指标：`areal/utils/stats_tracker.py` + `areal/utils/stats_logger.py`
  - tracing：`areal/utils/perf_tracer.py`
- 找到 trainer 侧的调用链（哪里 init、哪里 commit）。

## Steps (commands actually used)

### 1) 确认 AReaL clone 与版本

- `cd /home/john/github/MLLM-JAX/workdir/areal`
- `git rev-parse --short HEAD`
- `git show -s --oneline HEAD`

### 2) 快速定位 logging/metrics/tracing 核心文件

- `find areal -maxdepth 4 -type f \( -name '*log*' -o -name '*logger*' -o -name '*wandb*' -o -name '*tensorboard*' -o -name '*tb*' \) | sort`
- `grep -RIn "wandb" areal | head -n 200`
- `grep -RIn "setup_file_logging" areal | head -n 80`

### 3) 阅读核心模块（建议按顺序）

- 文本日志（颜色/handler/落盘）：`sed -n '1,260p' areal/utils/logging.py`
- 指标 sink（wandb/swanlab/tensorboard）：`sed -n '1,260p' areal/utils/stats_logger.py`
- 指标采集与分布式聚合：`sed -n '1,260p' areal/utils/stats_tracker.py`
- PerfTracer/SessionTracer（trace 输出 JSONL）：`sed -n '1,120p' areal/utils/perf_tracer.py`

### 4) 找到 trainer 调用链（init + commit）

- RL trainer 初始化（file logging + StatsLogger）：`sed -n '1,120p' areal/experimental/trainer/rl.py`
- RL trainer commit（export_stats + StatsLogger.commit）：`sed -n '640,780p' areal/experimental/trainer/rl.py`
- SFT trainer commit（对比）：`sed -n '340,420p' areal/experimental/trainer/sft.py`

### 5) 查看 config（StatsLoggerConfig / WandBConfig / PerfTracerConfig）

- `sed -n '1450,1620p' areal/api/cli_args.py`
- `sed -n '1760,1875p' areal/api/cli_args.py`
- `sed -n '1880,1985p' areal/api/cli_args.py`

### 6) 查看 export_stats 如何跨 worker 聚合（可选）

- RolloutController 的加权平均（使用 `__count`）：`sed -n '820,910p' areal/controller/rollout_controller.py`
- TrainController 的 stats 汇总：`sed -n '360,460p' areal/controller/train_controller.py`
- Train engine 的 export_stats（例：FSDP）：`sed -n '560,660p' areal/engine/fsdp_engine.py`

## Expected Result

- 能明确指出 AReaL 的三条观测链路：
  - `areal.utils.logging`（文本日志 + 可选落盘）
  - `areal.utils.stats_tracker`（指标采集/聚合）→ `export_stats`（聚合/汇总）→ `areal.utils.stats_logger.StatsLogger.commit`（写入 wandb/tb/swanlab）
  - `areal.utils.perf_tracer`（trace_scope/session tracer 输出 JSONL）

## References

- AReaL repo: https://github.com/inclusionAI/AReaL
- 本仓库调研输出：`answers/areal-logging-system.md`


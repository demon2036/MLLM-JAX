# SOP: Borrow AReaL logging patterns to modularize this repo’s observability

- **Title**: SOP: Borrow AReaL `StatsTracker`/`StatsLogger`/`PerfTracer` patterns to simplify and modularize logging
  **Prereqs**: Repo checkout; AReaL cloned to `workdir/areal`; familiarity with JAX multi-process (TPU multi-host)
  **Environment (verified)**: Ubuntu Linux; AReaL commit `b066584` (inspected 2026-01-25)

## Goal

- Reduce duplicated, ad-hoc metric logging (`wandb.log` + manual timing dicts) across runners.
- Standardize metric naming and multi-process aggregation.
- Keep changes non-invasive: new code under `plugins/` only (no upstream edits).

## Current state (this repo)

- Metrics/logging are implemented inside runner loops:
  - `plugins/training/runner/grpo_gsm8k.py`: builds large `train_log`/`eval_logs` dicts + manual timing vars; logs directly via `wandb.log`.
  - `plugins/sft/runner/sid_sft.py`: callback-based logging (already relatively modular), but still calls `wandb.log` directly.
  - `plugins/minionerec/rl/runner.py`: inline `wandb.log` calls.
- W&B init helper duplication:
  - `plugins/common/wandb_utils.py`: supports `process_index` gating (preferred).
  - `plugins/sft/wandb_utils.py`: legacy duplicate (currently used by `plugins/minionerec/rl/runner.py`).

## AReaL patterns worth borrowing

### 1) Text logging (Python `logging` + optional file logging)

- Core: `workdir/areal/areal/utils/logging.py`
- Key ideas:
  - Single `getLogger(...)` wrapper with consistent formatting.
  - Optional `setup_file_logging(log_dir)` that attaches streaming `FileHandler`s to all loggers (flush per message).
  - Optional multi-sink helper (`log_swanlab_wandb_tensorboard`) that maintains monotonic `step`.

### 2) Metrics collection vs sinks (the main win)

- Collection: `workdir/areal/areal/utils/stats_tracker.py`
- Sinks + printing: `workdir/areal/areal/utils/stats_logger.py`
- Key ideas:
  - Clear separation:
    - *Record anywhere* (components call `stats_tracker.scalar/stat/record_timing`).
    - *Commit centrally* (trainer calls `StatsLogger.commit(...)` once per step).
  - Hierarchical scoping: `with tracker.scope("train"):` → keys become `train/<metric>`.
  - Distributed-safe scalar aggregation:
    - `stats_tracker` emits `key__count` for SCALAR reduce type.
    - `StatsLogger.commit` filters `__count` keys before logging to W&B.
  - Timings are standardized under `timeperf/<stage>` via `record_timing(...)`.

### 3) Perf tracing (optional)

- Core: `workdir/areal/areal/utils/perf_tracer.py`
- Key ideas:
  - `trace_scope(...)` / `instant(...)` with Chrome-trace-compatible JSONL output.
  - Useful when you need nested timing breakdowns beyond simple `step_time`.
  - Probably optional for this repo because we already log `time/*` metrics; adopt only if needed.

## Proposed target layout (plugins-first)

Add a small observability package under `plugins/common/`:

- `plugins/common/observability/stats_tracker.py`: JAX-friendly metric recorder + aggregator (scopes + timing + summary stats).
- `plugins/common/observability/stats_logger.py`: sinks + printing; wraps W&B init and `wandb.log`.
- `plugins/common/observability/text_logging.py`: optional python logging setup + file handler.
- `plugins/common/observability/perf_tracer.py`: optional minimal tracer (JSONL scopes), or a thin adapter to JAX profiling.

## Minimal interfaces (proposed)

### `StatsTracker` (record-side API)

- `scope(name: str)`: context manager to prefix keys with `name/`.
- `scalar(**kvs: float)`: record host scalars (loss, entropy, etc).
- `summary(key: str, values: np.ndarray)`: record 1D arrays as `mean/std/min/max` without global full-gather.
- `masked(key: str, values: np.ndarray, mask: np.ndarray)`: record masked sums/counts (for token-level metrics).
- `record_timing(key: str)`: context manager; emits `time/<key>_s` (or `timeperf/<key>`).
- `export(*, reduce: str = "process", reset: bool = True) -> dict[str, float]`:
  - Aggregates across processes (rank 0) via small scalar allgathers (sum/count/min/max/sumsq).
  - Emits `__count` keys where needed (mirrors AReaL’s pattern).

### `StatsLogger` (commit-side API)

- Owns sink(s) and a monotonic-step guard:
  - `commit(step: int, stats: dict | list[dict]) -> None`
    - Filters out `__count` keys before passing to W&B.
    - Optionally prints a compact table (AReaL-style `tabulate_stats`).
    - Optionally appends JSONL to a local log file for offline debugging.

## Metric naming conventions (recommended)

- Use `/` separators only (avoid mixed `train-reward/...` + `train/...` styles):
  - `train/*`, `eval/*`, `time/*`, `throughput/*`
- Use stable suffixes:
  - `mean`, `std`, `min`, `max`, `count`
- Migration safety:
  - Keep backward-compatible aliases for 1–2 iterations (e.g., mirror `train-reward/...` to `train/reward/...`) so dashboards don’t break.

## Migration phases (safe increments)

1) **W&B helper consolidation**: make all runners use `plugins/common/wandb_utils.py` (no metric changes).
2) **Introduce `StatsTracker`**: start with timing + scalars in GRPO runner; keep existing metric keys.
3) **Introduce `StatsLogger`**: centralize `wandb.log` + printing; filter `__count`; enforce monotonic steps.
4) **Migrate GRPO metrics to summaries**: replace per-step `process_allgather(reward_vector)` with “local summary → global reduce” where possible.
5) **Migrate MiniOneRec RL runner**: replace inline `wandb.log` with tracker+logger commits (same metric keys).
6) **Optional**: add text/file logging + perf tracer if/when needed.

## Acceptance criteria (definition of done)

- Runners no longer call `wandb.log` directly (only `StatsLogger` does).
- Metrics are globally aggregated correctly on multi-host TPU (rank-0 logs reflect all hosts).
- Metric namespaces are consistent (`train/*`, `eval/*`, `time/*`), with alias mapping during transition.
- Logging overhead stays small (no unnecessary per-step full-gather of large arrays).

## References

- `docs/sops/areal-logging-system.md`
- AReaL repo: https://github.com/inclusionAI/AReaL


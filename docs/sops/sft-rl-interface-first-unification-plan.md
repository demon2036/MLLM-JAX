# SOP: SFT+RL interface-first unification plan (plugins-first, add-only backends)

- **Title**: SOP: Re-define interfaces to unify SFT + RL and eliminate duplication (add-only plugin backends)
  **Prereqs**: Repo checkout; familiarity with JAX multi-process; willingness to refactor under `plugins/` (non-invasive)
  **Environment (inspected)**: Ubuntu Linux; repo `/home/john/workdir/mllm-jax-minionerec` (2026-01-25)

## Why / Goal

This repo started RL-first, then added SFT with “interfaces-first” but split implementations. That achieved velocity, but now:

- SFT / RL duplicate core plumbing (tokenizer/model load, sharding, checkpoint, logging).
- Extensibility is inconsistent: rollout has a backend interface, but other areas still require editing factories/runners.

**Final target**:

- One shared “training platform” and two task pipelines (SFT, RL) built as compositions of stable interfaces.
- Adding a new backend (e.g., `sglang`) is **add-only**: add new file(s) under `plugins/` + new YAML config, without editing existing runners.

Non-goals (for the architecture phase):

- Don’t merge everything into a monolith; keep plugins modular.
- Don’t change upstream `MLLM_JAX/` internals; wrap via `plugins/`.

## Design principles (SOTA constraints)

1) **Stable contracts, unstable implementations**
   - Keep minimal interfaces (Protocols / dataclasses) stable.
   - Allow implementations to evolve without touching call sites.

2) **Add-only extensibility**
   - New components should be selectable via config using an import-path “target spec”.
   - Avoid “central factory edits” for new backends once the registry exists.

3) **Compositional pipelines**
   - RL already follows `rollout → reward → advantage → update`.
   - SFT is “dataset → update” but should reuse the same *platform* modules (model/sharding/opt/logging/ckpt).

4) **TPU-friendly by default**
   - Avoid per-step large `allgather` where summaries suffice.
   - Keep shapes stable across hosts (pad/round rules).

5) **Config traceability**
   - Every run is started from a committed YAML config file; path is stored in run metadata and W&B config.

## Target architecture (layers and folder layout)

Keep existing packages but “pull up” shared concerns into explicit layers:

- **Platform layer (shared)**: `plugins/common/`
  - config loading, env/dotenv, process gating, registries, observability, IO helpers
- **LLM layer (shared)**: `plugins/llm/` (new)
  - model family selection, HF config/tokenizer, weight loading, dtype/vocab padding, sharding/placement
- **Sampling layer (shared)**: `plugins/sample/` (already exists)
  - prompt formatting, decoding strategies, constraints, sampler backends (extend with `backends/`)
- **Training layer (shared RL algorithms)**: `plugins/training/`
  - RL phase modules, optimizer builder, generic train_step wrapper
  - (optionally) generic runner utilities shared with SFT
- **Task/pipeline layer**:
  - SFT: keep `plugins/sft/` as the task surface
  - RL: keep `plugins/training/runner/*` as the task surface (or move to `plugins/rl/` later)
  - Domain-specific RL (MiniOneRec): `plugins/minionerec/` becomes a pipeline reusing shared layers

## Core interfaces (minimal stable APIs)

### A) Component “target spec” (enables add-only backends)

Use a normalized component spec across the repo:

- `"<builtin_name>"` for stable built-ins (`naive`, `ppo`, …)
- or import-path string `"pkg.module:Symbol"` for add-only extensions
- or dict form:
  - `target: pkg.module:Symbol`
  - `kwargs: { ... }`

### B) Runtime & observability

- `StatsTracker` (record-side): scope + timing + scalar/summary recording.
- `StatsLogger` (sink-side): `commit(step, stats)` to W&B (+ optional JSONL/local prints).
- Optional `PerfTracer`: `trace_scope(name)` for nested trace events.

Reference: `docs/sops/logging-modularization-borrow-areal.md`.

### C) LLM platform

- `ModelFamily`:
  - `match(hf_config) -> bool`
  - `build(hf_config, jax_config) -> flax_module`
- `WeightsLoader`:
  - `load_state_dict(model_path, *, prefer_safetensors=True) -> dict`
  - (optional) `fallback_torch=True`
- `ParamsAdapter`:
  - `convert(state_dict) -> flax_params`
  - `patch_missing_keys(params)` (e.g., tied `lm_head`)
- `ShardingPlanner`:
  - `partition_rules(model_family) -> rules`
  - `place(params, mesh, rules, dtype) -> sharded_params`

### D) Sampling / decoding

Keep two levels:

- **Engine backend** (where generation happens):
  - `GenerationBackend.generate(...) -> {answers, token_ids?, lengths?}`
  - Implementations:
    - in-process JAX sampler (current)
    - remote engine (`sglang`) (future)
- **Decoding strategy** (how next token is chosen):
  - `greedy`, `sample`, `beam`, `constrained_beam`
  - Constraints are separate (`SidTrie`, future grammar constraints)

RL rollout remains expressed as `RolloutBackend` (already exists) but should *compose* an engine backend + batch builder.

### E) Training pipelines

- RL: keep current 4-phase module interfaces:
  - `RolloutModule`, `RewardModule`, `AdvantageModule`, `UpdateModule`
  - already defined in `plugins/training/api/interfaces.py`
- SFT: keep SFT-specific dataset/loss, but reuse:
  - `train_step` wrapper (`plugins/training/update/train_step.py`)
  - optimizer builder (`plugins/training/update/optimizer.py`)
  - shared `ModelBundle`/checkpoint/observability

## Plugin registry & factory design (critical for “add-only”)

Problem today:

- `rollout` and `algo` rely on hardcoded factories (`SUPPORTED_*`) so new implementations require editing existing files.

Target:

- One-time change: allow factories to accept import-path targets.
- After that, adding a new backend is add-only: create file + update YAML only.

Recommended pattern:

1) Introduce `plugins/common/component_loader.py`:
   - `load_symbol("pkg.module:Symbol")`
   - `instantiate(spec, **default_kwargs)`
2) Update factories (rollout backend, algorithm selection, reward fns, decode backends) to:
   - accept builtin alias OR import-path target
3) Keep builtin aliases for the stable core set; use import-path for experimental components.

## Sampling plan (incl. `sglang` without invasive edits)

### 1) Make rollout depend only on `RolloutBackend`

- Runner calls:
  - `backend.sync_weights(params)` (optional)
  - `backend.rollout(prompts=..., ...) -> RolloutResult`
- The backend owns its engine client/sampler.

### 2) Add `SGLangRolloutBackend` as add-only

Implementation sketch (future coding task):

- New file: `plugins/sample/backends/sglang_backend.py` (engine client)
- New file: `plugins/training/rollout/backends/sglang.py` (rollout backend that uses the engine backend)
- New YAML config selects it via import-path target (no runner edits).

Notes:

- For RL updates, the JAX side can still compute logprobs; the rollout backend primarily needs tokens + masks.
- Constrained decoding can stay JAX-only until the engine supports constraints.

## Unifying model loading + sharding (remove duplication)

Create a single “LLM bundle” builder (`plugins/llm/bundle.py` in the target design):

- Inputs: `model.name_or_path`, `jax.mesh_shape`, `jax.param_dtype`, `jax.compute_dtype`, `tokenizer.*`
- Outputs: `{mesh, tokenizer, model, params, shardings, pad_token_id, vocab_resize_info}`

Then:

- SFT runner, RL runner, PPO state, sampler all consume the bundle instead of duplicating:
  - HF config/tokenizer loading
  - vocab padding logic
  - `match_partition_rules/get_partition_rules_llama`
  - params placement and dtype casting

## Checkpointing / resume (one manager)

Target: `plugins/common/checkpoint/`

- `CheckpointBackend` interface:
  - `save(run_dir, step, payload)`
  - `load(path) -> payload`
- Provide two implementations:
  - `msgpack` (current, simplest)
  - `orbax` (recommended for large TPU training; future)
- `CheckpointManager`:
  - handles `save_every_steps`, `save_last`, `save_total_limit`, `resume_from`

## Observability unification (one way to log)

Target: `plugins/common/observability/`

- Adopt AReaL pattern:
  - `StatsTracker` records anywhere
  - `StatsLogger.commit(...)` logs once per step (rank 0)
  - multi-process aggregation via sum/count, filtering `__count` keys

Reference: `docs/sops/logging-modularization-borrow-areal.md`.

## Unified config schema (supports SFT-only, RL-only, or multi-stage)

Recommend a top-level multi-stage schema (even if you run only one stage today):

- `run.*`: `name`, `output_dir`, `seed`, `config_path`
- `jax.*`: `mesh_shape`, `param_dtype`, `compute_dtype`, `max_cache_length`
- `model.*`: `name_or_path`, `trust_remote_code`, `revision`
- `logging.*`: `wandb`, `print_every_steps`, `metrics_namespace`
- `checkpoint.*`: backend + save/resume policy
- `stages`: list of `{type, ...}` blocks

Example shape (illustrative only; not executed in this task):

- Stage `sft`: `data`, `train`, `eval`, `decode_backend`
- Stage `rl_grpo`: `data`, `rollout.backend`, `reward`, `algo`, `eval`

## Mapping from current code (what moves where)

High-value extraction targets:

- Model/tokenizer/params/sharding logic:
  - Today duplicated across `plugins/sft/runner/sid_sft.py`, `plugins/minionerec/rl/runner.py`,
    `plugins/training/ppo/state.py`, `plugins/sample/mllm_sampler.py`, `training2.py`.
  - Target: `plugins/llm/*` + shared bundle builder.

- Sampling engine:
  - Today: `plugins/sample/mllm_sampler.py` mixes model load + sampler.
  - Target: split into LLM bundle + sampler backend.

- Logging:
  - Today: direct `wandb.log` scattered across runners.
  - Target: `StatsLogger.commit` only.

## Phased migration plan (minimal conflicts; always runnable)

Phase 0 — **No behavior change**
- Add component loader (import-path targets) and keep existing aliases.
- Acceptance: existing YAML configs still work; tests unchanged.

Phase 1 — **Observability**
- Introduce `StatsTracker/StatsLogger`; route runner logs through commit-only.
- Acceptance: W&B keys unchanged; multi-host aggregation correct; no extra gathers.

Phase 2 — **LLM bundle**
- Centralize model/tokenizer/params/sharding; remove env-based dtype in favor of config.
- Acceptance: SFT + RL runs produce identical shapes; params placement stable.

Phase 3 — **Sampling backends**
- Add `GenerationBackend` abstraction; refactor `naive` to use it.
- Add `sglang` backend as a new file selected via import-path.
- Acceptance: RL rollout works with both backends; no runner edits for new backend.

Phase 4 — **Checkpoint manager**
- Unify checkpoint logic (SFT/RL/MiniOneRec); add save policy and resume.
- Acceptance: resume works on TPU; checkpoints are compatible across stages.

Phase 5 — **Multi-stage pipeline**
- Add optional “SFT then RL” pipeline driven by `stages:` config.
- Acceptance: one command can run stage1+stage2; shared run_dir; consistent logging.

## Evidence (commands actually used for inspection)

- Duplication scans:
  - `rg -n "AutoTokenizer\\.from_pretrained|AutoConfig\\.from_pretrained|AutoModelForCausalLM\\.from_pretrained" -S plugins`
  - `rg -n "convert_torch_to_flax|load_hf_safetensors_state_dict|get_params\\(" -S plugins`
  - `rg -n "get_partition_rules_llama|match_partition_rules|NamedSharding\\(|PartitionSpec" -S plugins/training plugins/sft plugins/sample training2.py`
  - `rg -n "save_checkpoint\\(|load_checkpoint\\(|msgpack|checkpoint" -S plugins/sft plugins/minionerec plugins/training training2.py | head -n 120`

## References

- Logging borrowing notes: `docs/sops/logging-modularization-borrow-areal.md`
- Existing RL module interfaces: `plugins/training/api/interfaces.py`
- Existing rollout backend adapter: `plugins/training/rollout/modules.py`
- Sampling shared package: `docs/sops/plugins-sample-shared-sampling.md`
- Shared abstractions (`plugins/common/`): `docs/sops/plugins-common-shared-abstractions.md`


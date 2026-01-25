# SOP: Architecture survey (AReaL/VERL/TuNIX/MaxText/Slime) for SFT+RL unification

- **Title**: SOP: Clone + compare AReaL/VERL/TuNIX/MaxText/Slime architectures and derive a SOTA “add-only” unification roadmap for this repo
  **Prereqs**: Ubuntu Linux; `git`; outbound network access
  **Environment (verified)**:
  - OS: Ubuntu (kernel `6.14.0-37-generic`)
  - Python: `3.12.2`
  - git: `2.48.1`
  - Repo: `/home/john/workdir/mllm-jax-minionerec` (branch `unify-sft-rl-phases1-5-v6e8`, commit `16f931e`)

## Goal

- Have local, gitignored clones under `workdir/` for browsing upstream architecture.
- Extract the “best ideas” for:
  - logging/observability modularity
  - add-only extensibility (new backend = add files + update config)
  - SFT+RL unification (minimize duplicated plumbing)
- Produce a concrete next-phase roadmap (Phase 6+).

## Steps (commands actually used in this repo)

### 1) Clone reference repos into `workdir/` (gitignored)

- `mkdir -p workdir`
- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/google/tunix.git workdir/tunix`
- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/volcengine/verl.git workdir/verl`
- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext`
- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/THUDM/slime.git workdir/slime`
- `GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/inclusionAI/AReaL.git workdir/areal` (may already exist)

### 2) Record inspected revisions (commit + remote)

- `for d in tunix areal verl maxtext slime; do echo \"--- $d\"; git -C \"workdir/$d\" rev-parse --short HEAD; git -C \"workdir/$d\" remote -v | head -n 2; done`
- `du -sh workdir/* | sort -h`

### 3) Inspect architecture touchpoints (minimal file list)

- AReaL:
  - `sed -n '1,220p' workdir/areal/areal/README.md`
  - `sed -n '1,220p' workdir/areal/areal/api/engine_api.py`
  - `sed -n '1,260p' workdir/areal/areal/api/workflow_api.py`
- VERL:
  - `sed -n '1,260p' workdir/verl/docs/blog/v0.7.md`
  - `sed -n '1,260p' workdir/verl/verl/workers/rollout/base.py`
- TuNIX:
  - `sed -n '1,260p' workdir/tunix/tunix/rl/rl_cluster.py`
  - `sed -n '1,260p' workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py`
  - `sed -n '1,220p' workdir/tunix/tunix/sft/metrics_logger.py`
- MaxText:
  - `sed -n '320,620p' workdir/maxtext/src/MaxText/rl/train_rl.py`
  - `sed -n '1,260p' workdir/maxtext/src/MaxText/integration/tunix/tunix_adapter.py`
- Slime:
  - `sed -n '1,240p' workdir/slime/README.md`
  - `sed -n '1,260p' workdir/slime/slime/rollout/data_source.py`
  - `sed -n '1,220p' workdir/slime/slime/rollout/sft_rollout.py`
  - `sed -n '1,220p' workdir/slime/train_async.py`

## Key takeaways (what to borrow)

### AReaL: “interfaces-first, composition-first”

- 4-layer design (`api/` → `engine/` → `workflow/` → `examples/`) is a clean mental model.
- Keep “contracts” minimal and stable; implementations can be swapped.

### VERL: “core components + trainer”

- Split core pluggable components from trainer/dataflow orchestration.
- Explicitly names “model engine / rollout engine / checkpoint engine / transfer queue” as core units.

### TuNIX: “RLCluster on TPU”

- `RLCluster` composes roles and meshes; rollout engine selectable as string or class for add-only engines.
- Logging uses backend factories and `jax.process_index()==0` gating.

### MaxText: “adapter pattern”

- Wraps MaxText model via `TunixMaxTextAdapter` to match Tunix call signature (keeps core non-invasive).

### Slime: “data-generation interface unifies SFT and RL”

- Treat SFT as a special rollout/data generator; unify the data plane with RL.
- Use a `DataSource/DataBuffer` abstraction for prompt scheduling, partial rollouts, and filtering.

## Mapping to this repo (problems + roadmap)

### Current gaps

- SFT and RL still use different data-plane abstractions and runner skeletons.
- Sharding/placement utilities are repeated in multiple runners.
- Sampler code still mixes model/weights utilities with generation logic.
- Observability `StatsTracker` exists but is not the default path for runner metrics.

### Proposed next phases (Phase 6+)

- Phase 6: introduce a shared `DataSource/DataGenerator` API and implement SFT+RL generators (Slime-style).
- Phase 7: refactor a shared “trainer skeleton” with task-specific loss modules (Tunix-style).
- Phase 8: consolidate sharding/placement helpers so checkpoint-load + placement is one reusable utility.
- Phase 9: finish sampling modularization and implement real add-only `sglang` backend.
- Phase 10: standardize metric naming + aggregation via `StatsTracker` scopes and one sink (`StatsLogger`).

## References

- `docs/sops/clone-reference-repos-into-workdir.md`
- `docs/sops/rl-four-phase-interface-research.md`
- `docs/sops/sft-rl-interface-first-unification-plan.md`
- `docs/sops/logging-modularization-borrow-areal.md`

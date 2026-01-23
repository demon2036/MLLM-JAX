# Task
- Define interfaces first for multi-RL-algorithm development: `rollout`, `reward`, `advantages-estimate`, `update`.
- Reference repos to study: Tunix, AReaL, VERL, MaxText (cloned into `workdir/`).

# Plan
1) Create task memory folder.
2) Record cloned repos revision info.
3) Analyze Tunix RL interfaces.
4) Analyze AReaL RL interfaces.
5) Analyze VERL RL interfaces.
6) Analyze MaxText RL interfaces.
7) Write unified interface proposal.
8) Run tests and update SOP.

## Step 1 - Create task memory folder
Completion criteria: `memory/<task>/README.md` exists and `memory/README.md` indexes the task.
Evidence:
- Added `memory/20260123_rl-four-phase-interfaces/README.md`.
- Updated `memory/README.md` with new task entry.

## Step 2 - Record cloned repos revision info
Completion criteria: each reference repo has a recorded origin URL + commit SHA.
Evidence:
- Commands (exit 0):
  - `git -C workdir/tunix rev-parse --short HEAD; git -C workdir/tunix remote -v`
  - `git -C workdir/areal rev-parse --short HEAD; git -C workdir/areal remote -v`
  - `git -C workdir/verl rev-parse --short HEAD; git -C workdir/verl remote -v`
  - `git -C workdir/maxtext rev-parse --short HEAD; git -C workdir/maxtext remote -v`
- Output:
  - Tunix: `dd833a9` (`https://github.com/google/tunix.git`)
  - AReaL: `b066584` (`https://github.com/inclusionAI/AReaL.git`)
  - VERL: `c0975d8` (`https://github.com/volcengine/verl.git`)
  - MaxText: `4bcee99` (`https://github.com/google/maxtext.git`)

## Step 3 - Analyze Tunix RL interfaces
Completion criteria: identify Tunix’s stable contracts for rollout/reward/adv/update, and note what varies across algos (GRPO vs PPO).
Evidence:
- Inspected files:
  - `workdir/tunix/tunix/rl/rollout/base_rollout.py` (defines `BaseRollout`, `RolloutConfig`, `RolloutOutput`)
  - `workdir/tunix/tunix/rl/rollout/sglang_jax_rollout.py` (sglang-jax rollout backend implementation)
  - `workdir/tunix/tunix/rl/rl_cluster.py` (roles + cluster orchestration; colocated vs disaggregated)
  - `workdir/tunix/tunix/rl/grpo/grpo_learner.py` (GRPO config/learner; shows which tensors are required)
  - `workdir/tunix/tunix/rl/ppo/ppo_learner.py` (PPO config/learner; critic + GAE)
  - `workdir/tunix/tunix/rl/rl_learner.py` (base learner; pipeline scheduling + reward manager hookup)
  - `workdir/tunix/tunix/rl/common.py` (trajectory data contract: `TrainExample`)
  - `workdir/tunix/tunix/rl/function_registry.py` (registries: `policy_loss_fn`, `advantage_estimator`, `reward_manager`)
  - `workdir/tunix/tunix/rl/reward_manager.py` (reward manager abstraction; default sequence-level composition)
- Key interface patterns observed:
  - **Rollout backend is a pluggable engine**: `BaseRollout.generate(prompts, rollout_config) -> RolloutOutput` with tokens + (optional) logprobs/logits.
  - **Central “trajectory” struct**: `common.TrainExample` carries `prompt_ids/prompt_mask/completion_ids/completion_mask/advantages` and optional `ref_per_token_logps/old_per_token_logps`.
  - **Algorithm-specific variability stays in (advantage + loss/update)**:
    - GRPO: no critic; advantage estimator is group-relative; policy loss is PPO-like with group advantage + optional KL-to-reference.
    - PPO: requires critic values; advantage estimator defaults to GAE; update includes actor+critic losses.
  - **Registry pattern for swapping components** (avoid inheritance-heavy design): `function_registry` registers policy losses, advantage estimators, reward managers by string key.
  - **Disaggregated training is first-class**: `rl_cluster.Role` + `ClusterConfig.role_to_mesh` make rollout/training resource split explicit; `RLLearner.should_sync_weights` is derived from whether actor/rollout share weights.

## Step 4 - Analyze AReaL RL interfaces
Completion criteria: identify AReaL’s “API ↔ backend ↔ workflow ↔ algo” boundaries and what their async rollout contract requires.
Evidence:
- Inspected files:
  - `workdir/areal/areal/README.md` (design doc: API/backend/customization/entrypoint layering)
  - `workdir/areal/areal/api/engine_api.py` (TrainEngine/InferenceEngine contracts; includes `prepare_batch`)
  - `workdir/areal/areal/api/workflow_api.py` (`RolloutWorkflow.arun_episode(...) -> trajectory|None`)
  - `workdir/areal/areal/engine/ppo/actor.py` (algorithm module: compute_logp/advantages/update; decoupled vs standard)
  - `workdir/areal/areal/workflow/rlvr.py` (example workflow: generation+reward; returns tensors incl. `rewards`)
  - `workdir/areal/areal/core/workflow_executor.py` (trajectory schema check + async task dispatcher)
  - `workdir/areal/areal/core/staleness_manager.py` (versioned staleness/capacity control)
- Key interface patterns observed:
  - **Workflow is the unit of rollout**: AReaL’s core contract is `workflow.arun_episode(engine, data) -> dict[str, Any] | None`.
  - **Trajectory dict is the shared data carrier**: keys like `input_ids/attention_mask/logprobs/loss_mask/versions/rewards` can be produced by workflow.
  - **Reward can be fused into workflow** (important for agentic RL): `RLVRWorkflow` computes reward and returns it as part of trajectory; suggests designing our interfaces so reward is optionally “inline”.
  - **Async rollout is a first-class system component**: `workflow_executor.BatchTaskDispatcher` + `StalenessManager` implement `submit/wait/prepare_batch`-style primitives with staleness bounds.
  - **Strong emphasis on schema validation**: `check_trajectory_format(...)` ensures required keys and consistent tensor shapes across episodes.

## Step 5 - Analyze VERL RL interfaces
Completion criteria: identify VERL’s rollout/reward/adv/update separations, plus its batch/trajectory data contract.
Evidence:
- Inspected files:
  - `workdir/verl/verl/protocol.py` (`DataProto`: batch carrier + concat/slice/pad utilities)
  - `workdir/verl/verl/trainer/ppo/core_algos.py` (advantage estimator registry + policy loss registry)
  - `workdir/verl/verl/trainer/ppo/ray_trainer.py` (single-controller trainer: rollout → reward → advantage → update)
  - `workdir/verl/verl/trainer/ppo/reward.py` (reward manager loading + async reward worker)
  - `workdir/verl/verl/trainer/config/algorithm.py` (`RolloutCorrectionConfig` for off-policy mismatch)
  - `workdir/verl/verl/workers/rollout/base.py` (`BaseRollout`: update_weights/resume/release + `generate_sequences`)
  - `workdir/verl/verl/workers/rollout/hf_rollout.py` (example rollout producing `DataProto` with tokens/masks)
  - `workdir/verl/verl/workers/actor/base.py` (`BasePPOActor`: compute_log_prob + update_policy)
  - `workdir/verl/verl/workers/critic/base.py` (`BasePPOCritic`: compute_values + update_critic)
  - `workdir/verl/verl/workers/rollout/schemas.py` (async rollout request schema; carries ids/masks/logprobs/reward_scores/etc)
- Key interface patterns observed:
  - **Canonical batch object (`DataProto`)**: phases mutate/extend the same object by adding keys (e.g. `token_level_rewards`, `advantages`, `returns`).
  - **Rollout is a service-like component**: `BaseRollout` explicitly manages weight residency (`resume/update_weights/release`) and returns generated sequences as `DataProto`.
  - **Advantage estimator is a registry**: `AdvantageEstimator` enum + `register_adv_est/get_adv_estimator_fn` allow “add new adv estimator without touching trainer”.
  - **Update is split by role**: actor/critic update methods live in worker classes; trainer orchestrates sequencing and metrics.
  - **Off-policy mismatch is treated as first-class** (training-inference mismatch): `RolloutCorrectionConfig` + helpers compute IS weights / rejection masks.

## Step 6 - Analyze MaxText RL interfaces
Completion criteria: identify how MaxText structures RL training and what interfaces it borrows/adopts (esp. Tunix integration + weight mapping).
Evidence:
- Inspected files:
  - `workdir/maxtext/src/MaxText/rl/train_rl.py` (MaxText RL entrypoint; builds Tunix `RLCluster` + `GrpoLearner`)
  - `workdir/maxtext/src/MaxText/integration/tunix/tunix_adapter.py` (adapts MaxText NNX Transformer to Tunix call signature)
  - `workdir/maxtext/src/MaxText/integration/tunix/utils.py` (MaxText↔HF/vLLM weight mapping + sharding knowledge)
  - `workdir/maxtext/src/MaxText/experimental/rl/grpo_trainer.py` (older GRPO implementation; producer/consumer generation+train)
- Key interface patterns observed:
  - **MaxText “outsources” RL interfaces to Tunix**: `train_rl.py` constructs Tunix `ClusterConfig` + `RolloutConfig`, then uses `GrpoLearner(rl_cluster=..., reward_fns=..., algo_config=...)`.
  - **Explicit device split (trainer vs sampler)**: `setup_configs_and_devices` can allocate disjoint device sets (and even multislice), reinforcing the need for rollout/training decoupling hooks.
  - **Adapter pattern for model compatibility**: `TunixMaxTextAdapter` makes a MaxText model look like a Tunix model (forward signature), and exposes HF/vLLM mapping hooks.
  - **Weight mapping is a first-class integration concern**: `VllmWeightMapping` generalizes param names + supplies transpose/hook/sharding info, enabling on-policy “sync_weights” into a separate rollout engine.

## Step 7 - Write unified interface proposal
Completion criteria: a written proposal exists in-repo (clickable path) that unifies rollout/reward/adv/update and maps the 4 reference repos’ lessons into a plugins-first architecture.
Evidence:
- Added proposal doc: `answers/rl-four-phase-interface-proposal.md`.
- Proposal explicitly documents:
  - 4-phase interfaces + “batch/trajectory schema”
  - optional capabilities for disaggregated/async rollout
  - mapping from Tunix/AReaL/VERL/MaxText patterns to this repo’s `plugins/` layout

## Step 8 - Run tests and update SOP
Completion criteria: local tests pass; SOP is added/updated with commands actually used.
Evidence:
- Tests:
  - `python -m pytest -q` (exit 0): `15 passed in 0.85s`
- SOP updates:
  - Added `docs/sops/rl-four-phase-interface-research.md`
  - Updated `docs/sops.md` to index the new SOP

---

# Implementation follow-up (make interfaces real in active runner)

## Step 9 - Audit GRPO runner phase calls
Completion criteria: identify all direct calls that bypass the 4-phase modules (rollout/reward/advantage/update) in the active runner.
Evidence:
- Command (exit 0): `Select-String -Path plugins/training/runner/grpo_gsm8k.py -Pattern "compute_weighted_rewards|compute_grpo_advantages_by_group_id|ppo_update" ...`
- Findings (line numbers):
  - Direct imports: `compute_grpo_advantages_by_group_id` (L13), `compute_weighted_rewards` (L14), `ppo_update` (L15)
  - Direct calls: reward (L580, L832, L967), advantage (L590), update (L658)
- Command (exit 0): `Select-String -Path plugins/training/runner/grpo_gsm8k.py -Pattern "rollout_backend\\.rollout" ...`
- Findings (line numbers):
  - Direct rollout backend calls: L562, L816, L951

## Step 10 - Add rollout-backend module adapter
Completion criteria: a `RolloutModule`-compatible wrapper exists so runners can use a `RolloutBackend` via the same 4-phase module API.
Evidence:
- Added `plugins/training/modules/rollout_backend_module.py` (`RolloutBackendModule`).
- Updated `plugins/training/modules/__init__.py` to export `RolloutBackendModule`.

## Step 11 - Refactor runner to use modules
Completion criteria: `plugins/training/runner/grpo_gsm8k.py` runs the pipeline via the 4-phase module interfaces (rollout/reward/advantage/update) instead of calling phase functions directly.
Evidence:
- Updated `plugins/training/runner/grpo_gsm8k.py`:
  - Uses `RolloutBackendModule` for rollout (train + eval).
  - Uses `WeightedRewardModule` for reward (train + eval).
  - Uses `GroupIdGRPOAdvantageModule` for advantages (train).
  - Uses `PPOUpdateModule` for update (train).
- Command (exit 0): `Select-String -Path plugins/training/runner/grpo_gsm8k.py -Pattern "compute_weighted_rewards|compute_grpo_advantages_by_group_id|ppo_update" ...`
- Output: no matches (direct phase-function calls removed).

## Step 12 - Move jit8 code into deprecated
Completion criteria: legacy `jit8_train` code is removed from the active `plugins/` tree and preserved under `deprecated/` without editing its contents.
Evidence:
- Commands (exit 0):
  - `git mv plugins/jit8_train deprecated/jit8_train`
  - `git mv test_jit8.py deprecated/test_jit8.py`
- Resulting paths:
  - `plugins/jit8_train/` no longer exists.
  - `deprecated/jit8_train/` contains the legacy jit8 implementation.

## Step 13 - Adjust tests to drop jit8 CLI
Completion criteria: the active test suite no longer imports/executes the deprecated `jit8_train` CLI.
Evidence:
- Updated `tests/test_jit8_schema_and_cli.py`:
  - Removed `from plugins.jit8_train.run import cli_main` and the CLI smoke test.
  - Kept the batch schema validator tests (now `TestGrpoBatchSchemaValidator`).

## Step 14 - Run pytest and verify pass
Completion criteria: local `pytest` exits 0 with all tests passing.
Evidence:
- Command (exit 0): `python -m pytest -q`
- Output: `14 passed in 0.84s`

## Step 15 - Update memory and SOP docs
Completion criteria: a new SOP exists for the implemented 4-phase module wiring, and the SOP index reflects it.
Evidence:
- Added `docs/sops/rl-four-phase-interface-implementation.md`.
- Updated `docs/sops.md` to index the new SOP.
- Marked `docs/sops/grpo-gsm8k-jit8-yaml-config.md` as deprecated/historical (jit8 moved to `deprecated/`).

---

# Follow-up: phase-folder layout (remove `grpo/` folder)

## Step 16 - Inventory grpo-folder dependencies
Completion criteria: enumerate all imports that reference `plugins.training.grpo.*` so we can migrate safely.
Evidence:
- Command (exit 0): `git grep -n "plugins\\.training\\.grpo"`
- Findings:
  - `plugins/training/modules/grpo_sync.py` imports `advantages/rewarding/sampling/update`.
  - `plugins/training/rollout_backends/naive_sampler.py` imports `sampling`.
  - `plugins/training/runner/grpo_gsm8k.py` imports `train_step`.
  - `tests/test_grpo_batching_inference.py` imports `batching`.

## Step 17 - Create four phase packages
Completion criteria: phase folders exist under `plugins/training/` for `rollout`, `reward`, `advantage`, and `update`.
Evidence:
- Added:
  - `plugins/training/rollout/__init__.py`
  - `plugins/training/reward/__init__.py`
  - `plugins/training/advantage/__init__.py`
  - `plugins/training/update/__init__.py`

## Step 18 - Move GRPO phase functions
Completion criteria: remove the algorithm-specific `plugins/training/grpo/` folder by relocating its per-phase utilities into the 4 phase folders.
Evidence:
- Commands (exit 0):
  - `git mv plugins/training/grpo/batching.py plugins/training/rollout/batching.py`
  - `git mv plugins/training/grpo/sampling.py plugins/training/rollout/sampling.py`
  - `git mv plugins/training/grpo/rewarding.py plugins/training/reward/weighted.py`
  - `git mv plugins/training/grpo/advantages.py plugins/training/advantage/grpo.py`
  - `git mv plugins/training/grpo/update.py plugins/training/update/ppo.py`
  - `git mv plugins/training/grpo/train_step.py plugins/training/update/train_step.py`
  - `git rm -f plugins/training/grpo/__init__.py`
- Cleanup (exit 0): removed residual `plugins/training/grpo/` dir (only `__pycache__` remained locally).

## Step 19 - Move rollout backends/optimizations under rollout
Completion criteria: rollout backends and rollout optimizations live under the rollout phase folder.
Evidence:
- Commands (exit 0):
  - `git mv plugins/training/rollout_backends plugins/training/rollout/backends`
  - `git mv plugins/training/rollout_optimizations plugins/training/rollout/optimizations`

## Step 20 - Split/move phase modules into phase folders
Completion criteria: rollout/reward/advantage/update module implementations live under their respective phase folders (and `plugins/training/modules` becomes a compatibility re-export only).
Evidence:
- Added:
  - `plugins/training/rollout/modules.py`
  - `plugins/training/reward/modules.py`
  - `plugins/training/advantage/modules.py`
  - `plugins/training/update/modules.py`
- Updated `plugins/training/modules/__init__.py` to re-export from phase modules.
- Removed legacy module implementations:
  - `git rm -f plugins/training/modules/grpo_sync.py`
  - Deleted untracked `plugins/training/modules/rollout_backend_module.py` (replaced by `plugins/training/rollout/modules.py`).

## Step 21 - Update imports and tests for new phase folders
Completion criteria: no code imports `plugins.training.grpo`, `plugins.training.rollout_backends`, or `plugins.training.rollout_optimizations`; tests import new phase folders.
Evidence:
- Updated imports:
  - `plugins/training/runner/grpo_gsm8k.py` now imports:
    - `plugins.training.rollout.backends` (backend factory)
    - `plugins.training.rollout.optimizations` (fast rollout patches)
    - `plugins.training.update.train_step` (training_step)
    - phase modules directly (`rollout/reward/advantage/update`)
  - Tests updated:
    - `tests/test_grpo_batching_inference.py` -> `plugins.training.rollout.batching`
    - `tests/test_rollout_backend_factory.py` -> `plugins.training.rollout.backends`
    - `tests/test_fast_sampler_patch.py` -> `plugins.training.rollout.optimizations`
- Command (exit 1 = no matches): `git grep -n -E "plugins\\.training\\.(grpo|rollout_backends|rollout_optimizations)"`

## Step 22 - Run pytest and update SOP docs
Completion criteria: tests pass and SOPs reflect the new phase-folder layout.
Evidence:
- Compile check (exit 0):
  - `python -m py_compile plugins/training/config.py scripts/run_grpo_gsm8k_training.py plugins/training/runner/grpo_gsm8k.py plugins/training/rollout/backends/__init__.py plugins/training/rollout/backends/base.py plugins/training/rollout/backends/naive_sampler.py plugins/training/rollout/backends/factory.py tests/test_rollout_backend_factory.py`
- Tests (exit 0):
  - `python -m pytest -q` -> `14 passed in 0.96s`
- SOP updates:
  - Updated `docs/sops/rl-four-phase-interface-implementation.md` (new phase module paths).
  - Added `docs/sops/rl-phase-folder-layout.md`.
  - Updated `docs/sops.md` to index the new SOP.

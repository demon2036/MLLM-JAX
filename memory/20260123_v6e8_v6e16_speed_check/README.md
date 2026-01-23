# Task Log: v6e-8 vs v6e-16 speed check (W&B)

## Goal

- 在不改代码的前提下，分别在 `v6e-8` 与 `v6e-16` 上跑同一套 GRPO/GSM8K baseline（W&B online），提取并对比 `time/train/step_avg_last10_s` 等速度指标，解释当前观察到的 13s vs 102s（~8x）差距来源。

## Context

- Repo: `MLLM-JAX` (local folder: `mllm-jax-improve-rollout`)
- Branch/commit: `improve-rollout` / `3d3dc0d`
- Local env: Windows PowerShell, `gcloud 551.0.0`
- Relevant SOPs:
  - `docs/sops/tpu-vm-v6e-8-v6e-16-grpo-gsm8k-speed-gap-debug.md`
  - `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-bs128-steps100.md`

## Plan (from update_plan)

1. Create task memory folder
2. Record pre-plan investigation notes
3. Check local W&B env readiness
4. Select TPU zones and names
5. Create v6e-8 TPU VM
6. Create v6e-16 TPU VM
7. Fetch SSH host key fingerprints
8. Bootstrap conda on v6e-8
9. Bootstrap conda on v6e-16 workers
10. Git-sync repo on v6e-8
11. Git-sync repo on v6e-16 workers
12. Install deps on v6e-8
13. Install deps on v6e-16 workers
14. Sync .env to both TPUs
15. Launch v6e-8 W&B run
16. Launch v6e-16 W&B run
17. Monitor logs and exit codes
18. Extract step-time metrics
19. Compare results and bottlenecks
20. Update SOP and report status

## Progress Log

### Step 1: Create task memory folder

- 完成判据：
  - 新建 `memory/20260123_v6e8_v6e16_speed_check/` 并包含 `README.md`
  - 在 `memory/README.md` 的 Task Index 登记该任务
- 证据：
  - Files changed:
    - `memory/20260123_v6e8_v6e16_speed_check/README.md`
    - `memory/README.md`

### Step 2: Record pre-plan investigation notes

- 完成判据：
  - 明确“v6e-16 看起来比 v6e-8 慢很多”的最可能原因，并确认仓库内已有对应 SOP / guardrails 可复用
- 证据：
  - Notes (from repo SOP `docs/sops/tpu-vm-v6e-8-v6e-16-grpo-gsm8k-speed-gap-debug.md`):
    - `v6e-16` 是 **4-host / 16 chips**；若只启动 worker0，则实际只用到 **4 chips**，会比 `v6e-8`（8 chips）更慢
    - 若叠加“每 step 计算量 ~8x”，会出现“v6e-8 ~13s vs v6e-16 ~102s（~8x）”这种表象差距
    - 跨 host FSDP（例如 `mesh_shape: 1,-1,1` 在 v6e-16 上变成 fsdp=16 跨 host）可能进一步拖慢 rollout/decode；推荐 v6e-16 用 `mesh_shape: 4,4,1`（dp=4 跨 host，fsdp=4 本地）
  - Code/Script guardrails confirmed:
    - `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh` 默认 `export REQUIRE_MULTIHOST=1`
    - `plugins/training/runner/grpo_gsm8k.py` 在 `REQUIRE_MULTIHOST=1` 且 `jax.process_count()==1` 时会 fail-fast（提示需 `--worker=all`）

### Step 3: Check local W&B env readiness

- 完成判据：
  - 确认训练依赖包含 `wandb`，并准备好以“只在 TPU 上注入 secret”的方式让 run 上报到 W&B（不把 key 写入仓库/提交记录）
- 证据：
  - Repo deps: `requirements-tpu.txt` 包含 `wandb`
  - W&B secret handling plan:
    - 在 TPU VM 上写入 `/root/.env`（权限 `600`），包含 `WANDB_API_KEY`（值不记录到 repo）与 `WANDB_ENTITY=johntitordemon2036`
    - 训练入口 `scripts/run_grpo_gsm8k_training.py` 会读取 `/root/.env` 并在检测到 `WANDB_API_KEY` 时自动设置 `WANDB_MODE=online`

### Step 4: Select TPU zones and names

- 完成判据：
  - 选定可同时创建 `v6e-8` 与 `v6e-16` 的 zone，并确定本次实验 TPU 名称（便于后续 ssh/scp/日志定位）
- 证据：
  - Zone: `us-east1-d`（加速器类型列表中存在 `v6e-8` 与 `v6e-16`）
  - Names:
    - `mllm-jax-v6e-8-speed-260123013221`
    - `mllm-jax-v6e-16-speed-260123013221`

### Step 5: Create v6e-8 TPU VM

- 完成判据：
  - `v6e-8` TPU VM 创建成功且 `state=READY health=HEALTHY`
- 证据：
  - Create attempt 1 (failed: zone capacity):
    - Command: `gcloud alpha compute tpus tpu-vm create mllm-jax-v6e-8-speed-260123013221 --zone us-east1-d --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --quiet`
    - Result: `There is no more capacity in the zone "us-east1-d"`
  - Create attempt 2 (success):
    - Command: `gcloud alpha compute tpus tpu-vm create mllm-jax-v6e-8-speed-260123013221 --zone us-east5-b --accelerator-type v6e-8 --version v6e-ubuntu-2404 --spot --quiet`
    - Describe (key fields): `acceleratorType=v6e-8 state=READY health=HEALTHY` (zone `us-east5-b`)

### Step 6: Fetch v6e-8 SSH host key

- 完成判据：
  - 获取并记录 v6e-8 worker0 的 `ssh-ed25519` host key fingerprint，确保后续可用 `--ssh-flag=-batch --ssh-flag=-hostkey` 无交互执行命令
- 证据：
  - Interactive probe (printed fingerprint then aborted):
    - Command: `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-speed-260123013221 --zone us-east5-b --worker 0 --quiet --command "set -euo pipefail; hostname; echo ok"`
    - Fingerprint: `SHA256:ZHJ2dckRAB6fXKSjdN8KjyvdxHce0A1UqDakePYI0Vc`
  - Non-interactive SSH (success):
    - Command: `gcloud alpha compute tpus tpu-vm ssh root@mllm-jax-v6e-8-speed-260123013221 --zone us-east5-b --worker 0 --quiet --ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:ZHJ2dckRAB6fXKSjdN8KjyvdxHce0A1UqDakePYI0Vc --command "set -euo pipefail; hostname; echo ok"`
    - Key output: `ok`

### Step 7: Snapshot branch diff set

- 完成判据：
  - 明确 `improve-rollout` 相对 `origin/main` 的变更清单，并确认“只保留到 main 的目标文件集合”
- 证据：
  - Working tree (uncommitted) changes:
    - `AGENTS.md` (M)
    - `docs/sops.md` (M)
    - `docs/sops/task-memory-workflow.md` (untracked)
    - `memory/` (untracked)
  - Committed branch diff (`origin/main..improve-rollout`) 覆盖多个模块（rollout 优化、runner、脚本、SOP、tests 等），因此按用户要求只挑选两个文件进入 main：
    - Keep #1: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
    - Keep #2: `scripts/tpu_vm_start_grpo_gsm8k_qwen25_3b_bs128_steps100_v6e16_multihost_nohup.sh`

### Step 8: Restore `memory/` and `AGENTS.md` onto `main`

- 完成判据：
  - `main` 工作区恢复出 `memory/` 与 `docs/sops/task-memory-workflow.md`，且 `AGENTS.md` 变更不丢失
  - 无 merge conflict / 无冲突标记残留
- 证据：
  - Stash contained (from `git stash show -u --name-status "stash@{1}"`):
    - `A docs/sops/task-memory-workflow.md`
    - `A memory/20260123_v6e8_v6e16_speed_check/README.md`
    - `A memory/README.md`
    - `A memory/_template/README.md`
    - `M AGENTS.md`
    - `M docs/sops.md`
  - Applied onto `main`:
    - Command: `git stash apply "stash@{1}"`
    - Result: `memory/` 与 `docs/sops/task-memory-workflow.md` 以 untracked 形式恢复，`AGENTS.md`/`docs/sops.md` 为 modified；无冲突产生

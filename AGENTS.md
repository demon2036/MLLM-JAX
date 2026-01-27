# Repository Guidelines

## Assistant Execution Policy (Required)

- 默认按用户需求“一口气”完成：实现、TPU 上完整验证（端到端能跑）、更新 SOP。
- 交付前必须在 TPU 上把对应的测试/验证 case 全部跑通，并开启 W&B `wandb_mode=online`（除非用户未提供/无法使用 W&B Key，需在交付说明里明确）；任何 traceback / 非零退出都视为未完成。
- “完整验证”以 TPU 上目标配置跑足为准（例如目标 batch size/steps/eval 集）；调试/单测阶段允许小规模 smoke/test，但不算交付验证。
- 接到用户指派任务后：优先检查/创建 `memory/`；先读 `memory/README.md`（若现有记录与本任务无关则删除对应条目/文件夹），再复用/新建 `memory/YYYYMMDD_<slug>/`，并在 `memory/README.md` 登记用途。
- 无论任务大小，若提供 plan/update_plan（或 start plan）函数，每次任务都必须调用；若工具不可用，执行前以清晰优雅的列表列出计划并直接开始执行，不等待用户确认。
- 一旦开了 plan：允许按批次记录（建议每 5–10 个 step，或每个关键 milestone 记录一次），在 `memory/<task>/README.md` 追加“完成判据 + 证据”（例如：命令+exit code、关键输出摘要、改动文件、通过的测试/验证），便于后续复用。
- 尽量不反问用户：先查 `memory/` 里潜在相关经验（优先看 `memory/README.md`），再查 `docs/` 里潜在相关经验（优先看 `docs/sops.md`；若是分支专用 SOP，再看 `docs/sops/<branch>/`），再在仓库内搜索（如 `rg`/`git`；无 `rg` 时用 `git grep`/`Select-String`），必要时做网络搜索（官方文档/GitHub/PyPI），并把已用链接或命令写入 SOP。
- 仅在搜索后仍无法推进时才提问，并一次问清最少必要信息。
- 保持无侵入开发：自定义代码一律放在 `plugins/`，不要直接改动 `easydel/` 或其他上游仓库。
- 配置语义清晰优先：禁止在 `scripts/` 里用环境变量/拼接参数去“覆盖”训练超参；如果要改配置，必须新建一个新的 YAML config，并用 `*.sh --config <path>.yaml` 启动，让 W&B 配置可追踪、可复现。
- W&B：TPU 验证/交付默认 `wandb_mode=online`；仅当用户未提供/无法使用 W&B Key 时允许 `disabled`（交付需注明）；本地调试可临时 `disabled`。
- TPU 使用隔离：除非用户明确指定复用已有 TPU/同一集群，否则每个任务新建独立 TPU（按任务名命名），避免在无关 TPU 上复用环境导致相互影响。
- TPU 上的覆盖/替换通过自写 shell 脚本完成（例如同步到覆盖目录 + `PYTHONPATH`），避免改动原始仓库内容。
- TPU 开发/运行采用 Git 同步：本地修改代码 → `git commit`/`git push` 到 GitHub → TPU VM 通过 `scripts/ssh_tpu_vm_root.sh`（或 `gcloud ... tpu-vm ssh`）执行 `git clone`/`git pull` 获取代码并运行；不要用 `gcloud ... scp` / `scp` 手动拷贝代码（见 `docs/sops.md#tpu-vm-repo-sync`）。（例外：同步本地 secret 如 `.env`，用 `scripts/sync_env_to_tpu_vm.sh` 分发到 worker=all。）
- 只能在自己分支开发；除非用户明确下达命令，禁止 push 到 main（main 绝对禁止）。
- `memory/` 为临时工作区：只保留与本任务相关记录；不再使用 `memo/`（草稿也写到 `memory/<task>/`，必要时可直接删除）。

## 大局观与接口先行（Required）

核心目标：**让后人能在不改训练主流程的前提下，按配置替换/新增组件**（loss、optimizer、rollout backend、reward、sampler、data 等），并且在 W&B 上可追踪、可复现。

### 原则（Why）

- 把 “配置 + 接口” 当成 API：稳定、可扩展、可测试。
- 训练 loop/runner 尽量只依赖接口，不依赖具体实现细节（避免到处 `if/else` 特判）。
- 默认实现可以是最简单的（例如 SFT 的 CE loss），但必须预留扩展点（例如 label smoothing / focal / distillation / 多任务 loss / mix-in loss）。

### 怎么做（How）

- **先写接口/契约，再写实现**：先定义输入/输出、shape/dtype、mask 语义、返回的 metrics 命名，再落地默认实现。
- **可插拔（pluggable）优先**：用 registry/factory 通过配置选择实现；不要在 runner 里 hardcode 某个实现。
- **新增行为必须新增 config**：禁止用脚本参数拼接/环境变量暗改超参；新增 loss/optimizer 等必须新建 YAML config（W&B 才能完整追踪）。
- **自定义代码放 `plugins/`**：接口通常放 `plugins/**/api/` 或 `plugins/common/`；上游目录只做最小必要 glue（如 import / wiring），避免侵入式改动。
- **文档先行**：接口落地后，在 SOP/README 里写清楚 “扩展点 + 默认实现 + 如何新增一个实现 + 验证命令”。

### 例子：SFT 的 loss 不要 hardcode CE

我们希望：SFT 默认用 CE，但以后加别的 loss 时，不需要改训练主循环，只需要：
1) 新增一个实现文件；2) 在 registry 里注册；3) 新建 YAML config 指向它。

推荐结构（示意）：

```python
# runner/loop 里（只依赖接口，不关心具体 loss）
loss_fn = build_loss(config.train.loss)
loss, metrics = loss_fn(model, batch, rng=rng)
```

```yaml
# configs/sft/qwen2_5_3b_ce.yaml
train:
  loss:
    name: ce
    label_smoothing: 0.0
```

```yaml
# configs/sft/qwen2_5_3b_focal.yaml
train:
  loss:
    name: focal
    gamma: 2.0
```

落地时优先复用现有接口与拆分方式：
- `plugins/training/api/interfaces.py`（接口定义）
- `docs/sops/training-modularization-plan.md`（接口先行的拆分路线）
- `docs/sops/rl-pluggable-optimizer.md`（“把 hardcode 变成可配置扩展点”的具体范例）

### PR 自查清单（Checklist）

- 是否先定义了接口契约（输入/输出/语义/metrics）而不是直接写实现？
- 新增实现是否能通过配置启用（而不是改 runner）？
- 新增实现是否有独立 YAML config（可复现、可在 W&B 对齐）？
- 新增实现是否放在 `plugins/` 且侵入改动最小？
- 默认配置/不配置时，行为是否保持与历史一致（避免无意破坏已有实验）？
- metrics 命名是否稳定、可对比（避免随意改 key 导致 W&B 面板断档）？
- 是否补充了最小验证步骤（smoke 或目标配置），并记录在 SOP/memory？

## Project Structure & Module Organization

This repository is focused on validating and documenting `sglang-jax`. The repo is currently minimal; keep the layout simple and documented as it grows. Recommended folders:

- `docs/` for SOPs, validation notes, and references (branch-specific SOPs under `docs/sops/<branch>/`, e.g., A → `docs/sops/A/`, B → `docs/sops/B/`).
- `plugins/` for reusable, non-invasive overrides and integration code (loss/model/optimizer/sampler/backends).
- `projects/` for project-scoped configs, data pointers, and entrypoints (e.g., nano-specific datasets/configs); avoid placing non-reusable configs in `plugins/`.
- `scripts/` for repeatable setup or test helpers.
- `tests/` for any local verification scripts.

Keep this file as the primary contributor guide and a quick entry point.

## Build, Test, and Development Commands

Commands are intentionally captured as SOPs so they stay accurate. Do not guess; only record commands you have actually run. Use placeholders until verified.

Example template (replace placeholders with real, validated steps):

```bash
git clone <sglang-jax-repo-url>
cd <sglang-jax-dir>
<install-command>
<test-command>
```

When you add a new command, also note the environment (OS, Python/JAX versions, GPU/CPU).

## Coding Style & Naming Conventions

Until the project adopts a formatter/linter, keep files consistent and readable:

- Use 2-space indentation for config files; 4-space indentation for code unless the language ecosystem dictates otherwise.
- Prefer descriptive file and module names (no abbreviations).
- Keep helper scripts small and single-purpose; name them with verbs (e.g., `setup_env.sh`).

## Testing Guidelines

Testing requirements must be derived from actual `sglang-jax` usage. Record the framework and commands in SOPs once verified. Use stable naming patterns such as `test_*.py` or `*_test.py` depending on the framework.

## Commit & Pull Request Guidelines

Until a project-specific convention is defined, use Conventional Commits (e.g., `feat: add env setup SOP`). Pull requests should include: purpose, scope, and test results (or explain why tests were not run).

## SOP Capture (Required)

Before answering any user question, first review existing SOPs to reuse prior experience. If no SOP applies, create a new one. After work is done, summarize new learnings as an SOP entry so others can reuse them. Each SOP should be brief, deterministic, and easy to follow.

Recommended format:

- **Title**: Short, action-focused (e.g., "SOP: Clone and bootstrap sglang-jax")
- **Prereqs**: OS, Python/JAX versions, hardware notes
- **Steps**: Exact commands run, in order
- **Expected Result**: What success looks like
- **Troubleshooting**: Common errors and fixes
- **References**: Links or commit SHAs used

Add/maintain common/stable SOPs in `docs/sops.md`. If the SOP is branch-specific, create a dedicated folder per branch so they are “错开”: A branch → `docs/sops/A/`, B branch → `docs/sops/B/`.

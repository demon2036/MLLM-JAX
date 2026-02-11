- **Title**: SOP: 审计 RL runner 的隐式行为（自动改写/静默降级）
- **Prereqs**:
  - Repo: `MLLM-JAX-max-rl`
  - Python: `python3` 可用
  - 目标范围：`projects/gsm8k_grpo` + `plugins/training/rl` + 配置/日志辅助模块

- **Steps**:
  1. 先读现有索引与历史记录，避免重复排查。
     - `sed -n '1,220p' memory/README.md`
     - `sed -n '1,260p' docs/sops.md`

  2. 全局 grep 隐式行为关键词（override/fallback/except/env/pad/drop）。
     - `rg -n "override|fallback|deprecated|warning|except|drop|pad|auto|grad_accum|max_sequences_per_pass|jax\.distributed\.initialize|wandb_mode|ROLLOUT_FAST|EVAL_GREEDY|EVAL_FULL_SWEEP" projects/gsm8k_grpo plugins/training training2.py MLLM_JAX`

  3. 定点阅读关键代码路径。
     - `nl -ba projects/gsm8k_grpo/jax/train.py | sed -n '180,420p'`
     - `nl -ba projects/gsm8k_grpo/jax/train.py | sed -n '520,820p'`
     - `nl -ba projects/gsm8k_grpo/scripts/run_train.py | sed -n '184,340p'`
     - `nl -ba plugins/training/core/config/loader.py | sed -n '1,120p'`
     - `nl -ba plugins/training/core/logging/wandb.py | sed -n '1,60p'`
     - `nl -ba plugins/training/rl/reward/weighted.py | sed -n '1,80p'`

  4. 用 `git blame` 给“隐式逻辑”定位引入提交。
     - `git blame -L 232,265 projects/gsm8k_grpo/jax/train.py`
     - `git blame -L 267,317 projects/gsm8k_grpo/jax/train.py`

  5. 做最小复现验证（不启动完整训练，仅解析配置）。
     - typo key 静默忽略复现：
       - `python3 projects/gsm8k_grpo/scripts/run_train.py --config /tmp/rl_hidden_check_typo.yaml --print-config`
     - `${ENV}` 缺失导致回落复现：
       - `python3 projects/gsm8k_grpo/scripts/run_train.py --config /tmp/rl_hidden_check_env_blank.yaml --print-config`

  6. 输出风险分级清单：高/中/低，并记录“触发条件 + 代码位置 + 影响 + 可见性 + 建议修复”。

- **Expected Result**:
  - 能明确回答：
    - rollout 是否被自动切 pass / padding。
    - grad_accum 是否会被运行时重写。
    - 是否存在异常吞掉继续跑、或 W&B/distributed 的静默降级。
  - 给出至少两条可复现证据（命令 + exit code）。

- **Troubleshooting**:
  - `python: command not found`：改用 `python3`。
  - 如果想避免重负载，不要运行完整 `run_grpo_gsm8k`，优先用 `--print-config` 做语义复现。
  - 多机场景若担心误单机运行，设置：
    - `REQUIRE_MULTIHOST=1` 或 `REQUIRE_JAX_PROCESS_COUNT=<N>`。

- **References**:
  - `projects/gsm8k_grpo/jax/train.py`
  - `projects/gsm8k_grpo/scripts/run_train.py`
  - `plugins/training/core/config/loader.py`
  - `plugins/training/core/runtime/env.py`
  - `plugins/training/core/logging/wandb.py`
  - `plugins/training/rl/reward/weighted.py`

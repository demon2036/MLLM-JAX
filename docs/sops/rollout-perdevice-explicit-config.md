- **Title**: SOP: 把 rollout 改为 per-device 显式 micro-batch（去掉隐式 `*8`）
- **Prereqs**:
  - Repo: `MLLM-JAX-max-rl`
  - Python: `python3`
  - 目标模块：`projects/gsm8k_grpo`

- **Steps**:
  1. 定位旧隐式规则：
     - `rg -n "max_sequences_per_pass|round_up_passes_for_divisibility|Padding global rollout.batch_size" projects/gsm8k_grpo/jax/train.py`

  2. 在 rollout schema 增加显式字段：
     - `rollout.micro_batch_size`
     - `rollout.micro_batch_size_per_device`
     - 文件：`projects/gsm8k_grpo/config_schema.py`

  3. 在配置解析层强制 per-device 必填：
     - 文件：`projects/gsm8k_grpo/scripts/run_train.py`
     - 要求：缺失 `rollout.micro_batch_size_per_device` 直接 `ValueError`。

  4. 在训练主环替换 pass 规划逻辑：
     - 文件：`projects/gsm8k_grpo/jax/train.py`
     - 删除：`max_sequences_per_pass = train.micro_batch_size * 8` 及隐式 padding/round-up。
     - 改为：基于 `rollout.micro_batch_size_per_device` 的显式 pass 规划，且整除约束不满足时 fail-fast。

  5. 补齐 YAML 显式参数：
     - 在每个 `projects/gsm8k_grpo/configs/*.yaml` 的 `rollout` 下添加：
       - `micro_batch_size: null`
       - `micro_batch_size_per_device: 32`

  6. 运行验证：
     - 语法：
       - `python3 -m py_compile projects/gsm8k_grpo/config_schema.py projects/gsm8k_grpo/scripts/run_train.py projects/gsm8k_grpo/jax/train.py plugins/training/rl/config.py`
     - 配置解析：
       - `python3 projects/gsm8k_grpo/scripts/run_train.py --config projects/gsm8k_grpo/configs/<file>.yaml --print-config`
     - 缺失字段 fail-fast：
       - `python3 projects/gsm8k_grpo/scripts/run_train.py --config /tmp/rollout_missing_perdevice.yaml --print-config`

- **Expected Result**:
  - 代码中不再存在 rollout `micro_batch_size * 8` 隐式 cap。
  - rollout per-device micro-batch 成为显式必填项。
  - 配置不满足整除条件时立即报错，而不是静默改写 batch/passes。

- **Troubleshooting**:
  - 若报错 `rollout micro-batch is too small for one prompt-group pass`：提高 `rollout.micro_batch_size_per_device` 或降低 `rollout.n`。
  - 若报错 `rollout.batch_size must be divisible by process_count`：把 `rollout.batch_size` 调整为 `process_count` 的整数倍。
  - 若报错 `requested prompts per process must be divisible by prompts_per_pass`：调整 `rollout.batch_size` 或 rollout micro-batch 使 pass 整除。

- **References**:
  - `projects/gsm8k_grpo/jax/train.py`
  - `projects/gsm8k_grpo/scripts/run_train.py`
  - `projects/gsm8k_grpo/config_schema.py`
  - `projects/gsm8k_grpo/configs/*.yaml`

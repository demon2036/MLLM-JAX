# SOP: Deep-dive official `ReMax` (DeepSpeed-Chat) implementation in `workdir/remax`

- **Title**: SOP: Read and audit the official `ReMax` implementation (DeepSpeed-Chat based) cloned under `workdir/remax`
  **Prereqs**: Repo already cloned into `workdir/remax` (see `docs/sops/clone-reference-repos-into-workdir.md`)
  **Scope**: `workdir/remax` (external reference repo), especially `step3_rlhf_finetuning/`

## Goal

快速回答：

- ReMax 在官方代码里到底怎么做 baseline / advantage / loss？
- 关键实现在哪些文件/函数里？
- 复刻（到 JAX/TPU 或别的框架）时有哪些坑要提前规避？

## Steps (commands used)

确认来源与版本：

```bash
cd workdir/remax
git remote -v
git rev-parse HEAD
git log -1 --oneline
```

定位 ReMax trainer 与入口：

```bash
rg -n "remax|ReMax" -S . | head -n 200
find step3_rlhf_finetuning -maxdepth 2 -type f
sed -n '1,260p' step3_rlhf_finetuning/remax_trainer.py
nl -ba step3_rlhf_finetuning/main.py | sed -n '640,1120p'
```

确认 reward model 的 end-score 语义：

```bash
sed -n '1,260p' utils/model/reward_model.py
```

## Expected Result

- 入口为 `deepspeed step3_rlhf_finetuning/main.py ... --algo remax`
- 训练 loop 调用：
  - `DeepSpeedReMaxTrainer.generate_experience()` 生成 sample 与 greedy baseline，并用 reward model 算 `reward` 和 `baseline_reward`
  - `DeepSpeedReMaxTrainer.compute_loss()` 做 `adv = reward - baseline_reward`，再做 token-level policy gradient
- 代码中强约束 `pad_token == eos_token`，并对 EOS 的 action_mask 做了修正（否则 EOS token 无梯度）。

## Notes (implementation details & pitfalls)

核心实现语义（step3）：

- baseline：当前 actor **greedy** 解码的 reward（而不是 value model）
- advantage：`reward(sample) - reward(greedy)`
- return：对 advantage 做折扣传播，并叠加每 token 的 KL penalty（与 reference model 的 token-level logprob 差）
- loss：`-sum(return * logp * mask) / sum(mask)`（REINFORCE 风格；无 PPO ratio/clipping）

常见坑：

- **必须用 DeepSpeed 启动**（强依赖 `torch.distributed` 初始化；`create_prompt_dataset()` 会做 all_reduce）。
- step3 的 `--ppo_epochs`/`--clip_reward_value` 等参数存在“看起来能调，但训练逻辑没用到/被硬编码”的情况；`--ppo_epochs` 还会影响 scheduler 的 `num_total_iters`，易导致 step 计数不一致。
- `tests/test_training.py` 引用的 sweep 路径在当前 `workdir/remax` 结构下不存在（以 `training_scripts/*/run_*.sh` 为准）。

## References

- `workdir/remax` upstream: `https://github.com/liziniu/ReMax`
- step3 entry: `workdir/remax/step3_rlhf_finetuning/main.py`
- ReMax trainer: `workdir/remax/step3_rlhf_finetuning/remax_trainer.py`
- Reward model head + end-score: `workdir/remax/utils/model/reward_model.py`

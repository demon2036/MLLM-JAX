# Merge `sft` into `minionerec` + enable GRPO rolloutfast launcher defaults

- **Title**: SOP: Merge `origin/sft` into `minionerec` and make GRPO rollout optimizations default in the TPU nohup launcher
  **Prereqs**: `git` + `pytest` available; you have push access to the repo; working tree clean
  **Environment (verified)**: Ubuntu; git `2.48.1`; pytest `8.x`

## Goal

- Keep `minionerec` updated with `sft` branch work.
- Make GRPO rollout optimizations default when launching via:
  - `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
  - `scripts/tpu_vm_start_grpo_gsm8k_from_config_multihost_nohup.sh` (wraps the single-host script)

## Steps (commands actually used)

### 1) Create a working branch from `origin/minionerec`

```bash
git fetch --all --prune
git checkout -b john/minionerec-grpo-rolloutfast origin/minionerec
```

### 2) (Optional) Add GRPO rolloutfast configs + fixes

This branch included two GRPO configs and a small HF tied-weights fix:

- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps2_v6e8_rolloutfast_smoke.yaml`
- `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_v6e8_rolloutfast.yaml`
- `plugins/sample/mllm_sampler.py` handles missing `lm_head.weight` by using `model.embed_tokens.weight`.

### 3) Merge `origin/sft` into the working branch

```bash
git merge --no-edit origin/sft
```

### 4) Enable rollout speedups by default in the GRPO nohup launcher

Edit `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh` to export defaults:

- `ROLLOUT_FAST_GENERATE=1`
- `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=1`

Then commit:

```bash
git add scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh
git commit -m "chore: enable rolloutfast by default in GRPO launcher"
```

### 5) Run local tests

```bash
pytest -q
```

### 6) Push the branch and fast-forward `origin/minionerec`

```bash
git push -u origin john/minionerec-grpo-rolloutfast
git push origin HEAD:minionerec
```

## Expected Result

- `origin/minionerec` contains the `origin/sft` updates.
- Running `bash scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh --config <...>.yaml` defaults to fast rollout.
  - Override: set `ROLLOUT_FAST_GENERATE=0` and/or `ROLLOUT_FAST_QWEN2_DECODE_ATTENTION=0` before launching.
- `pytest -q` exits `0`.

## Troubleshooting

- If a merge conflict happens (often around sampler/generate code):
  - Prefer keeping the `sft` side intact, then re-apply only the minimal GRPO launcher/config changes as separate commits.
  - Re-run `pytest -q` after resolving.

## References

- https://git-scm.com/docs/git-merge
- https://git-scm.com/docs/git-push


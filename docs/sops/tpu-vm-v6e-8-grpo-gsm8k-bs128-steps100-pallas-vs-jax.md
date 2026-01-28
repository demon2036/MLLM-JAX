# TPU VM v6e-8 GRPO/GSM8K 100-step Train: Pallas kernel vs JAX baseline (W&B online)

- **Title**: SOP: Run GRPO/GSM8K 100 steps on v6e-8 and compare `policy_loss_impl=pallas` vs `jax`
  **Prereqs**: `gcloud` authenticated; TPU VM already exists; repo branch pushed; W&B API key available on TPU; outbound internet (HF + datasets + wandb)

## Scope

This SOP verifies that swapping GRPO policy-loss implementation from pure JAX to a JAX Pallas kernel:

- runs end-to-end on TPU
- logs to W&B (`wandb_mode=online`)
- produces aligned GSM8K correctness metrics after 100 steps

## Target TPU VM (this run)

- TPU name: `minionerec-sft-subsetdiff-v6e8-euwest4a-260126160555`
- Zone: `europe-west4-a`
- Type: `v6e-8` (spot)

## Configs

- Baseline (JAX): `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml`
- Pallas: `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100_pallas.yaml` (`train.policy_loss_impl: pallas`)

## Steps (commands actually used)

> NOTE: Replace placeholders below **only after** you run the commands. Keep this SOP deterministic.

### 0) SSH sanity check

- TBD

### 1) Git sync (no SCP for code)

- TBD

### 2) Ensure WANDB_API_KEY is available on TPU (online mode)

- TBD (verify without printing the key)

### 3) Run baseline (JAX) 100 steps

- TBD: start command (nohup launcher)
- TBD: monitor command(s)
- TBD: exit code file (expect `0`)
- TBD: W&B run URL

### 4) Run Pallas 100 steps

- TBD: start command (nohup launcher)
- TBD: monitor command(s)
- TBD: exit code file (expect `0`)
- TBD: W&B run URL

### 5) Extract final metrics from `wandb-summary.json`

- TBD: command(s)
- Keys to record:
  - `eval/reward/func/reward_correct/mean`
  - `eval/reward/total/mean`
  - `time/train/step_avg_last10_s`

## Expected result

- Both runs complete with exit code `0`.
- Both runs appear in W&B (online) and have comparable training/eval curves.
- Final `eval/reward/func/reward_correct/mean` is aligned (report deltas).

## Troubleshooting

- If TPU SSH prompts for host key (Windows plink): use `--ssh-flag=-batch --ssh-flag=-hostkey --ssh-flag=SHA256:<fingerprint>`.
- If W&B init fails: verify `WANDB_API_KEY` is set on TPU and `wandb_mode: online` in YAML.

## References

- `scripts/tpu_vm_start_grpo_gsm8k_from_config_nohup.sh`
- `scripts/run_grpo_gsm8k_training.py`
- `plugins/training/kernels/grpo_loss_pallas.py`


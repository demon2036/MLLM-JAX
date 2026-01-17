# SOP: Run GSM8K GRPO Qwen2.5-7B (3 steps) on TPU VM

- **Title**: SOP: Smoke-run `gsm8k + GRPO + Qwen2.5-7B` training for 3 steps on a Cloud TPU VM
  **Prereqs**: `gcloud` installed + authenticated; TPU quota/capacity; internet egress from TPU VM (HF + datasets); local changes pushed to GitHub
  **Environment (verified)**:
  - TPU VM: `v4-8` (`us-central2-b`), runtime `tpu-ubuntu2204-base`
  - OS: Ubuntu `22.04.2` (kernel `5.19.0-1022-gcp`)
  - Python: `3.12.12` (conda env `mllm-jax`)
  - JAX: `0.8.2`, jaxlib `0.8.2`, libtpu `0.0.32`
  - JAX devices: `4` (TPU v4-8 in megacore mode)
  **Steps**:
  - Create TPU VM (example used in verification):
    - `cd /home/john/github/MLLM-JAX`
    - `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b --name mllm-jax-v4-8-260117090531`
  - Run the 3-step GRPO+GSM8K smoke training (verified):
    - `cd /home/john/github/MLLM-JAX`
    - `scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh --name mllm-jax-v4-8-260117090531 --zone us-central2-b --ref fcaa7b0`
  **Expected Result**:
  - TPU VM prints 3 training lines similar to:
    - `step=0 ...`
    - `step=1 ...`
    - `step=2 ...`
  **Troubleshooting**:
  - TPU shows `PREEMPTED`: delete and recreate TPU VM.
  - `ModuleNotFoundError: No module named 'MLLM_JAX'`: ensure you run via `scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh` (it checks out the right commit and runs from repo root).
  - `TypeError: _form_global_array() missing ...`: indicates cache sharding helper signature mismatch; use commit `fcaa7b0` or newer.
  **References**:
  - `scripts/run_grpo_gsm8k_qwen25_7b_3steps_on_tpu_vm.sh`
  - `scripts/run_smoke_grpo_gsm8k_qwen25_7b.py`
  - `docs/sops/tpu-vm-repo-sync.md`
  - `docs/sops/tpu-vm-bootstrap.md`

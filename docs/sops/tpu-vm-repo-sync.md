# TPU VM Repo Sync SOPs

- **Title**: SOP: Sync this repo to a TPU VM via Git (no SCP)
  **Prereqs**: Local changes pushed to GitHub; TPU VM reachable via `gcloud ... tpu-vm ssh`; `git` installed on the TPU VM
  **Environment (verified)**: Local Ubuntu kernel `6.14.0-37-generic`; git `2.48.1`
  **Steps**:
  - Push your local changes to GitHub (source of truth):
    - See `docs/sops/github-push.md`
  - Record the exact commit you want to run on TPU:
    - `cd /home/john/github/MLLM-JAX`
    - `git rev-parse HEAD`
  - (Optional) Verify the GitHub HTTPS remote is reachable (useful for TPU VMs without SSH keys):
    - `git ls-remote https://github.com/demon2036/MLLM-JAX.git HEAD`
  - SSH to the TPU VM and clone/pull the repo via Git (avoid `scp` for code sync):
    - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>`
    - `COMMIT_OR_BRANCH=<main|<commit-sha>>`
    - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command 'set -euo pipefail; REPO_URL=https://github.com/demon2036/MLLM-JAX.git; REPO_DIR=/root/MLLM-JAX; if [ ! -d "$REPO_DIR/.git" ]; then git clone \"$REPO_URL\" \"$REPO_DIR\"; fi; cd \"$REPO_DIR\"; git fetch --all --prune; git checkout \"$COMMIT_OR_BRANCH\"; git status -sb'`
  - Run your workload from the checked-out tree:
    - Use your TPU VM environment bootstrap SOPs as needed (for example `docs/sops/tpu-vm-bootstrap.md`)
    - Example placeholder: `python <entrypoint>.py`
  **Expected Result**: TPU VM runs code from a Git checkout that matches your pushed commit/branch, without any manual `scp` file copying.
  **Troubleshooting**:
  - If `git clone` fails due to network/DNS, run `docs/sops/network-checks.md` and confirm the TPU VM has egress.
  - If the repo becomes private, prefer HTTPS + token or configure SSH keys on the TPU VM.
  - If `scripts/ssh_tpu_vm_root.sh` fails, see `docs/sops/tpu-vm-lifecycle.md` for direct `gcloud ... tpu-vm ssh` recipes.
  **References**:
  - `docs/sops/github-push.md`
  - `docs/sops/tpu-vm-lifecycle.md`
  - `docs/sops/tpu-vm-bootstrap.md`
  - `scripts/ssh_tpu_vm_root.sh`

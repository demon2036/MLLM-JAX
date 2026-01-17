# TPU VM Create (v4-8 / v6e-8) SOPs

- **Title**: SOP: Create a TPU VM v4-8 (spot) or v6e-8 (spot) in a known-good zone
  **Prereqs**: `gcloud` installed and authenticated; TPU API enabled; quota in the target zone
  **Steps**:
  - Confirm project/account:
    - `gcloud --version`
    - `gcloud auth list --format='table(account,status)'`
    - `gcloud config get-value project`
  - Pick a zone that supports both `v4-8` and `v6e-8` (example: `us-central2-b`):
    - `ZONE=us-central2-b; gcloud compute tpus accelerator-types list --zone="$ZONE" --format='value(name)' | rg -n 'acceleratorTypes/(v4-8|v6e-8)$'`
  - Try creating a `v6e-8` spot TPU (may fail if the project's v6e spot quota is 0):
    - `scripts/create_tpu_vm.sh --type v6e-8 --zone us-central2-b`
  - If `v6e-8` spot creation fails with a quota error, create a `v4-8` spot TPU instead:
    - `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b`
  - Record the printed `TPU name` and verify it's `READY`:
    - `TPU_NAME=<TPU_NAME>; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='value(state,acceleratorType)'`
  - Verify SSH works (root):
    - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true'`
  - Delete the TPU VM when finished (to stop billing):
    - `scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE"`
  **Expected Result**: A TPU VM reaches `READY`, SSH works, and cleanup deletes the VM successfully
  **Troubleshooting**:
  - If `v6e-8` spot errors with `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI` quota `Limit: 0`, either request v6e spot quota or use `v4-8` spot.
  - If the TPU stays in `CREATING`, check `describe` output and retry in another zone.
  **References**:
  - `docs/sops/tpu-vm-lifecycle.md`
  - `docs/sops/tpu-vm-delete-all.md`


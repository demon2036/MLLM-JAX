# TPU VM Lifecycle SOPs

## Discovery and access

- **Title**: SOP: Locate TPU VMs in project (all zones)
  **Prereqs**: gcloud configured; TPU API enabled
  **Steps**:
  - `gcloud compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'gcloud compute tpus tpu-vm list --zone="{}" --format="value(name,acceleratorType,state)" 2>/dev/null'`
  **Expected Result**: Prints TPU VM names with accelerator type and state across zones
  **Troubleshooting**: If a zone errors, confirm permissions and that the TPU API is enabled
  **References**: N/A

- **Title**: SOP: SSH as root via gcloud (TPU VM)
  **Prereqs**: gcloud configured; TPU VM exists
  **Steps**:
  - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami' --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command 'sudo -n whoami' --quiet`
  **Expected Result**: Commands print `root`
  **Troubleshooting**: If `root@...` fails, use the sudo command and/or check `sudo -n true` on the VM
  **References**: N/A

## Provisioning

- **Title**: SOP: Create spot TPU VM (generic)
  **Prereqs**: gcloud configured; spot quota/capacity in the target zone
  **Steps**:
  - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>; ACCELERATOR_TYPE=<v4-8|v5litepod-4|...>; RUNTIME_VERSION=<tpu-ubuntu2204-base|...>`
  - `gcloud alpha compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" --accelerator-type="$ACCELERATOR_TYPE" --version="$RUNTIME_VERSION" --spot --quiet`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true' --quiet`
  **Expected Result**: TPU VM reaches `READY` and SSH works
  **Troubleshooting**: If capacity is unavailable, try another zone or a different accelerator type
  **References**: N/A

## Cleanup

- **Title**: SOP: Delete PREEMPTED TPU VMs in a zone
  **Prereqs**: gcloud configured
  **Steps**:
  - `ZONE=<ZONE>`
  - `gcloud alpha compute tpus tpu-vm list --zone="$ZONE" --filter='state=PREEMPTED' --format='value(name)' | xargs -r -n1 -I{} gcloud alpha compute tpus tpu-vm delete "{}" --zone="$ZONE" --quiet`
  **Expected Result**: PREEMPTED nodes are removed from the zone listing
  **Troubleshooting**: If a delete fails, re-run for the remaining node name(s)
  **References**: N/A

- **Title**: SOP: Delete ALL TPU VMs in project (all zones)
  **Prereqs**: gcloud configured
  **Steps**:
  - See `docs/sops/tpu-vm-delete-all.md`
  **Expected Result**: N/A
  **Troubleshooting**: N/A
  **References**: N/A

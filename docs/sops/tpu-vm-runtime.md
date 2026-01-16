# TPU VM Runtime SOPs

## Runtime inspection

- **Title**: SOP: Inspect TPU VM runtime image and tpu-runtime container
  **Prereqs**: TPU VM reachable via `gcloud ... tpu-vm ssh`
  **Steps**:
  - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>`
  - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format=json`
  - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; echo "==os-release=="; cat /etc/os-release; echo "==uname=="; uname -r; echo "==tpu-runtime status=="; systemctl status tpu-runtime --no-pager | sed -n "1,18p"; echo "==tpu-runtime image=="; systemctl status tpu-runtime --no-pager | sed -n "18,26p"; echo "==docker images=="; docker image ls --digests | head -n 8; echo "==/etc tpu=="; ls /etc | grep -i tpu || true'`
  **Expected Result**: Runtime version and `tpu-runtime` status are visible
  **Troubleshooting**: N/A
  **References**: N/A

## Runtime version mapping

- **Title**: SOP: List TPU VM runtime versions (zone)
  **Prereqs**: gcloud configured
  **Steps**:
  - `ZONE=<ZONE>; gcloud compute tpus tpu-vm versions list --zone="$ZONE" --format='table(name,version,releaseDate)'`
  **Expected Result**: Output lists available runtime versions
  **Troubleshooting**: If a zone returns fewer runtime versions, confirm TPU API is enabled and project is set correctly
  **References**: N/A

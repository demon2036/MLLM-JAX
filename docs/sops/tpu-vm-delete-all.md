# TPU VM Delete-All SOPs

- **Title**: SOP: Delete all TPU VMs across locations
  **Prereqs**: gcloud configured; `gcloud alpha` commands installed; project `civil-rarity-482610-s5`
  **Steps**:
  - Enumerate and delete all TPU VMs (if any):
    - `set -euo pipefail; tmp_file=/tmp/tpu_vms_to_delete.txt; gcloud alpha compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'z="{}"; gcloud alpha compute tpus tpu-vm list --zone="$z" --format="value(name)" 2>/dev/null | awk -v z="$z" "{print z, $0}"' > "$tmp_file"; if [ -s "$tmp_file" ]; then while read -r zone name; do echo "Deleting $name in $zone"; gcloud alpha compute tpus tpu-vm delete "$name" --zone="$zone" --quiet; done < "$tmp_file"; else echo "No TPU VMs found."; fi`
  - Verify all TPU VMs are gone:
    - `gcloud alpha compute tpus locations list --format='value(locationId)' | xargs -P6 -I{} bash -lc 'z="{}"; gcloud alpha compute tpus tpu-vm list --zone="$z" --format="value(name,state,acceleratorType)" 2>/dev/null | awk -v z="$z" "{print z, $0}"'`
  **Expected Result**: The deletion loop prints either delete messages or `No TPU VMs found.`, and the verification command prints nothing.
  **Troubleshooting**: If a delete fails, re-run the deletion loop for the remaining `(zone, name)` pair(s).
  **References**: N/A

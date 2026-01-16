# SOP: Check Cloud TPU is alive (gcloud, Windows PowerShell)

- **Title**: SOP: Check whether any Cloud TPU VMs / queued resources exist in the current gcloud project
- **Prereqs**: Windows PowerShell 5.1+; `gcloud` installed and authenticated

## Steps

- Confirm you are looking at the right project:
  - `gcloud --version`
  - `gcloud config get-value project`

- (Optional) If you are *on* a TPU VM and have JAX installed, verify runtime sees TPU devices:
  - `python -c "import jax; print(jax.default_backend()); print(jax.devices())"`

- List TPU VMs across all TPU-capable zones in the current project (prints nothing if none exist):
  - PowerShell:
    - `$env:CLOUDSDK_CORE_DISABLE_PROMPTS="1"`
    - `$zones = gcloud compute tpus locations list --format="value(locationId)"`
    - `foreach ($z in $zones) { $vms = gcloud compute tpus tpu-vm list --zone=$z --format="value(name,acceleratorType,state)" --verbosity=error 2>$null; if ($vms) { $vms | ForEach-Object { "$z`t$_" } } }`

- List TPU queued resources across all TPU-capable zones (prints nothing if none exist):
  - PowerShell:
    - `$env:CLOUDSDK_CORE_DISABLE_PROMPTS="1"`
    - `$zones = gcloud compute tpus locations list --format="value(locationId)"`
    - `foreach ($z in $zones) { $qr = gcloud compute tpus queued-resources list --zone=$z --format="value(name,acceleratorType,state)" --verbosity=error 2>$null; if ($qr) { $qr | ForEach-Object { "$z`t$_" } } }`

## Expected Result

- If TPU is “alive” in this project: at least one line is printed for a `tpu-vm` (usually `READY`) and/or a `queued-resources` entry (often `CREATING`/`ACCEPTED`/`PROVISIONING` depending on the API version).
- If nothing prints: there are no TPU VMs or queued TPU resources in the current gcloud project.

## Troubleshooting

- If you expect a TPU VM but nothing prints:
  - Double-check project: `gcloud config get-value project`
  - Double-check account: `gcloud auth list`
  - If you know the zone, list directly: `gcloud compute tpus tpu-vm list --zone=<zone>`
- If the optional JAX probe fails with `ModuleNotFoundError: No module named 'jax'`, run the check from the TPU VM environment (or install JAX in your local environment).

## References

- `docs/sops/tpu-vm-lifecycle.md`
- `docs/sops/tpu-tests.md`


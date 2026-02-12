# TPU VM v6e-32 spot create fails (code=13), fallback to v6e-16 (spot)

- **Title**: SOP: When `v6e-32` spot TPU VM create fails with `code=13` internal error, fall back to `v6e-16` (us-east1-d)
  **Prereqs**: `gcloud` authenticated; TPU API enabled; quota available; default VPC/network OK
  **Context**: project `civil-rarity-482610-s5`, date `2026-02-12`

## What we observed

### 1) Spot v6e quota is zone-scoped and non-zero only in two zones

This project had non-zero spot v6e quota only in:
- `us-east1-d`: `tpu-v6e-preemptible` effectiveLimit `64`
- `europe-west4-a`: `tpu-v6e-preemptible` effectiveLimit `64`

Command used:

```bash
gcloud alpha services quota list \
  --consumer=projects/civil-rarity-482610-s5 \
  --service=tpu.googleapis.com \
  --filter='metric:tpu-v6e-preemptible' \
  --format=json
```

### 2) `v6e-32` spot create failed with GCP internal error in both quota-enabled zones

Both of the following ended with:
- `ERROR: ... { "code": 13, "message": "an internal error has occurred" }`

Commands used:

```bash
scripts/create_tpu_vm.sh --type v6e-32 --zone us-east1-d --project civil-rarity-482610-s5 --spot
scripts/create_tpu_vm.sh --type v6e-32 --zone europe-west4-a --project civil-rarity-482610-s5 --spot
```

## Fallback: create `v6e-16` spot (works)

In `us-east1-d`, `v6e-16` spot created successfully and reached `READY/HEALTHY`.

Command used:

```bash
scripts/create_tpu_vm.sh --type v6e-16 --zone us-east1-d --project civil-rarity-482610-s5 --spot
```

Verify:

```bash
gcloud alpha compute tpus tpu-vm list --project civil-rarity-482610-s5 --zone us-east1-d \
  --format='table(name,acceleratorType,state,health)'
```

## Notes / Troubleshooting

- `code=13 internal error` is a backend error. Retrying later, changing zone, or using queued-resources may help, but this run reproduced the error in both quota-enabled zones.
- `v6e-64` with external IPs is often blocked by regional `IN_USE_ADDRESSES` quota (default limit `8`); multi-host TPU slices can require multiple external IPs.
- Delete unused TPU VMs to avoid burning quotas/cost:

```bash
scripts/delete_tpu_vm.sh --name <TPU_NAME> --zone <ZONE> --project civil-rarity-482610-s5
```


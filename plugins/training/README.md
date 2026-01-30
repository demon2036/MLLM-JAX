# `plugins.training` layout (canonical)

- Training infra: `plugins.training.core.*`
- RL implementation: `plugins.training.rl.*`
- SFT implementation: `plugins.training.sft.*`
- Contracts (APIs/schemas): `plugins.api.*`
- Sampling/generation: `plugins.sample.*` and `plugins.api.sample.*`

As of 2026-01-30, the legacy compatibility shims under `plugins/common` and
`plugins/training/*` were removed. Update any older imports/SOPs to use the
canonical paths above.

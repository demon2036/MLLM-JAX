# 20260123_default-config-and-launcher

Goal: reduce ambiguity by keeping a single default GRPO/GSM8K YAML config and a single launcher entrypoint (`*.sh --config <path>.yaml`), then re-validate on TPU v6e-8 with W&B online.

## Plan

1. Keep only `plugins/training/configs/grpo_gsm8k_qwen25_3b_bs128_steps100.yaml` as default config.
2. Delete confusing per-config launch scripts; keep one `--config`-based launcher.
3. Make W&B mode config-driven (allow disabled; validate online before reporting).
4. Update AGENTS.md policy (no env-var hyperparam overrides; memory summaries can batch 5–10 steps).
5. Run local tests + CLI print-config checks.
6. Push, then run the default config on v6e-8 (online) and confirm no OOM.

## Progress log (batched)

- 2026-01-23: Created this memory folder and registered it in `memory/README.md`.


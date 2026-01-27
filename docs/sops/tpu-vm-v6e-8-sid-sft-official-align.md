# SOP: TPU v6e-8 SID SFT official-alignment run (W&B online)

- **Title**: SOP: TPU v6e-8 SID SFT official-alignment run (W&B online)
- **Prereqs**: gcloud auth; TPU quota; W&B API key in `.env`; repo pushed to GitHub
- **Environment (verified)**: `<to be filled>`

## Goal

- Create a fresh v6e-8 TPU VM and run official-aligned SID SFT evals.
- Log runs to W&B online and cross-check with upstream `calc.py`.

## Steps (commands actually run)

- `<command>` (exit `<code>`)

## Expected Result

- TPU run exits `0`, writes `eval_predictions.json` + metrics, and logs W&B URLs.
- `calc.py` metrics match `eval_predictions.metrics.json` for each dataset.

## Troubleshooting

- If W&B auth fails, strip CRLF in `/root/.env` (`sed -i "s/\\r$//"`).

## References

- `docs/sops/minionerec-sid-sft-jax-tpu.md`
- `memory/20260127_tpu-v6e8-sft-official-align/README.md`

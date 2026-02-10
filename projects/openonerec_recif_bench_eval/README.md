# OpenOneRec RecIF-Bench Eval (TPU)

This project reproduces the **RecIF-Bench** evaluation pipeline described in the
OpenOneRec technical report, targeting the public Hugging Face checkpoints:

- `OpenOneRec/OneRec-1.7B`
- `OpenOneRec/OneRec-1.7B-pro`

Goals:

1. Follow the upstream OpenOneRec `benchmarks/` task definitions and evaluators.
2. Run the full 8-task RecIF-Bench evaluation on a TPU VM.
3. Log metrics to Weights & Biases with `wandb_mode=online`.

## Entrypoint

Run from a YAML config:

```bash
python -u projects/openonerec_recif_bench_eval/run.py --config projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b.yaml
```

On TPU VMs, prefer the `nohup` wrapper:

```bash
bash scripts/tpu_vm_start_openonerec_recif_bench_eval_from_config_nohup.sh --config projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b.yaml
```

## Notes

- This project treats `workdirs/OpenOneRec/benchmarks` as the “code of truth” for
  task configs + evaluators, and only swaps the generation backend to run on TPU.
- Secrets (HF/W&B keys) are loaded from `.env` or `/root/.env` via
  `plugins.training.core.runtime.env.load_dotenv_if_present`.


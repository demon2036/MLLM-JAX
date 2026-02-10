# SOP: TPU VM run OpenOneRec RecIF-Bench eval (OneRec-1.7B / OneRec-1.7B-pro)

- **Title**: SOP: Run OpenOneRec RecIF-Bench eval on a TPU VM with W&B online
- **Prereqs**:
  - `gcloud` installed and authenticated; TPU API enabled; quota in the target zone
  - A Hugging Face token available as `HF_TOKEN` (for model + dataset download)
  - A W&B key available as `WANDB_API_KEY` (required: `wandb_mode=online`)
  - Vertex AI / Gemini access (for `item_understand` + `rec_reason` LLM judge)

## Environment (fill after the first successful TPU run)

- TPU:
  - name:
  - type:
  - zone:
  - runtime:
- Python:
- JAX:
- torch/torch_xla:

## Steps (fill with commands actually used after the first successful TPU run)

TBD.

## Expected Result

- Two full runs finish with exit code `0` on TPU:
  - `projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b.yaml`
  - `projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b_pro.yaml`
- Each run writes:
  - per-task `*_generated.json`
  - `eval_results.json`
- W&B runs are visible (mode `online`) and include task metrics (e.g. `video/recall@32`, `label_pred/auc`).

## References

- `projects/openonerec_recif_bench_eval/README.md`
- `projects/openonerec_recif_bench_eval/run.py`
- Upstream benchmark entrypoint: `workdirs/OpenOneRec/benchmarks/eval_script.sh`
- TPU lifecycle helpers:
  - `docs/sops/tpu-vm-create-v4-8-or-v6e-8.md`
  - `docs/sops/tpu-vm-repo-sync.md`
  - `docs/sops/tpu-vm-bootstrap.md`


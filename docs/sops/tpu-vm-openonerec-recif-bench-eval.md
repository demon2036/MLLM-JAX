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


## 2026-02-11 实跑更新（OpenOneRec 对齐）

### 已验证命令（实际执行）

1) 在 TPU VM 上同步分支并安装评测依赖：

```bash
cd /root/MLLM-JAX-openonerec
git fetch --all --prune
git checkout openonerec
git pull --ff-only
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
python -m pip install --upgrade "openai>=1.66.3,<2" "anthropic>=0.49.0,<1"
```

2) 启动 long-task 分批评测（CPU backend, YAML 可追踪）：

```bash
bash scripts/tpu_vm_start_openonerec_recif_bench_eval_from_config_nohup.sh \
  --config projects/openonerec_recif_bench_eval/configs/recif_bench_onerec_1p7b_split_long_cpu_bs5.yaml
```

3) 运行态检查：

```bash
cat logs/nohup_openonerec_recif_recif_bench_onerec_1p7b_split_long_cpu_bs5_latest.exit
cat logs/nohup_openonerec_recif_recif_bench_onerec_1p7b_split_long_cpu_bs5_latest.pid
tail -n 120 logs/nohup_openonerec_recif_recif_bench_onerec_1p7b_split_long_cpu_bs5_latest.log
```

4) TPU 资源状态检查：

```bash
gcloud alpha compute tpus tpu-vm describe openonerec-recif-eval-v6e-8-2602110631 \
  --zone=us-east5-b --format='value(state,health,acceleratorType,runtimeVersion)'
```

### 关键结果

- `openai` 与 `anthropic` 依赖已补齐，`item_understand/rec_reason` 的 evaluator import 阻塞已解除。
- `hf_generator.py` 已提交修复（`b091c8c`）：TPU bucketed decode 支持 legacy tuple KV cache。
- 运行中 TPU 被维护抢占：节点状态从 `READY/UNHEALTHY_MAINTENANCE` 转为 `PREEMPTED`，导致执行中断。

### 当前阻塞

- 新建 TPU 多次失败（`v6e-8`、`v6e-4`、`v4-8`）：
  - `Insufficient capacity`
  - `Reservation not found`
  - `internal error`

### 恢复建议（可直接执行）

1) 先抢占可用 TPU（优先 `v6e-8`，次选 `v6e-4`）：

```bash
gcloud alpha compute tpus tpu-vm create <TPU_NAME> \
  --zone=<ZONE> \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base \
  --project=civil-rarity-482610-s5
```

2) 成功后立即同步代码并恢复分批跑：

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone <ZONE> --command '
  set -euo pipefail
  cd /root
  [ -d /root/MLLM-JAX-openonerec/.git ] || git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX-openonerec
  cd /root/MLLM-JAX-openonerec
  git fetch --all --prune
  git checkout openonerec
  git pull --ff-only
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate mllm-jax
  python -m pip install --upgrade "openai>=1.66.3,<2" "anthropic>=0.49.0,<1"
'
```

3) 恢复运行顺序（W&B online）：

- `recif_bench_onerec_1p7b_split_long_cpu_bs5.yaml`
- `recif_bench_onerec_1p7b_split_short.yaml`
- `recif_bench_onerec_1p7b_finalize.yaml`
- `recif_bench_onerec_1p7b_pro_split_long_cpu_bs5.yaml`
- `recif_bench_onerec_1p7b_pro_split_short.yaml`
- `recif_bench_onerec_1p7b_pro_finalize.yaml`

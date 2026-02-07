# SOP: OpenOneRec JAX full-align run on TPU v6e-8 (eval + train + post-train eval)

- **Title**: SOP: OpenOneRec JAX full-align run on TPU v6e-8 (eval + train + post-train eval)
- **Prereqs**:
  - Local code pushed to non-main branch (GitHub is TPU source-of-truth)
  - GCP project/quota for `v6e-8`
  - `HF_TOKEN` with accepted access to `OpenOneRec/OpenOneRec-RecIF`
  - `WANDB_API_KEY` for `wandb_mode=online`
- **Environment (verified)**:
  - Local host: Ubuntu, `gcloud` + SSH scripts in `scripts/`
  - TPU runtime: `v6e-ubuntu-2404`, 8 TPU devices
  - Conda env `mllm-jax`: `jax 0.9.0.1`, `torch 2.10.0+cpu`, `transformers 4.57.1`

## Steps

1) **Push local branch for TPU sync**

```bash
git checkout john/nano-fork-d
git push origin john/nano-fork-d
```

2) **Create dedicated TPU (do not reuse unrelated TPU)**

```bash
scripts/create_tpu_vm.sh --type v6e-8 --zone us-east1-d --name openonerec-jax-v6e8-<timestamp> --spot
```

3) **Bootstrap TPU python env**

```bash
scripts/bootstrap_miniconda_on_tpu_vm.sh --name <TPU_NAME> --zone us-east1-d --env-name mllm-jax --python 3.12
```

4) **Git-sync repo on TPU (no scp for code)**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --command '
set -euo pipefail
if [ ! -d /root/MLLM-JAX/.git ]; then
  git clone https://github.com/demon2036/MLLM-JAX.git /root/MLLM-JAX
fi
cd /root/MLLM-JAX
git fetch --all --prune
git checkout -B john/nano-fork-d origin/john/nano-fork-d
git reset --hard <commit_sha>
git clean -fd
git status -sb
'
```

5) **Install runtime deps on TPU**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --command '
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -U -r requirements-tpu.txt pandas pyarrow scikit-learn sentencepiece pydantic pydantic-settings transformers==4.57.1 pyfiglet pytest
'
```

6) **Clone OpenOneRec on TPU**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --command '
set -euo pipefail
mkdir -p /root/MLLM-JAX/workdir
cd /root/MLLM-JAX/workdir
if [ ! -d OpenOneRec/.git ]; then
  git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
fi
'
```

7) **Prepare secrets (`/root/.env`)**

```bash
# local .env must contain WANDB_API_KEY + HF_TOKEN
scripts/sync_env_to_tpu_vm.sh --name <TPU_NAME> --zone us-east1-d --worker 0
```

8) **Download benchmark data (requires HF_TOKEN)**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --env-file /root/.env --command '
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
python - <<"PY"
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import shutil
repo_id = "OpenOneRec/OpenOneRec-RecIF"
repo_type = "dataset"
out = Path("/root/MLLM-JAX/workdir/OpenOneRec/benchmarks/data")
out.mkdir(parents=True, exist_ok=True)
files = [f for f in HfApi().list_repo_files(repo_id, repo_type=repo_type) if f.startswith("benchmark_data/")]
for f in files:
    src = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=f)
    dst = out / f[len("benchmark_data/"):]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
print("downloaded", len(files), "files")
PY
'
```

9) **Run 1.7B official-aligned full eval (W&B online)**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --env-file /root/.env --command '
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
python scripts/run_openonerec_eval.py \
  --config projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_full_online.yaml \
  --run-mode eval
'
```

10) **Run full aligned training (W&B online)**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --env-file /root/.env --command '
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
python scripts/run_openonerec_train.py \
  --config projects/openonerec_train/configs/openonerec_train_jax_v6e8_full_online.yaml \
  --run-mode train
'
```

11) **Run post-train eval against trained checkpoint**

```bash
scripts/ssh_tpu_vm_root.sh --name <TPU_NAME> --zone us-east1-d --env-file /root/.env --command '
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mllm-jax
cd /root/MLLM-JAX
python scripts/run_openonerec_eval.py \
  --config projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_posttrain_full_online.yaml \
  --run-mode eval
'
```

## Expected Result

- `eval_results.json` + `paper_alignment.json` produced for pre-train and post-train eval runs
- `sft_state_last.msgpack` produced by train run
- W&B runs are online and linkable
- No traceback, all commands exit `0`

## Troubleshooting

- `GatedRepoError 401` on dataset download:
  - Ensure `HF_TOKEN` exists in `/root/.env`
  - Ensure the token account has accepted access to `OpenOneRec/OpenOneRec-RecIF`
- `wandb disabled due to init error: No API key configured`:
  - Ensure `WANDB_API_KEY` exists in `/root/.env`
  - `scripts/sync_env_to_tpu_vm.sh` after any local `.env` change
- `ModuleNotFoundError: pyfiglet` during benchmark import:
  - `pip install -U pyfiglet`

## References

- `docs/sops/tpu-vm-repo-sync.md`
- `docs/sops/tpu-vm-bootstrap.md`
- `scripts/ssh_tpu_vm_root.sh`
- `scripts/sync_env_to_tpu_vm.sh`
- `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_full_online.yaml`
- `projects/openonerec_train/configs/openonerec_train_jax_v6e8_full_online.yaml`
- `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_posttrain_full_online.yaml`

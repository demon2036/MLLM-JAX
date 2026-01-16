# TPU VM Bootstrap SOPs

- **Title**: SOP: Bootstrap Miniconda on TPU VM (generic)
  **Prereqs**: gcloud configured; TPU VM reachable; Ubuntu on TPU VM
  **Steps**:
  - `TPU_NAME=<TPU_NAME>; ZONE=<ZONE>`
  - Install Miniconda (if missing) and verify `conda`:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; if [ ! -d /root/miniconda3 ]; then curl -fsSL -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; bash /root/miniconda.sh -b -p /root/miniconda3; rm -f /root/miniconda.sh; fi; /root/miniconda3/bin/conda --version'`
  - Create a Python env (choose a name) and upgrade pip:
    - `gcloud alpha compute tpus tpu-vm ssh root@"$TPU_NAME" --zone="$ZONE" --quiet --command 'set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true; ENV_NAME=mllm-jax; if ! conda env list | awk \"{print \\$1}\" | grep -qx \"$ENV_NAME\"; then conda create -y -n \"$ENV_NAME\" python=3.12; fi; conda activate \"$ENV_NAME\"; python --version; pip install -U pip'`
  **Expected Result**: Miniconda installed and the conda env exists
  **Troubleshooting**: If conda prompts for Terms of Service, re-run the `conda tos accept` commands
  **References**: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


git clone --quiet --branch=main --depth=1 \
   https://github.com/google-research/big_vision big_vision_repo

# 1. Install miniconda3 (with removing legacy).
rm -rf ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
#bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -u
#rm Miniconda3-py310_24.1.2-0-Linux-x86_64.sh

wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b -u
rm Miniconda3-py311_24.7.1-0-Linux-x86_64.sh



~/miniconda3/bin/conda init bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"


# 2. Install requirements.
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U flax optax chex
#
pip install -U webdataset timm wandb
pip install einops gcsfs tensorflow tpu-info


pip3 install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece" transformers datasets kaggle kagglehub
pip install httpx

pip install --upgrade transformers  aqtp


pip install cloud_tpu_client fastapi uvicorn math_verify huggingface_hub[hf_transfer]



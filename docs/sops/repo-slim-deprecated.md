# SOP: Slim repo by moving unused code to `deprecated/`

- **Title**: SOP: Move non-Qwen2.5 / multimodal / unused code into `deprecated/` while keeping the TPU GRPO smoke-run working
  **Prereqs**: Working tree clean enough to move files; `git` installed; you already have a verified smoke-run baseline
  **Environment (verified)**:
  - Local: Ubuntu kernel `6.14.0-37-generic`; Python `3.13.9`; git `2.48.1`
  - Verified TPU smoke-run commit after slimming: `f886cf8`
  **Steps**:
  - Create holding area (example):
    - `cd /home/john/github/MLLM-JAX`
    - `mkdir -p deprecated/MLLM_JAX/language deprecated/MLLM_JAX/language/llama`
  - Move non-Qwen2.5 language model code into `deprecated/` (example set):
    - `git mv MLLM_JAX/language/gemma deprecated/MLLM_JAX/language/`
    - `git mv MLLM_JAX/language/gemma3 deprecated/MLLM_JAX/language/`
    - `git mv MLLM_JAX/language/qwen3 deprecated/MLLM_JAX/language/`
    - `git mv MLLM_JAX/language/qwen3_moe deprecated/MLLM_JAX/language/`
  - Keep only the LLaMA primitives needed by Qwen2.5, and move unused LLaMA helper files (example set):
    - `git mv MLLM_JAX/language/llama/configuration_llama.py deprecated/MLLM_JAX/language/llama/`
    - `git mv MLLM_JAX/language/llama/modeling_llama.py deprecated/MLLM_JAX/language/llama/`
    - `git mv MLLM_JAX/language/llama/ref.py deprecated/MLLM_JAX/language/llama/`
    - `git mv MLLM_JAX/language/llama/test.py deprecated/MLLM_JAX/language/llama/`
  - Remove accidental dependencies from the active Qwen2 path (this run removed unused imports from):
    - `MLLM_JAX/language/qwen2/configuration_qwen2.py`
  - Move multimodal/vision code (example set):
    - `git mv MLLM_JAX/vision deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/mutinomial deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/multinomial_sample.py deprecated/MLLM_JAX/`
  - Move unused kernels (example set):
    - `git mv MLLM_JAX/kernels deprecated/MLLM_JAX/`
  - Move unused misc modules / alternate sampler states (example set):
    - `git mv MLLM_JAX/activations.py deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/efficient.py deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/efficient2.py deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/efficient3.py deprecated/MLLM_JAX/`
    - `git mv MLLM_JAX/mask.py deprecated/MLLM_JAX/`
    - `mkdir -p deprecated/MLLM_JAX/sample`
    - `git mv MLLM_JAX/sample/sample_state_left_padding.py deprecated/MLLM_JAX/sample/`
    - `git mv MLLM_JAX/sample/sample_state_right_padding.py deprecated/MLLM_JAX/sample/`
    - `git mv MLLM_JAX/sample/sample_state_right_padding3.py deprecated/MLLM_JAX/sample/`
  - Sanity check (syntax only):
    - `python -m py_compile training2.py scripts/run_grpo_gsm8k_training.py test_jit8.py`
    - `find MLLM_JAX -type f -name '*.py' -print0 | xargs -0 python -m py_compile`
  - Keep only `test_jit8.py` outside `deprecated/` (this run moved the other jit scripts + old runner):
    - `mkdir -p deprecated/tests && git mv test_jit9.py test_jit10.py test_jit11.py deprecated/tests/`
    - `mkdir -p deprecated/scripts && git mv grpo_test.sh deprecated/scripts/`
  - Commit and push (so TPU can sync via Git):
    - `git add -A`
    - `git commit -m "refactor: move extra jit tests to deprecated"`
    - `git push origin john`
  **Expected Result**:
  - Active code needed for the TPU smoke-run stays under `MLLM_JAX/` + `scripts/` + `training2.py`
  - Only `test_jit8.py` remains at repo root; other jit experiments are under `deprecated/`
  - Deprecated code is kept under `deprecated/` for reference
  - TPU smoke-run still completes 3 steps on the new commit (this run: `f886cf8`)
  **Troubleshooting**:
  - `fatal: destination already exists ...` during `git mv <dir> deprecated/MLLM_JAX/`:
    - Do not pre-create `deprecated/MLLM_JAX/<dir>` when moving a whole directory; remove the empty destination dir (e.g. `rmdir deprecated/MLLM_JAX/vision`) and retry.
  **References**:
  - `deprecated/README.md`
  - `docs/sops/tpu-vm-v6e-8-grpo-gsm8k-bs128-steps100.md`
  - Commit `f886cf8`

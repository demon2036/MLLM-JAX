# SOP: Slim repo by moving unused code to `deprecated/`

- **Title**: SOP: Move non-Qwen2.5 / multimodal / unused code into `deprecated/` while keeping the TPU GRPO smoke-run working
  **Prereqs**: Working tree clean enough to move files; `git` installed; you already have a verified smoke-run baseline
  **Environment (verified)**:
  - Local: Ubuntu kernel `6.14.0-37-generic`; Python `3.13.9`; git `2.48.1`
  - Verified TPU smoke-run commit after slimming: `b6d9a5b`
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
    - `python -m py_compile training2.py scripts/run_smoke_grpo_gsm8k_qwen25_7b.py`
    - `find MLLM_JAX -type f -name '*.py' -print0 | xargs -0 python -m py_compile`
  - Commit and push (so TPU can sync via Git):
    - `git add -A MLLM_JAX deprecated`
    - `git commit -m "refactor: move unused code to deprecated"`
    - `git push origin john`
  **Expected Result**:
  - Active code needed for the TPU smoke-run stays under `MLLM_JAX/` + `scripts/` + `training2.py`
  - Deprecated code is kept under `deprecated/` for reference
  - TPU smoke-run still completes 3 steps on the new commit (this run: `b6d9a5b`)
  **Troubleshooting**:
  - `fatal: destination already exists ...` during `git mv <dir> deprecated/MLLM_JAX/`:
    - Do not pre-create `deprecated/MLLM_JAX/<dir>` when moving a whole directory; remove the empty destination dir (e.g. `rmdir deprecated/MLLM_JAX/vision`) and retry.
  **References**:
  - `deprecated/README.md`
  - `docs/sops/tpu-grpo-gsm8k-qwen25-7b-3steps.md`
  - Commit `b6d9a5b`


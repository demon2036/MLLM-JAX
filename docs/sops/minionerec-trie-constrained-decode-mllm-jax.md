# SOP: Clone MiniOneRec and extract trie constrained decode details for mllm-jax

## Prereqs
- OS: Windows (PowerShell)
- Python: <not checked>
- JAX: <not checked>
- Hardware: CPU (no TPU/GPU validation in this SOP)

## Steps
1) Create the clone destination.
```powershell
New-Item -ItemType Directory -Force plugins2
```

2) Clone MiniOneRec into `plugins2/`.
```powershell
git clone https://github.com/AkaliKong/MiniOneRec.git plugins2\MiniOneRec
```

3) Locate constrained decoding hooks.
```powershell
rg -n "ConstrainedLogitsProcessor|prefix_allowed_tokens_fn" -S plugins2\MiniOneRec
```

4) Inspect MiniOneRec constrained decode implementation.
```powershell
Get-Content plugins2\MiniOneRec\LogitProcessor.py
Get-Content plugins2\MiniOneRec\evaluate.py | Select-Object -First 260
Get-Content plugins2\MiniOneRec\minionerec_trainer.py | Select-Object -Skip 520 -First 220
```

5) Inspect SID token formats used to build the trie.
```powershell
Get-Content plugins2\MiniOneRec\rq\generate_indices.py | Select-Object -First 220
Get-Content plugins2\MiniOneRec\rq\generate_indices_plus.py | Select-Object -First 120
```

6) Inspect mllm-jax sampling pipeline for integration points.
```powershell
rg -n "decode|decoding|sampling|beam|logits|prefix|constraint" -S MLLM_JAX
Get-Content MLLM_JAX\sample\sample_state_right_padding2.py | Select-Object -First 260
Get-Content MLLM_JAX\sample\sanple_utils.py
```

7) Inspect right-padding, prefill buckets, and cache padding details.
```powershell
rg -n "padding|pad_cache_right|prefill|bucket|right" -S MLLM_JAX\sample\sample_state_right_padding2.py
rg -n "pad_cache_right" -S MLLM_JAX
Get-Content MLLM_JAX\language\qwen2\configuration_qwen2.py | Select-Object -Skip 300 -First 120
Get-Content MLLM_JAX\utils.py | Select-Object -Skip 300 -First 90
```

## Expected Result
- `plugins2/MiniOneRec` exists with a clean clone.
- Constrained decode logic located in:
  - `LogitProcessor.py`
  - `evaluate.py`
  - `minionerec_trainer.py`
- SID token formats confirmed in `rq/generate_indices*.py`.
- mllm-jax sampling entry points identified in `sample_state_right_padding2.py`.
- right padding / prefill bucket / cache padding locations identified.

## Troubleshooting
- If `git clone` fails, confirm network access and Git availability.
- If `rg` is unavailable, use `Select-String` or `Get-Content` with `findstr`.
- If using the `sid3_constrained_beam_search` implementation with **right-padded prompts**, note it builds
  `attention_mask = ones` during prefill. This means padding tokens are treated as real tokens, so **uniform
  padding to a fixed length is not safe** without modification.
  - Recommended fix: accept/pass `attention_mask` derived from `prompt_true_len` (mask out padding), or build it
    internally as `mask = arange(prompt_len) < prompt_true_len[:, None]` before prefill.

## References
- MiniOneRec commit inspected: `ba385c6827177c7b6b849d8b68f6349890099de3`

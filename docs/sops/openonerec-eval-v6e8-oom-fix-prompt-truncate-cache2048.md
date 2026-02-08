# SOP: OpenOneRec eval v6e-8 OOM fix (prompt truncation + cache 2048)

- **Title**: SOP: OpenOneRec eval v6e-8 OOM fix (prompt truncation + cache 2048)
- **Prereqs**:
  - Repo checkout contains `projects/openonerec_eval/*`
  - Python environment can run `py_compile`
- **Environment (verified)**:
  - Local Linux shell (`bash`)
  - Command verified on 2026-02-08

## Steps

1) Add generation prompt truncation control in eval defaults/dataclasses/generator:
- `generation.max_prompt_tokens` default `1024`
- In JAX generator fast path, right-truncate tokenized prompt IDs to the last `max_prompt_tokens`.
- Validate `max_prompt_tokens > 0` during generator init.

2) Reduce eval JAX cache length defaults:
- Set code defaults `jax.max_cache_length` from `8192` to `2048`.
- Set YAML values `jax.max_cache_length: 2048` for:
  - `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_full_online.yaml`
  - `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_posttrain_full_online.yaml`
  - `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_smoke_online.yaml`

3) Keep beam behavior unchanged:
- Ensure `num_beams: 32`
- Ensure `num_return_sequences: 32`

4) Validate syntax:

```bash
python -m py_compile projects/openonerec_eval/*.py scripts/run_openonerec_eval.py
```

## Expected Result

- Fast path truncates long prompts to prevent oversized prefill memory.
- Eval cache default is `2048` in code and target v6e-8 configs.
- `num_beams/num_return_sequences` remain `32/32`.
- `py_compile` exits `0`.

## Troubleshooting

- If `ValueError: max_prompt_tokens must be > 0`:
  - Set `generation.max_prompt_tokens` to a positive integer.
- If `max_cache_length too small for bucket` errors:
  - Increase `jax.max_cache_length` above selected prefill bucket + 2.

## References

- `projects/openonerec_eval/jax_generator.py`
- `projects/openonerec_eval/config.py`
- `projects/openonerec_eval/runner.py`
- `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_full_online.yaml`
- `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_posttrain_full_online.yaml`
- `projects/openonerec_eval/configs/openonerec_eval_jax_v6e8_smoke_online.yaml`

# Tiled cross-entropy Pallas kernel (fix + validation)

## Goal

- Fix correctness of the TPU Pallas tiled cross-entropy/logprob kernel (forward + backward).
- Keep everything non-invasive under `plugins/`.
- Prepare for TPU gradcheck on `mllm-jax-v6e-8-spot-260124132428` with `wandb_mode=online`.

## Key finding (root cause)

- **BlockSpec `index_map` semantics were wrong**: for `pl.BlockSpec((..., BLOCK), index_map)`, the `index_map`
  should return **block indices** (e.g. `(b, t, k)`), not element start offsets (e.g. `(b, t * BLOCK_T, k * BLOCK)`).
- The previous code multiplied by block sizes twice, leading to **out-of-bounds reads** under strict TPU interpret
  mode and silent wrong numerics under the loose `interpret=True` path.

## Reference comparison (MaxText)

- Cloned: `workdir/maxtext` @ `b646a53` (`https://github.com/google/maxtext.git`)
- File: `workdir/maxtext/src/MaxText/vocabulary_tiling.py`
- MaxText’s “vocab tiling” here is **token-chunking** + `jax.lax.scan` + `custom_vjp` (recompute logits in bwd),
  not a Pallas fused kernel. It’s still a good baseline pattern for “no full-logits residency”.

## Evidence (commands actually run)

### Local unit tests

- `python -m pytest -q`
  - Result: `33 passed, 1 warning` (exit `0`)

## Files changed

- `plugins/training/kernels/tiled_cross_entropy_pallas.py`
  - Fix BlockSpecs to use `(b, t, k)` block indices.
  - Pass through `interpret` (allow `pltpu.InterpretParams`) instead of forcing `bool`.
- `plugins/training/kernels/grpo_loss_pallas.py`
  - Same BlockSpec fix for the GRPO kernel (it used the same pattern).
  - Pass through `interpret`.
- `tests/test_tiled_cross_entropy_pallas_kernel.py`
  - Use `pltpu.InterpretParams(out_of_bounds_reads='raise')` for strict checking.
- `tests/test_grpo_pallas_kernel.py`
  - Same strict interpret params; also parametrize vocab to cover `blocks > 2`.

## Next: TPU validation (to fill)

- TPU command(s): `<to run>`
- Gradcheck command: `python -u scripts/cross_entropy_kernel_gradcheck.py --config plugins/training/configs/cross_entropy_kernel_gradcheck_qwen25_1p5b.yaml`
- Expected: exit `0`, loss + dlogits diffs under configured tolerances, W&B online run recorded.


# SOP: Scan plugins/sft for Torch remnants

- **Title**: SOP: Scan `plugins/sft/` for Torch/PyTorch remnants (line numbers + file list)
- **Prereqs**: `rg` (ripgrep) installed
- **Environment (verified)**: Linux `6.14.0-37-generic`; Python `3.12.2`; ripgrep `15.1.0`; JAX `0.8.2` (not required for scanning)

## Steps (commands actually run)

  - `rg --files plugins/sft`
  - `rg -n "\\btorch\\b" plugins/sft`
  - `rg -n "torch|pytorch" plugins/sft`
  - `nl -ba plugins/sft/trainer.py | sed -n '1,200p'`
  - `nl -ba plugins/sft/evaluator.py | sed -n '1,240p'`
  - `nl -ba plugins/sft/constrained_decoding.py | sed -n '1,220p'`
  - `nl -ba plugins/sft/runner/sid_sft.py | sed -n '1,220p'`
  - `nl -ba plugins/sft/runner/sid_sft.py | sed -n '240,420p'`
  - `nl -ba plugins/sft/runner/sid_sft.py | sed -n '500,580p'`
  - `nl -ba plugins/sft/datasets/eval_sid_next_item.py | sed -n '1,160p'`
  - `nl -ba plugins/sft/datasets/sid_item_alignment.py | sed -n '1,200p'`
  - `nl -ba plugins/sft/datasets/sid_next_item.py | sed -n '1,200p'`
  - `nl -ba plugins/sft/datasets/fusion_seq_rec.py | sed -n '1,240p'`

## Expected Result

- Torch/PyTorch references and their line numbers are enumerated for all `plugins/sft/` files.

## Troubleshooting

  - If `rg` is missing, use `git grep -n "torch" plugins/sft` and `nl -ba` for line numbers.
  - If no matches are found, confirm `plugins/sft` exists and rerun `rg --files plugins/sft`.

## References

- None.

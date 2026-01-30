# SOP: TPU v6e-8 SID SFT (official-align) baseline vs EMA (W&B online)

- **Title**: SOP: Run MiniOneRec SID SFT official-align on v6e-8 (baseline vs EMA) with W&B online
- **Prereqs**: TPU VM `READY`; conda env `mllm-jax`; valid `WANDB_API_KEY` in local `.env` (gitignored); outbound internet (HF + W&B); `workdir/MiniOneRec` present on TPU
- **Environment (verified)**:
  - TPU VM: `mllm-jax-sft-ema-v6e-8-260130172559` (`v6e-8`, 1 host), zone `europe-west4-a`, project `civil-rarity-482610-s5`
  - TPU OS: Ubuntu `24.04.2`
  - Python: `3.12.12`
  - JAX/JAXLIB: `0.9.0` / `0.9.0` (backend `tpu`, `device_count=8`)
  - Repo: `https://github.com/demon2036/MLLM-JAX.git`, branch `john/sft-rl-ema-20260130`, commit `a571716`

## Goal

- Run the official-aligned SID SFT config in `projects/sid_sft` on TPU with W&B online.
- Compare **baseline** vs **EMA-for-eval** (enabled via YAML only).
- Cross-check HR/NDCG using upstream `calc.py` from MiniOneRec.

## Steps (commands actually run)

### 0) Sync secrets to TPU (W&B key)

Run locally (workstation):

```bash
scripts/sync_env_to_tpu_vm.sh --name mllm-jax-sft-ema-v6e-8-260130172559 --zone europe-west4-a --src .env --dest /root/.env --worker all
```

### 1) Prepare MiniOneRec under repo `workdir/`

Run on TPU:

```bash
mkdir -p /root/MLLM-JAX/workdir
git clone --depth 1 https://github.com/AkaliKong/MiniOneRec /root/MLLM-JAX/workdir/MiniOneRec
```

### 2) Baseline run (official-align config)

In this verified run, baseline was started via a small `nohup` wrapper (before
`scripts/tpu_vm_start_sid_sft_from_config_nohup.sh` existed). The core command
executed on TPU was (train+eval):

```bash
cd /root/MLLM-JAX
bash scripts/run_sid_sft.sh \
  --config projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full.yaml \
  --run-mode train_eval
```

Observed:
- W&B run: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/rgpp5bd2
- Outputs in `runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full/`:
  - `eval_predictions.json`
  - `eval_predictions.metrics.json`
  - `sft_state_last.msgpack`

Baseline metrics (`eval_predictions.metrics.json`):
- HR@K: `{1: 0.06485771, 3: 0.08162365, 5: 0.09088904, 10: 0.11272888, 20: 0.13699537, 50: 0.18508714}`
- NDCG@K: `{1: 0.06485771, 3: 0.07465598, 5: 0.07841434, 10: 0.08546677, 20: 0.09147101, 50: 0.10087945}`

### 3) EMA run (same hyperparams, EMA-for-eval enabled via YAML)

Run on TPU (nohup helper):

```bash
cd /root/MLLM-JAX
bash scripts/tpu_vm_start_sid_sft_from_config_nohup.sh \
  --env-name mllm-jax \
  --config projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full_ema.yaml \
  --run-mode train_eval
```

Observed:
- W&B run: https://wandb.ai/johntitordemon2036/minionerec-sid-sft/runs/cwfnsa6t
- Outputs in `runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full_ema/`:
  - `eval_predictions.json`
  - `eval_predictions.metrics.json`
  - `sft_state_last.msgpack` (includes both params + ema_params)

EMA metrics (`eval_predictions.metrics.json`, eval uses EMA weights):
- HR@K: `{1: 0.0, 3: 0.00132363, 5: 0.00419148, 10: 0.00727995, 20: 0.01323627, 50: 0.02382528}`
- NDCG@K: `{1: 0.0, 3: 0.00071958, 5: 0.00186769, 10: 0.00286121, 20: 0.00436225, 50: 0.00646229}`

### 4) Cross-check HR/NDCG via upstream `calc.py`

Baseline:

```bash
cd /root/MLLM-JAX
python workdir/MiniOneRec/calc.py \
  --path runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full/eval_predictions.json \
  --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

EMA:

```bash
cd /root/MLLM-JAX
python workdir/MiniOneRec/calc.py \
  --path runs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full_ema/eval_predictions.json \
  --item_path workdir/MiniOneRec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

Expected: printed HR/NDCG match `eval_predictions.metrics.json` exactly.

## Notes

- For short runs (here: `780` steps), a high decay like `0.9998` keeps EMA very close to the initial weights
  (so evaluation can look much worse even if training converges). Consider:
  - smaller `ema.decay` for short runs, or
  - bias-corrected EMA (not implemented here).

## References

- `projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full.yaml`
- `projects/sid_sft/configs/sid_sft_jax_qwen25_1p5b_instruct_industrial_v6e8_full_ema.yaml`
- `scripts/run_sid_sft.sh`, `scripts/tpu_vm_start_sid_sft_from_config_nohup.sh`
- `docs/sops/minionerec-sid-sft-and-eval.md`

# MiniOneRec-JAX v2 (config scaffold)

This folder is the phase-1 design scaffold for the new parallel implementation path.

Scope of this phase:
- deterministic config defaults and YAML merge loading,
- typed dataclass conversion for downstream runner wiring,
- no dataset/decoder/runner behavior implementation.

## Provided API

- `DEFAULT_CONFIG`: deterministic default config dictionary.
- `load_config(config_path, overrides=None)`: wrapper around `plugins.training.core.config.loader.load_config`.
- `config_from_dict(cfg, config_path=...)`: convert merged dict to typed dataclass object.

## Required fields prepared for official-checkpoint eval flow

- Checkpoint source: `checkpoint.repo_id`, `checkpoint.revision`
- Dataset source: `dataset.source_root`, `dataset.dataset_name` (`industrial` / `office`)
- Decode settings: `decode.num_beams`, `decode.length_penalty`, `decode.max_new_tokens`, `decode.do_sample`, `decode.temperature`
- Eval metric cutoffs: `eval.topk`
- Logging mode: `wandb.mode` (default `disabled` in this phase)

## Phase-1 acceptance note

Official-checkpoint alignment acceptance in this phase uses Table1 `K={3,5,10}` with tolerance `±0.001`.

## Smoke check

```bash
python - <<'PY'
from projects.minionerec_jax_v2.config import DEFAULT_CONFIG, load_config, config_from_dict
cfg = load_config(None)
obj = config_from_dict(cfg, config_path='<default>')
print(bool(DEFAULT_CONFIG), obj.dataset_name)
PY
```

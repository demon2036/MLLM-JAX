# SOP: Inspect mesh/sharding/optimizer/checkpoint duplication

- **Title**: Inspect mesh/sharding/optimizer/checkpoint duplication
- **Prereqs**: Local repo clone at `/home/john/workdir/minionerec`.
- **Steps**:
  ```bash
  ls
  cat memory/README.md
  date +%Y%m%d
  cat docs/sops.md
  sed -n '1,240p' plugins/training/mesh.py
  sed -n '1,240p' plugins/training/update/optimizer.py
  sed -n '1,240p' plugins/training/update/modules.py
  sed -n '1,240p' plugins/training/update/train_step.py
  sed -n '1,240p' plugins/training/update/ppo.py
  sed -n '1,240p' training2.py
  rg -n "mesh|shard|sharding|partition|checkpoint|ckpt|save|load|optimizer|optax|TrainState" training2.py
  sed -n '1,260p' MLLM_JAX/utils.py
  sed -n '260,620p' MLLM_JAX/utils.py
  rg -n "mesh|shard|sharding|checkpoint|ckpt|save|load" MLLM_JAX/utils.py
  rg -n "batch|shard|sharding|mesh|partition|optimizer|checkpoint|ckpt|save|load" plugins/training/mesh.py plugins/training/update training2.py MLLM_JAX/utils.py
  rg -n "slice_data" -S
  rg -n "create_mesh|get_jax_mesh2|get_jax_mesh" -S
  rg -n "OptimizerConfig|build_tx|LRScheduleConfig" -S
  rg -n "msgpack|checkpoint|ckpt|save_checkpoint" -S
  ```
- **Expected Result**: Identify duplicated mesh creation, batch slicing, optimizer configuration, and checkpoint naming/serialization helpers.
- **Troubleshooting**: If `rg` is unavailable, install ripgrep or use a repo-local search alternative.
- **References**: `plugins/training/mesh.py`, `plugins/training/update/optimizer.py`, `plugins/training/update/train_step.py`, `plugins/training/update/ppo.py`, `training2.py`, `MLLM_JAX/utils.py`.

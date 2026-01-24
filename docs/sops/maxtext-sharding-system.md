# SOP: Inspect MaxText sharding system (mesh + logical_axis_rules)

- **Title**: SOP: Clone MaxText and map its `data/fsdp/tensor` sharding to this repo’s `dp/fsdp/tp`
  **Prereqs**: Linux/macOS shell; `git`; outbound network access
  **Scope**: Code reading only (no TPU run)

## Steps (commands actually used)

### 1) Clone MaxText into gitignored `workdir/`

From repo root:

```bash
cd /home/john/workdir/multi-host
mkdir -p workdir
git clone --depth 1 https://github.com/google/maxtext.git workdir/maxtext
git -C workdir/maxtext rev-parse --short HEAD
git -C workdir/maxtext remote -v
```

Revision inspected in this run: `b646a53`

### 2) Locate mesh axes + parallelism knobs

```bash
nl -ba workdir/maxtext/src/MaxText/configs/base.yml | sed -n '360,540p'
```

Look for:
- `mesh_axes: ['data', ..., 'fsdp', ..., 'tensor', ...]`
- `logical_axis_rules: [...]`
- `ici_*_parallelism` / `dcn_*_parallelism` (how mesh dimensions are chosen)

### 3) See how MaxText builds the physical device mesh

```bash
nl -ba workdir/maxtext/src/MaxText/maxtext_utils.py | sed -n '1080,1150p'
```

Key function: `create_device_mesh(config, devices=None)`

### 4) See how logical axes become `PartitionSpec` / `NamedSharding`

```bash
sed -n '1,240p' workdir/maxtext/src/MaxText/sharding.py
```

Key functions:
- `logical_to_mesh_axes(...)`
- `create_sharding(...)`
- `maybe_shard_with_name(...)` (auto: `with_sharding_constraint`, explicit: `reshard`)

### 5) Confirm “all-gather over fsdp” is explicit (not implicit)

```bash
nl -ba workdir/maxtext/src/MaxText/sharding.py | sed -n '540,640p'
nl -ba workdir/maxtext/src/MaxText/vocabulary_tiling.py | sed -n '1,170p'
```

Key signal: `all_gather_over_fsdp(...)` is a dedicated helper used in specific codepaths (e.g. vocab tiling).

### 6) Example layer showing parameter sharding via logical names

```bash
sed -n '1,220p' workdir/maxtext/src/MaxText/layers/embeddings.py
sed -n '1,160p' workdir/maxtext/src/MaxText/layers/simple_layer.py
```

## Expected Result

- You can explain MaxText’s sharding in one line:
  - **`data` = DP**, **`fsdp` = parameter/model sharding**, **`tensor` = tensor-parallel-like axis** (plus extra axes like `sequence/context/...`).
- You can map MaxText axis names to this repo:
  - MaxText `data` ↔ this repo `dp`
  - MaxText `fsdp` ↔ this repo `fsdp`
  - MaxText `tensor` ↔ this repo `tp`

## Notes

- MaxText’s `fsdp` naming is a **mesh axis name**, not a guarantee of “PyTorch FSDP runtime semantics”.
- In JAX/MaxText, “gathering” FSDP-sharded weights is something they do when needed (e.g. `all_gather_over_fsdp`), otherwise the compiler lowers sharded compute SPMD.


# SOP: Verify JAX sharded matmul uses all-reduce vs all-gather

- **Title**: SOP: Demonstrate when JAX SPMD inserts `all-reduce` vs `all-gather` for a sharded matmul
  **Prereqs**: Python + JAX installed
  **Scope**: Local CPU-only compile inspection (no TPU run)

## Why this exists

When weights are sharded (e.g. on an `fsdp` axis), JAX/XLA has multiple valid SPMD strategies to implement `dot_general`:

- **Keep weights sharded** → compute partial dot locally → **`all-reduce`** partial outputs.
- **Replicate weights** (remove sharding constraint) → **`all-gather`** weights → compute full dot locally.

This SOP gives a tiny, deterministic repro that you can use to reason about memory/perf tradeoffs and to interpret helpers like MaxText’s `all_gather_over_fsdp`.

## Environment (verified in this repo)

- JAX: `0.8.2`
- Backend: `cpu`
- Multi-device CPU simulation: `XLA_FLAGS=--xla_force_host_platform_device_count=8`

## Steps (commands actually used)

### 1) Compile a matmul with weights sharded on `fsdp` → expect `all-reduce`

```bash
cd /home/john/workdir/multi-host
XLA_FLAGS=--xla_force_host_platform_device_count=8 python - <<'PY'
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape((1, 8, 1)), ("dp", "fsdp", "tp"))

x = jnp.ones((2, 64), dtype=jnp.float32)
w = jnp.ones((64, 64), dtype=jnp.float32)

x_arr = jax.device_put(x, NamedSharding(mesh, PS()))               # replicated
w_arr = jax.device_put(w, NamedSharding(mesh, PS("fsdp", "tp")))   # sharded on fsdp

@jax.jit
def f(a, b):
  return a @ b

with mesh:
  compiled = f.lower(x_arr, w_arr).compile()

text = compiled.as_text()
print("all-reduce", "all-reduce" in text)
print("all-gather", "all-gather" in text)
PY
```

Expected result:
- `all-reduce True`
- `all-gather False`

### 2) Force weights to be replicated → expect `all-gather`

```bash
cd /home/john/workdir/multi-host
XLA_FLAGS=--xla_force_host_platform_device_count=8 python - <<'PY'
import re
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape((1, 8, 1)), ("dp", "fsdp", "tp"))
replicated_w = NamedSharding(mesh, PS())

x = jnp.ones((2, 64), dtype=jnp.float32)
w = jnp.ones((64, 64), dtype=jnp.float32)

x_arr = jax.device_put(x, NamedSharding(mesh, PS()))
w_arr = jax.device_put(w, NamedSharding(mesh, PS("fsdp", "tp")))

@jax.jit
def f(a, b):
  b_full = jax.lax.with_sharding_constraint(b, replicated_w)  # remove fsdp sharding
  return a @ b_full

with mesh:
  compiled = f.lower(x_arr, w_arr).compile()

text = compiled.as_text()
print("all-gather", "all-gather" in text)
print("all-reduce", "all-reduce" in text)
PY
```

Expected result:
- `all-gather True`
- `all-reduce False`

## How this maps to MaxText and this repo

- MaxText `all_gather_over_fsdp(...)` explicitly constructs a “no-fsdp” sharding and applies a constraint so XLA inserts **all-gather**:
  - `workdir/maxtext/src/MaxText/sharding.py`
- This repo’s MLLM-JAX-style sharding (e.g. `PS('fsdp','tp')`) normally relies on SPMD lowering that can keep weights sharded and use **all-reduce** instead:
  - `MLLM_JAX/utils.py`


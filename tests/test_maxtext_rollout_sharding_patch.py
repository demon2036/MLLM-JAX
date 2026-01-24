from __future__ import annotations

import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _run_py(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def test_print_config_accepts_rollout_sharding_style_maxtext() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "scripts", "run_grpo_gsm8k_training.py"),
            "--print-config",
            "--set",
            "rollout.sharding_style=maxtext",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
    assert "sharding_style: maxtext" in proc.stdout


def test_dp_only_rollout_sharding_patch_smoke() -> None:
    code = r"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from plugins.training.rollout.optimizations import patch_sampler_rollout_dp_only_sharding

devices = np.asarray(jax.devices(), dtype=object)
mesh = Mesh(devices.reshape((1, devices.size, 1)), ("dp", "fsdp", "tp"))

class DummySampler:
  def __init__(self, mesh):
    self.mesh = mesh

sampler = DummySampler(mesh)
patch_sampler_rollout_dp_only_sharding(sampler)

local = np.arange(8, dtype=np.int32).reshape((2, 4))
arr = sampler.global_collect_method(local)

assert tuple(arr.shape) == (2, 4), arr.shape
assert hasattr(arr, "addressable_shards")
shards = list(arr.addressable_shards)
assert len(shards) == len(jax.local_devices())

first = np.asarray(shards[0].data)
for shard in shards[1:]:
  np.testing.assert_array_equal(np.asarray(shard.data), first)

logits = jnp.zeros((2, 64), dtype=jnp.float32)
with mesh:
  out = sampler.sample_fn(jax.random.PRNGKey(0), logits)
  jax.block_until_ready(out)

assert tuple(out.shape) == (2,), out.shape
out_shards = list(out.addressable_shards)
assert out_shards, "expected shards"
ref = np.asarray(out_shards[0].data)
for shard in out_shards[1:]:
  np.testing.assert_array_equal(np.asarray(shard.data), ref)

print("ok")
"""
    proc = _run_py(
        code,
        env={
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        },
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n"
    assert "ok" in proc.stdout


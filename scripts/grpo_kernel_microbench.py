from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.training.kernels.entropy_pallas import EntropyKernelConfig, entropy_per_token_pallas_sharded
from plugins.training.kernels.grpo_loss_pallas import (
    GRPOKernelConfig,
    GRPOKernelShardingSpec,
    grpo_per_token_loss_pallas_sharded,
    grpo_per_token_loss_reference,
)
from plugins.training.mesh import create_mesh


def _maybe_git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


@dataclass(frozen=True)
class BenchShapeConfig:
    # Batch per device; global batch = batch_per_device * (dp*fsdp) for the mesh.
    batch_per_device: int = 1
    seq_len: int = 4096
    vocab: int = 151936
    logits_dtype: str = "bfloat16"


@dataclass(frozen=True)
class KernelSectionConfig:
    enabled: bool = False
    kernel: GRPOKernelConfig = field(default_factory=GRPOKernelConfig)
    sharding: GRPOKernelShardingSpec = field(default_factory=GRPOKernelShardingSpec)
    entropy: EntropyKernelConfig = field(default_factory=EntropyKernelConfig)


@dataclass(frozen=True)
class MicrobenchConfig:
    config_path: str

    mesh_shape: str = "auto"
    shape: BenchShapeConfig = field(default_factory=BenchShapeConfig)

    # Benchmark settings (after compilation).
    warmup_steps: int = 2
    steps: int = 10

    # GRPO hyperparams.
    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    temperature: float = 1.0

    # Old-logp initialization: constant baseline (no expensive seed compute).
    # This keeps the objective well-defined without needing a separate log-softmax pass.
    old_logp_value: float = 0.0

    include_entropy: bool = True
    kernel: KernelSectionConfig = field(default_factory=KernelSectionConfig)

    wandb_project: str = "mllm-jax-grpo-kernel-microbench"
    wandb_mode: str = "online"
    wandb_name: str | None = None


def _parse_dtype(name: str):
    import jax.numpy as jnp

    n = str(name or "bfloat16").strip().lower()
    if n in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if n in {"f16", "float16"}:
        return jnp.float16
    if n in {"f32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported logits_dtype: {name!r}")


def _cfg_from_dict(raw: dict[str, Any], *, config_path: str) -> MicrobenchConfig:
    mesh_shape = str(raw.get("mesh_shape") or "auto")

    shape_raw = raw.get("shape") or {}
    if not isinstance(shape_raw, dict):
        raise TypeError("shape must be a dict")
    shape = BenchShapeConfig(
        batch_per_device=int(shape_raw.get("batch_per_device") or 1),
        seq_len=int(shape_raw.get("seq_len") or 4096),
        vocab=int(shape_raw.get("vocab") or 151936),
        logits_dtype=str(shape_raw.get("logits_dtype") or "bfloat16"),
    )

    warmup_steps = int(raw.get("warmup_steps") or 2)
    steps = int(raw.get("steps") or 10)

    epsilon_low = float(raw.get("epsilon_low") or 0.2)
    epsilon_high = float(raw.get("epsilon_high") or 0.3)
    temperature = float(raw.get("temperature") or 1.0)
    old_logp_value = float(raw.get("old_logp_value") or 0.0)

    include_entropy = bool(raw.get("include_entropy", True))

    kernel_raw = raw.get("kernel") or {}
    if not isinstance(kernel_raw, dict):
        raise TypeError("kernel must be a dict")
    enabled = bool(kernel_raw.get("enabled", False))

    kernel_cfg_raw = kernel_raw.get("kernel") or {}
    if not isinstance(kernel_cfg_raw, dict):
        raise TypeError("kernel.kernel must be a dict")
    kernel_cfg = GRPOKernelConfig(
        block_size=int(kernel_cfg_raw.get("block_size") or 2048),
        time_block=int(kernel_cfg_raw.get("time_block") or 8),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
        bwd_impl=str(kernel_cfg_raw.get("bwd_impl") or "pallas"),
    )

    sharding_raw = kernel_raw.get("sharding") or {}
    if not isinstance(sharding_raw, dict):
        raise TypeError("kernel.sharding must be a dict")
    batch_axes_raw = sharding_raw.get("batch_axes") or ["dp", "fsdp"]
    if not isinstance(batch_axes_raw, (list, tuple)):
        raise TypeError("kernel.sharding.batch_axes must be a list")
    vocab_axis = sharding_raw.get("vocab_axis", None)
    kernel_sharding = GRPOKernelShardingSpec(
        batch_axes=tuple(str(x) for x in batch_axes_raw),
        vocab_axis=str(vocab_axis) if vocab_axis is not None else None,
    )

    entropy_raw = kernel_raw.get("entropy") or {}
    if not isinstance(entropy_raw, dict):
        raise TypeError("kernel.entropy must be a dict")
    entropy_cfg = EntropyKernelConfig(
        block_size=int(entropy_raw.get("block_size") or int(kernel_cfg.block_size)),
        time_block=int(entropy_raw.get("time_block") or int(kernel_cfg.time_block)),
        temperature=temperature,
    )

    wandb_project = str(raw.get("wandb_project") or "mllm-jax-grpo-kernel-microbench")
    wandb_mode = str(raw.get("wandb_mode") or "online")
    wandb_name = raw.get("wandb_name", None)
    wandb_name = str(wandb_name) if wandb_name else None

    return MicrobenchConfig(
        config_path=config_path,
        mesh_shape=mesh_shape,
        shape=shape,
        warmup_steps=warmup_steps,
        steps=steps,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
        old_logp_value=old_logp_value,
        include_entropy=include_entropy,
        kernel=KernelSectionConfig(enabled=enabled, kernel=kernel_cfg, sharding=kernel_sharding, entropy=entropy_cfg),
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        wandb_name=wandb_name,
    )


def _load_config(path: str) -> MicrobenchConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a dict, got {type(raw).__name__}")
    return _cfg_from_dict(raw, config_path=str(path))


def _device_memory_stats_max() -> dict[str, int]:
    import jax

    peaks: list[int] = []
    in_use: list[int] = []
    for dev in jax.local_devices():
        stats = dev.memory_stats()
        if not stats:
            continue
        peaks.append(int(stats.get("peak_bytes_in_use", 0) or 0))
        in_use.append(int(stats.get("bytes_in_use", 0) or 0))
    return {
        "peak_bytes_in_use_max": int(max(peaks) if peaks else 0),
        "bytes_in_use_max": int(max(in_use) if in_use else 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args(argv)

    load_dotenv_if_present(repo_root=REPO_ROOT)

    cfg = _load_config(args.config)
    git_sha = _maybe_git_short_sha()
    if git_sha and cfg.wandb_name:
        cfg = MicrobenchConfig(**{**cfg.__dict__, "wandb_name": f"{cfg.wandb_name}_{git_sha}"})

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as PS

    try:
        jax.distributed.initialize()
    except Exception as e:
        if os.environ.get("PRINT_JAX_DISTRIBUTED_INIT_ERROR") == "1":
            print(f"jax.distributed.initialize() skipped: {e}")

    mesh = create_mesh(cfg.mesh_shape)
    batch_axes = tuple(cfg.kernel.sharding.batch_axes)
    logits_spec = PS(batch_axes, None, None)
    bt_spec = PS(batch_axes, None)
    b_spec = PS(batch_axes)

    logits_dtype = _parse_dtype(cfg.shape.logits_dtype)
    batch_shards = 1
    for ax in batch_axes:
        batch_shards *= int(mesh.shape.get(ax, 1))
    batch_global = int(cfg.shape.batch_per_device) * int(batch_shards)
    seq_len = int(cfg.shape.seq_len)
    vocab = int(cfg.shape.vocab)

    print(
        "devices:",
        {
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
            "device_count": int(jax.device_count()),
            "backend": str(jax.default_backend()),
        },
    )
    print("mesh:", {"shape": dict(mesh.shape), "axis_names": tuple(mesh.axis_names)})
    print(
        "bench_shape:",
        {
            "batch_per_device": int(cfg.shape.batch_per_device),
            "batch_global": int(batch_global),
            "seq_len": int(seq_len),
            "vocab": int(vocab),
            "logits_dtype": str(logits_dtype),
        },
    )

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        process_index=int(jax.process_index()),
    )

    # --- Create sharded synthetic inputs (on device, no host materialization) ---
    key = jax.random.PRNGKey(0)

    logits_sharding = NamedSharding(mesh, logits_spec)
    bt_sharding = NamedSharding(mesh, bt_spec)
    b_sharding = NamedSharding(mesh, b_spec)

    @jax.jit(out_shardings=(logits_sharding, bt_sharding, bt_sharding, b_sharding))
    def make_inputs(rng):
        key_logits, key_ids, key_adv = jax.random.split(rng, 3)
        logits = jax.random.normal(key_logits, (batch_global, seq_len, vocab), dtype=logits_dtype)
        chosen_ids = jax.random.randint(key_ids, (batch_global, seq_len), 0, vocab, dtype=jnp.int32)
        old_per_token_logps = jnp.full((batch_global, seq_len), float(cfg.old_logp_value), dtype=jnp.float32)
        advantages = jax.random.normal(key_adv, (batch_global,), dtype=jnp.float32)
        return logits, chosen_ids, old_per_token_logps, advantages

    t_inputs0 = time.perf_counter()
    logits, chosen_ids, old_per_token_logps, advantages = make_inputs(key)
    # Ensure inputs are resident on device.
    logits.block_until_ready()
    t_inputs = time.perf_counter() - t_inputs0

    # --- Define baseline vs kernel paths ---
    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)

    def _baseline_entropy_mean(l):
        x = l.astype(jnp.float32) / float(temperature)
        probs = jax.nn.softmax(x, axis=-1)
        # Match repo baseline's epsilon-in-log behavior.
        ent = -jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1)
        return jax.lax.stop_gradient(jnp.mean(ent.astype(jnp.float32)))

    def _kernel_entropy_mean(l):
        ent = entropy_per_token_pallas_sharded(
            logits=l,
            mesh=mesh,
            cfg=cfg.kernel.entropy,
            batch_axes=tuple(cfg.kernel.sharding.batch_axes),
            vocab_axis=cfg.kernel.sharding.vocab_axis,
            interpret=False,
            debug=False,
            check_vma=False,
        )
        return jax.lax.stop_gradient(jnp.mean(ent.astype(jnp.float32)))

    def loss_and_aux(params_logits):
        if cfg.kernel.enabled:
            per_loss, per_logps = grpo_per_token_loss_pallas_sharded(
                logits=params_logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                mesh=mesh,
                cfg=cfg.kernel.kernel,
                sharding=cfg.kernel.sharding,
                interpret=False,
                debug=False,
                check_vma=False,
                use_self_old=False,
            )
            entropy_mean = _kernel_entropy_mean(params_logits) if cfg.include_entropy else jnp.asarray(0.0, jnp.float32)
        else:
            per_loss, per_logps = grpo_per_token_loss_reference(
                logits=params_logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                epsilon_low=eps_low,
                epsilon_high=eps_high,
                temperature=temperature,
            )
            entropy_mean = _baseline_entropy_mean(params_logits) if cfg.include_entropy else jnp.asarray(0.0, jnp.float32)
        loss = jnp.mean(per_loss.astype(jnp.float32))
        aux = {
            "entropy_mean": entropy_mean,
            "logp_mean": jax.lax.stop_gradient(jnp.mean(per_logps.astype(jnp.float32))),
        }
        return loss, aux

    loss_grad_fn = jax.jit(jax.value_and_grad(loss_and_aux, has_aux=True))

    # Compile + one warmup run.
    t0 = time.perf_counter()
    (loss0, aux0), grad0 = loss_grad_fn(logits)
    loss0.block_until_ready()
    grad0.block_until_ready()
    t_compile = time.perf_counter() - t0

    # Timed runs.
    times: list[float] = []
    for _ in range(int(cfg.warmup_steps)):
        (loss_w, _aux_w), grad_w = loss_grad_fn(logits)
        loss_w.block_until_ready()
        grad_w.block_until_ready()

    for _ in range(int(cfg.steps)):
        t1 = time.perf_counter()
        (loss_i, aux_i), grad_i = loss_grad_fn(logits)
        loss_i.block_until_ready()
        grad_i.block_until_ready()
        times.append(time.perf_counter() - t1)

    times_np = np.asarray(times, dtype=np.float64)
    time_mean = float(times_np.mean()) if times_np.size else float("nan")
    time_p50 = float(np.quantile(times_np, 0.50)) if times_np.size else float("nan")
    time_p90 = float(np.quantile(times_np, 0.90)) if times_np.size else float("nan")

    mem = _device_memory_stats_max()

    metrics = {
        "bench/inputs_build_s": float(t_inputs),
        "bench/compile_s": float(t_compile),
        "bench/step_s_mean": float(time_mean),
        "bench/step_s_p50": float(time_p50),
        "bench/step_s_p90": float(time_p90),
        "bench/loss": float(jnp.asarray(loss0)),
        "bench/entropy_mean": float(jnp.asarray(aux0["entropy_mean"])),
        "bench/logp_mean": float(jnp.asarray(aux0["logp_mean"])),
        "mem/peak_bytes_in_use_max": int(mem["peak_bytes_in_use_max"]),
        "mem/bytes_in_use_max": int(mem["bytes_in_use_max"]),
        "shape/batch_per_device": int(cfg.shape.batch_per_device),
        "shape/batch_global": int(batch_global),
        "shape/seq_len": int(seq_len),
        "shape/vocab": int(vocab),
        "shape/logits_dtype": str(logits_dtype),
        "kernel/enabled": int(bool(cfg.kernel.enabled)),
        "kernel/block_size": int(cfg.kernel.kernel.block_size),
        "kernel/time_block": int(cfg.kernel.kernel.time_block),
        "kernel/bwd_impl": str(cfg.kernel.kernel.bwd_impl),
        "entropy/block_size": int(cfg.kernel.entropy.block_size),
        "entropy/time_block": int(cfg.kernel.entropy.time_block),
    }

    print("metrics:", metrics)
    if wandb is not None:
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"wandb.log failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


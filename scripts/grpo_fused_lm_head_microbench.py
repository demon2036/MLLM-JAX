from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.training.kernels.grpo_fused_lm_head import GRPOLmHeadFusedConfig, grpo_per_token_loss_fused_lm_head
from plugins.training.kernels.grpo_loss_pallas import grpo_per_token_loss_reference
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
    batch_per_device: int = 1
    seq_len: int = 8192
    hidden: int = 4096
    vocab: int = 151936
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class KernelSectionConfig:
    enabled: bool = False
    vocab_block_size: int = 2048


@dataclass(frozen=True)
class MicrobenchConfig:
    config_path: str
    mesh_shape: str = "auto"
    shape: BenchShapeConfig = BenchShapeConfig()

    warmup_steps: int = 2
    steps: int = 10

    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    temperature: float = 1.0
    old_logp_value: float = 0.0

    kernel: KernelSectionConfig = KernelSectionConfig()

    wandb_project: str = "mllm-jax-grpo-fused-lmhead-microbench"
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
    raise ValueError(f"Unsupported dtype: {name!r}")


def _cfg_from_dict(raw: dict[str, Any], *, config_path: str) -> MicrobenchConfig:
    mesh_shape = str(raw.get("mesh_shape") or "auto")

    shape_raw = raw.get("shape") or {}
    if not isinstance(shape_raw, dict):
        raise TypeError("shape must be a dict")
    shape = BenchShapeConfig(
        batch_per_device=int(shape_raw.get("batch_per_device") or 1),
        seq_len=int(shape_raw.get("seq_len") or 8192),
        hidden=int(shape_raw.get("hidden") or 4096),
        vocab=int(shape_raw.get("vocab") or 151936),
        dtype=str(shape_raw.get("dtype") or "bfloat16"),
    )

    warmup_steps = int(raw.get("warmup_steps") or 2)
    steps = int(raw.get("steps") or 10)

    epsilon_low = float(raw.get("epsilon_low") or 0.2)
    epsilon_high = float(raw.get("epsilon_high") or 0.3)
    temperature = float(raw.get("temperature") or 1.0)
    old_logp_value = float(raw.get("old_logp_value") or 0.0)

    kernel_raw = raw.get("kernel") or {}
    if not isinstance(kernel_raw, dict):
        raise TypeError("kernel must be a dict")
    enabled = bool(kernel_raw.get("enabled", False))
    vocab_block_size = int(kernel_raw.get("vocab_block_size") or kernel_raw.get("block_size") or 2048)

    wandb_project = str(raw.get("wandb_project") or "mllm-jax-grpo-fused-lmhead-microbench")
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
        kernel=KernelSectionConfig(enabled=enabled, vocab_block_size=vocab_block_size),
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

    try:
        jax.distributed.initialize()
    except Exception as e:
        if os.environ.get("PRINT_JAX_DISTRIBUTED_INIT_ERROR") == "1":
            print(f"jax.distributed.initialize() skipped: {e}")

    mesh = create_mesh(cfg.mesh_shape)
    dtype = _parse_dtype(cfg.shape.dtype)

    batch = int(cfg.shape.batch_per_device)
    seq_len = int(cfg.shape.seq_len)
    hidden = int(cfg.shape.hidden)
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
    print("bench_shape:", {"batch": batch, "seq_len": seq_len, "hidden": hidden, "vocab": vocab, "dtype": str(dtype)})

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        process_index=int(jax.process_index()),
    )

    key = jax.random.PRNGKey(0)
    key_h, key_w, key_ids, key_adv = jax.random.split(key, 4)

    hidden_states = jax.random.normal(key_h, (batch, seq_len, hidden), dtype=dtype)
    lm_head_kernel = jax.random.normal(key_w, (hidden, vocab), dtype=dtype)
    chosen_ids = jax.random.randint(key_ids, (batch, seq_len), 0, vocab, dtype=jnp.int32)
    old_per_token_logps = jnp.full((batch, seq_len), float(cfg.old_logp_value), dtype=jnp.float32)
    advantages = jax.random.normal(key_adv, (batch,), dtype=jnp.float32)

    eps_low = float(cfg.epsilon_low)
    eps_high = float(cfg.epsilon_high)
    temperature = float(cfg.temperature)

    def loss_and_aux(h, w):
        if cfg.kernel.enabled:
            fused_cfg = GRPOLmHeadFusedConfig(
                vocab_block_size=int(cfg.kernel.vocab_block_size),
                epsilon_low=eps_low,
                epsilon_high=eps_high,
                temperature=temperature,
            )
            per_loss, per_logps = grpo_per_token_loss_fused_lm_head(
                hidden_states=h,
                lm_head_kernel=w,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                cfg=fused_cfg,
                use_self_old=False,
            )
        else:
            logits = jax.lax.dot_general(
                h,
                w,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(h.dtype)
            per_loss, per_logps = grpo_per_token_loss_reference(
                logits=logits,
                chosen_ids=chosen_ids,
                old_per_token_logps=old_per_token_logps,
                advantages=advantages,
                epsilon_low=eps_low,
                epsilon_high=eps_high,
                temperature=temperature,
            )
        loss = jnp.mean(per_loss.astype(jnp.float32))
        aux = {
            "logp_mean": jax.lax.stop_gradient(jnp.mean(per_logps.astype(jnp.float32))),
        }
        return loss, aux

    fn = jax.jit(jax.value_and_grad(loss_and_aux, argnums=(0, 1), has_aux=True))

    t0 = time.perf_counter()
    (loss0, aux0), (gh0, gw0) = fn(hidden_states, lm_head_kernel)
    loss0.block_until_ready()
    gh0.block_until_ready()
    gw0.block_until_ready()
    t_compile = time.perf_counter() - t0

    for _ in range(int(cfg.warmup_steps)):
        (loss_w, _aux_w), (gh_w, gw_w) = fn(hidden_states, lm_head_kernel)
        loss_w.block_until_ready()
        gh_w.block_until_ready()
        gw_w.block_until_ready()

    times: list[float] = []
    for _ in range(int(cfg.steps)):
        t1 = time.perf_counter()
        (loss_i, aux_i), (gh_i, gw_i) = fn(hidden_states, lm_head_kernel)
        loss_i.block_until_ready()
        gh_i.block_until_ready()
        gw_i.block_until_ready()
        times.append(time.perf_counter() - t1)

    import numpy as np

    times_np = np.asarray(times, dtype=np.float64)
    time_mean = float(times_np.mean()) if times_np.size else float("nan")
    time_p50 = float(np.quantile(times_np, 0.50)) if times_np.size else float("nan")
    time_p90 = float(np.quantile(times_np, 0.90)) if times_np.size else float("nan")

    mem = _device_memory_stats_max()

    metrics = {
        "bench/compile_s": float(t_compile),
        "bench/step_s_mean": float(time_mean),
        "bench/step_s_p50": float(time_p50),
        "bench/step_s_p90": float(time_p90),
        "bench/loss": float(jnp.asarray(loss0)),
        "bench/logp_mean": float(jnp.asarray(aux0["logp_mean"])),
        "mem/peak_bytes_in_use_max": int(mem["peak_bytes_in_use_max"]),
        "mem/bytes_in_use_max": int(mem["bytes_in_use_max"]),
        "shape/batch": int(batch),
        "shape/seq_len": int(seq_len),
        "shape/hidden": int(hidden),
        "shape/vocab": int(vocab),
        "shape/dtype": str(dtype),
        "kernel/enabled": int(bool(cfg.kernel.enabled)),
        "kernel/vocab_block_size": int(cfg.kernel.vocab_block_size),
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


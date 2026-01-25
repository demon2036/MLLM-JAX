from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.common.env import load_dotenv_if_present
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.sample.mllm_sampler import get_model
from plugins.training.kernels.grpo_loss_pallas import GRPOKernelConfig, grpo_per_token_loss_pallas, grpo_per_token_loss_reference
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
class GradcheckTolerances:
    loss_atol: float = 1e-4
    dlogits_max_abs: float = 5e-3
    dlogits_max_rel: float = 5e-2


@dataclass(frozen=True)
class GRPOKernelGradcheckConfig:
    config_path: str

    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mesh_shape: str = "1,-1,1"

    batch_size: int = 1
    seq_len: int = 64
    seed: int = 0

    epsilon_low: float = 0.2
    epsilon_high: float = 0.3
    temperature: float = 1.0
    old_logp_noise_scale: float = 0.3

    kernel: GRPOKernelConfig = GRPOKernelConfig(block_size=2048, epsilon_low=0.2, epsilon_high=0.3, temperature=1.0)
    tolerances: GradcheckTolerances = GradcheckTolerances()

    wandb_project: str = "mllm-jax-grpo-kernel"
    wandb_mode: str = "online"
    wandb_name: str = "grpo_kernel_gradcheck_qwen25_1p5b"


def _cfg_from_dict(cfg: dict[str, Any], *, config_path: str) -> GRPOKernelGradcheckConfig:
    model_path = str(cfg.get("model_path") or "Qwen/Qwen2.5-1.5B-Instruct")
    mesh_shape = str(cfg.get("mesh_shape") or "1,-1,1")

    batch_size = int(cfg.get("batch_size") or 1)
    seq_len = int(cfg.get("seq_len") or 64)
    seed = int(cfg.get("seed") or 0)

    epsilon_low = float(cfg.get("epsilon_low") or 0.2)
    epsilon_high = float(cfg.get("epsilon_high") or 0.3)
    temperature = float(cfg.get("temperature") or 1.0)
    old_logp_noise_scale = float(cfg.get("old_logp_noise_scale") or 0.3)

    kernel_raw = cfg.get("kernel") or {}
    if not isinstance(kernel_raw, dict):
        raise TypeError("kernel must be a dict")
    kernel_block_size = int(kernel_raw.get("block_size") or 2048)

    tol_raw = cfg.get("tolerances") or {}
    if not isinstance(tol_raw, dict):
        raise TypeError("tolerances must be a dict")
    tolerances = GradcheckTolerances(
        loss_atol=float(tol_raw.get("loss_atol") or 1e-4),
        dlogits_max_abs=float(tol_raw.get("dlogits_max_abs") or 5e-3),
        dlogits_max_rel=float(tol_raw.get("dlogits_max_rel") or 5e-2),
    )

    wandb_project = str(cfg.get("wandb_project") or "mllm-jax-grpo-kernel")
    wandb_mode = str(cfg.get("wandb_mode") or "online")
    wandb_name = str(cfg.get("wandb_name") or "grpo_kernel_gradcheck_qwen25_1p5b")

    return GRPOKernelGradcheckConfig(
        config_path=config_path,
        model_path=model_path,
        mesh_shape=mesh_shape,
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        temperature=temperature,
        old_logp_noise_scale=old_logp_noise_scale,
        kernel=GRPOKernelConfig(
            block_size=kernel_block_size,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            temperature=temperature,
        ),
        tolerances=tolerances,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        wandb_name=wandb_name,
    )


def _load_config(config_path: str) -> GRPOKernelGradcheckConfig:
    raw = yaml.safe_load(Path(config_path).read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a dict, got {type(raw).__name__}")
    return _cfg_from_dict(raw, config_path=str(config_path))


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args(argv)

    load_dotenv_if_present(repo_root=REPO_ROOT)

    cfg = _load_config(args.config)
    git_sha = _maybe_git_short_sha()
    if git_sha:
        cfg = GRPOKernelGradcheckConfig(**{**cfg.__dict__, "wandb_name": f"{cfg.wandb_name}_{git_sha}"})

    import jax
    import jax.numpy as jnp

    try:
        jax.distributed.initialize()
    except Exception as e:
        if os.environ.get("PRINT_JAX_DISTRIBUTED_INIT_ERROR") == "1":
            print(f"jax.distributed.initialize() skipped: {e}")

    mesh = create_mesh(cfg.mesh_shape)
    process_index = int(getattr(jax, "process_index", lambda: 0)())
    print(
        "devices:",
        {
            "process_index": process_index,
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
            "device_count": int(jax.device_count()),
            "backend": str(jax.default_backend()),
        },
    )

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        process_index=process_index,
    )

    t0 = time.time()
    model, params, tokenizer = get_model(mesh, model_path=cfg.model_path)
    t_load = time.time() - t0
    vocab = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    print("model:", {"model_path": cfg.model_path, "vocab": vocab, "pad_token_id": int(pad_token_id or 0)})
    print("load_s:", round(t_load, 3))

    key = jax.random.PRNGKey(int(cfg.seed))
    key_ids, key_adv, key_noise = jax.random.split(key, 3)

    input_ids = jax.random.randint(
        key_ids,
        (int(cfg.batch_size), int(cfg.seq_len)),
        0,
        vocab,
        dtype=jnp.int32,
    )
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.int32)
    labels = attention_mask

    @jax.jit
    def _forward(p, ids, mask):
        return model.apply({"params": p}, input_ids=ids, attention_mask=mask)

    t1 = time.time()
    logits, _ = _forward(params, input_ids, attention_mask)
    t_fwd = time.time() - t1
    print("forward_s:", round(t_fwd, 3))

    logits_for_loss = logits[:, :-1, :]
    chosen_ids = input_ids[:, 1:]
    mask_loss = labels[:, 1:].astype(jnp.float32)
    device0 = jax.devices()[0]

    def _to_device0(x):
        return jax.device_put(jax.device_get(x), device0)

    # Mosaic kernels cannot be auto-partitioned by SPMD; run the kernel on a
    # single TPU device by materializing single-device arrays.
    logits_for_loss = _to_device0(logits_for_loss)
    chosen_ids = _to_device0(chosen_ids)
    mask_loss = _to_device0(mask_loss)
    denom = mask_loss.sum()

    advantages = jax.random.normal(key_adv, (int(cfg.batch_size),), dtype=jnp.float32)
    advantages = _to_device0(advantages)

    _, logps_seed = grpo_per_token_loss_reference(
        logits=logits_for_loss,
        chosen_ids=chosen_ids,
        old_per_token_logps=jnp.zeros_like(mask_loss, dtype=jnp.float32),
        advantages=advantages,
        epsilon_low=cfg.epsilon_low,
        epsilon_high=cfg.epsilon_high,
        temperature=cfg.temperature,
    )
    noise = jax.random.normal(key_noise, logps_seed.shape, dtype=jnp.float32) * float(cfg.old_logp_noise_scale)
    old_per_token_logps = jax.lax.stop_gradient(logps_seed + _to_device0(noise))

    fwd_ref = jax.jit(
        lambda l: grpo_per_token_loss_reference(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=cfg.epsilon_low,
            epsilon_high=cfg.epsilon_high,
            temperature=cfg.temperature,
        )
    )
    fwd_kernel = jax.jit(
        lambda l: grpo_per_token_loss_pallas(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg.kernel,
            interpret=False,
            debug=False,
        )
    )

    per_loss_ref_fwd, per_logp_ref_fwd = fwd_ref(logits_for_loss)
    per_loss_kernel_fwd, per_logp_kernel_fwd = fwd_kernel(logits_for_loss)

    fwd_loss_ref = float(jnp.asarray((per_loss_ref_fwd * mask_loss).sum() / (denom + 1e-8)))
    fwd_loss_kernel = float(jnp.asarray((per_loss_kernel_fwd * mask_loss).sum() / (denom + 1e-8)))
    fwd_logp_max_abs = float(jnp.asarray(jnp.max(jnp.abs(per_logp_ref_fwd - per_logp_kernel_fwd))))
    fwd_loss_max_abs = float(jnp.asarray(jnp.max(jnp.abs(per_loss_ref_fwd - per_loss_kernel_fwd))))

    def _scalar_loss_ref(l):
        per_loss, _ = grpo_per_token_loss_reference(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            epsilon_low=cfg.epsilon_low,
            epsilon_high=cfg.epsilon_high,
            temperature=cfg.temperature,
        )
        return (per_loss * mask_loss).sum() / (denom + 1e-8)

    def _scalar_loss_kernel(l):
        per_loss, _ = grpo_per_token_loss_pallas(
            logits=l,
            chosen_ids=chosen_ids,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            cfg=cfg.kernel,
            interpret=False,
            debug=False,
        )
        return (per_loss * mask_loss).sum() / (denom + 1e-8)

    loss_and_grad_ref = jax.jit(jax.value_and_grad(_scalar_loss_ref))
    loss_and_grad_kernel = jax.jit(jax.value_and_grad(_scalar_loss_kernel))

    t2 = time.time()
    loss_ref, dlogits_ref = loss_and_grad_ref(logits_for_loss)
    t_ref = time.time() - t2

    t3 = time.time()
    loss_kernel, dlogits_kernel = loss_and_grad_kernel(logits_for_loss)
    t_kernel = time.time() - t3

    loss_ref_f = float(jnp.asarray(loss_ref))
    loss_kernel_f = float(jnp.asarray(loss_kernel))

    d_ref_f32 = dlogits_ref.astype(jnp.float32)
    d_kernel_f32 = dlogits_kernel.astype(jnp.float32)
    diff = jnp.abs(d_ref_f32 - d_kernel_f32)

    max_abs = float(jnp.asarray(jnp.max(diff)))
    max_rel = float(jnp.asarray(jnp.max(diff / (jnp.abs(d_ref_f32) + 1e-8))))
    mean_abs = float(jnp.asarray(jnp.mean(diff)))

    abs_diff_loss = abs(loss_ref_f - loss_kernel_f)

    metrics = {
        "fwd/loss_ref": fwd_loss_ref,
        "fwd/loss_kernel": fwd_loss_kernel,
        "fwd/logp_max_abs": fwd_logp_max_abs,
        "fwd/per_loss_max_abs": fwd_loss_max_abs,
        "loss_ref": loss_ref_f,
        "loss_kernel": loss_kernel_f,
        "abs_diff_loss": abs_diff_loss,
        "dlogits_max_abs": max_abs,
        "dlogits_max_rel": max_rel,
        "dlogits_mean_abs": mean_abs,
        "time/load_s": round(t_load, 3),
        "time/forward_s": round(t_fwd, 3),
        "time/ref_grad_s": round(t_ref, 3),
        "time/kernel_grad_s": round(t_kernel, 3),
        "kernel/block_size": int(cfg.kernel.block_size),
        "shape/batch": int(cfg.batch_size),
        "shape/seq_len": int(cfg.seq_len),
        "shape/vocab": int(vocab),
    }

    print("metrics:", metrics)
    if wandb is not None:
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"wandb.log failed: {e}")

    if not isfinite(loss_ref_f) or not isfinite(loss_kernel_f):
        raise SystemExit(f"non-finite loss: loss_ref={loss_ref_f} loss_kernel={loss_kernel_f}")
    if not isfinite(abs_diff_loss):
        raise SystemExit(f"non-finite loss diff: abs_diff_loss={abs_diff_loss}")
    if not isfinite(max_abs) or not isfinite(max_rel) or not isfinite(mean_abs):
        raise SystemExit(
            f"non-finite dlogits diff: max_abs={max_abs} max_rel={max_rel} mean_abs={mean_abs}"
        )

    if abs_diff_loss > float(cfg.tolerances.loss_atol):
        raise SystemExit(f"loss mismatch: abs_diff_loss={abs_diff_loss} > loss_atol={cfg.tolerances.loss_atol}")
    if max_abs > float(cfg.tolerances.dlogits_max_abs):
        raise SystemExit(
            f"dlogits max abs mismatch: max_abs={max_abs} > dlogits_max_abs={cfg.tolerances.dlogits_max_abs}"
        )
    if max_rel > float(cfg.tolerances.dlogits_max_rel):
        raise SystemExit(
            f"dlogits max rel mismatch: max_rel={max_rel} > dlogits_max_rel={cfg.tolerances.dlogits_max_rel}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

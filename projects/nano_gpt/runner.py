from __future__ import annotations

import functools
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np


def run_nano_gpt(cfg: dict[str, Any], *, config_path: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import checkpoints, train_state

    from projects.nano_gpt.data import prepare_tinyshakespeare_char, sample_batch
    from plugins.training.core.optim.optimizer import _scale_by_muon
    from plugins.nano_gpt.model import GPT, GPTConfig, parse_dtype

    seed = int(cfg.get("seed", 1337))
    output_dir = os.path.abspath(str(cfg.get("output_dir") or "runs/nano_gpt"))
    cfg = dict(cfg)
    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    resolved_cfg_path = os.path.join(output_dir, "config_resolved.json")
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")

    data_cfg = dict(cfg.get("data") or {})
    dataset = prepare_tinyshakespeare_char(
        cache_dir=str(data_cfg.get("cache_dir") or "workdir/nano_gpt_data"),
        url=str(data_cfg.get("url")),
        train_ratio=float(data_cfg.get("train_ratio") or 0.9),
    )

    model_cfg_raw = dict(cfg.get("model") or {})
    block_size = int(model_cfg_raw.get("block_size") or 256)
    model_cfg = GPTConfig(
        vocab_size=int(dataset.vocab_size),
        block_size=block_size,
        n_layer=int(model_cfg_raw.get("n_layer") or 6),
        n_head=int(model_cfg_raw.get("n_head") or 6),
        n_embd=int(model_cfg_raw.get("n_embd") or 384),
        dropout=float(model_cfg_raw.get("dropout") or 0.0),
        bias=bool(model_cfg_raw.get("bias") if model_cfg_raw.get("bias") is not None else True),
        param_dtype=parse_dtype(model_cfg_raw.get("param_dtype") or "float32"),
        compute_dtype=parse_dtype(model_cfg_raw.get("compute_dtype") or "float32"),
    )

    train_cfg = dict(cfg.get("train") or {})
    max_steps = int(train_cfg.get("max_steps") or 0)
    if max_steps <= 0:
        raise ValueError(f"train.max_steps must be > 0, got: {max_steps}")

    devices = jax.local_devices()
    num_devices = len(devices)
    global_batch_size = int(train_cfg.get("global_batch_size") or 0)
    if global_batch_size <= 0:
        raise ValueError(f"train.global_batch_size must be > 0, got: {global_batch_size}")
    if global_batch_size % num_devices != 0:
        raise ValueError(f"global_batch_size={global_batch_size} must be divisible by num_devices={num_devices}")
    per_device_batch_size = global_batch_size // num_devices

    learning_rate = float(train_cfg.get("learning_rate") or 0.0)
    min_lr = float(train_cfg.get("min_lr") if train_cfg.get("min_lr") is not None else 0.0)
    warmup_steps = int(train_cfg.get("warmup_steps") or 0)
    weight_decay = float(train_cfg.get("weight_decay") if train_cfg.get("weight_decay") is not None else 0.0)
    beta1 = float(train_cfg.get("beta1") if train_cfg.get("beta1") is not None else 0.9)
    beta2 = float(train_cfg.get("beta2") if train_cfg.get("beta2") is not None else 0.95)
    grad_clip_norm = float(train_cfg.get("grad_clip_norm") if train_cfg.get("grad_clip_norm") is not None else 1.0)

    optimizer_name = str(train_cfg.get("optimizer") or "adamw").strip().lower()
    muon_cfg = dict(train_cfg.get("muon") or {})
    muon_aux_learning_rate = float(muon_cfg.get("aux_learning_rate") if muon_cfg.get("aux_learning_rate") is not None else 3e-4)
    muon_momentum = float(muon_cfg.get("momentum") if muon_cfg.get("momentum") is not None else 0.95)
    muon_nesterov = bool(muon_cfg.get("nesterov") if muon_cfg.get("nesterov") is not None else True)
    muon_ns_steps = int(muon_cfg.get("ns_steps") if muon_cfg.get("ns_steps") is not None else 5)
    muon_eps = float(muon_cfg.get("eps") if muon_cfg.get("eps") is not None else 1e-7)
    muon_max_dim = int(muon_cfg.get("max_dim") if muon_cfg.get("max_dim") is not None else 10_000)
    muon_exclude_embeddings = bool(muon_cfg.get("exclude_embeddings") if muon_cfg.get("exclude_embeddings") is not None else True)

    log_every = int(train_cfg.get("log_every") or 10)
    eval_every = int(train_cfg.get("eval_every") or 0)
    eval_iters = int(train_cfg.get("eval_iters") or 0)
    sample_every = int(train_cfg.get("sample_every") or 0)
    sample_tokens = int(train_cfg.get("sample_tokens") or 0)
    temperature = float(train_cfg.get("temperature") if train_cfg.get("temperature") is not None else 1.0)
    top_k = int(train_cfg.get("top_k") if train_cfg.get("top_k") is not None else 0)
    ckpt_every = int(train_cfg.get("ckpt_every") or 0)
    keep_ckpts = int(train_cfg.get("keep_ckpts") or 2)

    def make_lr_schedule(*, peak_lr: float, min_lr_val: float):
        peak_lr = float(peak_lr)
        min_lr_val = float(min_lr_val)

        def schedule(step: jnp.ndarray) -> jnp.ndarray:
            stepf = step.astype(jnp.float32)
            if warmup_steps > 0:
                warmup = jnp.asarray(peak_lr, dtype=jnp.float32) * stepf / jnp.maximum(1.0, float(warmup_steps))
            else:
                warmup = jnp.asarray(peak_lr, dtype=jnp.float32)

            denom = max(1, max_steps - warmup_steps)
            progress = (stepf - float(warmup_steps)) / float(denom)
            progress = jnp.clip(progress, 0.0, 1.0)
            cosine = jnp.asarray(min_lr_val, dtype=jnp.float32) + 0.5 * (jnp.asarray(peak_lr, dtype=jnp.float32) - jnp.asarray(min_lr_val, dtype=jnp.float32)) * (
                1.0 + jnp.cos(jnp.pi * progress)
            )
            return jnp.where(stepf < float(warmup_steps), warmup, cosine)

        return schedule

    lr_schedule = make_lr_schedule(peak_lr=learning_rate, min_lr_val=min_lr)
    aux_min_lr = 0.0
    if optimizer_name == "muon":
        if learning_rate <= 0.0:
            raise ValueError(f"train.learning_rate must be > 0 for optimizer='muon', got: {learning_rate}")
        aux_min_lr = float(muon_aux_learning_rate) * float(min_lr) / float(learning_rate)
    aux_lr_schedule = make_lr_schedule(peak_lr=muon_aux_learning_rate, min_lr_val=aux_min_lr)

    model = GPT(model_cfg)

    init_rng = jax.random.PRNGKey(seed)
    dummy_idx = jnp.zeros((1, model_cfg.block_size), dtype=jnp.int32)
    params = model.init({"params": init_rng}, dummy_idx, deterministic=True)["params"]

    # Align optimizer semantics with the common nanoGPT AdamW setup:
    # apply weight decay only to parameters with ndim >= 2 (e.g., kernels/embeddings),
    # and exclude biases / LayerNorm scale/bias (ndim == 1).
    decay_mask = jax.tree_util.tree_map(lambda x: getattr(x, "ndim", 0) >= 2, params)

    if optimizer_name == "adamw":
        tx_base = optax.adamw(
            learning_rate=lr_schedule,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
            mask=decay_mask,
        )
    elif optimizer_name == "muon":
        if muon_max_dim <= 0:
            raise ValueError(f"train.muon.max_dim must be > 0, got: {muon_max_dim}")

        def is_embedding_path(path) -> bool:
            if not muon_exclude_embeddings:
                return False
            for k in path:
                if getattr(k, "key", None) in {"wte", "wpe"}:
                    return True
            return False

        def label_leaf(path, p):
            if getattr(p, "ndim", 0) != 2:
                return "adamw_nodecay"
            if int(max(p.shape[0], p.shape[1])) > int(muon_max_dim):
                return "adamw_decay"
            if is_embedding_path(path):
                return "adamw_decay"
            return "muon"

        leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
        param_labels = jax.tree_util.tree_unflatten(treedef, [label_leaf(path, p) for path, p in leaves_with_path])

        muon_chain = [
            _scale_by_muon(
                momentum=float(muon_momentum),
                nesterov=bool(muon_nesterov),
                ns_steps=int(muon_ns_steps),
                eps=float(muon_eps),
                mu_dtype=jnp.float32,
            )
        ]
        if float(weight_decay) != 0.0:
            muon_chain.append(optax.add_decayed_weights(float(weight_decay)))
        muon_chain.extend([optax.scale_by_schedule(lr_schedule), optax.scale(-1.0)])
        muon_tx = optax.chain(*muon_chain)
        adamw_decay_tx = optax.adamw(
            learning_rate=aux_lr_schedule,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
        )
        adamw_nodecay_tx = optax.adamw(
            learning_rate=aux_lr_schedule,
            b1=beta1,
            b2=beta2,
            weight_decay=0.0,
        )
        tx_base = optax.multi_transform(
            {
                "muon": muon_tx,
                "adamw_decay": adamw_decay_tx,
                "adamw_nodecay": adamw_nodecay_tx,
            },
            param_labels,
        )
    else:
        raise ValueError(f"Unsupported train.optimizer={optimizer_name!r} (expected 'adamw'|'muon')")

    if grad_clip_norm > 0.0:
        tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), tx_base)
    else:
        tx = tx_base
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax.device_put_replicated(state, devices)

    rng = np.random.default_rng(seed)
    dropout_rng = jax.random.PRNGKey(seed + 1)
    dropout_rngs = jax.random.split(dropout_rng, num_devices)

    def _loss_from_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
        return jnp.mean(losses)

    def train_step(
        state: train_state.TrainState, batch: dict[str, jnp.ndarray], rng_key: jnp.ndarray
    ) -> tuple[train_state.TrainState, dict[str, jnp.ndarray], jnp.ndarray]:
        rng_key, dropout_key = jax.random.split(rng_key)

        def loss_fn(params):
            logits = model.apply({"params": params}, batch["x"], deterministic=False, rngs={"dropout": dropout_key})
            loss = _loss_from_logits(logits, batch["y"])
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")

        grad_norm = optax.global_norm(grads)
        grad_norm = jax.lax.pmean(grad_norm, axis_name="data")

        state = state.apply_gradients(grads=grads)
        metrics = {
            "train/loss": loss,
            "train/grad_norm": grad_norm,
            "train/lr": lr_schedule(state.step),
        }
        if optimizer_name == "muon":
            metrics["train/aux_lr"] = aux_lr_schedule(state.step)
        return state, metrics, rng_key

    p_train_step = jax.pmap(train_step, axis_name="data", donate_argnums=(0,))

    def eval_step(params: Any, batch: dict[str, jnp.ndarray]) -> jnp.ndarray:
        logits = model.apply({"params": params}, batch["x"], deterministic=True)
        loss = _loss_from_logits(logits, batch["y"])
        return jax.lax.pmean(loss, axis_name="data")

    p_eval_step = jax.pmap(eval_step, axis_name="data")

    @functools.partial(jax.jit, static_argnames=("num_tokens", "top_k"))
    def generate_tokens(
        params: Any,
        idx: jnp.ndarray,
        rng_key: jnp.ndarray,
        num_tokens: int,
        temperature: float,
        top_k: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def step_fn(carry, _):
            idx, rng_key = carry
            rng_key, sample_key = jax.random.split(rng_key)
            logits = model.apply({"params": params}, idx, deterministic=True)
            logits = logits[:, -1, :].astype(jnp.float32)
            logits = logits / jnp.maximum(1e-6, jnp.asarray(temperature, dtype=jnp.float32))

            if top_k and top_k > 0:
                kth = jnp.sort(logits, axis=-1)[:, -top_k]
                logits = jnp.where(logits < kth[:, None], jnp.full_like(logits, -1e10), logits)

            next_id = jax.random.categorical(sample_key, logits, axis=-1).astype(jnp.int32)
            idx = jnp.concatenate([idx[:, 1:], next_id[:, None]], axis=1)
            return (idx, rng_key), next_id

        (idx, rng_key), out = jax.lax.scan(step_fn, (idx, rng_key), xs=None, length=num_tokens)
        return idx, rng_key, out

    wandb_cfg = dict(cfg.get("wandb") or {})
    wandb_mode = str(wandb_cfg.get("mode") or "disabled").strip().lower()
    wandb_project = str(wandb_cfg.get("project") or "nano-gpt-jax")
    wandb_name = wandb_cfg.get("name")
    if wandb_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_name = f"nano-gpt-{ts}"

    wandb_run_url: str | None = None
    wandb_obj: Any | None = None
    if jax.process_index() == 0 and wandb_mode not in {"disabled", "disable", "off"}:
        import wandb  # type: ignore

        wandb_obj = wandb
        wandb.init(project=wandb_project, name=str(wandb_name), mode=wandb_mode, config=cfg)
        wandb_run_url = getattr(getattr(wandb, "run", None), "url", None)

    last_log_t = time.time()
    last_log_step = 0

    def host_metrics(m: dict[str, jnp.ndarray]) -> dict[str, float]:
        return {k: float(v[0]) for k, v in m.items()}

    def save_checkpoint(step: int) -> None:
        if jax.process_index() != 0:
            return
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        state0 = jax.tree_util.tree_map(lambda x: x[0], state)
        checkpoints.save_checkpoint(
            ckpt_dir,
            target=state0,
            step=step,
            keep=keep_ckpts,
            overwrite=True,
        )

    best_val_loss = float("inf")
    best_val_step = 0
    for _ in range(max_steps):
        x, y = sample_batch(
            rng=rng,
            ids=dataset.train_ids,
            batch_size=global_batch_size,
            block_size=model_cfg.block_size,
        )
        x = x.reshape(num_devices, per_device_batch_size, model_cfg.block_size)
        y = y.reshape(num_devices, per_device_batch_size, model_cfg.block_size)
        batch = {"x": jnp.asarray(x), "y": jnp.asarray(y)}

        state, metrics, dropout_rngs = p_train_step(state, batch, dropout_rngs)
        step = int(state.step[0])

        if log_every > 0 and (step % log_every == 0 or step == 1):
            now = time.time()
            dt = now - last_log_t
            steps_delta = step - last_log_step
            tok_delta = steps_delta * global_batch_size * model_cfg.block_size
            tps = tok_delta / dt if dt > 0 else 0.0

            m = host_metrics(metrics)
            m["perf/tokens_per_sec"] = float(tps)
            m["perf/step_time_sec"] = float(dt / max(1, steps_delta))
            print(f"[train] step={step} loss={m['train/loss']:.4f} lr={m['train/lr']:.3e} tps={tps:.1f}")
            if wandb_obj is not None:
                wandb_obj.log(m, step=step)

            last_log_t = now
            last_log_step = step

        if eval_every > 0 and eval_iters > 0 and (step % eval_every == 0 or step == max_steps):
            losses = []
            for _ in range(eval_iters):
                x, y = sample_batch(
                    rng=rng,
                    ids=dataset.val_ids,
                    batch_size=global_batch_size,
                    block_size=model_cfg.block_size,
                )
                x = x.reshape(num_devices, per_device_batch_size, model_cfg.block_size)
                y = y.reshape(num_devices, per_device_batch_size, model_cfg.block_size)
                batch = {"x": jnp.asarray(x), "y": jnp.asarray(y)}
                loss = p_eval_step(state.params, batch)
                losses.append(float(loss[0]))

            val_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"[eval] step={step} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = step
            if wandb_obj is not None:
                wandb_obj.log({"eval/loss": val_loss}, step=step)

        if sample_every > 0 and sample_tokens > 0 and (step % sample_every == 0 or step == max_steps):
            params0 = jax.tree_util.tree_map(lambda x: x[0], state.params)
            prompt_ch = "\n"
            start_id = dataset.stoi.get(prompt_ch, 0)
            idx = jnp.full((1, model_cfg.block_size), start_id, dtype=jnp.int32)
            gen_rng = jax.random.PRNGKey(seed + step + 999)
            _, _, out = generate_tokens(params0, idx, gen_rng, sample_tokens, temperature, top_k)
            sample_text = dataset.decode(np.asarray(out))
            print(f"[sample] step={step}\n{sample_text}\n")
            if wandb_obj is not None:
                wandb_obj.log({"sample/text": sample_text}, step=step)

        if ckpt_every > 0 and (step % ckpt_every == 0 or step == max_steps):
            save_checkpoint(step)

    final_metrics = host_metrics(metrics)
    summary = {
        "config_path": config_path,
        "output_dir": output_dir,
        "resolved_config_json": resolved_cfg_path,
        "final_step": int(state.step[0]),
        "wandb_run_url": wandb_run_url,
        "final_train_loss": float(final_metrics.get("train/loss", float("nan"))),
        "best_eval_loss": float(best_val_loss) if best_val_step > 0 else None,
        "best_eval_step": int(best_val_step) if best_val_step > 0 else None,
    }

    if wandb_obj is not None:
        wandb_obj.finish()

    summary_path = os.path.join(output_dir, "run_summary.json")
    if jax.process_index() == 0:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"summary_json={summary_path}")

    return summary


__all__ = ["run_nano_gpt"]

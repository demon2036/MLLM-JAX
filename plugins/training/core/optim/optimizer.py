from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


@dataclass(frozen=True)
class LRScheduleConfig:
    """Learning-rate schedule config (Optax).

    Defaults match the hardcoded schedule in `training2.get_state`.
    """

    # Supported: warmup_cosine, warmup_linear, constant
    type: str = "warmup_cosine"

    init_value: float = 0.0
    peak_value: float = 1e-6
    end_value: float = 0.0

    # If warmup_steps is None, derive: round(training_steps * warmup_ratio).
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer config (Optax).

    Defaults match the hardcoded optimizer in `training2.get_state`.
    """

    # Supported: lion, adamw, sgd, muon
    name: str = "lion"

    clip_norm: float = 1.0
    weight_decay: float = 1e-8

    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Muon (https://kellerjordan.github.io/posts/muon/) specific options.
    #
    # When `name == "muon"`, Muon is applied to 2D parameters with
    # max(shape) <= `muon_max_dim` and AdamW is used as an auxiliary optimizer
    # for all other params.
    muon_aux_lr: float = 3e-4
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-7
    muon_max_dim: int = 10_000


def build_lr_schedule(*, training_steps: int, cfg: LRScheduleConfig):
    import optax

    steps = int(training_steps)
    if steps <= 0:
        raise ValueError(f"training_steps must be > 0, got {steps}")

    schedule_type = str(cfg.type).strip().lower()
    if schedule_type in {"warmup_cosine", "warmup_cosine_decay"}:
        warmup_steps = cfg.warmup_steps
        if warmup_steps is None:
            warmup_steps = int(round(float(steps) * float(cfg.warmup_ratio)))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        return optax.warmup_cosine_decay_schedule(
            init_value=float(cfg.init_value),
            peak_value=float(cfg.peak_value),
            warmup_steps=warmup_steps,
            decay_steps=steps,
            end_value=float(cfg.end_value),
        )

    if schedule_type in {"warmup_linear", "warmup_linear_decay", "linear_warmup"}:
        warmup_steps = cfg.warmup_steps
        if warmup_steps is None:
            warmup_steps = int(round(float(steps) * float(cfg.warmup_ratio)))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        warmup_steps = min(warmup_steps, steps)
        if warmup_steps == 0:
            return optax.linear_schedule(init_value=float(cfg.peak_value), end_value=float(cfg.end_value), transition_steps=steps)

        warmup = optax.linear_schedule(init_value=float(cfg.init_value), end_value=float(cfg.peak_value), transition_steps=warmup_steps)
        decay_steps = max(1, steps - warmup_steps)
        decay = optax.linear_schedule(init_value=float(cfg.peak_value), end_value=float(cfg.end_value), transition_steps=decay_steps)
        return optax.join_schedules([warmup, decay], [warmup_steps])

    if schedule_type in {"constant", "const"}:
        return optax.constant_schedule(float(cfg.peak_value))

    raise ValueError(f"Unsupported lr_schedule.type={cfg.type!r} (expected warmup_cosine|warmup_linear|constant)")


def _muon_frob_norm(x):
    import jax.numpy as jnp

    x32 = x.astype(jnp.float32)
    return jnp.sqrt(jnp.sum(x32 * x32))


def _muon_newton_schulz_5(*, g, steps: int, eps: float):
    """Approximate an orthogonalized update for a 2D matrix (Muon).

    This is a JAX port of the `newtonschulz5` reference from
    https://kellerjordan.github.io/posts/muon/.
    """
    import jax
    import jax.numpy as jnp

    if int(getattr(g, "ndim", 0)) != 2:
        raise ValueError(f"Muon update expects a 2D matrix, got ndim={getattr(g, 'ndim', None)}")

    a, b, c = (3.4445, -4.7750, 2.0315)

    x = g.astype(jnp.bfloat16)
    x = x / (_muon_frob_norm(x) + jnp.asarray(float(eps), dtype=jnp.float32))

    transposed = bool(int(x.shape[0]) > int(x.shape[1]))
    if transposed:
        x = x.T

    def body(_i, x_in):
        a_in = x_in @ x_in.T
        b_in = (jnp.asarray(b, dtype=jnp.bfloat16) * a_in) + (jnp.asarray(c, dtype=jnp.bfloat16) * (a_in @ a_in))
        return (jnp.asarray(a, dtype=jnp.bfloat16) * x_in) + (b_in @ x_in)

    x = jax.lax.fori_loop(0, int(steps), body, x)
    if transposed:
        x = x.T
    return x


def _scale_by_muon(
    *,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    eps: float,
    mu_dtype,
):
    import jax
    import jax.numpy as jnp
    import optax

    MaskedNode = getattr(optax, "MaskedNode", None)

    def is_masked(x: Any) -> bool:
        return MaskedNode is not None and isinstance(x, MaskedNode)

    class _MuonState(NamedTuple):
        momentum: Any

    def init_fn(params):
        def init_leaf(p):
            if is_masked(p):
                return MaskedNode()
            return jnp.zeros_like(p, dtype=mu_dtype)

        return _MuonState(momentum=jax.tree_util.tree_map(init_leaf, params, is_leaf=is_masked))

    def update_fn(updates, state, params=None):
        del params

        def update_leaf(g, m):
            if is_masked(g):
                return MaskedNode(), m
            g = g.astype(mu_dtype)

            m_new = (jnp.asarray(float(momentum), dtype=mu_dtype) * m) + g
            if bool(nesterov):
                g_eff = g + (jnp.asarray(float(momentum), dtype=mu_dtype) * m_new)
            else:
                g_eff = m_new

            u = _muon_newton_schulz_5(g=g_eff, steps=int(ns_steps), eps=float(eps))
            return u, m_new

        pairs = jax.tree_util.tree_map(update_leaf, updates, state.momentum, is_leaf=is_masked)
        is_pair = lambda x: isinstance(x, tuple) and len(x) == 2
        new_updates = jax.tree_util.tree_map(lambda x: x[0], pairs, is_leaf=is_pair)
        new_momentum = jax.tree_util.tree_map(lambda x: x[1], pairs, is_leaf=is_pair)
        return new_updates, _MuonState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def build_tx(*, training_steps: int, cfg: OptimizerConfig, params: Any | None = None):
    """Build an Optax optimizer transformation.

    This is intended to be passed to `training2.get_state(..., tx=...)`.
    """
    import jax
    import jax.numpy as jnp
    import optax

    lr_schedule = build_lr_schedule(training_steps=training_steps, cfg=cfg.lr_schedule)
    name = str(cfg.name).strip().lower()
    weight_decay = float(cfg.weight_decay)

    if name == "lion":
        base = optax.lion(lr_schedule, weight_decay=weight_decay)
    elif name == "adamw":
        base = optax.adamw(lr_schedule, weight_decay=weight_decay)
    elif name == "sgd":
        base = optax.sgd(lr_schedule)
        if weight_decay != 0.0:
            base = optax.chain(optax.add_decayed_weights(weight_decay), base)
    elif name == "muon":
        if params is None:
            raise ValueError("optimizer.name='muon' requires passing `params=` to build the parameter mask.")

        muon_max_dim = int(cfg.muon_max_dim)
        if muon_max_dim <= 0:
            raise ValueError(f"muon_max_dim must be > 0, got {muon_max_dim}")

        aux_lr_cfg = LRScheduleConfig(
            type=str(cfg.lr_schedule.type),
            init_value=float(cfg.lr_schedule.init_value),
            peak_value=float(cfg.muon_aux_lr),
            end_value=float(cfg.lr_schedule.end_value),
            warmup_ratio=float(cfg.lr_schedule.warmup_ratio),
            warmup_steps=cfg.lr_schedule.warmup_steps,
        )
        aux_lr_schedule = build_lr_schedule(training_steps=training_steps, cfg=aux_lr_cfg)

        # NOTE: We intentionally use a custom Muon + auxiliary AdamW split via
        # `optax.multi_transform` so we can tune the auxiliary LR separately.
        # Some Optax versions ship `optax.contrib.muon`, but its signature does
        # not consistently expose an auxiliary learning-rate knob.
        muon_chain = [
            _scale_by_muon(
                momentum=float(cfg.muon_momentum),
                nesterov=bool(cfg.muon_nesterov),
                ns_steps=int(cfg.muon_ns_steps),
                eps=float(cfg.muon_eps),
                mu_dtype=jnp.float32,
            )
        ]
        if weight_decay != 0.0:
            # Decoupled weight decay: do NOT include it inside the orthogonalization.
            muon_chain.append(optax.add_decayed_weights(weight_decay))
        muon_chain.extend([optax.scale_by_schedule(lr_schedule), optax.scale(-1.0)])
        muon_tx = optax.chain(*muon_chain)
        aux_tx = optax.adamw(aux_lr_schedule, weight_decay=weight_decay)

        def label(p):
            if getattr(p, "ndim", 0) != 2:
                return "adamw"
            m = int(p.shape[0])
            n = int(p.shape[1])
            if max(m, n) > int(muon_max_dim):
                return "adamw"
            return "muon"

        param_labels = jax.tree_util.tree_map(label, params)
        base = optax.multi_transform({"muon": muon_tx, "adamw": aux_tx}, param_labels)
    else:
        raise ValueError(f"Unsupported optimizer.name={cfg.name!r} (expected lion|adamw|sgd|muon)")

    clip_norm = float(cfg.clip_norm)
    if clip_norm <= 0.0:
        return base
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)


__all__ = ["LRScheduleConfig", "OptimizerConfig", "build_lr_schedule", "build_tx"]

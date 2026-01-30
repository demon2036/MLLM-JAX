import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from plugins.training.core.optim.optimizer import LRScheduleConfig, OptimizerConfig, build_tx


def test_muon_weight_decay_is_decoupled_for_zero_gradients():
    params = {
        "w": jnp.arange(4, dtype=jnp.float32).reshape(2, 2),
        "b": jnp.arange(2, dtype=jnp.float32),
    }
    peak_lr = 2e-2
    aux_lr = 3e-3
    weight_decay = 1e-1

    cfg = OptimizerConfig(
        name="muon",
        clip_norm=0.0,
        weight_decay=float(weight_decay),
        lr_schedule=LRScheduleConfig(type="constant", peak_value=float(peak_lr)),
        muon_aux_lr=float(aux_lr),
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        muon_eps=1e-7,
        muon_max_dim=10_000,
    )
    tx = build_tx(training_steps=10, cfg=cfg, params=params)

    state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    updates, _ = tx.update(grads, state, params)

    expected_w = -float(peak_lr) * float(weight_decay) * params["w"]
    expected_b = -float(aux_lr) * float(weight_decay) * params["b"]
    np.testing.assert_allclose(np.array(updates["w"]), np.array(expected_w), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(updates["b"]), np.array(expected_b), rtol=1e-6, atol=1e-6)


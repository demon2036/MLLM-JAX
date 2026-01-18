from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.config import DEFAULT_CONFIG
from plugins.training.rollout_backends import SUPPORTED_ROLLOUT_BACKENDS, create_rollout_backend


class _DummySampler:
    # Only needed so the backend can hold a reference.
    tokenizer = object()

    def find_ceil(self, length: int) -> int:
        return int(length)

    def generate(self, *args, **kwargs):  # noqa: ANN001, D401
        raise NotImplementedError


class TestRolloutBackendFactory(unittest.TestCase):
    def test_default_config_includes_rollout_backend(self) -> None:
        self.assertIn("rollout", DEFAULT_CONFIG)
        self.assertIn("backend", DEFAULT_CONFIG["rollout"])
        self.assertEqual(DEFAULT_CONFIG["rollout"]["backend"], "naive")

    def test_create_rollout_backend_naive(self) -> None:
        backend = create_rollout_backend(name="naive", sampler=_DummySampler())
        self.assertEqual(backend.__class__.__name__, "NaiveSamplerRolloutBackend")

    def test_create_rollout_backend_sglang_jax(self) -> None:
        backend = create_rollout_backend(
            name="sglang_jax",
            sampler=_DummySampler(),
            model_path="Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(backend.__class__.__name__, "SglangJaxRolloutBackend")

    def test_create_rollout_backend_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            create_rollout_backend(name="does_not_exist", sampler=_DummySampler())
        msg = str(ctx.exception)
        self.assertIn("rollout.backend", msg)
        for key in SUPPORTED_ROLLOUT_BACKENDS:
            self.assertIn(key, msg)


if __name__ == "__main__":
    unittest.main()

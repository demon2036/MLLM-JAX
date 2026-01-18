from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from plugins.training.rollout_backends.sglang_jax import _engine_key_to_train_key, _nnx_key_to_dotted_path


class TestSglangJaxKeyMapping(unittest.TestCase):
    def test_nnx_key_to_dotted_path_flattens_nested_tuples(self) -> None:
        key = ("model", ("layers", 0), "self_attn", "q_proj", "weight")
        self.assertEqual(_nnx_key_to_dotted_path(key), "model.layers.0.self_attn.q_proj.weight")

    def test_engine_key_mapping_layer_weight(self) -> None:
        self.assertEqual(
            _engine_key_to_train_key("model.layers.0.self_attn.q_proj.weight"),
            ("model.layers_0.self_attn.q_proj.kernel", False),
        )

    def test_engine_key_mapping_layer_scale(self) -> None:
        self.assertEqual(
            _engine_key_to_train_key("model.layers.12.input_layernorm.scale"),
            ("model.layers_12.input_layernorm.scale", False),
        )

    def test_engine_key_mapping_embed_and_norm(self) -> None:
        self.assertEqual(
            _engine_key_to_train_key("model.embed_tokens.embedding"),
            ("model.embed_tokens.embedding", False),
        )
        self.assertEqual(
            _engine_key_to_train_key("model.norm.scale"),
            ("model.norm.scale", False),
        )

    def test_engine_key_mapping_lm_head(self) -> None:
        self.assertEqual(_engine_key_to_train_key("lm_head.embedding"), ("lm_head.kernel", True))

    def test_engine_key_mapping_skips_unknown(self) -> None:
        self.assertEqual(_engine_key_to_train_key("model.layers.0.self_attn.rotary_emb.inv_freq"), (None, False))


if __name__ == "__main__":
    unittest.main()


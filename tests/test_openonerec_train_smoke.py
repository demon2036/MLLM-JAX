from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from projects.openonerec_train.config import load_config
from projects.openonerec_train.runner import run_openonerec_train
from scripts.run_openonerec_train import _cfg_from_dict


def _build_tiny_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    base_vocab = {
        "[UNK]": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|im_start|>": 4,
        "<|im_end|>": 5,
        "<think>": 6,
        "</think>": 7,
    }
    tok = Tokenizer(WordLevel(base_vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tok.save_pretrained(model_dir)

    config = LlamaConfig(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
        attention_bias=True,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir, safe_serialization=True)


def _build_synthetic_openonerec_data(data_dir: Path) -> None:
    task_dir = data_dir / "video"
    task_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "messages": json.dumps(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": "Recommend next video id."}]},
                ],
                ensure_ascii=False,
            ),
            "metadata": json.dumps(
                {
                    "answer": "<|sid_begin|>123<|sid_end|><|sid_begin|>999<|sid_end|>",
                    "source": "synthetic",
                },
                ensure_ascii=False,
            ),
        },
        {
            "messages": json.dumps(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": "Recommend another video id."}]},
                ],
                ensure_ascii=False,
            ),
            "metadata": json.dumps(
                {
                    "answer": "<|sid_begin|>456<|sid_end|>",
                    "source": "synthetic",
                },
                ensure_ascii=False,
            ),
        },
    ]

    pd.DataFrame(rows).to_parquet(task_dir / "video_test.parquet", index=False)


def _prepare_smoke_assets(repo_root: Path) -> None:
    smoke_root = repo_root / "memory" / "20260207_openonerec_jax_full_align"
    _build_tiny_model(smoke_root / "smoke_model")
    _build_synthetic_openonerec_data(smoke_root / "smoke_data")


def test_openonerec_train_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _prepare_smoke_assets(repo_root)

    cfg_path = repo_root / "projects" / "openonerec_train" / "configs" / "openonerec_train_jax_smoke.yaml"
    cfg_dict = load_config(str(cfg_path))
    cfg = _cfg_from_dict(cfg_dict, config_path=str(cfg_path))
    cfg = replace(cfg, output_dir=str(tmp_path / "openonerec_train"))

    assert not torch.cuda.is_available()

    result = run_openonerec_train(cfg, run_mode="train")

    assert int(result["dataset_size"]) > 0
    assert result["train"] is not None
    assert int(result["train"]["steps"]) > 0

    checkpoint_path = result.get("checkpoint_path") or str(Path(cfg.output_dir) / "sft_state_last.msgpack")
    assert Path(checkpoint_path).exists()

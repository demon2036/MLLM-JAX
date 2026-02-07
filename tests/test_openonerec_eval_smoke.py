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

from projects.openonerec_eval.config import load_config
from projects.openonerec_eval.runner import run_openonerec_eval
from scripts.run_openonerec_eval import _cfg_from_dict


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


def _build_replay_data(replay_dir: Path) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)

    replay_payload = {
        "total_time": 0.01,
        "samples": {
            "0": {
                "generations": [
                    "<|sid_begin|>123<|sid_end|>",
                    "<|sid_begin|>888<|sid_end|>",
                ],
                "logprobs": [0.0, -0.2],
            },
            "1": {
                "generations": [
                    "<|sid_begin|>456<|sid_end|>",
                    "<|sid_begin|>111<|sid_end|>",
                ],
                "logprobs": [0.0, -0.3],
            },
        },
    }

    (replay_dir / "video_test_generated.json").write_text(
        json.dumps(replay_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_smoke_assets(repo_root: Path) -> None:
    smoke_root = repo_root / "memory" / "20260207_openonerec_jax_full_align"
    _build_tiny_model(smoke_root / "smoke_model")
    _build_synthetic_openonerec_data(smoke_root / "smoke_data")
    _build_replay_data(smoke_root / "smoke_replay")


def test_openonerec_eval_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    _prepare_smoke_assets(repo_root)

    cfg_path = repo_root / "projects" / "openonerec_eval" / "configs" / "openonerec_eval_jax_smoke.yaml"
    cfg_dict = load_config(str(cfg_path))
    cfg = _cfg_from_dict(cfg_dict, config_path=str(cfg_path))
    cfg = replace(cfg, output_dir=str(tmp_path / "openonerec_eval"))

    assert not torch.cuda.is_available()

    result = run_openonerec_eval(cfg, run_mode="eval")

    eval_results_path = Path(result["eval_results_path"])
    paper_alignment_path = Path(result["paper_alignment_path"])

    assert eval_results_path.exists()
    assert paper_alignment_path.exists()

    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    model_key = next(iter(eval_results.keys()))
    split_metrics = eval_results[model_key]["video"]["test"]

    assert "pass@1" in split_metrics
    assert "position1_pass@1" in split_metrics
    assert "recall@1" in split_metrics

    generated_json = Path(split_metrics["generation_file"])
    generated_payload = json.loads(generated_json.read_text(encoding="utf-8"))
    assert "samples" in generated_payload
    assert isinstance(generated_payload["samples"], dict)
    assert "0" in generated_payload["samples"]

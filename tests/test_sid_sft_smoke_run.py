import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from plugins.sft.runner.sid_sft import (
    SidSftConfig,
    SidSftDataConfig,
    SidSftEvalConfig,
    SidSftJaxConfig,
    SidSftTasksConfig,
    SidSftTrainConfig,
    SidSftWandbConfig,
    run_sid_sft,
)


def test_sid_sft_train_eval_smoke(tmp_path: Path):
    # --- minimal SID universe (2 items) ---
    sid_tokens = {
        "0": ["<a_1>", "<b_1>", "<c_1>"],
        "1": ["<a_2>", "<b_2>", "<c_2>"],
    }
    item_meta = {
        "0": {"title": "Item Zero", "description": "", "brand": "", "categories": []},
        "1": {"title": "Item One", "description": "", "brand": "", "categories": []},
    }
    info_lines = [
        "<a_1><b_1><c_1>\tItem Zero\t0",
        "<a_2><b_2><c_2>\tItem One\t1",
    ]

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sid_index_path = data_dir / "toy.index.json"
    item_meta_path = data_dir / "toy.item.json"
    info_file = data_dir / "toy.info.txt"

    sid_index_path.write_text(json.dumps(sid_tokens), encoding="utf-8")
    item_meta_path.write_text(json.dumps(item_meta), encoding="utf-8")
    info_file.write_text("\n".join(info_lines) + "\n", encoding="utf-8")

    header = "user_id,history_item_sid,item_sid\n"
    train_csv = data_dir / "train.csv"
    eval_csv = data_dir / "eval.csv"
    test_csv = data_dir / "test.csv"
    train_csv.write_text(header + 'U1,"[\'<a_1><b_1><c_1>\']","<a_2><b_2><c_2>"\n', encoding="utf-8")
    eval_csv.write_text(header + 'U2,"[\'<a_2><b_2><c_2>\']","<a_1><b_1><c_1>"\n', encoding="utf-8")
    test_csv.write_text(header + 'U3,"[\'<a_1><b_1><c_1>\']","<a_2><b_2><c_2>"\n', encoding="utf-8")

    # --- local tiny model/tokenizer (offline) ---
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    base_vocab = {"[UNK]": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
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
    model.save_pretrained(model_dir)

    # --- run ---
    cfg = SidSftConfig(
        config_path="<pytest>",
        backend="jax",
        base_model=str(model_dir),
        output_dir=str(tmp_path / "out"),
        seed=0,
        device="cpu",
        data=SidSftDataConfig(
            category="toy",
            train_file=str(train_csv),
            eval_file=str(eval_csv),
            test_file=str(test_csv),
            info_file=str(info_file),
            sid_index_path=str(sid_index_path),
            item_meta_path=str(item_meta_path),
            max_len=64,
            sample_train=-1,
            sample_eval=-1,
            sample_test=-1,
        ),
        jax=SidSftJaxConfig(mesh_shape="1,-1,1", param_dtype="float32", compute_dtype="float32", max_cache_length=256),
        tasks=SidSftTasksConfig(sid_next_item=True, sid_item_alignment=False, fusion_seq_rec=False),
        train=SidSftTrainConfig(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            optimizer="adamw",
            weight_decay=0.0,
            num_train_epochs=1,
            max_steps=1,
            warmup_steps=0,
            logging_steps=1,
            eval_steps=1,
            save_steps=1,
            save_total_limit=1,
            group_by_length=False,
            freeze_LLM=False,
            train_from_scratch=False,
            resume_from_checkpoint=None,
            early_stopping_patience=0,
            bf16=False,
            fp16=False,
        ),
        eval=SidSftEvalConfig(
            enabled=True,
            batch_size=1,
            num_beams=2,
            max_new_tokens=8,
            length_penalty=0.0,
            topk=(1, 2),
            constrained=True,
            save_predictions_json=True,
        ),
        wandb=SidSftWandbConfig(project="minionerec-sid-sft", mode="disabled", name=None),
    )

    # Avoid accidental GPU usage in CI-like environments.
    assert not torch.cuda.is_available()

    result = run_sid_sft(cfg, run_mode="train_eval")
    assert result["train"] is not None
    assert result["eval"] is not None
    assert "hr" in result["eval"]
    assert "ndcg" in result["eval"]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MINIONEREC_DIR = os.path.join(REPO_ROOT, "workdir", "MiniOneRec")
if MINIONEREC_DIR not in sys.path:
    sys.path.insert(0, MINIONEREC_DIR)

from data import EvalSidDataset  # noqa: E402
from LogitProcessor import ConstrainedLogitsProcessor  # noqa: E402


def get_hash(x: Iterable[int]) -> str:
    return "-".join(str(v) for v in x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _build_prefix_allowed_tokens_fn(*, base_model: str, tokenizer, info_file: str) -> Callable[[int, list[int]], list[int]]:
    with open(info_file, "r", encoding="utf-8") as f:
        info = f.readlines()
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]
        item_titles = [line.split("\t")[1].strip() + "\n" for line in info if len(line.split("\t")) >= 2]

        info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
        info_titles = [f"### Response:\n{_}" for _ in item_titles]

    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]

    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3

    hash_dict: dict[str, set[int]] = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        _ = get_hash(ID[prefix_index:])

    hash_dict_title: dict[str, set[int]] = {}
    for ID in prefixTitleID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        _ = get_hash(ID[prefix_index:])

    hash_dict_list: dict[str, list[int]] = {k: list(v) for k, v in hash_dict.items()}
    hash_dict_title_list: dict[str, list[int]] = {k: list(v) for k, v in hash_dict_title.items()}

    def prefix_allowed_tokens_fn_semantic(batch_id: int, input_ids: list[int]) -> list[int]:
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_list:
            return hash_dict_list[hash_number]
        return []

    def prefix_allowed_tokens_fn_title(batch_id: int, input_ids: list[int]) -> list[int]:
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title_list:
            return hash_dict_title_list[hash_number]
        return []

    _ = prefix_allowed_tokens_fn_title
    return prefix_allowed_tokens_fn_semantic


def _evaluate_batch(
    *,
    model,
    tokenizer,
    base_model: str,
    device: str,
    encodings: list[dict],
    prefix_allowed_tokens_fn,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
) -> list[list[str]]:
    maxLen = max(len(_["input_ids"]) for _ in encodings)

    padding_encodings = {"input_ids": []}
    attention_mask: list[list[int]] = []

    for _ in encodings:
        L = len(_["input_ids"])
        padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
        attention_mask.append([0] * (maxLen - L) + [1] * L)

    generation_config = GenerationConfig(
        num_beams=num_beams,
        length_penalty=length_penalty,
        num_return_sequences=num_beams,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        max_new_tokens=max_new_tokens,
        top_k=None,
        top_p=None,
    )

    with torch.no_grad():
        clp = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            base_model=base_model,
            eos_token_id=model.config.eos_token_id,
        )
        logits_processor = LogitsProcessorList([clp])

        generation_output = model.generate(
            torch.tensor(padding_encodings["input_ids"]).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logits_processor,
        )

    batched_completions = generation_output.sequences[:, maxLen:]

    if base_model.lower().find("llama") > -1:
        output = tokenizer.batch_decode(
            batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    else:
        output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)

    output = [_.split("Response:\n")[-1].strip() for _ in output]
    real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]
    return real_outputs


def _iter_slices(indices: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Torch+HF generate eval equivalent to workdir/MiniOneRec/evaluate.py (EvalSidDataset), "
            "but optionally bucket by prompt length so batches have zero left-padding."
        )
    )

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--info_file", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--result_json_data", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=50)

    parser.add_argument(
        "--bucket-by-length",
        dest="bucket_by_length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bucket samples by prompt length so each batch has pad_len=0 (default: true).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only evaluate the first N rows from the CSV (default: all).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    set_seed(int(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category = category_dict.get(str(args.category), str(args.category))
    print(category)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prefix_allowed_tokens_fn = _build_prefix_allowed_tokens_fn(
        base_model=args.base_model,
        tokenizer=tokenizer,
        info_file=args.info_file,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    val_dataset = EvalSidDataset(
        train_file=args.test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=category,
        test=True,
        K=int(args.K),
        seed=int(args.seed),
    )

    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    max_samples = None if args.max_samples is None else int(args.max_samples)
    if max_samples is not None and max_samples > 0:
        encodings = encodings[:max_samples]
        test_data = test_data[:max_samples]

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model = model.to(device)

    outputs_by_index: list[list[str] | None] = [None] * len(encodings)

    if bool(args.bucket_by_length):
        length_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, encoding in enumerate(encodings):
            length_to_indices[len(encoding["input_ids"])].append(idx)

        for length in sorted(length_to_indices.keys()):
            indices = length_to_indices[length]
            for batch_indices in _iter_slices(indices, int(args.batch_size)):
                batch_encodings = [encodings[i] for i in batch_indices]
                batch_outputs = _evaluate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    base_model=args.base_model,
                    device=device,
                    encodings=batch_encodings,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=int(args.num_beams),
                    max_new_tokens=int(args.max_new_tokens),
                    length_penalty=float(args.length_penalty),
                )
                if len(batch_outputs) != len(batch_indices):
                    raise RuntimeError(
                        f"Internal error: got {len(batch_outputs)} outputs for {len(batch_indices)} inputs"
                    )
                for i, out in zip(batch_indices, batch_outputs, strict=True):
                    outputs_by_index[i] = out
    else:
        for start in range(0, len(encodings), int(args.batch_size)):
            batch_encodings = encodings[start : start + int(args.batch_size)]
            batch_outputs = _evaluate_batch(
                model=model,
                tokenizer=tokenizer,
                base_model=args.base_model,
                device=device,
                encodings=batch_encodings,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=int(args.num_beams),
                max_new_tokens=int(args.max_new_tokens),
                length_penalty=float(args.length_penalty),
            )
            for offset, out in enumerate(batch_outputs):
                outputs_by_index[start + offset] = out

    if any(o is None for o in outputs_by_index):
        missing = [i for i, o in enumerate(outputs_by_index) if o is None]
        raise RuntimeError(f"Internal error: missing predictions for indices: {missing[:16]}")

    outputs: list[list[str]] = [o for o in outputs_by_index if o is not None]
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]

    for i in range(len(test_data)):
        if "dedup" in test_data[i]:
            test_data[i].pop("dedup")

    out_path = Path(str(args.result_json_data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(test_data, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()

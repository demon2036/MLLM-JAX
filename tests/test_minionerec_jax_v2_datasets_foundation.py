from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_datasets = _load_module("_minionerec_jax_v2_datasets_under_test", "projects/minionerec_jax_v2/datasets.py")

OFFICIAL_INSTRUCTION = _datasets.OFFICIAL_INSTRUCTION
OfficialEvalSidDataset = _datasets.OfficialEvalSidDataset
OfficialTokenizerAdapter = _datasets.OfficialTokenizerAdapter
build_official_eval_prompt = _datasets.build_official_eval_prompt


class _TokenizerWithSpecials:
    bos_token_id = 101
    eos_token_id = 102

    def encode(self, text: str) -> list[int]:
        if text == OFFICIAL_INSTRUCTION:
            return [self.bos_token_id, 11, 12, self.eos_token_id]
        if text.startswith("### User Input:"):
            return [self.bos_token_id, 21, 22, self.eos_token_id]
        raise AssertionError(f"Unexpected text passed to encode: {text!r}")


class _TokenizerFixed:
    bos_token_id = 101
    eos_token_id = 102

    def encode(self, text: str) -> list[int]:
        return [101, 101, 7, 8, 102, 102]


def test_build_official_eval_prompt_matches_expected_official_format() -> None:
    prompt = build_official_eval_prompt(["sid-001", "sid-002"])
    expected = (
        "### User Input: \n"
        "Can you predict the next possible item the user may expect, "
        "given the following chronological interaction history: sid-001, sid-002\n\n"
        "### Response:\n"
    )
    assert prompt == expected


def test_official_tokenizer_adapter_handles_bos_eos_flags() -> None:
    adapter = OfficialTokenizerAdapter(_TokenizerFixed())

    assert adapter.encode("x", bos=False, eos=False) == [7, 8]
    assert adapter.encode("x", bos=True, eos=False) == [101, 7, 8]
    assert adapter.encode("x", bos=False, eos=True) == [7, 8, 102]
    assert adapter.encode("x", bos=True, eos=True) == [101, 7, 8, 102]


def test_eval_dataset_extracts_targets_prompts_and_prompt_only_tokens(tmp_path) -> None:
    csv_path = tmp_path / "eval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["history_item_sid", "item_sid"])
        writer.writeheader()
        writer.writerow({"history_item_sid": "['sid-1', 'sid-2']", "item_sid": "sid-3"})
        writer.writerow({"history_item_sid": "['dup-sid']", "item_sid": "dup-sid"})
        writer.writerow({"history_item_sid": "not-a-list", "item_sid": ""})

    dataset = OfficialEvalSidDataset(
        csv_path=str(csv_path),
        tokenizer=_TokenizerWithSpecials(),
        dedup=True,
        pretokenize=True,
        max_len=256,
    )

    assert len(dataset) == 2
    assert dataset.get_targets() == ["sid-3", ""]
    assert dataset.get_prompts() == [
        build_official_eval_prompt(["sid-1", "sid-2"]),
        build_official_eval_prompt([]),
    ]

    encoded = dataset[0]
    assert encoded["input_ids"] == [101, 11, 12, 21, 22]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]

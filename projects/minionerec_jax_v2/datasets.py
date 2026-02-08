from __future__ import annotations

import ast
import csv
import random
from dataclasses import dataclass
from typing import Any


OFFICIAL_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""


def _read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _sample_rows(rows: list[dict[str, str]], *, sample: int, seed: int) -> list[dict[str, str]]:
    if sample <= 0 or sample >= len(rows):
        return rows
    sampler = random.Random(seed)
    return sampler.sample(rows, sample)


def _safe_parse_history_item_sid(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(token) for token in parsed]


def _build_official_user_input(history_item_sid: list[str]) -> str:
    history = ", ".join(history_item_sid)
    return (
        "Can you predict the next possible item the user may expect, "
        f"given the following chronological interaction history: {history}"
    )


def build_official_eval_prompt(history_item_sid: list[str]) -> str:
    user_input = _build_official_user_input(history_item_sid)
    return f"### User Input: \n{user_input}\n\n### Response:\n"


class OfficialTokenizerAdapter:
    """Tokenization aligned with upstream MiniOneRec Tokenizer.encode behavior."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._bos_id = getattr(tokenizer, "bos_token_id", None)
        self._eos_id = getattr(tokenizer, "eos_token_id", None)

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        token_ids = [int(token_id) for token_id in self._tokenizer.encode(text)]

        if self._bos_id is not None:
            bos_id = int(self._bos_id)
            while token_ids and token_ids[0] == bos_id:
                token_ids = token_ids[1:]

        if self._eos_id is not None:
            eos_id = int(self._eos_id)
            while token_ids and token_ids[-1] == eos_id:
                token_ids = token_ids[:-1]

        if bos and self._bos_id is not None:
            token_ids = [int(self._bos_id)] + token_ids
        if eos and self._eos_id is not None:
            token_ids = token_ids + [int(self._eos_id)]

        return token_ids


@dataclass(frozen=True)
class OfficialEvalSidExample:
    prompt: str
    target_sid: str


class OfficialEvalSidDataset:
    """Eval dataset with official prompt + prompt-only tokenization outputs."""

    def __init__(
        self,
        *,
        csv_path: str,
        tokenizer: Any,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 0,
        dedup: bool = False,
        pretokenize: bool = True,
        truncate_to_max_len: bool = False,
    ):
        rows = _read_csv_rows(csv_path)
        rows = _sample_rows(rows, sample=int(sample), seed=int(seed))

        self._max_len = int(max_len)
        if self._max_len <= 0:
            raise ValueError("max_len must be > 0")

        self._truncate_to_max_len = bool(truncate_to_max_len)
        self._tokenizer = OfficialTokenizerAdapter(tokenizer)

        examples: list[OfficialEvalSidExample] = []
        for row in rows:
            history_item_sid = _safe_parse_history_item_sid(row.get("history_item_sid"))
            target_sid = str(row.get("item_sid") or "")

            if dedup and history_item_sid and target_sid == history_item_sid[-1]:
                continue

            prompt = build_official_eval_prompt(history_item_sid)
            examples.append(OfficialEvalSidExample(prompt=prompt, target_sid=target_sid))

        self._examples = examples
        self._encoded: list[dict[str, list[int]]] | None = None
        if pretokenize:
            self._encoded = [self._encode_prompt_only(example.prompt) for example in self._examples]

    def _encode_prompt_only(self, prompt: str) -> dict[str, list[int]]:
        token_ids = self._tokenizer.encode(OFFICIAL_INSTRUCTION, bos=True, eos=False)
        token_ids = token_ids + self._tokenizer.encode(prompt, bos=False, eos=False)

        if self._truncate_to_max_len:
            token_ids = token_ids[-self._max_len :]

        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode_prompt_only(self._examples[idx].prompt)

    def get_targets(self) -> list[str]:
        return [example.target_sid for example in self._examples]

    def get_prompts(self) -> list[str]:
        return [example.prompt for example in self._examples]


__all__ = [
    "OFFICIAL_INSTRUCTION",
    "OfficialEvalSidDataset",
    "OfficialEvalSidExample",
    "OfficialTokenizerAdapter",
    "build_official_eval_prompt",
]


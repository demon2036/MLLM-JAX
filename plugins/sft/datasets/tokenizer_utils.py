from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TokenizerAdapter:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

    def encode(self, text: str, *, bos: bool, eos: bool) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        # Avoid model-specific default special token behavior.
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if bos and self.bos_id is not None:
            token_ids = [int(self.bos_id)] + token_ids
        if eos and self.eos_id is not None:
            token_ids = token_ids + [int(self.eos_id)]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids))


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int] | None = None

    def as_trainer_batch(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        if self.labels is not None:
            out["labels"] = self.labels
        return out


def encode_supervised_example(
    *,
    tokenizer: TokenizerAdapter,
    instruction: str,
    prompt: str,
    completion: str,
    max_len: int,
    include_labels: bool,
) -> EncodedExample:
    if int(max_len) <= 0:
        raise ValueError("max_len must be > 0")

    tokens = tokenizer.encode(instruction, bos=True, eos=False) + tokenizer.encode(prompt, bos=False, eos=False)
    prompt_len = len(tokens)
    attention_mask = [1] * prompt_len

    if not include_labels:
        return EncodedExample(
            input_ids=tokens[-int(max_len) :],
            attention_mask=attention_mask[-int(max_len) :],
            labels=None,
        )

    completion_ids = tokenizer.encode(completion, bos=False, eos=True)
    tokens = tokens + completion_ids
    attention_mask = [1] * len(tokens)
    labels = [-100] * prompt_len + tokens[prompt_len:]

    return EncodedExample(
        input_ids=tokens[-int(max_len) :],
        attention_mask=attention_mask[-int(max_len) :],
        labels=labels[-int(max_len) :],
    )


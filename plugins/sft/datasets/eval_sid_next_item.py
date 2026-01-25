from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.sft.datasets.csv_utils import parse_sid_example, read_csv_rows, sample_rows
from plugins.sft.datasets.tokenizer_utils import TokenizerAdapter, encode_supervised_example


_INSTRUCTION_NEXT_ITEM = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""


def _format_prompt(history_item_sid: list[str]) -> str:
    history = ", ".join(history_item_sid)
    user_input = f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}"
    return f"""### User Input: 
{user_input}

### Response:\n"""


@dataclass(frozen=True)
class SidNextItemEvalExample:
    prompt: str
    target_sid: str


class SidNextItemEvalDataset:
    """Eval dataset: prompt-only encodings + ground-truth SID (for HR/NDCG)."""

    def __init__(
        self,
        *,
        csv_path: str,
        tokenizer: Any,
        max_len: int,
        sample: int = -1,
        seed: int = 0,
        pretokenize: bool = True,
        dedup: bool = False,
    ):
        rows = read_csv_rows(csv_path)
        rows = sample_rows(rows, sample=sample, seed=seed)

        self._tokenizer = TokenizerAdapter(tokenizer)
        self._max_len = int(max_len)

        examples: list[SidNextItemEvalExample] = []
        for row in rows:
            ex = parse_sid_example(row)
            if not ex.history_item_sid:
                continue
            if dedup and ex.item_sid == ex.history_item_sid[-1]:
                continue
            examples.append(SidNextItemEvalExample(prompt=_format_prompt(ex.history_item_sid), target_sid=ex.item_sid))

        self._examples = examples
        self._encoded: list[dict[str, list[int]]] | None = None
        if pretokenize:
            self._encoded = [self._encode_prompt_only(e) for e in self._examples]

    def _encode_prompt_only(self, example: SidNextItemEvalExample) -> dict[str, list[int]]:
        return encode_supervised_example(
            tokenizer=self._tokenizer,
            instruction=_INSTRUCTION_NEXT_ITEM,
            prompt=example.prompt,
            completion="",
            max_len=self._max_len,
            include_labels=False,
        ).as_trainer_batch()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode_prompt_only(self._examples[idx])

    def get_targets(self) -> list[str]:
        return [e.target_sid for e in self._examples]

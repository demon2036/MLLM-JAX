from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.sft.datasets.csv_utils import parse_sid_example, read_csv_rows, sample_rows
from plugins.sft.datasets.tokenizer_utils import TokenizerAdapter, encode_supervised_example


_INSTRUCTION_NEXT_ITEM = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""


def _format_prompt_next_item(history_item_sid: list[str]) -> str:
    history = ", ".join(history_item_sid)
    user_input = (
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    return f"""### User Input: 
{user_input}

### Response:\n"""


@dataclass(frozen=True)
class SidNextItemSftExample:
    prompt: str
    completion: str


class SidNextItemSftDataset:
    """Next-item prediction in SID space (MiniOneRec SFT core task)."""

    def __init__(
        self,
        *,
        csv_path: str,
        tokenizer: Any,
        max_len: int,
        sample: int = -1,
        seed: int = 0,
        include_labels: bool = True,
        pretokenize: bool = True,
        dedup: bool = False,
    ):
        rows = read_csv_rows(csv_path)
        rows = sample_rows(rows, sample=sample, seed=seed)

        self._tokenizer = TokenizerAdapter(tokenizer)
        self._max_len = int(max_len)
        self._include_labels = bool(include_labels)

        examples: list[SidNextItemSftExample] = []
        for row in rows:
            ex = parse_sid_example(row)
            if not ex.history_item_sid:
                continue
            if dedup and ex.item_sid == ex.history_item_sid[-1]:
                continue
            prompt = _format_prompt_next_item(ex.history_item_sid)
            completion = f"{ex.item_sid}\n"
            examples.append(SidNextItemSftExample(prompt=prompt, completion=completion))

        self._examples = examples
        self._encoded: list[dict[str, list[int]]] | None = None
        if pretokenize:
            self._encoded = [self._encode(e).as_trainer_batch() for e in self._examples]

    def _encode(self, example: SidNextItemSftExample):
        return encode_supervised_example(
            tokenizer=self._tokenizer,
            instruction=_INSTRUCTION_NEXT_ITEM,
            prompt=example.prompt,
            completion=example.completion,
            max_len=self._max_len,
            include_labels=self._include_labels,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode(self._examples[idx]).as_trainer_batch()

    def get_prompts_and_targets(self) -> tuple[list[str], list[str]]:
        prompts = [e.prompt for e in self._examples]
        targets = [e.completion for e in self._examples]
        return prompts, targets

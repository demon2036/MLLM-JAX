from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from projects.minionerec.sft.datasets.tokenizer_utils import TokenizerAdapter, encode_supervised_example


_INSTRUCTION_ITEM_ID = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""


TaskName = Literal["sid2title", "title2sid"]


@dataclass(frozen=True)
class SidItemAlignmentExample:
    task: TaskName
    prompt: str
    completion: str


def _combine_sid_tokens(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return f"{tokens[0]}{tokens[1]}{tokens[2]}"


def _format_prompt(task: TaskName, *, text: str) -> str:
    if task == "title2sid":
        user_input = f"Which item has the title: {text}?"
    else:
        user_input = f'What is the title of item "{text}"?'
    return f"""### User Input: 
{user_input}

### Response:\n"""


class SidItemAlignmentDataset:
    """SID↔title alignment tasks used during SFT (language grounding)."""

    def __init__(
        self,
        *,
        item_meta_path: str,
        sid_index_path: str,
        tokenizer: Any,
        max_len: int,
        sample: int = -1,
        seed: int = 0,
        include_labels: bool = True,
        pretokenize: bool = True,
    ):
        rng = random.Random(int(seed))
        item_feat = json.loads(Path(item_meta_path).read_text())
        indices = json.loads(Path(sid_index_path).read_text())

        sid2title: dict[str, str] = {}
        title2sid: dict[str, str] = {}
        for item_id, sid_tokens in indices.items():
            if item_id not in item_feat:
                continue
            combined_sid = _combine_sid_tokens([str(x) for x in sid_tokens])
            if not combined_sid:
                continue
            title = str(item_feat[item_id].get("title") or "")
            sid2title[combined_sid] = title
            title2sid[title] = combined_sid

        examples: list[SidItemAlignmentExample] = []
        for sid, title in sid2title.items():
            examples.append(SidItemAlignmentExample(task="sid2title", prompt=_format_prompt("sid2title", text=sid), completion=title + "\n"))
        for title, sid in title2sid.items():
            examples.append(SidItemAlignmentExample(task="title2sid", prompt=_format_prompt("title2sid", text=title), completion=sid + "\n"))

        if int(sample) > 0 and int(sample) < len(examples):
            examples = rng.sample(examples, int(sample))

        self._tokenizer = TokenizerAdapter(tokenizer)
        self._max_len = int(max_len)
        self._include_labels = bool(include_labels)
        self._examples = examples
        self._encoded: list[dict[str, list[int]]] | None = None
        if pretokenize:
            self._encoded = [self._encode(e) for e in self._examples]

    def _encode(self, example: SidItemAlignmentExample) -> dict[str, list[int]]:
        return encode_supervised_example(
            tokenizer=self._tokenizer,
            instruction=_INSTRUCTION_ITEM_ID,
            prompt=example.prompt,
            completion=example.completion,
            max_len=self._max_len,
            include_labels=self._include_labels,
        ).as_trainer_batch()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode(self._examples[idx])

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from plugins.sft.datasets.csv_utils import parse_sid_example, read_csv_rows, sample_rows
from plugins.sft.datasets.tokenizer_utils import TokenizerAdapter, encode_supervised_example


_INSTRUCTION_FUSION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""


def _combine_sid_tokens(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return f"{tokens[0]}{tokens[1]}{tokens[2]}"


def _format_prompt(history_item_sid: list[str]) -> str:
    history = ", ".join(history_item_sid)
    user_input = (
        f"The user has sequentially interacted with items {history}. "
        "Can you recommend the next item for him? Tell me the title of the item"
    )
    return f"""### User Input: 
{user_input}

### Response:\n"""


@dataclass(frozen=True)
class FusionSeqRecExample:
    prompt: str
    completion: str


class FusionSeqRecSftDataset(Dataset):
    """Auxiliary task: history SIDs -> next item title (language alignment)."""

    def __init__(
        self,
        *,
        csv_path: str,
        item_meta_path: str,
        sid_index_path: str,
        tokenizer: Any,
        max_len: int,
        sample: int = -1,
        seed: int = 0,
        include_labels: bool = True,
        pretokenize: bool = True,
        dedup: bool = False,
    ):
        rng = random.Random(int(seed))

        item_feat = json.loads(Path(item_meta_path).read_text())
        indices = json.loads(Path(sid_index_path).read_text())

        sid2title: dict[str, str] = {}
        for item_id, sid_tokens in indices.items():
            if item_id not in item_feat:
                continue
            combined_sid = _combine_sid_tokens([str(x) for x in sid_tokens])
            if not combined_sid:
                continue
            sid2title[combined_sid] = str(item_feat[item_id].get("title") or combined_sid)

        rows = read_csv_rows(csv_path)
        rows = sample_rows(rows, sample=sample, seed=seed)

        examples: list[FusionSeqRecExample] = []
        for row in rows:
            ex = parse_sid_example(row)
            if not ex.history_item_sid:
                continue
            if dedup and ex.item_sid == ex.history_item_sid[-1]:
                continue
            target_title = sid2title.get(ex.item_sid, ex.item_sid)
            examples.append(FusionSeqRecExample(prompt=_format_prompt(ex.history_item_sid), completion=target_title + "\n"))

        if int(sample) > 0 and int(sample) < len(examples):
            examples = rng.sample(examples, int(sample))

        self._tokenizer = TokenizerAdapter(tokenizer)
        self._max_len = int(max_len)
        self._include_labels = bool(include_labels)
        self._examples = examples
        self._encoded: list[dict[str, list[int]]] | None = None
        if pretokenize:
            self._encoded = [self._encode(e) for e in self._examples]

    def _encode(self, example: FusionSeqRecExample) -> dict[str, list[int]]:
        return encode_supervised_example(
            tokenizer=self._tokenizer,
            instruction=_INSTRUCTION_FUSION,
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


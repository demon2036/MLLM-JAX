from __future__ import annotations

from dataclasses import dataclass

from projects.minionerec.sft.datasets.csv_utils import parse_sid_example, read_csv_rows, sample_rows


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
class MiniOneRecNextItemRlExample:
    prompt: str
    target_sid: str


class MiniOneRecNextItemRlDataset:
    """RL dataset for SID next-item prediction (prompt-only + target SID)."""

    def __init__(
        self,
        *,
        csv_path: str,
        sample: int = -1,
        seed: int = 0,
        dedup: bool = False,
    ):
        rows = read_csv_rows(csv_path)
        rows = sample_rows(rows, sample=sample, seed=seed)

        examples: list[MiniOneRecNextItemRlExample] = []
        for row in rows:
            ex = parse_sid_example(row)
            if not ex.history_item_sid:
                continue
            if dedup and ex.item_sid == ex.history_item_sid[-1]:
                continue
            examples.append(MiniOneRecNextItemRlExample(prompt=_format_prompt_next_item(ex.history_item_sid), target_sid=ex.item_sid))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecNextItemRlExample:
        return self._examples[idx]


__all__ = ["MiniOneRecNextItemRlDataset", "MiniOneRecNextItemRlExample"]


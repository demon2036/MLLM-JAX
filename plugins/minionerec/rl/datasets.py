from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path

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
class MiniOneRecRlExample:
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

        examples: list[MiniOneRecRlExample] = []
        for row in rows:
            ex = parse_sid_example(row)
            if not ex.history_item_sid:
                continue
            if dedup and ex.item_sid == ex.history_item_sid[-1]:
                continue
            examples.append(MiniOneRecRlExample(prompt=_format_prompt_next_item(ex.history_item_sid), target_sid=ex.item_sid))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]

def _format_prompt_user_input(user_input: str) -> str:
    return f"""### User Input: 
{user_input}

### Response:\n"""


def _combine_sid_tokens(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return f"{tokens[0]}{tokens[1]}{tokens[2]}"


class MiniOneRecTitle2SidRlDataset:
    """RL dataset for title2sid + description2sid alignment tasks (prompt-only + target SID)."""

    def __init__(
        self,
        *,
        item_meta_path: str,
        sid_index_path: str,
        include_title: bool = True,
        include_description: bool = True,
        sample: int = -1,
        seed: int = 0,
    ):
        rng = random.Random(int(seed))

        item_feat = json.loads(Path(item_meta_path).read_text())
        indices = json.loads(Path(sid_index_path).read_text())

        title2sid: dict[str, str] = {}
        description2sid: dict[str, str] = {}
        for item_id, sid_tokens in indices.items():
            item = item_feat.get(str(item_id))
            if not isinstance(item, dict):
                continue
            sid = _combine_sid_tokens([str(x) for x in sid_tokens])
            if not sid:
                continue
            title = str(item.get("title") or "")
            if title:
                title2sid[title] = sid

            desc = item.get("description") or ""
            if isinstance(desc, str) and desc.startswith("['") and desc.endswith("']"):
                try:
                    parsed = ast.literal_eval(desc)
                    if isinstance(parsed, list) and parsed:
                        desc = parsed[0]
                except Exception:
                    pass
            desc = str(desc or "")
            if desc:
                description2sid[desc] = sid

        examples: list[MiniOneRecRlExample] = []
        if bool(include_title):
            for title, sid in title2sid.items():
                user_input = f"Which item has the title: {title}?"
                examples.append(MiniOneRecRlExample(prompt=_format_prompt_user_input(user_input), target_sid=sid))
        if bool(include_description):
            for desc, sid in description2sid.items():
                user_input = f'An item can be described as follows: "{desc}". Which item is it describing?'
                examples.append(MiniOneRecRlExample(prompt=_format_prompt_user_input(user_input), target_sid=sid))

        if int(sample) > 0 and int(sample) < len(examples):
            examples = rng.sample(examples, int(sample))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]


class MiniOneRecSeqTitle2SidRlDataset:
    """RL dataset for sequential title prompts -> next SID target."""

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

        examples: list[MiniOneRecRlExample] = []
        for row in rows:
            titles_raw = row.get("history_item_title")
            if titles_raw is None:
                continue
            try:
                history_titles = ast.literal_eval(str(titles_raw))
            except Exception:
                continue
            if not isinstance(history_titles, list) or not history_titles:
                continue

            target_sid = str(row.get("item_sid") or "").strip()
            if not target_sid:
                continue

            if dedup:
                try:
                    history_item_id = ast.literal_eval(str(row.get("history_item_id") or "[]"))
                    target_item_id = str(row.get("item_id") or "")
                    if isinstance(history_item_id, list) and history_item_id:
                        if str(history_item_id[-1]) == target_item_id:
                            continue
                except Exception:
                    pass

            inter_titles = ", ".join([f"\"{t}\"" for t in history_titles])
            user_input = (
                "Given the title sequence of user historical interactive items: "
                f"{inter_titles}, can you recommend a suitable next item for the user?"
            )
            examples.append(MiniOneRecRlExample(prompt=_format_prompt_user_input(user_input), target_sid=target_sid))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]


__all__ = [
    "MiniOneRecRlExample",
    "MiniOneRecNextItemRlDataset",
    "MiniOneRecTitle2SidRlDataset",
    "MiniOneRecSeqTitle2SidRlDataset",
]

from __future__ import annotations

import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_python_literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _sample_rows(rows: list[dict[str, str]], *, sample: int, seed: int) -> list[dict[str, str]]:
    if sample is None or int(sample) <= 0 or int(sample) >= len(rows):
        return rows
    rng = random.Random(int(seed))
    return rng.sample(rows, int(sample))


def _combine_sid_tokens(tokens: list[str]) -> str | None:
    if len(tokens) < 3:
        return None
    return f"{tokens[0]}{tokens[1]}{tokens[2]}"


def _format_prompt(prompt: str) -> str:
    return f"""### User Input: 
{prompt}

### Response:\n"""


def _format_prompt_next_item(history_item_sid: list[str]) -> str:
    history = ", ".join(history_item_sid)
    user_input = (
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )
    return _format_prompt(user_input)


def _format_prompt_title2sid(title: str) -> str:
    return _format_prompt(f"Which item has the title: {title}?")


def _format_prompt_description2sid(description: str) -> str:
    return _format_prompt(f'An item can be described as follows: "{description}". Which item is it describing?')


def _format_prompt_seqtitle2sid(inter_titles: str) -> str:
    prompt = f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
    return _format_prompt(prompt)


@dataclass(frozen=True)
class MiniOneRecRlExample:
    prompt: str
    target_sid: str


class MiniOneRecRlDataset:
    def __init__(self, examples: list[MiniOneRecRlExample]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]


def build_next_item_examples(
    *,
    csv_path: str,
    sample: int,
    seed: int,
    dedup: bool = True,
) -> list[MiniOneRecRlExample]:
    rows = _read_csv_rows(csv_path)
    rows = _sample_rows(rows, sample=sample, seed=seed)

    examples: list[MiniOneRecRlExample] = []
    for row in rows:
        history_raw = row.get("history_item_sid") or "[]"
        history = _parse_python_literal(history_raw)
        if not isinstance(history, list):
            continue
        history_item_sid = [str(x) for x in history]
        target_sid = str(row.get("item_sid") or "")
        if not history_item_sid or not target_sid:
            continue
        if dedup and target_sid == history_item_sid[-1]:
            continue
        examples.append(MiniOneRecRlExample(prompt=_format_prompt_next_item(history_item_sid), target_sid=target_sid))
    return examples


def build_title_and_description_examples(
    *,
    item_meta_path: str,
    sid_index_path: str,
) -> tuple[list[MiniOneRecRlExample], dict[str, str]]:
    """Return (examples, sid2title) for the alignment-style RL tasks."""
    item_feat = json.loads(Path(item_meta_path).read_text())
    indices = json.loads(Path(sid_index_path).read_text())

    sid2title: dict[str, str] = {}
    title2sid: dict[str, str] = {}
    description2sid: dict[str, str] = {}

    for item_id, sid_tokens in indices.items():
        if item_id not in item_feat:
            continue
        combined_sid = _combine_sid_tokens([str(x) for x in sid_tokens])
        if not combined_sid:
            continue
        title = str(item_feat[item_id].get("title") or "")
        if title:
            sid2title[combined_sid] = title
            title2sid[title] = combined_sid

        description = item_feat[item_id].get("description") or ""
        if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
            parsed = _parse_python_literal(description)
            if isinstance(parsed, list) and parsed:
                description = parsed[0]
        description = str(description)
        if description:
            description2sid[description] = combined_sid

    examples: list[MiniOneRecRlExample] = []
    for title, sid in title2sid.items():
        examples.append(MiniOneRecRlExample(prompt=_format_prompt_title2sid(title), target_sid=sid))
    for description, sid in description2sid.items():
        examples.append(MiniOneRecRlExample(prompt=_format_prompt_description2sid(description), target_sid=sid))
    return examples, sid2title


def build_seqtitle2sid_examples(
    *,
    csv_path: str,
    sid2title: dict[str, str],
    sample: int,
    seed: int,
    dedup: bool = True,
) -> list[MiniOneRecRlExample]:
    rows = _read_csv_rows(csv_path)
    rows = _sample_rows(rows, sample=sample, seed=seed)

    examples: list[MiniOneRecRlExample] = []
    for row in rows:
        history_raw = row.get("history_item_sid") or "[]"
        history = _parse_python_literal(history_raw)
        if not isinstance(history, list):
            continue
        history_item_sid = [str(x) for x in history]
        target_sid = str(row.get("item_sid") or "")
        if not history_item_sid or not target_sid:
            continue
        if dedup and target_sid == history_item_sid[-1]:
            continue
        history_titles = [sid2title.get(sid, sid) for sid in history_item_sid]
        inter_titles = ", ".join([f'"{t}"' for t in history_titles])
        examples.append(MiniOneRecRlExample(prompt=_format_prompt_seqtitle2sid(inter_titles), target_sid=target_sid))
    return examples


__all__ = [
    "MiniOneRecRlDataset",
    "MiniOneRecRlExample",
    "build_next_item_examples",
    "build_title_and_description_examples",
    "build_seqtitle2sid_examples",
]


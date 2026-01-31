from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from projects.sid_sft.datasets.csv_utils import parse_python_literal, parse_sid_example, read_csv_rows, sample_rows


def _format_prompt(user_input: str) -> str:
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
            history = ", ".join(ex.history_item_sid)
            user_input = (
                f"The user has interacted with items {history} in chronological order. "
                "Can you predict the next possible item that the user may expect?"
            )
            examples.append(MiniOneRecRlExample(prompt=_format_prompt(user_input), target_sid=ex.item_sid))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]

    @property
    def examples(self) -> list[MiniOneRecRlExample]:
        return self._examples


class MiniOneRecTitle2SidRlDataset:
    """RL dataset for title/description -> SID prediction (uses item_meta + sid index)."""

    def __init__(
        self,
        *,
        item_meta_path: str,
        sid_index_path: str,
        sample: int = -1,
        seed: int = 0,
        include_description: bool = True,
    ):
        with open(item_meta_path, "r", encoding="utf-8") as f:
            item_feat = json.load(f)
        with open(sid_index_path, "r", encoding="utf-8") as f:
            indices = json.load(f)

        data: list[dict[str, Any]] = []
        for item_id, sids in indices.items():
            feat = item_feat.get(item_id)
            if not isinstance(feat, dict):
                continue
            if not isinstance(sids, list) or len(sids) < 3:
                continue

            combined_sid = "".join(str(x) for x in sids[:3])
            title = feat.get("title")
            if isinstance(title, str) and title.strip():
                data.append({"task": "title2sid", "input": title.strip(), "output": combined_sid})

            if include_description:
                description = feat.get("description")
                if isinstance(description, str):
                    # The upstream dataset sometimes stores a Python-literal list in a string.
                    parsed = parse_python_literal(description)
                    if isinstance(parsed, list) and parsed:
                        description = str(parsed[0])
                if isinstance(description, str) and description.strip():
                    data.append({"task": "description2sid", "input": description.strip(), "output": combined_sid})

        data = sample_rows(data, sample=sample, seed=seed)

        examples: list[MiniOneRecRlExample] = []
        for dp in data:
            task = str(dp.get("task") or "")
            inp = str(dp.get("input") or "")
            out = str(dp.get("output") or "")
            if not inp or not out:
                continue
            if task == "title2sid":
                user_input = f"Which item has the title: {inp}?"
            else:
                user_input = f"An item can be described as follows: \"{inp}\". Which item is it describing?"
            examples.append(MiniOneRecRlExample(prompt=_format_prompt(user_input), target_sid=out))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]


class MiniOneRecSeqTitle2SidRlDataset:
    """RL dataset for title-sequence -> next SID prediction (from the train CSV)."""

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
            hist_titles_raw = row.get("history_item_title") or "[]"
            hist_titles = parse_python_literal(hist_titles_raw)
            if not isinstance(hist_titles, list):
                continue
            history_item_title = [str(x) for x in hist_titles if str(x).strip()]
            if not history_item_title:
                continue

            target_sid = str(row.get("item_sid") or "").strip()
            if not target_sid:
                continue

            if dedup:
                last_title = history_item_title[-1]
                # We don't have an item_title column consistently across files, so
                # keep this heuristic minimal.
                if str(row.get("item_title") or "").strip() == last_title:
                    continue

            inter_titles = ", ".join([f'"{t}"' for t in history_item_title])
            user_input = (
                f"Given the title sequence of user historical interactive items: {inter_titles}, "
                "can you recommend a suitable next item for the user?"
            )
            examples.append(MiniOneRecRlExample(prompt=_format_prompt(user_input), target_sid=target_sid))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        return self._examples[idx]


class MiniOneRecMixedRlDataset:
    """Concat-like dataset that preserves `.prompt`/`.target_sid` contract."""

    def __init__(self, datasets: list[Any]):
        datasets = [d for d in (datasets or []) if d is not None and len(d) > 0]
        if not datasets:
            raise ValueError("Empty datasets")

        offsets: list[int] = [0]
        total = 0
        for d in datasets:
            total += int(len(d))
            offsets.append(total)

        self._datasets = datasets
        self._offsets = offsets
        self._total = total

    def __len__(self) -> int:
        return int(self._total)

    def __getitem__(self, idx: int) -> MiniOneRecRlExample:
        idx = int(idx)
        if idx < 0 or idx >= int(self._total):
            raise IndexError(idx)
        # Linear scan is OK for small number of datasets (3-4).
        for di in range(len(self._datasets)):
            start = int(self._offsets[di])
            end = int(self._offsets[di + 1])
            if start <= idx < end:
                return self._datasets[di][idx - start]
        raise IndexError(idx)

__all__ = [
    "MiniOneRecRlExample",
    "MiniOneRecNextItemRlDataset",
    "MiniOneRecTitle2SidRlDataset",
    "MiniOneRecSeqTitle2SidRlDataset",
    "MiniOneRecMixedRlDataset",
]

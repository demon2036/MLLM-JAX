from __future__ import annotations

import ast
import csv
import random
from dataclasses import dataclass
from typing import Any, Iterable


def parse_python_literal(text: str) -> Any:
    """Parse a Python literal safely (no eval).

    The MiniOneRec data files often store lists as stringified Python literals
    (e.g. "['<a_1><b_2><c_3>', ...]" or "[1, 2, 3]").
    """
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def sample_rows(rows: list[dict[str, str]], *, sample: int, seed: int) -> list[dict[str, str]]:
    if sample is None or int(sample) <= 0 or int(sample) >= len(rows):
        return rows
    rng = random.Random(int(seed))
    return rng.sample(rows, int(sample))


@dataclass(frozen=True)
class MiniOneRecSidExample:
    user_id: str
    history_item_sid: list[str]
    item_sid: str


def parse_sid_example(row: dict[str, str]) -> MiniOneRecSidExample:
    user_id = str(row.get("user_id") or "")
    history_raw = row.get("history_item_sid") or "[]"
    history = parse_python_literal(history_raw)
    if not isinstance(history, list):
        history = []
    history_item_sid = [str(x) for x in history]
    item_sid = str(row.get("item_sid") or "")
    return MiniOneRecSidExample(user_id=user_id, history_item_sid=history_item_sid, item_sid=item_sid)


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from projects.sid_sft.datasets.tokenizer_utils import TokenizerAdapter


@dataclass(frozen=True)
class OpenOneRecPromptGroundTruth:
    task_name: str
    sample_id: str
    prompt: str
    ground_truth: str


@dataclass(frozen=True)
class EncodedSftExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int] | None

    def as_trainer_batch(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
        }
        if self.labels is not None:
            out["labels"] = list(self.labels)
        return out


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def extract_prompt_ground_truth_pairs(
    *,
    task_name: str,
    loader_samples: Mapping[str, Mapping[str, Any]],
) -> list[OpenOneRecPromptGroundTruth]:
    pairs: list[OpenOneRecPromptGroundTruth] = []
    for sample_id, sample in loader_samples.items():
        prompt = _safe_str(sample.get("prompt"))
        ground_truth = _safe_str(sample.get("ground_truth"))
        if not prompt.strip() or not ground_truth.strip():
            continue
        pairs.append(
            OpenOneRecPromptGroundTruth(
                task_name=str(task_name),
                sample_id=str(sample_id),
                prompt=prompt,
                ground_truth=ground_truth,
            )
        )
    return pairs


def aggregate_loader_pairs(
    *,
    task_types: Sequence[str],
    task_samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[OpenOneRecPromptGroundTruth]:
    out: list[OpenOneRecPromptGroundTruth] = []
    for task_name in task_types:
        samples = task_samples.get(str(task_name), {})
        out.extend(extract_prompt_ground_truth_pairs(task_name=str(task_name), loader_samples=samples))
    return out


class OpenOneRecSftDataset:
    """SFT dataset built from OpenOneRec loader prompt+ground_truth outputs."""

    def __init__(
        self,
        *,
        pairs: Sequence[OpenOneRecPromptGroundTruth],
        tokenizer: Any,
        max_len: int,
        include_labels: bool = True,
        pretokenize: bool = True,
    ):
        max_len_i = int(max_len)
        if max_len_i <= 0:
            raise ValueError("max_len must be > 0")

        self._pairs = list(pairs)
        self._tokenizer = TokenizerAdapter(tokenizer)
        self._max_len = max_len_i
        self._include_labels = bool(include_labels)
        self._eos_id = getattr(tokenizer, "eos_token_id", None)

        self._encoded: list[dict[str, list[int]]] | None = None
        if bool(pretokenize):
            self._encoded = [self._encode_pair(pair).as_trainer_batch() for pair in self._pairs]

    def _truncate_tail(self, values: list[int]) -> list[int]:
        return values[-int(self._max_len) :]

    def _encode_pair(self, pair: OpenOneRecPromptGroundTruth) -> EncodedSftExample:
        prompt_ids = self._tokenizer.encode(pair.prompt, bos=True, eos=False)
        completion_ids = self._tokenizer.encode(pair.ground_truth, bos=False, eos=False)

        if self._eos_id is not None:
            completion_ids = completion_ids + [int(self._eos_id)]

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        labels: list[int] | None = None
        if self._include_labels:
            labels = ([-100] * len(prompt_ids)) + completion_ids

        input_ids = self._truncate_tail(input_ids)
        attention_mask = self._truncate_tail(attention_mask)
        if labels is not None:
            labels = self._truncate_tail(labels)

        return EncodedSftExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        if self._encoded is not None:
            return self._encoded[idx]
        return self._encode_pair(self._pairs[idx]).as_trainer_batch()

    def iter_pairs(self) -> Iterable[OpenOneRecPromptGroundTruth]:
        return iter(self._pairs)


__all__ = [
    "EncodedSftExample",
    "OpenOneRecPromptGroundTruth",
    "OpenOneRecSftDataset",
    "aggregate_loader_pairs",
    "extract_prompt_ground_truth_pairs",
]

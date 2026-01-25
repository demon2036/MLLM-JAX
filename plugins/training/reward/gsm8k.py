from __future__ import annotations

import re
from typing import Any, Mapping

from math_verify import ExprExtractionConfig, parse, verify


def reward_correct(item: Mapping[str, Any], answer: str, **kwargs: Any) -> float:
    del kwargs
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, answer)
    if len(nums) == 0:
        return 0.0
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(str(item.get("A", "")), extraction_config=[ExprExtractionConfig()])
    return 1.0 if verify(ans, ground_truth) else 0.0


def reward_format(item: Mapping[str, Any], answer: str, **kwargs: Any) -> float:
    del item, kwargs
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
    match = re.match(pattern, answer, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def tag_count_reward(item: Mapping[str, Any], answer: str, **kwargs: Any) -> float:
    """Reward based on producing the desired tag count for `reward_format()`."""
    del item, kwargs

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        if text.count("<answer>") == 1:
            count += 0.25
        if text.count("</answer>") == 1:
            count += 0.25
        return count

    return count_tags(answer)


__all__ = ["reward_correct", "reward_format", "tag_count_reward"]


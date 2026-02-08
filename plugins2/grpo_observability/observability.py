from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np



def compute_token_observability_tensors(
    *,
    labels: Any,
    advantages: Any,
    per_token_logps: Any,
    total_valid_token_count: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_np = np.asarray(labels, dtype=np.float32)
    per_token_logps_np = np.asarray(per_token_logps, dtype=np.float32)
    advantages_np = np.asarray(advantages, dtype=np.float32)

    if labels_np.ndim != 2:
        raise ValueError(f"labels must be rank-2 [B, T], got {labels_np.shape}")
    if per_token_logps_np.ndim != 2:
        raise ValueError(f"per_token_logps must be rank-2 [B, T-1], got {per_token_logps_np.shape}")

    completion_mask = labels_np[:, 1:]
    if completion_mask.shape != per_token_logps_np.shape:
        raise ValueError(
            "completion mask and per_token_logps must have same shape, "
            f"got {completion_mask.shape} vs {per_token_logps_np.shape}"
        )

    if advantages_np.ndim == 1:
        if advantages_np.shape[0] != completion_mask.shape[0]:
            raise ValueError(
                f"advantages shape [B] must match batch size, got {advantages_np.shape[0]} vs {completion_mask.shape[0]}"
            )
        advantages_per_token = np.repeat(advantages_np[:, None], completion_mask.shape[1], axis=1)
    elif advantages_np.ndim == 2:
        if advantages_np.shape != completion_mask.shape:
            raise ValueError(
                "advantages shape [B, T-1] must match completion mask shape, "
                f"got {advantages_np.shape} vs {completion_mask.shape}"
            )
        advantages_per_token = advantages_np
    else:
        raise ValueError(f"advantages must be rank-1 or rank-2, got {advantages_np.shape}")

    denom = max(float(total_valid_token_count), 1.0)
    token_grad_logprob = -(advantages_per_token * completion_mask) / denom
    token_loss_contrib = -(advantages_per_token * per_token_logps_np * completion_mask) / denom
    token_probs = np.exp(per_token_logps_np) * completion_mask
    return token_grad_logprob.astype(np.float32), token_loss_contrib.astype(np.float32), token_probs.astype(np.float32)



def build_token_rows(
    *,
    input_ids: Any,
    labels: Any,
    per_token_logps: Any,
    token_grad_logprob: Any,
    token_loss_contrib: Any,
    token_probs: Any,
    tokenizer: Any,
) -> list[list[dict[str, Any]]]:
    input_ids_np = np.asarray(input_ids)
    labels_np = np.asarray(labels)
    logps_np = np.asarray(per_token_logps, dtype=np.float32)
    grad_np = np.asarray(token_grad_logprob, dtype=np.float32)
    loss_np = np.asarray(token_loss_contrib, dtype=np.float32)
    probs_np = np.asarray(token_probs, dtype=np.float32)

    if input_ids_np.ndim != 2:
        raise ValueError(f"input_ids must be rank-2 [B, T], got {input_ids_np.shape}")
    if labels_np.shape != input_ids_np.shape:
        raise ValueError(f"labels shape must match input_ids shape, got {labels_np.shape} vs {input_ids_np.shape}")

    chosen_ids = input_ids_np[:, 1:]
    completion_mask = labels_np[:, 1:] > 0
    if chosen_ids.shape != logps_np.shape:
        raise ValueError(f"chosen_ids and per_token_logps mismatch: {chosen_ids.shape} vs {logps_np.shape}")

    decoded_cache: dict[int, str] = {}

    def decode_token(token_id: int) -> str:
        if token_id not in decoded_cache:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            decoded_cache[token_id] = str(text)
        return decoded_cache[token_id]

    all_rows: list[list[dict[str, Any]]] = []
    for sample_idx in range(chosen_ids.shape[0]):
        rows: list[dict[str, Any]] = []
        positions = np.where(completion_mask[sample_idx])[0]
        for position in positions.tolist():
            token_id = int(chosen_ids[sample_idx, position])
            rows.append(
                {
                    "position": int(position),
                    "token_id": token_id,
                    "token_text": decode_token(token_id),
                    "logprob": float(logps_np[sample_idx, position]),
                    "prob": float(probs_np[sample_idx, position]),
                    "grad_logprob": float(grad_np[sample_idx, position]),
                    "loss_contrib": float(loss_np[sample_idx, position]),
                }
            )
        all_rows.append(rows)
    return all_rows



def summarize_rewards(*, rewards: Any, rewards_per_func: Any, reward_names: list[str]) -> tuple[dict[str, float], list[dict[str, float]]]:
    rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
    per_func_np = np.asarray(rewards_per_func, dtype=np.float32)

    if per_func_np.ndim != 2:
        raise ValueError(f"rewards_per_func must be rank-2 [F, B], got {per_func_np.shape}")
    if per_func_np.shape[0] != len(reward_names):
        raise ValueError(f"reward_names length mismatch: {len(reward_names)} vs {per_func_np.shape[0]}")
    if per_func_np.shape[1] != rewards_np.shape[0]:
        raise ValueError(f"batch size mismatch: rewards {rewards_np.shape[0]} vs per_func {per_func_np.shape[1]}")

    reward_stats = {
        "mean": float(rewards_np.mean()) if rewards_np.size else float("nan"),
        "std": float(rewards_np.std()) if rewards_np.size else float("nan"),
        "min": float(rewards_np.min()) if rewards_np.size else float("nan"),
        "max": float(rewards_np.max()) if rewards_np.size else float("nan"),
    }

    per_sample: list[dict[str, float]] = []
    for sample_idx in range(rewards_np.shape[0]):
        values: dict[str, float] = {
            name: float(per_func_np[name_idx, sample_idx]) for name_idx, name in enumerate(reward_names)
        }
        values["total"] = float(rewards_np[sample_idx])
        per_sample.append(values)
    return reward_stats, per_sample


__all__ = [
    "build_token_rows",
    "compute_token_observability_tensors",
    "summarize_rewards",
]

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import GenerationConfig, LogitsProcessorList

from plugins.sft.constrained_decoding import ConstrainedLogitsProcessor, build_sid_constraint, load_valid_sids_from_info
from plugins.sft.metrics import RankingMetrics, compute_hr_ndcg


def evaluate_sid_next_item(
    *,
    model: Any,
    tokenizer: Any,
    eval_dataset: Any,
    info_file: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
    topk: list[int],
    constrained: bool,
    output_predictions_json: str | None,
    device: str,
) -> tuple[list[list[str]], RankingMetrics]:
    model.eval()

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Unable to resolve eos_token_id from tokenizer/model")
    eos_token_id = int(eos_token_id)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    pad_token_id = int(pad_token_id)

    valid_sids = load_valid_sids_from_info(info_file)
    constraint = build_sid_constraint(tokenizer=tokenizer, valid_sids=valid_sids, eos_token_id=eos_token_id)

    all_predictions: list[list[str]] = []
    device_obj = torch.device(device)
    model = model.to(device_obj)

    targets = list(getattr(eval_dataset, "get_targets")())
    n = len(eval_dataset)
    if n != len(targets):
        raise ValueError("eval_dataset.get_targets() length mismatch")

    for start in range(0, n, int(batch_size)):
        batch = [eval_dataset[i] for i in range(start, min(n, start + int(batch_size)))]
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        for x in batch:
            ids = list(x["input_ids"])
            mask = list(x["attention_mask"])
            pad = max_len - len(ids)
            input_ids.append([pad_token_id] * pad + ids)
            attention_mask.append([0] * pad + mask)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device_obj)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device_obj)

        generation_config = GenerationConfig(
            num_beams=int(num_beams),
            num_return_sequences=int(num_beams),
            length_penalty=float(length_penalty),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
            max_new_tokens=int(max_new_tokens),
        )

        logits_processor = None
        if constrained:
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=constraint.prefix_allowed_tokens_fn(),
                num_beams=int(num_beams),
                prefix_token_count=int(constraint.prefix_token_count),
                eos_token_id=int(eos_token_id),
            )
            logits_processor = LogitsProcessorList([clp])

        with torch.no_grad():
            out = model.generate(
                input_ids_t,
                attention_mask=attention_mask_t,
                generation_config=generation_config,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=False,
            )

        completions = out.sequences[:, max_len:]
        decoded = tokenizer.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for i in range(len(batch)):
            group = decoded[i * int(num_beams) : (i + 1) * int(num_beams)]
            all_predictions.append([str(x).strip() for x in group])

    metrics = compute_hr_ndcg(predictions=all_predictions, targets=targets, topk=topk, valid_items=constraint.valid_sid_set)

    if output_predictions_json:
        payload = []
        for target, preds in zip(targets, all_predictions, strict=True):
            payload.append({"target": target, "predict": preds})
        Path(output_predictions_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_predictions_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        metrics_path = str(Path(output_predictions_json).with_suffix(".metrics.json"))
        Path(metrics_path).write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return all_predictions, metrics


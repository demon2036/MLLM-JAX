from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from transformers import EarlyStoppingCallback


@dataclass(frozen=True)
class SftTrainResult:
    output_dir: str
    final_checkpoint_dir: str
    best_checkpoint: str | None


def _supports_bf16(device: str) -> bool:
    if device.startswith("cuda") and torch.cuda.is_available():
        return bool(torch.cuda.is_bf16_supported())
    # On recent CPUs, bf16 may work but is not always efficient; keep conservative.
    return False


def run_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    output_dir: str,
    run_name: str | None,
    seed: int,
    train_cfg: dict[str, Any],
    wandb_enabled: bool,
    device: str,
) -> SftTrainResult:
    os.makedirs(output_dir, exist_ok=True)

    per_device_train_batch_size = int(train_cfg.get("per_device_train_batch_size") or 1)
    per_device_eval_batch_size = int(train_cfg.get("per_device_eval_batch_size") or per_device_train_batch_size)
    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps") or 1)
    learning_rate = float(train_cfg.get("learning_rate") or 3e-4)
    num_train_epochs = float(train_cfg.get("num_train_epochs") or 1)
    max_steps = int(train_cfg.get("max_steps") or -1)
    warmup_steps = int(train_cfg.get("warmup_steps") or 0)
    logging_steps = int(train_cfg.get("logging_steps") or 10)
    eval_steps = int(train_cfg.get("eval_steps") or 0)
    save_steps = int(train_cfg.get("save_steps") or 0)
    save_total_limit = int(train_cfg.get("save_total_limit") or 1)
    group_by_length = bool(train_cfg.get("group_by_length") or False)
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint") or None
    early_stopping_patience = int(train_cfg.get("early_stopping_patience") or 0)

    # Resolve dtype flags.
    bf16_requested = bool(train_cfg.get("bf16") or False)
    fp16_requested = bool(train_cfg.get("fp16") or False)
    bf16 = bool(bf16_requested and _supports_bf16(device))
    fp16 = bool(fp16_requested and device.startswith("cuda") and torch.cuda.is_available())

    report_to = ["wandb"] if wandb_enabled else []
    eval_strategy = "steps" if eval_dataset is not None and int(eval_steps) > 0 else "no"
    save_strategy = "steps" if int(save_steps) > 0 else "no"

    args = transformers.TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        seed=int(seed),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        bf16=bf16,
        fp16=fp16,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=save_total_limit,
        load_best_model_at_end=bool(eval_strategy != "no"),
        report_to=report_to,
        group_by_length=group_by_length,
        remove_unused_columns=False,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    callbacks: list[Any] = []
    if early_stopping_patience and early_stopping_patience > 0 and eval_strategy != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    trainer.model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    return SftTrainResult(output_dir=output_dir, final_checkpoint_dir=final_checkpoint_dir, best_checkpoint=best_checkpoint)


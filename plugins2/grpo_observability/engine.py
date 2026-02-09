from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from plugins.sample.workflows.grpo_sync import generate_answers_and_training_batch
from plugins.training.core.logging.wandb import maybe_init_wandb
from plugins.training.core.mesh.mesh import create_mesh
from plugins.training.core.optim.optimizer import build_tx
from plugins.training.core.sharding.batch import local_from_global, make_form_training_global_array
from plugins.training.core.step.train_step import training_step
from plugins.training.rl.advantage.modules import GroupIdGRPOAdvantageModule
from training2 import get_state

from plugins2.grpo_observability.config import Plugins2ObservabilityConfig
from plugins2.grpo_observability.observability import (
    build_token_rows,
    compute_input_attribution_tensors,
    compute_token_observability_tensors,
    summarize_rewards,
)
from plugins2.grpo_observability.rewarding import build_gsm8k_reward_inputs, build_gsm8k_reward_module
from plugins2.grpo_observability.types import RunRequest



def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(np.asarray(x))
    except Exception:
        return float(default)



def _tree_l2_norm(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    sq = [jnp.sum(jnp.square(jnp.asarray(x, dtype=jnp.float32))) for x in leaves]
    return float(jnp.sqrt(jnp.sum(jnp.asarray(sq))))



def _tree_abs_mean(tree: Any) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return 0.0
    abs_sum = jnp.asarray(0.0, dtype=jnp.float32)
    count = 0
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=jnp.float32)
        abs_sum = abs_sum + jnp.sum(jnp.abs(arr))
        count += int(arr.size)
    if count <= 0:
        return 0.0
    return float(abs_sum / float(count))



def _to_local_array(array: Any) -> np.ndarray:
    if hasattr(array, "addressable_shards"):
        return np.asarray(local_from_global(array))
    return np.asarray(array)


def _build_token_advantages(
    *,
    advantage_mode: str,
    sample_advantages: np.ndarray,
    per_token_logps: np.ndarray,
    labels: np.ndarray,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    sample_adv = np.asarray(sample_advantages, dtype=np.float32).reshape(-1)
    if sample_adv.shape[0] != labels.shape[0]:
        raise ValueError(f"sample advantages batch mismatch: {sample_adv.shape[0]} vs {labels.shape[0]}")

    completion_mask = np.asarray(labels[:, 1:], dtype=np.float32)
    if completion_mask.shape != per_token_logps.shape:
        raise ValueError(f"completion mask mismatch: {completion_mask.shape} vs {per_token_logps.shape}")

    if advantage_mode == "sample":
        return np.repeat(sample_adv[:, None], completion_mask.shape[1], axis=1).astype(np.float32)

    token_surprisal = np.maximum(-np.asarray(per_token_logps, dtype=np.float32), 0.0)
    masked_surprisal = token_surprisal * completion_mask

    token_weights = np.ones_like(masked_surprisal, dtype=np.float32)
    for sample_idx in range(masked_surprisal.shape[0]):
        valid = completion_mask[sample_idx] > 0
        if not np.any(valid):
            continue
        sample_vals = masked_surprisal[sample_idx, valid]
        mean_val = float(sample_vals.mean())
        if mean_val <= 1e-8:
            normalized = np.ones_like(sample_vals, dtype=np.float32)
        else:
            normalized = sample_vals / mean_val
        normalized = np.clip(normalized, min_weight, max_weight)
        token_weights[sample_idx, valid] = normalized

    token_adv = sample_adv[:, None] * token_weights
    token_adv = token_adv * completion_mask
    return token_adv.astype(np.float32)


class GRPOObservabilityEngine:
    def __init__(self, cfg: Plugins2ObservabilityConfig):
        self.cfg = cfg
        self.mesh = create_mesh(cfg.mesh_shape)
        self.form_training_global_array = make_form_training_global_array(self.mesh)

        tx = build_tx(training_steps=cfg.train.training_steps, cfg=cfg.train.optimizer)
        self.state, self.sampler, _ = get_state(
            self.mesh,
            training_steps=cfg.train.training_steps,
            grad_accum_steps=cfg.train.grad_accum_steps,
            model_path=cfg.model_path,
            num_pre_q=cfg.rollout.k,
            beta=cfg.train.beta,
            tx=tx,
        )
        self.train_step = jax.jit(training_step, donate_argnums=(0,))
        self.reward_module, self.reward_names = build_gsm8k_reward_module(cfg.reward_weights)
        self.advantage_module = GroupIdGRPOAdvantageModule()

        self.wandb = maybe_init_wandb(
            cfg=cfg,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            process_index=jax.process_index(),
        )
        self._lock = threading.Lock()

    def _loss_and_grads(
        self,
        batch: dict[str, Any],
        *,
        input_embed_delta: Any | None = None,
    ) -> tuple[float, dict[str, Any], Any, Any, float, float, Any | None]:
        def loss_fn(params: Any):
            variables = {"params": {"model": params}}
            if getattr(self.state, "ref_params", None) is not None:
                variables["params"]["ref_model"] = self.state.ref_params
            local_batch = batch
            if input_embed_delta is not None:
                local_batch = dict(batch)
                local_batch["input_embed_delta"] = input_embed_delta
            metrics = self.state.apply_fn(variables, local_batch)
            per_token_logps = metrics.get("per_token_logps")
            metrics_scalar = {k: v for k, v in metrics.items() if k != "per_token_logps"}
            metrics_scalar = jax.tree_util.tree_map(jnp.mean, metrics_scalar)
            return metrics_scalar["loss"], (metrics_scalar, per_token_logps)

        if input_embed_delta is None:
            (loss_value, (metrics_scalar, per_token_logps)), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.state.params)
            input_embed_grads = None
        else:
            (loss_value, (metrics_scalar, per_token_logps)), (grads, input_embed_grads) = jax.value_and_grad(
                loss_fn,
                argnums=(0, 1),
                has_aux=True,
            )(self.state.params, input_embed_delta)
        jax.block_until_ready(loss_value)
        grad_l2 = _tree_l2_norm(grads)
        grad_abs_mean = _tree_abs_mean(grads)
        metrics_np = {k: _as_float(v) for k, v in metrics_scalar.items()}
        return _as_float(loss_value), metrics_np, per_token_logps, grads, grad_l2, grad_abs_mean, input_embed_grads

    def run_request(self, request: RunRequest) -> dict[str, Any]:
        with self._lock:
            t_request0 = time.perf_counter()
            k = int(request.k if request.k is not None else self.cfg.rollout.k)
            if k <= 0:
                raise ValueError("k must be > 0")

            prompts = [str(request.user_prompt) for _ in range(k)]

            t0 = time.perf_counter()
            chat_prompts, answers, batch_np = generate_answers_and_training_batch(
                prompts=prompts,
                sampler=self.sampler,
                params=self.state.params,
                system_prompt=str(request.system_prompt),
                global_length=int(self.cfg.rollout.global_length),
                max_length_sample=int(self.cfg.rollout.max_length_sample),
            )
            t_rollout = time.perf_counter() - t0

            reward_inputs = build_gsm8k_reward_inputs(label=str(request.label), batch_size=k)

            t0 = time.perf_counter()
            rewards_out = self.reward_module.compute(inputs=reward_inputs, answers=answers)
            rewards_np = np.asarray(rewards_out.rewards, dtype=np.float32).reshape(-1)
            rewards_per_func = np.asarray(rewards_out.rewards_per_func, dtype=np.float32)
            t_reward = time.perf_counter() - t0

            group_ids = np.zeros((k,), dtype=np.int32)
            t0 = time.perf_counter()
            advantages_np = np.asarray(
                self.advantage_module.compute(rewards=rewards_np, group_ids=group_ids).advantages,
                dtype=np.float32,
            ).reshape(-1)
            t_advantage = time.perf_counter() - t0

            batch_np = dict(batch_np)
            batch_np["rewards"] = rewards_np
            batch_np["advantages"] = advantages_np
            batch_np["group_ids"] = group_ids

            t0 = time.perf_counter()
            batch = jax.tree_util.tree_map_with_path(self.form_training_global_array, batch_np)
            total_valid_token_count = batch["labels"][:, 1:].sum()
            t_shard = time.perf_counter() - t0

            t0 = time.perf_counter()
            loss_before_update, grad_metrics, per_token_logps, _grads, grad_l2, grad_abs_mean, _ = self._loss_and_grads(batch)
            t_grad = time.perf_counter() - t0

            per_token_logps_local = _to_local_array(per_token_logps)
            total_valid_value = _as_float(total_valid_token_count, default=1.0)

            advantages_for_loss = _build_token_advantages(
                advantage_mode=self.cfg.train.advantage_mode,
                sample_advantages=advantages_np,
                per_token_logps=per_token_logps_local,
                labels=np.asarray(batch_np["labels"], dtype=np.float32),
                min_weight=float(self.cfg.train.token_adv_min_weight),
                max_weight=float(self.cfg.train.token_adv_max_weight),
            )
            batch_np["advantages"] = advantages_for_loss
            batch = jax.tree_util.tree_map_with_path(self.form_training_global_array, batch_np)

            hidden_size = int(self.state.params["model"]["model"]["embed_tokens"]["embedding"].shape[1])
            input_embed_delta = np.zeros(
                (int(batch_np["input_ids"].shape[0]), int(batch_np["input_ids"].shape[1]), hidden_size),
                dtype=np.float32,
            )
            input_embed_delta_global = self.form_training_global_array(
                (jax.tree_util.DictKey("input_embed_delta"),),
                input_embed_delta,
            )
            _, _, _, _, _, _, input_embed_grads = self._loss_and_grads(
                batch,
                input_embed_delta=input_embed_delta_global,
            )
            if input_embed_grads is None:
                raise ValueError("input embedding gradients unavailable")
            input_embed_grads_local = _to_local_array(input_embed_grads)

            input_grad_norms, input_grad_dot, input_grad_cos = compute_input_attribution_tensors(
                input_ids=batch_np["input_ids"],
                labels=batch_np["labels"],
                embedding_table=self.state.params["model"]["model"]["embed_tokens"]["embedding"],
                input_embed_grads=input_embed_grads_local,
            )

            token_grad_logprob, token_loss_contrib, token_probs = compute_token_observability_tensors(
                labels=batch_np["labels"],
                advantages=advantages_for_loss,
                per_token_logps=per_token_logps_local,
                total_valid_token_count=total_valid_value,
            )
            token_rows_by_sample = build_token_rows(
                input_ids=batch_np["input_ids"],
                labels=batch_np["labels"],
                per_token_logps=per_token_logps_local,
                advantages=advantages_for_loss,
                token_grad_logprob=token_grad_logprob,
                token_loss_contrib=token_loss_contrib,
                token_probs=token_probs,
                input_grad_norms=input_grad_norms,
                input_grad_dot=input_grad_dot,
                input_grad_cos=input_grad_cos,
                tokenizer=self.sampler.tokenizer,
            )

            reward_stats, reward_per_sample = summarize_rewards(
                rewards=rewards_np,
                rewards_per_func=rewards_per_func,
                reward_names=self.reward_names,
            )

            step_before = int(np.asarray(self.state.step))
            t0 = time.perf_counter()
            self.state, train_metrics = self.train_step(self.state, batch)
            jax.block_until_ready(train_metrics["loss"])
            t_update = time.perf_counter() - t0
            step_after = int(np.asarray(self.state.step))

            train_metrics_out = {k: _as_float(v) for k, v in train_metrics.items() if k != "per_token_logps"}
            loss_after_update = train_metrics_out.get("loss", float("nan"))
            entropy_after_update = train_metrics_out.get("entropy", float("nan"))

            samples: list[dict[str, Any]] = []
            for idx, answer in enumerate(answers):
                samples.append(
                    {
                        "sample_index": int(idx),
                        "answer": str(answer),
                        "reward": reward_per_sample[idx],
                        "tokens": token_rows_by_sample[idx],
                    }
                )

            summary = {
                "config_path": self.cfg.config_path,
                "model_path": self.cfg.model_path,
                "mesh_shape": self.cfg.mesh_shape,
                "k": int(k),
                "advantage_mode": str(self.cfg.train.advantage_mode),
                "step_before": int(step_before),
                "step_after": int(step_after),
                "loss_before_update": float(loss_before_update),
                "loss_after_update": float(loss_after_update),
                "entropy_after_update": float(entropy_after_update),
                "grad_l2": float(grad_l2),
                "grad_abs_mean": float(grad_abs_mean),
                "total_valid_token_count": float(total_valid_value),
                "reward_stats": reward_stats,
                "timing_s": {
                    "rollout": float(t_rollout),
                    "reward": float(t_reward),
                    "advantage": float(t_advantage),
                    "shard": float(t_shard),
                    "grad": float(t_grad),
                    "update": float(t_update),
                    "request_total": float(time.perf_counter() - t_request0),
                },
                "train_metrics": train_metrics_out,
                "grad_metrics": grad_metrics,
            }

            response = {
                "request": {
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                    "label": request.label,
                    "k": int(k),
                },
                "chat_prompts": chat_prompts,
                "summary": summary,
                "samples": samples,
            }

            if self.wandb is not None and jax.process_index() == 0:
                self.wandb.log(
                    {
                        "plugins2/loss_before_update": float(loss_before_update),
                        "plugins2/loss_after_update": float(loss_after_update),
                        "plugins2/grad_l2": float(grad_l2),
                        "plugins2/reward_mean": float(reward_stats["mean"]),
                        "plugins2/request_total_s": float(summary["timing_s"]["request_total"]),
                        "plugins2/k": int(k),
                    },
                    step=step_after,
                )

            return response

    def describe(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "jax_backend": jax.default_backend(),
            "jax_process": f"{jax.process_index()}/{jax.process_count()}",
            "jax_device_count": int(jax.device_count()),
            "jax_local_device_count": int(jax.local_device_count()),
        }


__all__ = ["GRPOObservabilityEngine"]

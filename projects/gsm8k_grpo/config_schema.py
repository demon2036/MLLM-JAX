from __future__ import annotations

from dataclasses import dataclass, field

from plugins.training.core.optim.optimizer import OptimizerConfig, sanitize_optimizer_config_for_logging
from plugins.training.rl.algorithms import AlgoConfig, sanitize_algo_config_for_logging


@dataclass(frozen=True)
class GRPODynamicSamplingConfig:
    # Enable rollout-time dynamic re-sampling for homogeneous prompt-groups.
    enabled: bool = False
    # Trigger policy; currently only homogeneous-group trigger is supported.
    trigger: str = "homogeneous_group"
    # Metric used to detect group homogeneity.
    # - "acc": first reward function score (typically correctness)
    # - "seq_reward": weighted total sequence reward
    metric: str = "acc"
    # Dominant-value fraction threshold to mark a group as homogeneous.
    # 1.0 means all samples in a group must share the same metric value.
    homogeneity_threshold: float = 1.0
    # Minimum number of unique metric values required to treat a group as diverse.
    min_unique_reward_values: int = 2
    # Maximum number of extra rollout rounds for homogeneous groups.
    max_extra_roll_rounds: int = 10
    # Desired number of valid prompt-groups after dynamic sampling.
    # None falls back to `rollout.batch_size`.
    target_valid_groups: int | None = None
    # Budget-exhaust fallback for still-homogeneous groups.
    # - keep_last: keep latest sampled groups
    # - drop_group: zero out labels/rewards/advantages for dropped groups
    fallback_policy: str = "keep_last"


@dataclass(frozen=True)
class GRPORolloutConfig:
    # Prompt batch size per training step (global, across all processes).
    #
    # Each prompt is expanded to `n` sampled completions, so the global
    # sequence batch is: `batch_size * n`.
    batch_size: int = 32
    # Number of samples per prompt (GRPO group size, a.k.a. K / num_pre_q).
    n: int = 8
    global_length: int = 512
    max_length_sample: int = 64
    # Explicit rollout micro-batch controls (sequences per rollout pass).
    #
    # `micro_batch_size_per_device` is required by the runner so rollout pass
    # planning is fully explicit (no hidden heuristic cap).
    micro_batch_size: int | None = None
    micro_batch_size_per_device: int | None = None
    # Rollout backend selector (swappable generation engine).
    backend: str = "naive"
    dynamic_sampling: GRPODynamicSamplingConfig = field(default_factory=GRPODynamicSamplingConfig)


@dataclass(frozen=True)
class GRPOTrainConfig:
    # Optional: sequences per process per micro-step.
    micro_batch_size: int | None = None
    # Optional: sequences per device per micro-step.
    micro_batch_size_per_device: int | None = None
    max_length_total: int = 0
    ppo_epochs: int = 1
    grad_accum_steps: int = 1
    beta: float = 0.0
    # Whether to enable rematerialization (gradient checkpointing) in train modules.
    gradient_checkpointing: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class GRPOGsm8kConfig:
    # The YAML config path used to construct this run (as passed to the CLI).
    # Kept for W&B traceability (so runs can be mapped back to a committed file).
    config_path: str

    model_path: str
    steps: int
    rollout: GRPORolloutConfig
    train: GRPOTrainConfig
    mesh_shape: str

    wandb_project: str
    wandb_mode: str
    wandb_name: str
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    reward_weights: tuple[float, float, float] = (1.0, 0.5, 0.5)
    eval_every_steps: int = 0
    eval_batches_per_process: int = 1
    eval_split: str = "test"
    # Number of sampled completions per eval prompt.
    eval_rollout_n: int = 1
    # Run a full-split eval sweep every N training steps (0 disables).
    eval_full_every_steps: int = 0
    # Whether to run a full-split eval sweep once at the end of training.
    eval_full_sweep: bool = False

    def to_logging_dict(self) -> dict[str, object]:
        return {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "steps": int(self.steps),
            "rollout": {
                "batch_size": int(self.rollout.batch_size),
                "n": int(self.rollout.n),
                "global_length": int(self.rollout.global_length),
                "max_length_sample": int(self.rollout.max_length_sample),
                "micro_batch_size": self.rollout.micro_batch_size,
                "micro_batch_size_per_device": self.rollout.micro_batch_size_per_device,
                "backend": str(self.rollout.backend),
                "dynamic_sampling": {
                    "enabled": bool(self.rollout.dynamic_sampling.enabled),
                    "trigger": str(self.rollout.dynamic_sampling.trigger),
                    "metric": str(self.rollout.dynamic_sampling.metric),
                    "homogeneity_threshold": float(self.rollout.dynamic_sampling.homogeneity_threshold),
                    "min_unique_reward_values": int(self.rollout.dynamic_sampling.min_unique_reward_values),
                    "max_extra_roll_rounds": int(self.rollout.dynamic_sampling.max_extra_roll_rounds),
                    "target_valid_groups": self.rollout.dynamic_sampling.target_valid_groups,
                    "fallback_policy": str(self.rollout.dynamic_sampling.fallback_policy),
                },
            },
            "train": {
                "micro_batch_size": self.train.micro_batch_size,
                "micro_batch_size_per_device": self.train.micro_batch_size_per_device,
                "max_length_total": int(self.train.max_length_total),
                "ppo_epochs": int(self.train.ppo_epochs),
                "grad_accum_steps": int(self.train.grad_accum_steps),
                "beta": float(self.train.beta),
                "gradient_checkpointing": bool(self.train.gradient_checkpointing),
                "optimizer": sanitize_optimizer_config_for_logging(self.train.optimizer),
            },
            "mesh_shape": str(self.mesh_shape),
            "wandb_project": str(self.wandb_project),
            "wandb_mode": str(self.wandb_mode),
            "wandb_name": str(self.wandb_name),
            "algo": sanitize_algo_config_for_logging(self.algo),
            "reward_weights": [float(x) for x in self.reward_weights],
            "eval_every_steps": int(self.eval_every_steps),
            "eval_batches_per_process": int(self.eval_batches_per_process),
            "eval_split": str(self.eval_split),
            "eval_rollout_n": int(self.eval_rollout_n),
            "eval_full_every_steps": int(self.eval_full_every_steps),
            "eval_full_sweep": bool(self.eval_full_sweep),
        }


__all__ = ["GRPODynamicSamplingConfig", "GRPORolloutConfig", "GRPOTrainConfig", "GRPOGsm8kConfig"]

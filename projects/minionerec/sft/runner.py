from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from plugins.sft.hf_safetensors import load_hf_safetensors_state_dict
from projects.minionerec.sft.datasets.concat_dataset import ConcatDataset
from projects.minionerec.sft.datasets.eval_sid_next_item import SidNextItemEvalDataset
from projects.minionerec.sft.datasets.fusion_seq_rec import FusionSeqRecSftDataset
from projects.minionerec.sft.datasets.sid_item_alignment import SidItemAlignmentDataset
from projects.minionerec.sft.datasets.sid_next_item import SidNextItemSftDataset
from projects.minionerec.sft.jax.evaluator import SidNextItemJaxEvaluator, evaluate_sid_next_item_jax
from projects.minionerec.sft.tokens import maybe_extend_tokenizer
from plugins.common.wandb_utils import maybe_init_wandb
from plugins.common.tokenizer import prepare_tokenizer


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class SidSftDataConfig:
    category: str
    train_file: str
    eval_file: str
    test_file: str
    info_file: str
    sid_index_path: str
    item_meta_path: str
    max_len: int = 512
    padding_side: str = "left"
    # Training-only padding policy.
    # - "fixed": pad every step to `max_len`.
    # - "pow2_buckets": pad every step to the smallest bucket >= global max len.
    # - "max": pad every step to the global max len (can trigger many recompiles).
    train_pad_policy: str = "fixed"
    train_pad_buckets: tuple[int, ...] = (128, 256, 512)
    train_pad_to_multiple_of: int = 8
    sample_train: int = -1
    sample_eval: int = -1
    sample_test: int = -1


@dataclass(frozen=True)
class SidSftTasksConfig:
    sid_next_item: bool = True
    sid_item_alignment: bool = True
    fusion_seq_rec: bool = True


@dataclass(frozen=True)
class SidSftMuonConfig:
    """Muon optimizer hyperparameters (used when train.optimizer == 'muon')."""

    aux_learning_rate: float = 3e-4
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    eps: float = 1e-7
    max_dim: int = 10_000


@dataclass(frozen=True)
class SidSftTrainConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    global_batch_size: int = 0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    optimizer: str = "adamw"
    muon: SidSftMuonConfig = field(default_factory=SidSftMuonConfig)
    weight_decay: float = 0.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 1
    save_last: bool = True
    group_by_length: bool = False
    freeze_LLM: bool = False
    train_from_scratch: bool = False
    resume_from_checkpoint: str | None = None
    early_stopping_patience: int = 3
    bf16: bool = False
    fp16: bool = False


@dataclass(frozen=True)
class SidSftEvalConfig:
    enabled: bool = True
    batch_size: int = 4
    num_beams: int = 50
    max_new_tokens: int = 64
    length_penalty: float = 0.0
    topk: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    constrained: bool = True
    save_predictions_json: bool = True


@dataclass(frozen=True)
class SidSftWandbConfig:
    project: str = "minionerec-sid-sft"
    mode: str = "online"
    name: str | None = None


@dataclass(frozen=True)
class SidSftJaxConfig:
    mesh_shape: str = "1,-1,1"
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    max_cache_length: int = 2048


@dataclass(frozen=True)
class SidSftConfig:
    config_path: str
    backend: str
    base_model: str
    output_dir: str
    seed: int
    device: str
    data: SidSftDataConfig
    jax: SidSftJaxConfig = field(default_factory=SidSftJaxConfig)
    tasks: SidSftTasksConfig = field(default_factory=SidSftTasksConfig)
    train: SidSftTrainConfig = field(default_factory=SidSftTrainConfig)
    eval: SidSftEvalConfig = field(default_factory=SidSftEvalConfig)
    wandb: SidSftWandbConfig = field(default_factory=SidSftWandbConfig)


def _build_train_dataset(cfg: SidSftConfig, tokenizer: Any) -> Any:
    train_datasets = []
    if cfg.tasks.sid_next_item:
        train_datasets.append(
            SidNextItemSftDataset(
                csv_path=cfg.data.train_file,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if cfg.tasks.sid_item_alignment:
        train_datasets.append(
            SidItemAlignmentDataset(
                item_meta_path=cfg.data.item_meta_path,
                sid_index_path=cfg.data.sid_index_path,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if cfg.tasks.fusion_seq_rec:
        train_datasets.append(
            FusionSeqRecSftDataset(
                csv_path=cfg.data.train_file,
                item_meta_path=cfg.data.item_meta_path,
                sid_index_path=cfg.data.sid_index_path,
                tokenizer=tokenizer,
                max_len=cfg.data.max_len,
                sample=cfg.data.sample_train,
                seed=cfg.seed,
                include_labels=True,
                pretokenize=True,
            )
        )
    if not train_datasets:
        raise ValueError("No training tasks enabled under cfg.tasks.*")
    return train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)


def _build_eval_dataset(cfg: SidSftConfig, tokenizer: Any) -> SidNextItemEvalDataset:
    return SidNextItemEvalDataset(
        csv_path=cfg.data.test_file,
        tokenizer=tokenizer,
        max_len=cfg.data.max_len,
        sample=cfg.data.sample_test,
        seed=cfg.seed,
        pretokenize=True,
    )


def _run_sid_sft_jax(cfg: SidSftConfig, *, run_mode_norm: str) -> dict[str, Any]:
    import math
    import time

    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding

    from MLLM_JAX.language.llama.llama import LlamaJaxConfig, convert_torch_to_flax_llama
    from MLLM_JAX.language.qwen2.modular_qwen2 import Qwen2ForCausalLM
    from plugins.training.sft.jax.sharding import get_partition_rules_llama, match_partition_rules

    require_multihost = str(os.environ.get("REQUIRE_MULTIHOST", "")).strip().lower() in {"1", "true", "yes"}
    require_process_count_raw = str(os.environ.get("REQUIRE_JAX_PROCESS_COUNT", "")).strip()
    require_process_count = int(require_process_count_raw) if require_process_count_raw else 0

    # Avoid calling `jax.distributed.initialize()` unless multi-host is required.
    #
    # On multi-host TPU VMs (e.g. v6e-16), calling `jax.distributed.initialize()`
    # from only *one* worker (e.g. eval on worker 0) can hang waiting for missing
    # workers. We prefer an explicit opt-in via `REQUIRE_MULTIHOST=1` or
    # `REQUIRE_JAX_PROCESS_COUNT=<N>` (set by our multi-host launch wrappers).
    if require_multihost or require_process_count > 0:
        try:
            jax.distributed.initialize()
        except Exception as e:
            raise RuntimeError(
                "jax.distributed.initialize() failed but a multi-host runtime is required "
                "(REQUIRE_MULTIHOST=1 or REQUIRE_JAX_PROCESS_COUNT is set). "
                "Start the job on all workers (`gcloud ... tpu-vm ssh --worker=all`) "
                "or launch one process per worker."
            ) from e

    if require_multihost and int(jax.process_count()) <= 1:
        raise RuntimeError(
            "Expected multi-host JAX runtime (REQUIRE_MULTIHOST=1), but got jax.process_count()==1. "
            "Launch with --worker=all on multi-host TPUs."
        )
    if require_process_count > 0 and int(jax.process_count()) != int(require_process_count):
        raise RuntimeError(
            f"Expected jax.process_count()=={int(require_process_count)} (REQUIRE_JAX_PROCESS_COUNT), "
            f"got {int(jax.process_count())}."
        )

    if int(jax.process_index()) == 0:
        print(f"backend={jax.default_backend()} process={jax.process_index()}/{jax.process_count()}")
        print(f"device_count={jax.device_count()} local_device_count={jax.local_device_count()}")

    from plugins.training.sft.jax.checkpoint import load_checkpoint, save_checkpoint
    from plugins.training.sft.jax.params import resize_lm_vocab
    from plugins.training.sft.jax.train import SftLossEvaluator, create_mesh_from_config, run_sft_train

    def parse_dtype(name: str) -> Any:
        n = str(name or "float32").strip().lower()
        if n in {"float32", "f32"}:
            return jnp.float32
        if n in {"bfloat16", "bf16"}:
            return jnp.bfloat16
        if n in {"float16", "f16"}:
            return jnp.float16
        raise ValueError(f"Unsupported dtype: {name!r}")

    mesh = create_mesh_from_config(cfg.jax.mesh_shape)
    if int(jax.process_index()) == 0:
        print(f"mesh_shape={cfg.jax.mesh_shape} resolved_mesh={dict(mesh.shape)}")
    compute_dtype = parse_dtype(cfg.jax.compute_dtype)
    param_dtype = parse_dtype(cfg.jax.param_dtype)

    wandb = maybe_init_wandb(
        cfg=cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        process_index=jax.process_index(),
    )

    # Tokenizer + SID token extension (tokenizer-only; params resized below).
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer, pad_token_id = prepare_tokenizer(tokenizer, padding_side=cfg.data.padding_side)

    extension = maybe_extend_tokenizer(tokenizer=tokenizer, sid_index_path=cfg.data.sid_index_path)

    tokenizer_vocab_size = int(len(tokenizer))
    # When using FSDP/TP sharding, some params shard over the vocab axis. Pad the
    # model vocab size so the embedding table shards evenly.
    fsdp = int(mesh.shape.get("fsdp", 1))
    tp = int(mesh.shape.get("tp", 1))
    pad_multiple = max(1, fsdp * tp)
    padded_vocab_size = tokenizer_vocab_size
    if pad_multiple > 1:
        r = tokenizer_vocab_size % pad_multiple
        if r != 0:
            padded_vocab_size = tokenizer_vocab_size + (pad_multiple - r)
            if jax.process_index() == 0:
                print(f"[sft] pad_vocab_size {tokenizer_vocab_size} -> {padded_vocab_size} (multiple={pad_multiple})")

    # Model config must reflect the resized vocab (tokenizer + optional padding).
    base_config = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    base_config.vocab_size = int(padded_vocab_size)

    # Avoid TPU-only fused attention kernels on CPU/GPU backends.
    attention_mesh = mesh if jax.devices()[0].platform == "tpu" else None
    jax_config = LlamaJaxConfig(mesh=attention_mesh, dtype=compute_dtype, param_dtype=param_dtype)
    model = Qwen2ForCausalLM(base_config, jax_config)

    rng = jax.random.PRNGKey(int(cfg.seed))

    # Param init / load:
    # - For eval-only runs with `resume_from_checkpoint`, skip loading base model weights.
    # - Otherwise, load from base model or init from scratch.
    loaded_from_checkpoint = False
    if run_mode_norm == "eval" and cfg.train.resume_from_checkpoint:
        payload = load_checkpoint(str(cfg.train.resume_from_checkpoint))
        ckpt_params = payload.get("params")
        if ckpt_params is None:
            raise ValueError("Checkpoint payload missing 'params'")
        params = ckpt_params
        loaded_from_checkpoint = True
    elif cfg.train.train_from_scratch:
        dummy_len = 8
        dummy = jnp.zeros((1, dummy_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, dummy_len), dtype=jnp.int32)
        dummy_pos = jnp.arange(dummy_len, dtype=jnp.int32)[None, :]
        variables = model.init(rng, input_ids=dummy, attention_mask=dummy_mask, position_ids=dummy_pos, cache=None)
        params = flax.core.unfreeze(variables["params"])
    else:
        state_dict = load_hf_safetensors_state_dict(cfg.base_model)
        if "lm_head.weight" not in state_dict:
            embed_key = None
            for key in state_dict.keys():
                if str(key).endswith("embed_tokens.weight"):
                    embed_key = str(key)
                    break
            if embed_key is None:
                raise KeyError("lm_head.weight missing and no embed_tokens.weight found in safetensors state_dict")
            state_dict = dict(state_dict)
            state_dict["lm_head.weight"] = state_dict[embed_key]
        params = convert_torch_to_flax_llama(state_dict)
        params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)

    # Resize embeddings/lm_head for new SID tokens (+ optional padding for sharding divisibility).
    params, vocab_resize = resize_lm_vocab(params=params, new_vocab_size=int(padded_vocab_size), rng=rng)

    # Place params with sharding + dtype.
    params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), params)
    shapes = jax.eval_shape(lambda x: x, params)
    partitions = match_partition_rules(get_partition_rules_llama(), shapes)
    shardings = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), partitions)
    params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), params, shardings)

    # Train (optional).
    state = None
    train_stats = None
    eval_metrics = None
    if run_mode_norm in {"train", "train_eval"}:
        train_dataset = _build_train_dataset(cfg, tokenizer)

        replicas = int(mesh.shape.get("dp", 1)) * int(mesh.shape.get("fsdp", 1))
        micro_per_replica = int(cfg.train.per_device_train_batch_size)
        if replicas <= 0 or micro_per_replica <= 0:
            raise ValueError(f"Invalid replicas={replicas} or micro_batch_size_per_replica={micro_per_replica}")

        global_batch_size = int(getattr(cfg.train, "global_batch_size", 0) or 0)
        grad_accum_steps = int(cfg.train.gradient_accumulation_steps)
        if global_batch_size > 0:
            micro_global = micro_per_replica * replicas
            ga = int(global_batch_size) // int(micro_global) if micro_global > 0 else 1
            if ga < 1:
                ga = 1
            if jax.process_index() == 0:
                effective = int(micro_global) * int(ga)
                if effective != int(global_batch_size):
                    print(
                        f"[sft] global_batch_size={global_batch_size} not divisible by micro_global={micro_global}; "
                        f"using grad_accum_steps={ga} (effective_bs={effective})"
                    )
                else:
                    print(f"[sft] auto grad_accum_steps={ga} (effective_bs={effective})")
            grad_accum_steps = ga

        max_steps = int(cfg.train.max_steps)
        micro = micro_per_replica * replicas
        effective = micro * int(grad_accum_steps)
        steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, int(effective))))

        if max_steps <= 0:
            max_steps = int(math.ceil(float(cfg.train.num_train_epochs) * steps_per_epoch))

        save_steps = int(cfg.train.save_steps)
        save_total_limit = int(cfg.train.save_total_limit)
        saved_steps: list[int] = []

        def checkpoint_cb(step: int, st: Any) -> None:
            if jax.process_index() != 0:
                return
            save_checkpoint(output_dir=cfg.output_dir, state=st, name=f"step{int(step)}")
            if save_total_limit > 0:
                saved_steps.append(int(step))
                while len(saved_steps) > int(save_total_limit):
                    old = saved_steps.pop(0)
                    try:
                        os.remove(os.path.join(cfg.output_dir, f"sft_state_step{int(old)}.msgpack"))
                    except FileNotFoundError:
                        pass

        evaluator = None
        eval_every_steps = 0
        eval_cb_impl: Callable[[int, Any], None] | None = None
        eval_topk = sorted({int(k) for k in cfg.eval.topk if int(k) > 0 and int(k) <= int(cfg.eval.num_beams)})
        if not eval_topk:
            raise ValueError(f"No valid eval.topk <= num_beams={int(cfg.eval.num_beams)}: {list(cfg.eval.topk)}")

        process_count = int(jax.process_count())
        process_index = int(jax.process_index())
        process_allgather = None
        if process_count > 1:
            from jax.experimental.multihost_utils import process_allgather as _process_allgather

            process_allgather = _process_allgather

        # Local per-process eval dataset subset (for multi-host speed).
        eval_dataset_full = None

        # Paper-aligned: periodic validation loss + save best checkpoint.
        best_eval_loss = float("inf")
        best_eval_step = 0
        val_dataset = None
        val_loss_evaluator = None
        if run_mode_norm == "train":
            eval_every_steps = int(cfg.train.eval_steps)
            if eval_every_steps > 0:
                val_dataset = SidNextItemSftDataset(
                    csv_path=cfg.data.eval_file,
                    tokenizer=tokenizer,
                    max_len=cfg.data.max_len,
                    sample=cfg.data.sample_eval,
                    seed=cfg.seed,
                    include_labels=True,
                    pretokenize=True,
                )
                val_loss_evaluator = SftLossEvaluator(
                    mesh=mesh,
                    model=model,
                    pad_token_id=int(pad_token_id),
                    pad_to_length=int(cfg.data.max_len) if int(cfg.data.max_len) > 0 else None,
                    padding_side=str(cfg.data.padding_side or "left"),
                    micro_batch_size_per_replica=int(cfg.train.per_device_eval_batch_size),
                    pad_to_multiple_of=int(getattr(cfg.data, "train_pad_to_multiple_of", 8) or 8),
                    label_ignore_id=-100,
                )

                def _eval_loss_cb(step: int, st: Any) -> None:
                    nonlocal best_eval_loss, best_eval_step
                    assert val_loss_evaluator is not None
                    assert val_dataset is not None
                    stats = val_loss_evaluator.evaluate(params=st.params, eval_dataset=val_dataset)
                    if jax.process_index() == 0:
                        epoch = float(step) / max(1.0, float(steps_per_epoch))
                        print(
                            "[eval_loss] "
                            + " ".join(
                                [
                                    f"step={int(step)}/{int(max_steps)}",
                                    f"epoch={epoch:.2f}",
                                    f"loss={float(stats.loss):.6f}",
                                    f"tokens={int(stats.token_count)}",
                                    f"t={float(stats.dt_sec):.2f}s",
                                    f"best={float(best_eval_loss):.6f}@{int(best_eval_step)}",
                                ]
                            )
                        )
                        if wandb is not None:
                            wandb.log(
                                {
                                    "eval/loss": float(stats.loss),
                                    "eval/token_count": int(stats.token_count),
                                    "time/eval_loss_s": float(stats.dt_sec),
                                },
                                step=int(step),
                            )

                    if jax.process_index() == 0 and float(stats.loss) < float(best_eval_loss):
                        best_eval_loss = float(stats.loss)
                        best_eval_step = int(step)
                        save_checkpoint(output_dir=cfg.output_dir, state=st, name="best")
                        if wandb is not None:
                            wandb.log(
                                {
                                    "eval/best_loss": float(best_eval_loss),
                                    "eval/best_step": int(best_eval_step),
                                },
                                step=int(step),
                            )

                eval_cb_impl = _eval_loss_cb

        # For `train_eval`, run constrained-decoding eval periodically so W&B
        # shows metrics during training (at least once per epoch).
        if run_mode_norm == "train_eval" and cfg.eval.enabled:
            if not cfg.eval.constrained:
                raise NotImplementedError("JAX evaluator currently supports only constrained=true (SID trie).")

            eval_dataset_full = _build_eval_dataset(cfg, tokenizer)
            n_eval = int(len(eval_dataset_full))
            if n_eval <= 0:
                raise ValueError("Empty eval_dataset")

            if process_count > 1:

                class _EvalSubsetDataset:
                    def __init__(self, base: Any, indices: list[int]):
                        self._base = base
                        self._indices = indices
                        self._targets = list(getattr(base, "get_targets")())

                    def __len__(self) -> int:
                        return len(self._indices)

                    def __getitem__(self, idx: int) -> Any:
                        return self._base[self._indices[idx]]

                    def get_targets(self) -> list[str]:
                        return [self._targets[i] for i in self._indices]

                subset_indices = list(range(process_index, n_eval, process_count))
                if subset_indices:
                    eval_subset = _EvalSubsetDataset(eval_dataset_full, subset_indices)
                    evaluator = SidNextItemJaxEvaluator(
                        model=model,
                        tokenizer=tokenizer,
                        eval_dataset=eval_subset,
                        sid_index_path=cfg.data.sid_index_path,
                        info_file=cfg.data.info_file,
                        batch_size=cfg.eval.batch_size,
                        num_beams=cfg.eval.num_beams,
                        max_cache_length=cfg.jax.max_cache_length,
                        topk=eval_topk,
                        show_progress=False,
                    )
                elif jax.process_index() == 0:
                    print(f"[eval] WARNING: eval_dataset size={n_eval} < process_count={process_count}; some workers will eval 0 samples.")
            else:
                evaluator = SidNextItemJaxEvaluator(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataset=eval_dataset_full,
                    sid_index_path=cfg.data.sid_index_path,
                    info_file=cfg.data.info_file,
                    batch_size=cfg.eval.batch_size,
                    num_beams=cfg.eval.num_beams,
                    max_cache_length=cfg.jax.max_cache_length,
                    topk=eval_topk,
                    show_progress=False,
                )

            eval_every_steps = int(cfg.train.eval_steps)
            if eval_every_steps < 0:
                eval_every_steps = 0
            elif eval_every_steps == 0:
                eval_every_steps = int(steps_per_epoch)

        def eval_cb(step: int, st: Any) -> None:
            nonlocal eval_metrics
            from projects.minionerec.sft.metrics import RankingMetrics

            epoch = float(step) / max(1.0, float(steps_per_epoch))
            t0 = time.perf_counter()

            output_predictions_json = None
            if process_count == 1 and int(step) == int(max_steps) and bool(cfg.eval.save_predictions_json):
                output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json")
            elif process_count > 1 and int(step) == int(max_steps) and bool(cfg.eval.save_predictions_json) and jax.process_index() == 0:
                print("[eval] multi-host periodic eval does not write eval_predictions.json; run a single-host eval job if needed.")

            local_n = 0
            local_invalid = 0
            local_sum_hr = [0.0 for _ in eval_topk]
            local_sum_ndcg_raw = [0.0 for _ in eval_topk]

            if evaluator is not None:
                _preds, metrics_local = evaluator.evaluate(
                    params=st.params,
                    output_predictions_json=output_predictions_json,
                )
                local_n = int(metrics_local.n_samples)
                local_invalid = int(metrics_local.invalid_prediction_count)
                log2 = float(math.log(2.0))
                for i, k in enumerate(eval_topk):
                    local_sum_hr[i] = float(metrics_local.hr[int(k)]) * float(local_n)
                    local_sum_ndcg_raw[i] = (float(metrics_local.ndcg[int(k)]) * float(local_n)) / log2

            # Aggregate sums/counts across processes.
            local_vec = np.asarray(
                [*local_sum_hr, *local_sum_ndcg_raw, float(local_invalid), float(local_n)],
                dtype=np.float64,
            )
            global_vec = local_vec
            if process_allgather is not None:
                gathered = np.asarray(process_allgather(local_vec))
                global_vec = gathered.sum(axis=0)

            m = int(len(eval_topk))
            global_sum_hr = global_vec[:m]
            global_sum_ndcg_raw = global_vec[m : 2 * m]
            global_invalid = int(global_vec[2 * m])
            global_n = int(global_vec[2 * m + 1])

            log2 = float(math.log(2.0))
            hr = {int(k): float(global_sum_hr[i]) / float(global_n) if global_n > 0 else float("nan") for i, k in enumerate(eval_topk)}
            ndcg = {
                int(k): (float(global_sum_ndcg_raw[i]) / float(global_n) * log2) if global_n > 0 else float("nan")
                for i, k in enumerate(eval_topk)
            }
            eval_metrics = RankingMetrics(
                hr=hr,
                ndcg=ndcg,
                topk=list(eval_topk),
                n_samples=int(global_n),
                n_beams=int(cfg.eval.num_beams),
                invalid_prediction_count=int(global_invalid),
            )

            t_eval_s = float(time.perf_counter() - t0)
            if jax.process_index() == 0:
                print(
                    "[eval] "
                    + " ".join(
                        [
                            f"step={int(step)}/{int(max_steps)}",
                            f"epoch={epoch:.2f}",
                            f"samples={int(global_n)}",
                            f"hr@1={hr.get(1, float('nan')):.4f}",
                            f"ndcg@10={ndcg.get(10, float('nan')):.4f}",
                            f"t={t_eval_s:.2f}s",
                        ]
                    )
                )

            if wandb is not None:
                log = {"eval/epoch": epoch}
                for k, v in eval_metrics.hr.items():
                    log[f"eval/hr@{k}"] = v
                for k, v in eval_metrics.ndcg.items():
                    log[f"eval/ndcg@{k}"] = v
                log["eval/invalid_prediction_count"] = int(eval_metrics.invalid_prediction_count)
                log["eval/n_samples"] = int(eval_metrics.n_samples)
                log["time/eval_step_s"] = float(t_eval_s)
                wandb.log(log, step=int(step))

        if eval_cb_impl is None and run_mode_norm == "train_eval" and cfg.eval.enabled and int(eval_every_steps) > 0:
            eval_cb_impl = eval_cb

        state, train_stats = run_sft_train(
            mesh=mesh,
            model=model,
            params=params,
            train_dataset=train_dataset,
            pad_token_id=int(pad_token_id),
            pad_to_length=int(cfg.data.max_len) if int(cfg.data.max_len) > 0 else None,
            pad_policy=str(cfg.data.train_pad_policy or "fixed"),
            pad_buckets=tuple(int(x) for x in (cfg.data.train_pad_buckets or (128, 256, 512))),
            pad_to_multiple_of=int(getattr(cfg.data, "train_pad_to_multiple_of", 8) or 8),
            padding_side=str(cfg.data.padding_side or "left"),
            optimizer_name=cfg.train.optimizer,
            learning_rate=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay),
            muon_aux_learning_rate=float(cfg.train.muon.aux_learning_rate),
            muon_momentum=float(cfg.train.muon.momentum),
            muon_nesterov=bool(cfg.train.muon.nesterov),
            muon_ns_steps=int(cfg.train.muon.ns_steps),
            muon_eps=float(cfg.train.muon.eps),
            muon_max_dim=int(cfg.train.muon.max_dim),
            grad_accum_steps=int(grad_accum_steps),
            micro_batch_size_per_replica=int(cfg.train.per_device_train_batch_size),
            max_steps=int(max_steps),
            seed=int(cfg.seed),
            logging_steps=int(cfg.train.logging_steps),
            warmup_steps=int(cfg.train.warmup_steps),
            log_cb=(
                (
                    lambda step, loss, effective_bs, step_time_sec: wandb.log(
                        {
                            "train/loss": loss,
                            "train/effective_batch_size": effective_bs,
                            "train/step_time_sec": float(step_time_sec),
                            "train/samples_per_sec": float(effective_bs) / max(float(step_time_sec), 1e-9),
                        },
                        step=step,
                    )
                )
                if wandb is not None
                else None
            ),
            log_extra_cb=((lambda step, extra: wandb.log(extra, step=int(step))) if wandb is not None else None),
            eval_every_steps=int(eval_every_steps),
            eval_cb=(eval_cb_impl if eval_every_steps > 0 else None),
            checkpoint_every_steps=int(save_steps),
            checkpoint_cb=(checkpoint_cb if save_steps > 0 else None),
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        tokenizer.save_pretrained(cfg.output_dir)
        if bool(cfg.train.save_last) and jax.process_index() == 0:
            save_checkpoint(output_dir=cfg.output_dir, state=state, name="last")

    # Eval params: prefer trained state, else use placed params.
    eval_params = state.params if state is not None else params
    # For non-eval-only runs, allow eval to use a checkpoint if no trained state is available.
    if state is None and cfg.train.resume_from_checkpoint and not loaded_from_checkpoint:
        payload = load_checkpoint(str(cfg.train.resume_from_checkpoint))
        ckpt_params = payload.get("params")
        if ckpt_params is not None:
            ckpt_params = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.dtype(param_dtype)), ckpt_params)
            eval_params = jax.tree_util.tree_map(lambda x, sh: jax.device_put(jnp.asarray(x, dtype=param_dtype), sh), ckpt_params, shardings)

    if run_mode_norm == "train_eval" and cfg.eval.enabled and eval_metrics is None and int(jax.process_count()) == 1:
        if not cfg.eval.constrained:
            raise NotImplementedError("JAX evaluator currently supports only constrained=true (SID trie).")
        eval_dataset = _build_eval_dataset(cfg, tokenizer)
        output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json") if cfg.eval.save_predictions_json else None
        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=eval_params,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            sid_index_path=cfg.data.sid_index_path,
            info_file=cfg.data.info_file,
            batch_size=cfg.eval.batch_size,
            num_beams=cfg.eval.num_beams,
            max_cache_length=cfg.jax.max_cache_length,
            topk=list(cfg.eval.topk),
            output_predictions_json=output_predictions_json,
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log, step=int(getattr(state, "step", 0) or 0))

    if run_mode_norm == "eval" and cfg.eval.enabled:
        if not cfg.eval.constrained:
            raise NotImplementedError("JAX evaluator currently supports only constrained=true (SID trie).")
        eval_dataset = _build_eval_dataset(cfg, tokenizer)
        output_predictions_json = os.path.join(cfg.output_dir, "eval_predictions.json") if cfg.eval.save_predictions_json else None
        _preds, eval_metrics = evaluate_sid_next_item_jax(
            model=model,
            params=eval_params,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            sid_index_path=cfg.data.sid_index_path,
            info_file=cfg.data.info_file,
            batch_size=cfg.eval.batch_size,
            num_beams=cfg.eval.num_beams,
            max_cache_length=cfg.jax.max_cache_length,
            topk=list(cfg.eval.topk),
            output_predictions_json=output_predictions_json,
        )

        if wandb is not None:
            log = {}
            for k, v in eval_metrics.hr.items():
                log[f"eval/hr@{k}"] = v
            for k, v in eval_metrics.ndcg.items():
                log[f"eval/ndcg@{k}"] = v
            log["eval/invalid_prediction_count"] = eval_metrics.invalid_prediction_count
            wandb.log(log, step=0)

    if wandb is not None:
        wandb.finish()

    return {
        "config": asdict(cfg),
        "token_extension": asdict(extension),
        "vocab_resize": asdict(vocab_resize),
        "train": asdict(train_stats) if train_stats else None,
        "eval": asdict(eval_metrics) if eval_metrics else None,
    }


def run_sid_sft(cfg: SidSftConfig, *, run_mode: str) -> dict[str, Any]:
    _set_seed(cfg.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_mode_norm = str(run_mode).strip().lower()
    if run_mode_norm not in {"train", "eval", "train_eval"}:
        raise ValueError("run_mode must be one of: train, eval, train_eval")

    backend = str(cfg.backend or "jax").strip().lower()
    if backend in {"jax", "tpu"}:
        return _run_sid_sft_jax(cfg, run_mode_norm=run_mode_norm)
    raise ValueError(f"Unknown backend={cfg.backend!r} (expected jax|tpu)")

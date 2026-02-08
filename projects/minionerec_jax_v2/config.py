from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plugins.training.core.config.loader import load_config as _load_config

_ALLOWED_DATASET_NAMES = {"industrial", "office"}
_ALLOWED_PREFILL_MODES = {"bucket", "fixed", "exact"}
_ALLOWED_RUN_MODES = {"eval"}


DEFAULT_CONFIG: dict[str, Any] = {
    "checkpoint": {
        "repo_id": "kkknight/MiniOneRec",
        "revision": "main",
        "local_root": None,
        "subdir": None,
        "trust_remote_code": True,
    },
    "dataset": {
        "source_root": "data/minionerec",
        "dataset_name": "industrial",
        "test_file": None,
        "info_file": None,
        "sid_index_path": None,
        "max_len": 2560,
        "sample_test": -1,
        "dedup": False,
        "truncate_to_max_len": False,
    },
    "decode": {
        "batch_size": 4,
        "num_beams": 50,
        "length_penalty": 0.0,
        "max_new_tokens": 256,
        "prefill_mode": "bucket",
        "fixed_prefill_len": None,
        "do_sample": False,
        "temperature": 1.0,
    },
    "eval": {
        "topk": [3, 5, 10],
        "save_predictions_json": True,
        "output_predictions_name": "eval_predictions.json",
        "show_progress": False,
    },
    "runtime": {
        "output_dir": "runs/minionerec_jax_v2_eval",
        "run_mode": "eval",
        "seed": 42,
    },
    "jax": {
        "mesh_shape": "auto",
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        "max_cache_length": 4096,
    },
    "wandb": {
        "project": "minionerec-jax-v2",
        "mode": "disabled",
        "name": None,
    },
}


@dataclass(frozen=True)
class MiniOneRecCheckpointConfig:
    repo_id: str
    revision: str
    local_root: str | None
    subdir: str | None
    trust_remote_code: bool


@dataclass(frozen=True)
class MiniOneRecDatasetConfig:
    source_root: str
    dataset_name: str
    test_file: str | None
    info_file: str | None
    sid_index_path: str | None
    max_len: int
    sample_test: int
    dedup: bool
    truncate_to_max_len: bool

    def __post_init__(self) -> None:
        if self.dataset_name not in _ALLOWED_DATASET_NAMES:
            allowed = ", ".join(sorted(_ALLOWED_DATASET_NAMES))
            raise ValueError(f"dataset_name must be one of {{{allowed}}}, got: {self.dataset_name!r}")
        if int(self.max_len) <= 0:
            raise ValueError(f"dataset.max_len must be > 0, got: {int(self.max_len)}")


@dataclass(frozen=True)
class MiniOneRecDecodeConfig:
    batch_size: int
    num_beams: int
    length_penalty: float
    max_new_tokens: int
    prefill_mode: str
    fixed_prefill_len: int | None
    do_sample: bool
    temperature: float

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError(f"decode.batch_size must be > 0, got: {int(self.batch_size)}")
        if int(self.num_beams) <= 0:
            raise ValueError(f"decode.num_beams must be > 0, got: {int(self.num_beams)}")
        if int(self.max_new_tokens) <= 0:
            raise ValueError(f"decode.max_new_tokens must be > 0, got: {int(self.max_new_tokens)}")
        if self.prefill_mode not in _ALLOWED_PREFILL_MODES:
            allowed = ", ".join(sorted(_ALLOWED_PREFILL_MODES))
            raise ValueError(f"decode.prefill_mode must be one of {{{allowed}}}, got: {self.prefill_mode!r}")
        if self.fixed_prefill_len is not None and int(self.fixed_prefill_len) <= 0:
            raise ValueError(f"decode.fixed_prefill_len must be > 0, got: {self.fixed_prefill_len!r}")
        if float(self.temperature) <= 0.0:
            raise ValueError(f"decode.temperature must be > 0, got: {float(self.temperature)}")


@dataclass(frozen=True)
class MiniOneRecEvalConfig:
    topk: tuple[int, ...]
    save_predictions_json: bool
    output_predictions_name: str
    show_progress: bool

    def __post_init__(self) -> None:
        if len(self.topk) == 0:
            raise ValueError("eval.topk must be non-empty")
        if not str(self.output_predictions_name or "").endswith(".json"):
            raise ValueError(
                f"eval.output_predictions_name must end with '.json', got: {self.output_predictions_name!r}"
            )


@dataclass(frozen=True)
class MiniOneRecRuntimeConfig:
    output_dir: str
    run_mode: str
    seed: int

    def __post_init__(self) -> None:
        if self.run_mode not in _ALLOWED_RUN_MODES:
            allowed = ", ".join(sorted(_ALLOWED_RUN_MODES))
            raise ValueError(f"runtime.run_mode must be one of {{{allowed}}}, got: {self.run_mode!r}")


@dataclass(frozen=True)
class MiniOneRecJaxConfig:
    mesh_shape: str
    param_dtype: str
    compute_dtype: str
    max_cache_length: int

    def __post_init__(self) -> None:
        if int(self.max_cache_length) <= 0:
            raise ValueError(f"jax.max_cache_length must be > 0, got: {int(self.max_cache_length)}")


@dataclass(frozen=True)
class MiniOneRecWandbConfig:
    project: str
    mode: str
    name: str | None


@dataclass(frozen=True)
class MiniOneRecJaxV2Config:
    config_path: str
    checkpoint: MiniOneRecCheckpointConfig
    dataset: MiniOneRecDatasetConfig
    decode: MiniOneRecDecodeConfig
    eval: MiniOneRecEvalConfig
    runtime: MiniOneRecRuntimeConfig
    jax: MiniOneRecJaxConfig
    wandb: MiniOneRecWandbConfig

    @property
    def dataset_name(self) -> str:
        return self.dataset.dataset_name

    @property
    def dataset_source_root(self) -> str:
        return self.dataset.source_root


def _get_by_path(cfg: dict[str, Any], key_path: str) -> Any:
    keys = [k for k in key_path.split(".") if k]
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _get_or_default(cfg: dict[str, Any], key_path: str, default: Any) -> Any:
    value = _get_by_path(cfg, key_path)
    return default if value is None else value


def load_config(config_path: str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    return _load_config(DEFAULT_CONFIG, config_path, overrides=overrides)


def config_from_dict(cfg: dict[str, Any], *, config_path: str) -> MiniOneRecJaxV2Config:
    checkpoint = MiniOneRecCheckpointConfig(
        repo_id=str(_get_or_default(cfg, "checkpoint.repo_id", DEFAULT_CONFIG["checkpoint"]["repo_id"])),
        revision=str(_get_or_default(cfg, "checkpoint.revision", DEFAULT_CONFIG["checkpoint"]["revision"])),
        local_root=(
            None
            if _get_or_default(cfg, "checkpoint.local_root", DEFAULT_CONFIG["checkpoint"]["local_root"]) is None
            else str(_get_or_default(cfg, "checkpoint.local_root", DEFAULT_CONFIG["checkpoint"]["local_root"]))
        ),
        subdir=(
            None
            if _get_or_default(cfg, "checkpoint.subdir", DEFAULT_CONFIG["checkpoint"]["subdir"]) is None
            else str(_get_or_default(cfg, "checkpoint.subdir", DEFAULT_CONFIG["checkpoint"]["subdir"]))
        ),
        trust_remote_code=bool(
            _get_or_default(cfg, "checkpoint.trust_remote_code", DEFAULT_CONFIG["checkpoint"]["trust_remote_code"])
        ),
    )

    dataset = MiniOneRecDatasetConfig(
        source_root=str(_get_or_default(cfg, "dataset.source_root", DEFAULT_CONFIG["dataset"]["source_root"])),
        dataset_name=str(_get_or_default(cfg, "dataset.dataset_name", DEFAULT_CONFIG["dataset"]["dataset_name"])),
        test_file=(
            None
            if _get_or_default(cfg, "dataset.test_file", DEFAULT_CONFIG["dataset"]["test_file"]) is None
            else str(_get_or_default(cfg, "dataset.test_file", DEFAULT_CONFIG["dataset"]["test_file"]))
        ),
        info_file=(
            None
            if _get_or_default(cfg, "dataset.info_file", DEFAULT_CONFIG["dataset"]["info_file"]) is None
            else str(_get_or_default(cfg, "dataset.info_file", DEFAULT_CONFIG["dataset"]["info_file"]))
        ),
        sid_index_path=(
            None
            if _get_or_default(cfg, "dataset.sid_index_path", DEFAULT_CONFIG["dataset"]["sid_index_path"]) is None
            else str(_get_or_default(cfg, "dataset.sid_index_path", DEFAULT_CONFIG["dataset"]["sid_index_path"]))
        ),
        max_len=int(_get_or_default(cfg, "dataset.max_len", DEFAULT_CONFIG["dataset"]["max_len"])),
        sample_test=int(_get_or_default(cfg, "dataset.sample_test", DEFAULT_CONFIG["dataset"]["sample_test"])),
        dedup=bool(_get_or_default(cfg, "dataset.dedup", DEFAULT_CONFIG["dataset"]["dedup"])),
        truncate_to_max_len=bool(
            _get_or_default(cfg, "dataset.truncate_to_max_len", DEFAULT_CONFIG["dataset"]["truncate_to_max_len"])
        ),
    )

    decode = MiniOneRecDecodeConfig(
        batch_size=int(_get_or_default(cfg, "decode.batch_size", DEFAULT_CONFIG["decode"]["batch_size"])),
        num_beams=int(_get_or_default(cfg, "decode.num_beams", DEFAULT_CONFIG["decode"]["num_beams"])),
        length_penalty=float(
            _get_or_default(cfg, "decode.length_penalty", DEFAULT_CONFIG["decode"]["length_penalty"])
        ),
        max_new_tokens=int(_get_or_default(cfg, "decode.max_new_tokens", DEFAULT_CONFIG["decode"]["max_new_tokens"])),
        prefill_mode=str(_get_or_default(cfg, "decode.prefill_mode", DEFAULT_CONFIG["decode"]["prefill_mode"])),
        fixed_prefill_len=(
            None
            if _get_or_default(cfg, "decode.fixed_prefill_len", DEFAULT_CONFIG["decode"]["fixed_prefill_len"]) is None
            else int(_get_or_default(cfg, "decode.fixed_prefill_len", DEFAULT_CONFIG["decode"]["fixed_prefill_len"]))
        ),
        do_sample=bool(_get_or_default(cfg, "decode.do_sample", DEFAULT_CONFIG["decode"]["do_sample"])),
        temperature=float(_get_or_default(cfg, "decode.temperature", DEFAULT_CONFIG["decode"]["temperature"])),
    )

    topk_raw = _get_or_default(cfg, "eval.topk", DEFAULT_CONFIG["eval"]["topk"])
    if not isinstance(topk_raw, (list, tuple)):
        raise TypeError(f"eval.topk must be list/tuple[int], got: {type(topk_raw).__name__}")
    eval_cfg = MiniOneRecEvalConfig(
        topk=tuple(int(k) for k in topk_raw),
        save_predictions_json=bool(
            _get_or_default(cfg, "eval.save_predictions_json", DEFAULT_CONFIG["eval"]["save_predictions_json"])
        ),
        output_predictions_name=str(
            _get_or_default(cfg, "eval.output_predictions_name", DEFAULT_CONFIG["eval"]["output_predictions_name"])
        ),
        show_progress=bool(_get_or_default(cfg, "eval.show_progress", DEFAULT_CONFIG["eval"]["show_progress"])),
    )

    runtime = MiniOneRecRuntimeConfig(
        output_dir=str(_get_or_default(cfg, "runtime.output_dir", DEFAULT_CONFIG["runtime"]["output_dir"])),
        run_mode=str(_get_or_default(cfg, "runtime.run_mode", DEFAULT_CONFIG["runtime"]["run_mode"])),
        seed=int(_get_or_default(cfg, "runtime.seed", DEFAULT_CONFIG["runtime"]["seed"])),
    )

    jax_cfg = MiniOneRecJaxConfig(
        mesh_shape=str(_get_or_default(cfg, "jax.mesh_shape", DEFAULT_CONFIG["jax"]["mesh_shape"])),
        param_dtype=str(_get_or_default(cfg, "jax.param_dtype", DEFAULT_CONFIG["jax"]["param_dtype"])),
        compute_dtype=str(_get_or_default(cfg, "jax.compute_dtype", DEFAULT_CONFIG["jax"]["compute_dtype"])),
        max_cache_length=int(_get_or_default(cfg, "jax.max_cache_length", DEFAULT_CONFIG["jax"]["max_cache_length"])),
    )

    wandb_name_raw = _get_or_default(cfg, "wandb.name", DEFAULT_CONFIG["wandb"]["name"])
    wandb_cfg = MiniOneRecWandbConfig(
        project=str(_get_or_default(cfg, "wandb.project", DEFAULT_CONFIG["wandb"]["project"])),
        mode=str(_get_or_default(cfg, "wandb.mode", DEFAULT_CONFIG["wandb"]["mode"])),
        name=None if wandb_name_raw is None else str(wandb_name_raw),
    )

    return MiniOneRecJaxV2Config(
        config_path=config_path,
        checkpoint=checkpoint,
        dataset=dataset,
        decode=decode,
        eval=eval_cfg,
        runtime=runtime,
        jax=jax_cfg,
        wandb=wandb_cfg,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "MiniOneRecCheckpointConfig",
    "MiniOneRecDatasetConfig",
    "MiniOneRecDecodeConfig",
    "MiniOneRecEvalConfig",
    "MiniOneRecRuntimeConfig",
    "MiniOneRecJaxConfig",
    "MiniOneRecWandbConfig",
    "MiniOneRecJaxV2Config",
    "load_config",
    "config_from_dict",
]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from safetensors.numpy import load_file


def _load_local_safetensors_dir(model_dir: Path) -> dict[str, Any]:
    single = model_dir / "model.safetensors"
    if single.is_file():
        return dict(load_file(str(single)))

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shard_files = sorted(set(str(v) for v in weight_map.values()))
        if not shard_files:
            raise ValueError(f"Empty safetensors index: {index_path}")

        state: dict[str, Any] = {}
        for shard in shard_files:
            shard_path = model_dir / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing safetensors shard referenced by index: {shard_path}")
            state.update(load_file(str(shard_path)))
        return state

    candidates = sorted(model_dir.glob("*.safetensors"))
    if len(candidates) == 1:
        return dict(load_file(str(candidates[0])))
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple *.safetensors found under {model_dir} but no index file; "
            "expected model.safetensors or model.safetensors.index.json"
        )

    raise FileNotFoundError(
        f"No safetensors weights found under {model_dir}; "
        "expected model.safetensors or model.safetensors.index.json"
    )


def _load_hub_safetensors(repo_id: str) -> dict[str, Any]:
    api = HfApi()
    files = set(api.list_repo_files(repo_id=repo_id))

    index_name = "model.safetensors.index.json"
    if index_name in files:
        index_path = hf_hub_download(repo_id=repo_id, filename=index_name)
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shard_files = sorted(set(str(v) for v in weight_map.values()))
        if not shard_files:
            raise ValueError(f"Empty safetensors index in repo: {repo_id}")

        state: dict[str, Any] = {}
        for shard in shard_files:
            shard_path = hf_hub_download(repo_id=repo_id, filename=shard)
            state.update(load_file(str(shard_path)))
        return state

    single_name = "model.safetensors"
    if single_name in files:
        path = hf_hub_download(repo_id=repo_id, filename=single_name)
        return dict(load_file(str(path)))

    raise FileNotFoundError(
        f"Repo {repo_id!r} has no safetensors weights (no {single_name!r} or {index_name!r}). "
        "If it only provides pytorch_model.bin, either convert to safetensors, "
        "run with train.train_from_scratch=true, or provide train.resume_from_checkpoint."
    )


def load_hf_safetensors_state_dict(base_model: str) -> dict[str, Any]:
    model_dir = Path(base_model)
    if model_dir.is_dir():
        return _load_local_safetensors_dir(model_dir)
    return _load_hub_safetensors(str(base_model))


__all__ = ["load_hf_safetensors_state_dict"]


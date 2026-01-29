from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
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


def _looks_like_missing_hf_file_error(e: Exception) -> bool:
    # huggingface_hub exception types vary by version; keep this conservative.
    name = type(e).__name__
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return False
    if name in {"EntryNotFoundError", "LocalEntryNotFoundError"}:
        return True
    msg = str(e)
    if "404" in msg and ("Not Found" in msg or "not found" in msg):
        return True
    return False


def _hf_hub_download_with_retry(
    *,
    repo_id: str,
    filename: str,
    retries: int = 5,
    min_sleep_s: float = 2.0,
) -> str:
    last: Exception | None = None
    for attempt in range(int(retries)):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            last = e
            if attempt >= int(retries) - 1:
                break
            sleep_s = max(float(min_sleep_s), float(min_sleep_s) * (2.0**attempt))
            time.sleep(sleep_s)
    assert last is not None
    raise last


def _load_hub_safetensors(repo_id: str) -> dict[str, Any]:
    index_name = "model.safetensors.index.json"
    try:
        index_path = _hf_hub_download_with_retry(repo_id=repo_id, filename=index_name)
    except Exception as e:
        if _looks_like_missing_hf_file_error(e):
            index_path = None
        else:
            raise

    if index_path is not None:
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        weight_map = data.get("weight_map") or {}
        shard_files = sorted(set(str(v) for v in weight_map.values()))
        if not shard_files:
            raise ValueError(f"Empty safetensors index in repo: {repo_id}")

        state: dict[str, Any] = {}
        for shard in shard_files:
            shard_path = _hf_hub_download_with_retry(repo_id=repo_id, filename=shard)
            state.update(load_file(str(shard_path)))
        return state

    single_name = "model.safetensors"
    try:
        path = _hf_hub_download_with_retry(repo_id=repo_id, filename=single_name)
    except Exception as e:
        if _looks_like_missing_hf_file_error(e):
            raise FileNotFoundError(
                f"Repo {repo_id!r} has no safetensors weights (no {single_name!r} or {index_name!r}). "
                "If it only provides pytorch_model.bin, either convert to safetensors, "
                "run with train.train_from_scratch=true, or provide train.resume_from_checkpoint."
            ) from e
        raise

    if path:
        return dict(load_file(str(path)))

    raise FileNotFoundError(f"hf_hub_download returned empty path for {repo_id!r}/{single_name!r}.")


def load_hf_safetensors_state_dict(base_model: str) -> dict[str, Any]:
    model_dir = Path(base_model)
    if model_dir.is_dir():
        return _load_local_safetensors_dir(model_dir)
    return _load_hub_safetensors(str(base_model))


__all__ = ["load_hf_safetensors_state_dict"]

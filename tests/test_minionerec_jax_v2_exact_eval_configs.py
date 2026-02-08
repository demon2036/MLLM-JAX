from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> dict:
    cfg_path = REPO_ROOT / path
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def test_office_exact_config_fields() -> None:
    cfg = _load_yaml("projects/minionerec_jax_v2/configs/eval_official_office_paper_exact_batch8_f32bf16.yaml")
    assert cfg["dataset"]["dataset_name"] == "office"
    assert cfg["decode"]["prefill_mode"] == "exact"
    assert cfg["decode"]["batch_size"] == 8
    assert cfg["decode"]["do_sample"] is False
    assert cfg["decode"]["temperature"] == 1.0
    assert tuple(cfg["eval"]["topk"]) == (3, 5, 10)
    assert cfg["runtime"]["run_mode"] == "eval"


def test_industrial_exact_config_fields() -> None:
    cfg = _load_yaml("projects/minionerec_jax_v2/configs/eval_official_industrial_paper_exact_batch8_f32bf16.yaml")
    assert cfg["dataset"]["dataset_name"] == "industrial"
    assert cfg["decode"]["prefill_mode"] == "exact"
    assert cfg["decode"]["batch_size"] == 8
    assert cfg["decode"]["do_sample"] is False
    assert cfg["decode"]["temperature"] == 1.0
    assert tuple(cfg["eval"]["topk"]) == (3, 5, 10)
    assert cfg["runtime"]["run_mode"] == "eval"


def test_office_beam_sample_config_fields() -> None:
    cfg = _load_yaml("projects/minionerec_jax_v2/configs/eval_official_office_beam_sample_temp1p0.yaml")
    assert cfg["dataset"]["dataset_name"] == "office"
    assert cfg["decode"]["prefill_mode"] == "bucket"
    assert cfg["decode"]["do_sample"] is True
    assert cfg["decode"]["temperature"] == 1.0
    assert tuple(cfg["eval"]["topk"]) == (3, 5, 10)
    assert cfg["runtime"]["run_mode"] == "eval"


def test_industrial_beam_sample_config_fields() -> None:
    cfg = _load_yaml("projects/minionerec_jax_v2/configs/eval_official_industrial_beam_sample_temp1p0.yaml")
    assert cfg["dataset"]["dataset_name"] == "industrial"
    assert cfg["decode"]["prefill_mode"] == "bucket"
    assert cfg["decode"]["do_sample"] is True
    assert cfg["decode"]["temperature"] == 1.0
    assert tuple(cfg["eval"]["topk"]) == (3, 5, 10)
    assert cfg["runtime"]["run_mode"] == "eval"

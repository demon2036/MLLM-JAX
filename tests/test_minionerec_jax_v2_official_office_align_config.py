from __future__ import annotations

from pathlib import Path

import yaml


def test_official_office_align_config_required_fields() -> None:
    config_path = Path(__file__).resolve().parents[1] / "projects/minionerec_jax_v2/configs/eval_official_office_align.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["dataset"]["max_len"] == 512
    assert cfg["decode"]["batch_size"] == 2
    assert cfg["decode"]["prefill_mode"] == "fixed"
    assert cfg["decode"]["fixed_prefill_len"] is None
    assert cfg["decode"]["do_sample"] is False
    assert cfg["decode"]["temperature"] == 1.0
    assert cfg["jax"]["param_dtype"] == "bfloat16"
    assert cfg["jax"]["compute_dtype"] == "bfloat16"
    assert cfg["jax"]["max_cache_length"] == 512
    assert tuple(cfg["eval"]["topk"]) == (3, 5, 10)
    assert cfg["runtime"]["run_mode"] == "eval"

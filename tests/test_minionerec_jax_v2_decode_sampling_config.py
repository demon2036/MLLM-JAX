from __future__ import annotations

import pytest

from projects.minionerec_jax_v2.config import DEFAULT_CONFIG, config_from_dict


def _with_decode(**kwargs: object) -> dict[str, object]:
    cfg = {
        "checkpoint": dict(DEFAULT_CONFIG["checkpoint"]),
        "dataset": dict(DEFAULT_CONFIG["dataset"]),
        "decode": dict(DEFAULT_CONFIG["decode"]),
        "eval": dict(DEFAULT_CONFIG["eval"]),
        "runtime": dict(DEFAULT_CONFIG["runtime"]),
        "jax": dict(DEFAULT_CONFIG["jax"]),
        "wandb": dict(DEFAULT_CONFIG["wandb"]),
    }
    cfg["decode"].update(kwargs)
    return cfg


def test_decode_sampling_defaults() -> None:
    cfg = config_from_dict(_with_decode(), config_path="<test-defaults>")
    assert cfg.decode.do_sample is False
    assert cfg.decode.temperature == pytest.approx(1.0)


def test_decode_sampling_fields_are_read() -> None:
    cfg = config_from_dict(
        _with_decode(do_sample=True, temperature=0.7),
        config_path="<test-sampling-enabled>",
    )
    assert cfg.decode.do_sample is True
    assert cfg.decode.temperature == pytest.approx(0.7)


@pytest.mark.parametrize("bad_temperature", [0.0, -0.5])
def test_decode_temperature_must_be_positive(bad_temperature: float) -> None:
    with pytest.raises(ValueError, match="decode.temperature must be > 0"):
        _ = config_from_dict(
            _with_decode(temperature=bad_temperature),
            config_path="<test-bad-temperature>",
        )

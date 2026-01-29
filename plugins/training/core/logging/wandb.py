from __future__ import annotations

from dataclasses import asdict
from typing import Any


def maybe_init_wandb(
    *,
    cfg: Any,
    project: str,
    name: str | None,
    mode: str,
    process_index: int | None = None,
) -> Any | None:
    """Best-effort W&B init helper shared across plugin runners."""
    if process_index is not None and int(process_index) != 0:
        return None

    mode_norm = str(mode or "online").strip().lower()
    if mode_norm in {"disabled", "disable", "off"}:
        return None

    try:
        import wandb  # type: ignore

        wandb.init(
            project=str(project),
            name=name,
            config=asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg,
            mode=mode_norm,
        )
        return wandb
    except Exception as e:
        print(f"wandb disabled due to init error: {e}")
        return None


__all__ = ["maybe_init_wandb"]


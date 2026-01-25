from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from plugins.common.wandb_utils import maybe_init_wandb


@dataclass(frozen=True)
class WandbRunSpec:
    project: str
    mode: str = "online"
    name: str | None = None


class StatsLogger:
    """Single writer for metrics (W&B, console, etc).

    Design goal: runner code never calls `wandb.log` directly; it calls this.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        wandb: WandbRunSpec,
        process_index: int | None = None,
    ) -> None:
        self._wandb = maybe_init_wandb(
            cfg=cfg,
            project=str(wandb.project),
            name=wandb.name,
            mode=str(wandb.mode),
            process_index=process_index,
        )
        self._last_step: int = -1

    def commit(self, data: Mapping[str, Any] | Iterable[Mapping[str, Any]], *, step: int | None = None) -> int:
        """Commit one or more metric dicts and return the last used step."""
        items = [data] if isinstance(data, Mapping) else list(data)
        if not items:
            return self._last_step

        # W&B supports multiple logs at the same `step` (fields are merged). We only
        # guard against out-of-order steps going backwards by clamping to
        # `self._last_step`.
        base = self._last_step + 1 if step is None else max(int(step), self._last_step)
        for offset, item in enumerate(items):
            log_step = base + offset
            filtered = {k: v for k, v in dict(item).items() if not str(k).endswith("__count")}
            if self._wandb is not None:
                self._wandb.log(filtered, step=log_step)
            self._last_step = max(self._last_step, log_step)
        return self._last_step

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


__all__ = ["StatsLogger", "WandbRunSpec"]

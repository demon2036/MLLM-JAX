from __future__ import annotations

import os
from typing import Iterable


def load_dotenv_if_present(*, repo_root: str, extra_paths: Iterable[str] | None = None) -> None:
    """Load KEY=VALUE lines from common dotenv locations if present.

    This is a minimal helper shared across CLI scripts; it is not a full dotenv
    implementation (no export semantics, no multiline values).
    """
    candidates = [
        os.path.join(str(repo_root), ".env"),
        "/root/.env",
    ]
    if extra_paths:
        candidates.extend([str(p) for p in extra_paths])

    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                existing = os.environ.get(k)
                if existing is None or str(existing).strip() == "":
                    os.environ[k] = v
        return


__all__ = ["load_dotenv_if_present"]


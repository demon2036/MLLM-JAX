from __future__ import annotations

from pathlib import Path


def load_valid_sids_from_info(info_file: str) -> list[str]:
    lines = Path(info_file).read_text(encoding="utf-8").splitlines()
    sids: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        # expected format: sid \t title \t item_id
        sid = line.split("\t", 1)[0].strip()
        if sid:
            sids.append(sid)
    return sids


__all__ = ["load_valid_sids_from_info"]


"""Append-only JSONL records (flushed per line so a crash loses at most one record)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(_clean(rec), default=str) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

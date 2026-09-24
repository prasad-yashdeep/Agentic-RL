"""Append-only JSONL metrics for runs (so nothing is lost the way G4's judge scores were)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def write(self, record: dict) -> None:
        rec = {"ts": time.time(), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
        self.n += 1


def read_metrics(path: str | Path):
    """DataFrame of all records (column `kind` distinguishes rollout/update/sync/alarm/step_summary)."""
    import pandas as pd

    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

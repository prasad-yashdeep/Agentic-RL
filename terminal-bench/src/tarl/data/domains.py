"""Task -> domain mapping for the 89 Terminal-Bench 2.1 tasks (configs/tasks/tb2_domains.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from tarl.common.paths import TB2_DOMAINS

UNKNOWN = "unknown"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def task_domains(path: str | Path = TB2_DOMAINS) -> dict[str, str]:
    """{task_name: domain} built from the YAML; duplicates raise."""
    cfg = _load(str(path))
    out: dict[str, str] = {}
    for dom, tasks in (cfg.get("domains") or {}).items():
        for t in tasks or []:
            if t in out:
                raise ValueError(f"task {t!r} listed under both {out[t]!r} and {dom!r}")
            out[t] = dom
    return out


def domain_order(path: str | Path = TB2_DOMAINS) -> list[str]:
    return list((_load(str(path)).get("domains") or {}).keys())


def domain_of(task: str, path: str | Path = TB2_DOMAINS) -> str:
    return task_domains(path).get(task, UNKNOWN)


def domain_table(path: str | Path = TB2_DOMAINS):
    """DataFrame(domain, n_tasks, tasks) in YAML order."""
    import pandas as pd

    cfg = _load(str(path))
    rows = [
        {"domain": d, "n_tasks": len(ts or []), "tasks": list(ts or [])}
        for d, ts in (cfg.get("domains") or {}).items()
    ]
    return pd.DataFrame(rows)


def taxonomy_version(path: str | Path = TB2_DOMAINS) -> str:
    return str(_load(str(path)).get("version", "?"))

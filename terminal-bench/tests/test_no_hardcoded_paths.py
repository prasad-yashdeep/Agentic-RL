"""Guard for the 2026-09-24 monorepo cut-over: live code must not reference the old checkout path.

There is no compatibility symlink, so a script that still says ~/terminal-agent-rl either fails or, after the
read-only placeholder is removed, silently recreates a stray tree (Harbor `mkdir -p`s its --jobs-dir).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OLD = re.compile(r"(/home/yashp|\$HOME|~)/terminal-agent" r"-rl")
ALLOW = {"scripts/06_grpo_micro.py", "configs/lockfile.txt"}  # frozen gate-G4 artifact, frozen install record
CODE_SUFFIXES = {".sh", ".py", ".toml", ".yaml", ".yml", ".mk"}


def _tracked() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "--", "."], text=True, timeout=20)
    except Exception:  # pragma: no cover - not a git checkout
        pytest.skip("not a git checkout")
    return [REPO / p for p in out.splitlines()]


def test_live_code_has_no_old_checkout_path():
    offenders = []
    for p in _tracked():
        rel = p.relative_to(REPO).as_posix()
        if rel in ALLOW or not p.is_file():
            continue
        if p.suffix not in CODE_SUFFIXES and p.name != "Makefile":
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if OLD.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not offenders, "old checkout path in live code:\n" + "\n".join(offenders)


def test_scripts_locate_themselves():
    """Every driver-path script derives REPO from its own location (pwd -P), with a TARL_REPO override."""
    for name in ["09_experiments.sh", "watch.sh", "04_baseline_eval.sh", "04b_a0_driver.sh", "07_sync_node05.sh", "_node08_health.sh"]:
        text = (REPO / "scripts" / name).read_text()
        assert 'REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"' in text, name

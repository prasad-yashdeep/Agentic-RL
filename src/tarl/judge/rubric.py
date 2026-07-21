"""Pinned rubric loading with version hashing (every judge record carries the
rubric hash so scores across rubric revisions are never silently mixed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

RUBRIC_DIR = Path(__file__).resolve().parents[3] / "configs" / "judge"


def load_rubric(name: str = "rubric-v1.md") -> tuple[str, str]:
    """Returns (rubric_text, sha256_12)."""
    text = (RUBRIC_DIR / name).read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]

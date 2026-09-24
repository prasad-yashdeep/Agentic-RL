"""scripts/env.sh must put the venv first, then the singularity shims, and activate the shared monorepo venv."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARL = REPO.parent
VENV = ARL / ".venv"


def _source(expr: str) -> str:
    cmd = f'. "{REPO}/scripts/env.sh" >/dev/null 2>&1; {expr}'
    return subprocess.check_output(["bash", "-c", cmd], text=True, timeout=30, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}).strip()


def test_shims_on_path_from_real_location():
    parts = _source('echo "$PATH"').split(":")
    shims = str(REPO / "scripts" / "shims")
    assert shims in parts, parts[:4]
    if VENV.exists():
        assert parts[0] == str(VENV / "bin"), parts[:3]  # activate prepends the venv
        assert parts[1] == shims, parts[:3]
    else:
        assert parts[0] == shims


@pytest.mark.skipif(not (VENV / "bin" / "python").exists(), reason="shared venv not present")
def test_activates_shared_venv():
    assert _source('echo "$VIRTUAL_ENV"') == str(VENV)
    assert _source("command -v python") == str(VENV / "bin" / "python")

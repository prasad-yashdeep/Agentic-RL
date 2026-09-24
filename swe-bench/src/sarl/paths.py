"""Single source of every location sarl touches. Self-locating; nothing hard-codes the checkout path."""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
REPO = Path(os.environ.get("SARL_REPO") or Path(__file__).resolve().parents[2])  # swe-bench/
ARL = REPO.parent  # Agentic-RL monorepo root
VENV = Path(os.environ.get("ARL_VENV", str(ARL / ".venv")))

REFERENCE = REPO / "reference"
UPSTREAM = REFERENCE / "_upstream"  # fetched at the pinned SHA, gitignored (no upstream license)
UPSTREAM_SHA = "8cddfb60c93cc6b2c7d9dc047939084858f30f76"
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"
EXPERIMENTS = REPO / "experiments"
DOCS = REPO / "docs"
NOTEBOOKS = REPO / "notebooks"
HF_CACHE = Path(os.environ.get("HF_HOME", str(HOME / "hf-cache")))

# node-local (never /home: the NFS disk is shared and nearly full)
SIF_DIR = Path(os.environ.get("SARL_SIF_DIR", "/tmp/swe-sif"))
SCRATCH = Path(os.environ.get("SARL_SCRATCH", "/var/tmp/swe"))
APPTAINER_CACHEDIR = SCRATCH / "apptainer-cache"
APPTAINER_TMPDIR = SCRATCH / "apptainer-tmp"
GRADER_VENV = Path(os.environ.get("SARL_GRADER_VENV", str(SCRATCH / "grader-venv")))
APPTAINER = Path(os.environ.get("SARL_APPTAINER", str(HOME / "apptainer-env" / "bin" / "apptainer")))
FAKEROOT_COMPAT = HOME / "fakeroot-compat" / "bin"

# pinned upstream assets
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
DATASETS = {
    # name: (hf repo, revision, split)
    "dev": ("SWE-bench/SWE-bench", "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2", "dev"),  # training (repos disjoint from Verified)
    "verified": ("SWE-bench/SWE-bench_Verified", "78f471bf655a3137b2e8a75af1501690ec009ec3", "test"),  # eval only
    "verified_princeton": ("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "test"),  # the reference's copy
}


def refuse_on_home(path: Path, what: str) -> Path:
    """Big node-local artefacts (images, overlays, grader venv) must never land on the NFS /home."""
    p = Path(path).resolve()
    if str(p).startswith(str(HOME.resolve())):
        raise RuntimeError(f"{what} must be node-local, not under {HOME} (NFS, ~92% full): {p}")
    return p

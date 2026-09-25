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
# pinned policy models (key -> (hf id, revision)); the same Qwen2.5 tokenizer/chat template for all of them
MODELS = {
    "0.5b": (MODEL_ID, MODEL_REVISION),
    # 1.5b and 7b-awq: S9 probe 2026-09-24 — 0 valid <tool_call>s (they write JSON inside markdown fences); snapshots deleted
    "1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
    "7b-awq": ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", "8e8ed243bbe6f9a5aff549a0924562fc719b2b8a"),  # int4 serve; train would need QLoRA
    # Qwen3 (2025-04, non-thinking mode): native hermes <tool_call> JSON, weak at SWE -> headroom for RL
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-4b-awq": ("Qwen/Qwen3-4B-AWQ", "74d4bd2bd4bff9cafc9345221320bffb08b406a3"),
    # Qwen3-1.7B after SFT on ~400k Qwen3-Coder-30B-A3B mini-swe-agent trajectories (SWE-smith tasks; MIT). Same
    # tokenizer as Qwen3-1.7B; card: 18.6 % pass@1 on SWE-bench Verified (bash-only). Use protocol "mswea" and render
    # with enable_thinking=True (its SFT turns carry no <think> block). Trained on SWE-smith bugs from all four pool
    # repos (astroid, sqlfluff, pydicom, marshmallow) -> measure generalisation on configs/holdout_dev.txt.
    "mini-coder-1.7b": ("ricdomolm/mini-coder-1.7b", "5b7fbc90b6c4684c546de7f811865cff4414a96c"),
    # Gemma 4 E4B (Apache 2.0): Google's bf16 snapshot = tokenizer, chat template (2026-07-09; the FP8 copy's template is
    # older) and training reference; the third-party FP8 copy is what vLLM serves on the A4000s (W8A16 on sm_86,
    # --language-model-only). Same tokenizer.json / vocab. Render with enable_thinking=False; protocol mswea only (the
    # template renders tool results only after a native tool call).
    # mini-coder-1.7b's 4B sibling: Qwen3-4B-Instruct-2507 SFT'd on the same mini-swe-agent corpus (MIT); card: 26.8 %
    # pass@1 on Verified (bash-only). Same tokenizer family and protocol as mini-coder-1.7b.
    "mini-coder-4b": ("ricdomolm/mini-coder-4b", "c87892d046a62415a349a6b3aa244bb630fb7417"),
    "gemma4-e4b": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    "gemma4-e4b-fp8": ("prithivMLmods/gemma-4-E4B-it-FP8", "e991fcb6c8b3871071c0ad7f7343dc5e0e7e5639"),
}

# chat-template kwargs per model (ChatFormat defaults to enable_thinking=False)
TEMPLATE_KWARGS = {"mini-coder-1.7b": {"enable_thinking": True}, "mini-coder-4b": {"enable_thinking": True},
                   "gemma4-e4b": {"enable_thinking": False},
                   "gemma4-e4b-fp8": {"enable_thinking": False}}


def model_snapshot(key: str) -> Path:
    mid, rev = MODELS[key]
    return HF_CACHE / "hub" / f"models--{mid.replace('/', '--')}" / "snapshots" / rev
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

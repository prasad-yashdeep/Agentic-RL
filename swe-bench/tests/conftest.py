"""Markers skip cleanly when an asset is absent (see pyproject markers)."""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import pytest

from sarl.paths import APPTAINER, DATASETS, GRADER_VENV, HF_CACHE, MODEL_REVISION, SIF_DIR, UPSTREAM

SMALL_TASK = "marshmallow-code__marshmallow-1359"


def _have_data() -> bool:
    return all((HF_CACHE / "hub" / f"datasets--{r.replace('/', '--')}" / "snapshots" / rev).exists() for r, rev, _ in DATASETS.values())


def _have_qwen() -> bool:
    return (HF_CACHE / "hub" / "models--Qwen--Qwen2.5-Coder-0.5B-Instruct" / "snapshots" / MODEL_REVISION / "config.json").exists()


def _have_gemma() -> bool:
    from sarl.paths import MODELS

    mid, rev = MODELS["gemma4-e4b"]
    return (HF_CACHE / "hub" / f"models--{mid.replace('/', '--')}" / "snapshots" / rev / "tokenizer.json").exists()


def _have_sif() -> bool:
    return APPTAINER.exists() and (SIF_DIR / f"{SMALL_TASK}.sif").exists() and not socket.gethostname().startswith("bcm11-headnode")


def _have_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


SKIPS = {
    "upstream": (lambda: (UPSTREAM / "utils.py").exists(), "reference/_upstream not fetched (bash reference/fetch.sh)"),
    "swe_data": (_have_data, "pinned SWE-bench datasets not in the HF cache (make assets)"),
    "qwen": (_have_qwen, "Qwen2.5-Coder-0.5B-Instruct not in the HF cache (make assets)"),
    "gemma": (_have_gemma, "google/gemma-4-E4B-it tokenizer not in the HF cache"),
    "sif": (_have_sif, f"no apptainer image {SMALL_TASK}.sif in {SIF_DIR} on this host (node08: make sifs)"),
    "grader": (lambda: (GRADER_VENV / "bin" / "python").exists(), "grader venv missing (make grader)"),
    "gpu": (_have_gpu, "no CUDA"),
}


def pytest_collection_modifyitems(config, items):
    cache = {}
    for it in items:
        for name, (probe, why) in SKIPS.items():
            if it.get_closest_marker(name):
                if name not in cache:
                    cache[name] = probe()
                if not cache[name]:
                    it.add_marker(pytest.mark.skip(reason=why))


@pytest.fixture(scope="session")
def upstream():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sarl_upstream_utils", UPSTREAM / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def qwen_tok():
    from transformers import AutoTokenizer

    from sarl.paths import MODEL_ID

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


@pytest.fixture(scope="session")
def gemma_tok():
    from transformers import AutoTokenizer

    from sarl.paths import MODELS

    mid, rev = MODELS["gemma4-e4b"]
    return AutoTokenizer.from_pretrained(mid, revision=rev)


def make_tiny_qwen():
    """A randomly initialised 2-layer Qwen2 (no download) for trainer math on CPU; the same weights on every call."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).float()


@pytest.fixture(scope="session")
def tiny_qwen():
    return make_tiny_qwen()


@pytest.fixture
def fresh_tiny_qwen():
    """tiny_qwen's weights, but a new object per test: LoRATrainer wraps its model in place, so the session fixture
    carries every earlier test's LoRA layers and updates."""
    return make_tiny_qwen()

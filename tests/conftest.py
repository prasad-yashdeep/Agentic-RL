"""Shared fixtures. Real-data tests skip cleanly when the recorded artifacts are absent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", str(Path.home() / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

A0 = REPO / "jobs" / "2026-07-21__05-38-50"
G4 = REPO / "jobs-g4"
G4LOG = REPO / "g4.log"
PILOT = REPO / "experiments" / "judge-pilot.jsonl"
WATCH = REPO / "watch.log"


def _gemma_available() -> bool:
    root = Path(os.environ["HF_HOME"]) / "hub" / "models--google--gemma-4-E4B-it" / "snapshots"
    return root.exists() and any((s / "tokenizer_config.json").exists() for s in root.iterdir())


def _minilm_available() -> bool:
    from tarl.embed.model import snapshot_dir

    return snapshot_dir() is not None


def _cuda() -> bool:
    from tarl.common.env import has_cuda

    return has_cuda()


def pytest_collection_modifyitems(config, items):
    real = A0.exists() and G4.exists() and G4LOG.exists() and PILOT.exists()
    gemma = _gemma_available()
    minilm = _minilm_available()
    cuda = _cuda()
    for it in items:
        if "real_data" in it.keywords and not real:
            it.add_marker(pytest.mark.skip(reason="recorded jobs/ jobs-g4/ g4.log not on disk"))
        if "gemma_tok" in it.keywords and not gemma:
            it.add_marker(pytest.mark.skip(reason="Gemma tokenizer not in offline HF cache"))
        if "embed_model" in it.keywords and not minilm:
            it.add_marker(pytest.mark.skip(reason="MiniLM weights not in offline HF cache"))
        if "gpu" in it.keywords and not cuda:
            it.add_marker(pytest.mark.skip(reason="no CUDA GPU"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def a0_job() -> Path:
    return A0


@pytest.fixture(scope="session")
def g4_jobs() -> Path:
    return G4


@pytest.fixture(scope="session")
def g4_log() -> Path:
    return G4LOG


@pytest.fixture(scope="session")
def pilot_jsonl() -> Path:
    return PILOT


@pytest.fixture(scope="session")
def tiny_tok():
    """Whitespace WordLevel tokenizer with a Gemma-like chat template; unknown words -> <unk>."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "<|turn>", "<turn|>", "user", "model", "assistant", "system"]
    for w in "the a git status reflog checkout master merge commit ls cat analysis plan commands keystrokes duration task_complete true false hello world fix".split():
        words.append(w)
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>")
    tok.chat_template = (
        "{% for m in messages %}<|turn> {{ 'model' if m['role'] == 'assistant' else m['role'] }} "
        "{{ m['content'] }} <turn|> {% endfor %}"
    )
    return tok


@pytest.fixture(scope="session")
def gemma_tok():
    from tarl.data.examples import load_gemma_tokenizer

    return load_gemma_tokenizer()


@pytest.fixture
def synthetic_trajectory():
    """3-turn Terminus-like trajectory with tool_calls/observations/metrics."""
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "x", "extra": {"parser": "json", "temperature": 1.0}},
        "steps": [
            {"step_id": 1, "source": "user", "message": "SYSTEM PROMPT\n\nTask Description:\nfix the git repo\n\nCurrent terminal state:\nroot@x:/app#"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "Analysis: look around\nPlan: run git status",
                "reasoning_content": "think think",
                "tool_calls": [{"function_name": "bash_command", "arguments": {"keystrokes": "git status\n", "duration": 1.0}}],
                "observation": {"results": [{"content": "New Terminal Output:\nerror: not a git repository"}]},
                "metrics": {"prompt_tokens": 10, "completion_tokens": 7},
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "Analysis: error seen\nPlan: done",
                "tool_calls": [{"function_name": "mark_task_complete", "arguments": {}}],
                "observation": {"results": [{"content": "New Terminal Output:\nroot@x:/app#"}]},
                "metrics": {"prompt_tokens": 20, "completion_tokens": 5},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 30, "total_completion_tokens": 12, "total_cached_tokens": 0},
    }

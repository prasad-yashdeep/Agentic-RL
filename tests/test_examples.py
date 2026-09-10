import importlib.util
import json
from pathlib import Path

import pytest

from tarl.data.examples import build_example, render_conversation, synth_terminus_json, to_tensors


def test_masks_cover_assistant_spans_only(tiny_tok, synthetic_trajectory):
    ex = build_example(tiny_tok, synthetic_trajectory, render="legacy06", max_seq=None)
    assert ex.kept and ex.n_tokens == len(ex.mask)
    assert ex.prefix_stable
    roles = {sp.role for sp in ex.spans}
    assert roles == {"user", "assistant"}
    for sp in ex.spans:
        seg = ex.mask[sp.start : sp.end]
        assert all(v == (1 if sp.role == "assistant" else 0) for v in seg)
    assert ex.spans[-1].end == ex.n_tokens
    ids, mask = to_tensors(ex)
    assert ids.shape == mask.shape


def test_terminus_chat_includes_observations(tiny_tok, synthetic_trajectory):
    convo = render_conversation(synthetic_trajectory, render="terminus_chat")
    roles = [m["role"] for m in convo]
    assert roles == ["user", "assistant", "user", "assistant"]  # no observation after the final turn
    assert "not a git repository" in convo[2]["content"]
    obj = json.loads(convo[1]["content"])
    assert obj["commands"][0]["keystrokes"] == "git status\n" and "task_complete" not in obj
    assert json.loads(convo[3]["content"])["task_complete"] is True
    ex = build_example(tiny_tok, synthetic_trajectory, render="terminus_chat", max_seq=None)
    obs_span = ex.spans[2]
    assert obs_span.role == "user" and all(v == 0 for v in ex.mask[obs_span.start : obs_span.end])
    with_reasoning = render_conversation(synthetic_trajectory, render="terminus_chat", include_reasoning=True)
    assert with_reasoning[1].get("reasoning_content") == "think think"


def test_overlong_and_empty(tiny_tok, synthetic_trajectory):
    ex = build_example(tiny_tok, synthetic_trajectory, max_seq=3)
    assert ex.dropped_reason == "overlong" and not ex.kept
    only_user = {"steps": [synthetic_trajectory["steps"][0]]}
    ex2 = build_example(tiny_tok, only_user, max_seq=None)
    assert ex2.dropped_reason == "empty_mask"
    assert build_example(tiny_tok, {"steps": []}).dropped_reason == "no_steps"


def test_synth_json_roundtrip():
    s = synth_terminus_json({"message": "Analysis: a\nPlan: b", "tool_calls": [{"function_name": "bash_command", "arguments": {"keystrokes": "ls\n", "duration": 0.5}}]})
    d = json.loads(s)
    assert d == {"analysis": "a", "plan": "b", "commands": [{"keystrokes": "ls\n", "duration": 0.5}]}


def _load_06(repo: Path):
    spec = importlib.util.spec_from_file_location("g06", repo / "scripts" / "06_grpo_micro.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy06_parity(tiny_tok, synthetic_trajectory, repo, tmp_path):
    g06 = _load_06(repo)
    long_traj = {"steps": synthetic_trajectory["steps"] * 6}
    cases = {"short": synthetic_trajectory, "long": long_traj, "user_only": {"steps": [synthetic_trajectory["steps"][0]]}}
    for name, traj in cases.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(traj))
        theirs = g06.build_example(tiny_tok, str(p))
        ours = build_example(tiny_tok, traj, render="legacy06", max_seq=g06.MAX_SEQ)
        if theirs is None:
            assert not ours.kept, name
        else:
            assert ours.kept, name
            assert theirs[0].tolist() == ours.ids and theirs[1].tolist() == ours.mask, name


@pytest.mark.real_data
@pytest.mark.gemma_tok
def test_gemma_reconstruction_gap(gemma_tok, g4_jobs):
    from tarl.data.examples import reconstruction_gap
    from tarl.judge.compressor import load_trajectory

    p = g4_jobs / "2026-07-21__16-57-00" / "fix-git__2E66fzc" / "agent" / "trajectory.json"
    traj = load_trajectory(p)
    ex = build_example(gemma_tok, traj, render="legacy06", max_seq=None)
    assert ex.kept and ex.n_assistant_tokens > 0
    # finding #1b: consecutive `assistant` messages are merged by the template -> the incremental
    # prefix method is NOT stable for legacy06; the offsets fallback kicks in
    assert not ex.prefix_stable and ex.meta["mask_method"] == "offsets" and ex.meta["mask_tokens_changed"] > 0
    gap = reconstruction_gap(ex, traj)
    assert len(gap) == 7
    # finding #1: the prose rendering carries fewer tokens than the model actually generated
    assert (gap["delta"] < 0).all()
    ex2 = build_example(gemma_tok, traj, render="terminus_chat", max_seq=None)
    assert ex2.prefix_stable and ex2.meta["mask_method"] == "incremental"
    assert ex2.n_tokens > ex.n_tokens

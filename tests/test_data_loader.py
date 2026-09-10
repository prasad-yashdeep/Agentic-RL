import pytest

from tarl.data.cast import keystrokes, parse_cast, strip_ansi
from tarl.data.instruction import split_system_and_task, task_instruction
from tarl.data.loader import iter_trials, job_summary, legacy_rollout_order, load_episode, turns_table

pytestmark = pytest.mark.real_data


def test_a0_counts(a0_job):
    trials = list(iter_trials(a0_job))
    assert len(trials) == 176
    by_status = {}
    for t in trials:
        by_status[t.status] = by_status.get(t.status, 0) + 1
    assert by_status["clean"] == 150
    assert by_status["errored_with_trajectory"] == 13 and by_status["errored_no_trajectory"] == 13
    assert sum(t.passed for t in trials) == 5
    assert len(list(iter_trials(a0_job, include_incomplete=True))) == 180
    assert sum(t.has_trajectory for t in iter_trials(a0_job, include_incomplete=True)) == 166


def test_build_pmars_episode(a0_job):
    ep = load_episode(a0_job / "build-pmars__iqVnGoH")
    assert ep.passed and ep.reward == 1.0 and ep.status == "clean"
    assert ep.n_steps == 13 and ep.n_agent_steps == 12
    assert ep.commands[0] == "apt-get source pmars\n"
    assert ep.ctrf and ep.ctrf.label == "4/4"
    assert ep.tokens["total_prompt_tokens"] == 70035 and ep.tokens["total_completion_tokens"] == 6608
    assert ep.phases["agent_execution"] > 100 and ep.phases["verifier"] > 0
    assert ep.instruction().startswith("Build pMARS")
    assert len(ep.api_request_times_ms) == 12
    t = turns_table(ep)
    assert len(t) == 12 and t["cumulative_prompt_tokens"].iloc[-1] == 70035
    assert t["task_complete"].iloc[-1]


def test_instruction_parser(a0_job):
    ep = load_episode(a0_job / "build-pmars__iqVnGoH")
    sysm, inst, state = split_system_and_task(ep.steps[0]["message"])
    assert "Task Description" not in sysm and inst.startswith("Build pMARS") and "root@" in state
    assert task_instruction({"steps": [{"source": "user", "message": "no markers"}]}) is None


def test_cast_and_pane(a0_job):
    c = parse_cast(a0_job / "build-pmars__iqVnGoH" / "agent" / "recording.cast")
    assert c.header.get("version") == 2 and c.events
    first_raw = [d for _, k, d in c.events if k == "i"][0]
    assert first_raw.strip() == "clear"
    ks = keystrokes(c)
    assert ks[0] == "apt-get source pmars\n"
    assert strip_ansi("\x1b[?2004hroot@node08:/app# ") == "root@node08:/app# "


def test_job_summary_frame(a0_job):
    df = job_summary(a0_job)
    assert len(df) == 176 and df["passed"].sum() == 5
    assert set(["task", "domain", "status", "reward", "n_steps", "agent_s", "ctrf_passed"]) <= set(df.columns)
    assert (df["domain"] != "unknown").all()
    assert df["n_steps"].max() == 899


def test_legacy_rollout_order(g4_jobs):
    trials = legacy_rollout_order(g4_jobs / "2026-07-21__18-14-34")
    assert len(trials) == 8
    assert sorted(t.task for t in trials) == ["fix-git"] * 4 + ["regex-log"] * 4

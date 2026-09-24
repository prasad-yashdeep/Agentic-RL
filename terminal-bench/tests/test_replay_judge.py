import pytest

from tarl.judge.composer import RewardConfig, composite_reward
from tarl.judge.compressor import compress, load_trajectory
from tarl.judge.g4log import parse_g4_log
from tarl.judge.protocol import JudgeUnavailable
from tarl.judge.replay import ReplayJudge
from tarl.judge.rubric import load_rubric

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def rj():
    return ReplayJudge.from_artifacts()


def test_pilot_exact(rj, a0_job):
    p = a0_job / "gcode-to-text__AGNJpPu" / "agent" / "trajectory.json"
    rubric, _ = load_rubric()
    v = rj.judge(rubric, compress(load_trajectory(p)))
    assert v.score == 0.15 and v.raw["source"] == "pilot" and "misread" in v.rationale
    assert rj.lookup_trial(p.parent.parent).confidence == "exact"


def test_table_and_counts(rj):
    t = rj.table()
    assert (t["source"] == "pilot").sum() == 12
    assert (t["source"] == "g4log").sum() == 40
    assert t.loc[t["source"] == "pilot", "score"].mean() == pytest.approx(0.304, abs=1e-3)


def test_g4_composite_parity(rj, g4_log, g4_jobs):
    """Every one of the 40 replayed judge scores reproduces the logged composite reward."""
    from tarl.data.loader import legacy_rollout_order

    run = parse_g4_log(g4_log)
    cfg = RewardConfig(lambda_judge=0.5)
    n = 0
    for st in run.steps:
        trials = legacy_rollout_order(g4_jobs / st.job)
        by_task = {}
        for t in trials:
            by_task.setdefault(t.task, []).append(t)
        for r in st.rollouts:
            t = by_task[r.task][r.order]
            score = rj.score_for_path(t.trajectory_path)
            assert score is not None
            assert composite_reward(t.passed, score, cfg) == pytest.approx(r.R, abs=5e-4)
            n += 1
    assert n == 40


def test_miss_policies(rj):
    rubric, _ = load_rubric()
    with pytest.raises(JudgeUnavailable):
        rj.judge(rubric, "never seen")
    lenient = ReplayJudge(rj.entries, on_miss="mock")
    v = lenient.judge(rubric, "[step 2 · agent]\nAnalysis: x\n")
    assert 0.0 <= v.score <= 1.0 and v.raw.get("replay_miss")


def test_write_jsonl(rj, tmp_path):
    n = rj.write_jsonl(tmp_path / "replay.jsonl")
    assert n == 52

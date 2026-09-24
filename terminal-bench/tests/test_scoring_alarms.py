from pathlib import Path

import json

import pytest

from tarl.judge.alarms import JudgeAlarms
from tarl.judge.client import JudgeVerdict
from tarl.judge.composer import RewardConfig
from tarl.judge.cost import CostCappedJudge, CostMeter, JudgeBudgetExceeded
from tarl.judge.scoring import score_rollouts
from tarl.training.rollouts import RolloutRecord


class CountingJudge:
    name = "counting"

    def __init__(self, score=0.6):
        self.calls = 0
        self.score = score

    def judge(self, s, u):
        self.calls += 1
        return JudgeVerdict(self.score, "r", 100, 10)


def _rollouts(tmp_path: Path, rewards):
    out = []
    for i, r in enumerate(rewards):
        p = tmp_path / f"t{i}.json"
        p.write_text(json.dumps({"steps": [{"step_id": 1, "source": "user", "message": "x"}, {"step_id": 2, "source": "agent", "message": "y"}]}))
        out.append(RolloutRecord(step=1, task="t", trial_dir=tmp_path, trajectory=p, verifier=r))
    return out


def test_passes_never_judged(tmp_path):
    j = CountingJudge()
    ros = _rollouts(tmp_path, [1.0, 0.0, 0.0])
    score_rollouts(ros, j, "rubric", RewardConfig(0.5))
    assert j.calls == 2 and ros[0].R == 1.0 and ros[1].R == pytest.approx(0.3) and ros[1].judge_source == "counting"


def test_lambda_zero_skips_judge(tmp_path):
    j = CountingJudge()
    ros = _rollouts(tmp_path, [0.0, 0.0])
    score_rollouts(ros, j, "rubric", RewardConfig(0.0))
    assert j.calls == 0 and all(r.R == 0.0 for r in ros)


def test_budget_cap_falls_back(tmp_path):
    meter = CostMeter(5.0, 25.0, cap_usd=0.0005)  # 100 in + 10 out tokens = $0.00075 per call -> cap hit after call 1
    j = CostCappedJudge(CountingJudge(), meter)
    ros = _rollouts(tmp_path, [0.0, 0.0, 0.0])
    alarms = score_rollouts(ros, j, "rubric", RewardConfig(0.5))
    assert ros[0].judge == 0.6 and ros[1].judge is None and ros[2].judge is None
    assert any(a.startswith("BUDGET") for a in alarms)
    with pytest.raises(JudgeBudgetExceeded):
        j.judge("a", "b")


def test_alarms_window_and_audit(tmp_path):
    al = JudgeAlarms(window=4, audit_path=tmp_path / "audit.jsonl")
    fired = []
    for s in [0.2, 0.3, 0.2, 0.3]:
        fired += al.record(s, False)
    for s in [0.6, 0.7, 0.6, 0.7]:
        fired += al.record(s, False)
    assert any(f.startswith("DIVERGENCE") for f in fired)
    assert any(f.startswith("AUDIT") for f in al.record(0.9, False, meta={"task": "t"}))
    assert (tmp_path / "audit.jsonl").exists() and al.n_audits == 1

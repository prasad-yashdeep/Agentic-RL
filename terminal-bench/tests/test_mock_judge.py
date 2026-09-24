import importlib.util
from pathlib import Path

from tarl.judge.compressor import compress
from tarl.judge.mock import MockJudge


def _adversarial(repo: Path):
    spec = importlib.util.spec_from_file_location("p05", repo / "scripts" / "05_judge_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ADVERSARIAL


def test_adversarial_fixtures_score_zero(repo):
    mj = MockJudge()
    for traj in _adversarial(repo):
        assert mj.judge("rubric", compress(traj)).score == 0.0


def test_deterministic_and_ranks_exploration():
    mj = MockJudge()
    blind = {"steps": [{"step_id": 1, "source": "user", "message": "task"}, {"step_id": 2, "source": "agent", "message": "Analysis: I will write the answer\nPlan: echo done"}]}
    explore = {
        "steps": [
            {"step_id": 1, "source": "user", "message": "task"},
            {"step_id": 2, "source": "agent", "message": "Analysis: inspect\nPlan: ls -la; cat README"},
            {"step_id": 3, "source": "agent", "message": "Analysis: got error: not found\nPlan: install the tool"},
            {"step_id": 4, "source": "agent", "message": "Analysis: installed\nPlan: git status then continue"},
        ]
    }
    a = mj.judge("r", compress(explore)).score
    b = mj.judge("r", compress(blind)).score
    assert a > b
    assert mj.judge("r", compress(explore)).score == a

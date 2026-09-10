import pytest

from tarl.judge.client import JudgeError
from tarl.judge.config import load_judge_config, make_judge
from tarl.judge.rubric import load_rubric


def test_yaml_loaded():
    cfg = load_judge_config()
    assert cfg.model == "grok-4.5" and cfg.env_key == "XAI_API_KEY" and cfg.cost_cap_usd_per_run == 500
    assert cfg.rubric.exists()
    assert load_rubric()[1] == "61083938f358"


def test_live_requires_spend_and_key(monkeypatch, tmp_path):
    with pytest.raises(JudgeError):
        make_judge("live")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("tarl.judge.config.SECRETS_ENV", tmp_path / "nope")
    with pytest.raises(JudgeError):
        make_judge("live", allow_spend=True)


def test_mock_and_replay_construct():
    assert make_judge("mock").name == "mock"
    rj = make_judge("replay", on_miss="mock", pilot_jsonl=None, g4_log=None, jobs_g4=None)
    assert rj.name == "replay"

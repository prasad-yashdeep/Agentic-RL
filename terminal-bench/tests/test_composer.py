import pytest

from tarl.judge.composer import RewardConfig, composite_reward


def test_gate_semantics():
    cfg = RewardConfig(lambda_judge=0.5)
    assert composite_reward(True, 0.0, cfg) == 1.0
    assert composite_reward(True, None, cfg) == 1.0
    assert composite_reward(False, None, cfg) == 0.0
    assert composite_reward(False, 0.8, cfg) == pytest.approx(0.4)
    assert composite_reward(False, 1.0, cfg) < composite_reward(True, 0.0, cfg)


def test_lambda_gate_enforced():
    with pytest.raises(ValueError):
        RewardConfig(lambda_judge=0.6)
    with pytest.raises(ValueError):
        composite_reward(False, 1.2, RewardConfig())

import pytest

from tarl.training.advantages import group_advantages, group_is_informative, informative, useful_group_fraction


def test_dr_grpo_hand_values():
    adv = group_advantages([0.175, 0.15, 0.2, 0.16])
    assert adv == pytest.approx([0.00375, -0.02125, 0.02875, -0.01125], abs=1e-9)
    assert sum(adv) == pytest.approx(0.0, abs=1e-12)


def test_g4_step3_group():
    R = [0.15, 0.39, 0.325, 0.15]
    adv = group_advantages(R)
    assert adv == pytest.approx([-0.104, 0.136, 0.071, -0.104], abs=5e-4)


def test_std_mode_and_edges():
    z = group_advantages([0.0, 1.0, 1.0, 0.0], mode="grpo_std")
    assert sum(z) == pytest.approx(0.0, abs=1e-9)
    assert abs(z[1]) == pytest.approx(1.0, abs=1e-6)
    assert group_advantages([0.3, 0.3, 0.3]) == [0.0, 0.0, 0.0]
    assert group_advantages([]) == []
    assert group_advantages([0.7]) == [0.0]
    assert not group_is_informative([0.0, 0.0])
    assert informative(0.01) and not informative(1e-9)
    with pytest.raises(ValueError):
        group_advantages([1, 2], mode="nope")


def test_useful_group_fraction():
    assert useful_group_fraction(0.02, 8) == pytest.approx(0.149, abs=2e-3)
    assert useful_group_fraction(0.5, 2) == pytest.approx(0.5)
    assert useful_group_fraction(0.0, 8) == 0.0

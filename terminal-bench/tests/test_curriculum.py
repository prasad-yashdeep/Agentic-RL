import pytest

from tarl.training.curriculum import Curriculum, TaskStats, filter_informative_groups


def test_posterior_and_band():
    st = TaskStats()
    assert st.p_hat == 0.5
    st.n, st.successes = 10, 1
    assert st.p_hat == pytest.approx(2 / 12)
    cur = Curriculum(["a", "b", "c"], seed=1)
    for _ in range(20):
        cur.update("a", True)
        cur.update("b", False)
    cur.update("c", False)
    assert set(cur.in_band()) == {"c"}
    s = cur.sample(1)
    assert s == ["c"] and not cur.last_sample_fallback
    s2 = cur.sample(3)
    assert len(s2) == 3 and cur.last_sample_fallback
    assert 0 < cur.trainable_fraction() < 1
    assert Curriculum(["x", "y"], seed=0).sample(2, "uncertainty") == Curriculum(["x", "y"], seed=0).sample(2, "uncertainty")


def test_dynamic_sampling_filter():
    kept, dropped = filter_informative_groups({"t1": [0, 0, 0, 0], "t2": [0, 1, 0, 0], "t3": [1, 1, 1, 1]})
    assert list(kept) == ["t2"] and dropped == ["t1", "t3"]

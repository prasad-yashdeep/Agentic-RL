import pandas as pd
import pytest

from tarl.eval.stats import bootstrap_ci, eligible_ks, mcnemar, paired_compare, pass_at_k, pass_at_k_table, wilson_ci, wilson_halfwidth


def test_wilson():
    lo, hi = wilson_ci(5, 150)
    assert lo == pytest.approx(0.0143, abs=5e-4) and hi == pytest.approx(0.0757, abs=5e-4)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    assert wilson_halfwidth(0.08, 89) == pytest.approx(0.055, abs=0.005)


def test_pass_at_k():
    assert pass_at_k(5, 1, 2) == pytest.approx(0.4)
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(4, 0, 2) == 0.0
    assert eligible_ks(5) == [2, 4, 5]
    df = pd.DataFrame({"task": ["a"] * 4 + ["b"] * 4, "reward": [1, 0, 0, 0, 0, 0, 0, None]})
    t = pass_at_k_table(df)
    assert t.loc[t["task"] == "a", "pass@2"].iloc[0] == pytest.approx(0.5)
    assert t.loc[t["task"] == "mean", "pass@4"].iloc[0] == pytest.approx(0.5)


def test_mcnemar_and_bootstrap():
    r = mcnemar(3, 1)
    assert r.p == pytest.approx(0.625) and r.method == "exact"
    assert mcnemar(0, 0).p == 1.0
    lo, hi = bootstrap_ci([0.5] * 20)
    assert lo == hi == 0.5


def test_paired_compare():
    a = pd.DataFrame({"task": ["x", "x", "y", "y", "z"], "passed": [False, False, True, False, False], "status": ["clean"] * 5})
    b = pd.DataFrame({"task": ["x", "x", "y", "y", "z"], "passed": [True, True, True, True, False], "status": ["clean"] * 5})
    res = paired_compare(a, b, reduce="mean", n_boot=200)
    assert res["n_tasks"] == 3 and res["delta_pp"] == pytest.approx((1.0 + 0.5 + 0) / 3 * 100)
    assert res["gained"] == ["x", "y"] and res["mcnemar"]["b"] == 1

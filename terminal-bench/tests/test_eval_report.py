import pytest

from tarl.data.loader import job_summary
from tarl.eval.report import by_domain, error_taxonomy, pass_rate_by_task, summary_table

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def df(a0_job):
    return job_summary(a0_job)


def test_summary_reproduces_experiments_md(df):
    s = summary_table(df)
    assert s["n_trials"] == 176 and s["n_clean"] == 150 and s["n_pass"] == 5 and s["n_error"] == 26
    assert s["pass_at_1"] == pytest.approx(5 / 150)
    assert s["tasks_solved"] == 5 and s["solved_tasks"] == ["build-pmars", "extract-elf", "modernize-scientific-stack", "prove-plus-comm", "sqlite-with-gcov"]
    assert s["tasks_touched"] == 89 and s["tasks_with_clean_trial"] == 83
    assert s["errors_by_type"]["RuntimeError"] == 10 and s["errors_by_type"]["AgentTimeoutError"] == 9
    assert 0.03 <= s["mean_task_pass_rate"] <= 0.04


def test_tables(df):
    tax = error_taxonomy(df)
    n_err = tax.loc[tax["category"].str.startswith("error:"), "n"].sum()
    n_out = tax.loc[tax["category"].isin(["pass", "fail:verifier_partial", "fail:verifier_all_failed", "unverified"]), "n"].sum()
    assert n_err + n_out == 176
    bd = by_domain(df)
    assert bd["n_tasks"].sum() == 89 and bd["k_pass"].sum() == 5
    pt = pass_rate_by_task(df)
    assert len(pt) == 83 and pt["pass_rate"].iloc[0] > 0

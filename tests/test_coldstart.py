from tarl.training.coldstart import SimConfig, simulate_coldstart, useful_fraction_table


def test_table_monotone():
    t = useful_fraction_table(ps=(0.02,), Gs=(4, 8, 16))
    row = t.iloc[0]
    assert row["G=4"] < row["G=8"] < row["G=16"]


def test_simulation_runs_and_judge_helps():
    a = simulate_coldstart(SimConfig(p0=0.01, steps=40, n_tasks=10, judge_shaping=False, seed=0))
    b = simulate_coldstart(SimConfig(p0=0.01, steps=40, n_tasks=10, judge_shaping=True, seed=0))
    assert len(a) == 41 and a["mean_p"].between(0, 1).all()
    assert b["useful_group_frac"].mean() >= a["useful_group_frac"].mean()
    assert b["judged_calls"].iloc[-1] > 0 and a["judged_calls"].iloc[-1] == 0

"""Tables over tarl.data.job_summary frames: headline summary, per-task, per-domain, error taxonomy."""

from __future__ import annotations

from tarl.eval.stats import wilson_ci

ERROR_HINTS = {
    "RuntimeError": "infra: container image conversion / runtime",
    "AgentTimeoutError": "agent hit the wall-clock budget",
    "EnvironmentStartTimeoutError": "sandbox failed to start in time",
    "VerifierTimeoutError": "verifier hit its time budget",
    "BadRequestError": "model endpoint rejected a request (e.g. max_tokens=0 context overrun)",
    "MemoryLimitExceededError": "container exceeded its memory watchdog",
}


def summary_table(df) -> dict:
    """The numbers EXPERIMENTS.md quotes for A0, recomputed from disk."""
    clean = df[df["status"] == "clean"]
    errored = df[df["exception_type"].notna()]
    n_clean = int(len(clean))
    n_pass = int(clean["passed"].sum())
    lo, hi = wilson_ci(n_pass, n_clean)
    n_all = int(len(df))
    lo_all, hi_all = wilson_ci(n_pass, n_all)
    per_task = clean.groupby("task")["passed"].mean() if n_clean else None
    solved = sorted(clean.loc[clean["passed"], "task"].unique().tolist())
    return {
        "n_trials": n_all,
        "n_clean": n_clean,
        "n_pass": n_pass,
        "n_fail": n_clean - n_pass,
        "n_error": int(len(errored)),
        "n_unverified": int((df["status"] == "unverified").sum()),
        "errors_by_type": errored["exception_type"].value_counts().to_dict(),
        "pass_at_1": (n_pass / n_clean) if n_clean else None,
        "pass_at_1_ci95": [lo, hi],
        "pass_at_1_all_trials": (n_pass / n_all) if n_all else None,
        "pass_at_1_all_trials_ci95": [lo_all, hi_all],
        "tasks_touched": int(df["task"].nunique()),
        "tasks_with_clean_trial": int(clean["task"].nunique()),
        "tasks_solved": len(solved),
        "solved_tasks": solved,
        "mean_task_pass_rate": (float(per_task.mean()) if per_task is not None and len(per_task) else None),
        "n_with_trajectory": int(df["has_trajectory"].sum()),
        "n_summarized": int((df["summarization_count"].fillna(0) > 0).sum()) if "summarization_count" in df else None,
    }


def pass_rate_by_task(df):
    """DataFrame(task, domain, n, k_pass, pass_rate, wilson_lo, wilson_hi) over clean trials."""
    import pandas as pd

    clean = df[df["status"] == "clean"]
    rows = []
    for task, g in clean.groupby("task"):
        n, k = int(len(g)), int(g["passed"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"task": task, "domain": g["domain"].iloc[0], "n": n, "k_pass": k, "pass_rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows).sort_values(["pass_rate", "task"], ascending=[False, True]).reset_index(drop=True)


def by_domain(df):
    """DataFrame(domain, n_tasks, n_trials, k_pass, pass_rate, wilson_lo, wilson_hi, n_error)."""
    import pandas as pd

    rows = []
    for dom, g in df.groupby("domain"):
        clean = g[g["status"] == "clean"]
        n, k = int(len(clean)), int(clean["passed"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append(
            {
                "domain": dom,
                "n_tasks": int(g["task"].nunique()),
                "n_trials": int(len(g)),
                "n_clean": n,
                "k_pass": k,
                "pass_rate": (k / n) if n else float("nan"),
                "wilson_lo": lo,
                "wilson_hi": hi,
                "n_error": int(g["exception_type"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_tasks", ascending=False).reset_index(drop=True)


def error_taxonomy(df):
    """DataFrame(category, n, hint) — infra failures vs agent outcomes, per docs/04 gap 16."""
    import pandas as pd

    rows = []
    err = df[df["exception_type"].notna()]
    for et, n in err["exception_type"].value_counts().items():
        rows.append({"category": f"error:{et}", "n": int(n), "hint": ERROR_HINTS.get(et, "")})
    clean = df[df["status"] == "clean"]
    rows.append({"category": "pass", "n": int(clean["passed"].sum()), "hint": "verifier reward 1"})
    fail = clean[~clean["passed"]]
    partial = fail[(fail["ctrf_passed"].fillna(0) > 0)]
    rows.append({"category": "fail:verifier_partial", "n": int(len(partial)), "hint": "some pytest tests passed"})
    rows.append({"category": "fail:verifier_all_failed", "n": int(len(fail) - len(partial)), "hint": "no test passed / no ctrf"})
    if "summarization_count" in df:
        rows.append({"category": "context_summarized", "n": int((df["summarization_count"].fillna(0) > 0).sum()), "hint": "Terminus 2 compacted context at least once (overlaps other rows)"})
    rows.append({"category": "unverified", "n": int((df["status"] == "unverified").sum()), "hint": "no verifier result"})
    return pd.DataFrame(rows)


def pivot(frames: dict[str, "object"], value: str = "pass_rate"):
    """Per-task pivot across several runs: DataFrame(task, <run>...)."""
    import pandas as pd

    cols = {}
    for name, df in frames.items():
        t = pass_rate_by_task(df).set_index("task")[value]
        cols[name] = t
    return pd.DataFrame(cols).reset_index()

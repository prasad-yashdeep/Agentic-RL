"""Evaluation statistics (docs/02-architecture.md §7, docs/04-gaps-and-risks.md gap 11).

On 89 binary tasks a single run's 95% interval is ±6–8 pp — a +3–5 pp effect is invisible without
repeats and paired tests. Everything here is pure numpy/scipy and CPU-cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n Bernoulli trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width (in probability units) of the Wilson interval at observed rate p."""
    if n <= 0:
        return 0.5
    z2 = z * z
    return z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1.0 + z2 / n)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021) as Harbor computes it: 1 − C(n−c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def eligible_ks(min_n: int) -> list[int]:
    """Harbor's k schedule: powers of two and multiples of five up to the smallest trials/task."""
    ks: set[int] = set()
    k = 2
    while k <= min_n:
        ks.add(k)
        k *= 2
    k = 5
    while k <= min_n:
        ks.add(k)
        k += 5
    return sorted(ks)


def pass_at_k_table(df, ks: Sequence[int] | None = None, *, task_col: str = "task", reward_col: str = "reward"):
    """Per-task pass@k (rewards None -> 0, as Harbor does) plus a `mean` row."""
    import pandas as pd

    d = df[[task_col, reward_col]].copy()
    d[reward_col] = d[reward_col].fillna(0.0).astype(float)
    grp = d.groupby(task_col)[reward_col]
    n = grp.size()
    c = grp.apply(lambda s: int((s >= 1.0).sum()))
    if ks is None:
        ks = eligible_ks(int(n.min())) if len(n) else []
    rows = []
    for task in n.index:
        row = {"task": task, "n": int(n[task]), "c": int(c[task])}
        for k in ks:
            row[f"pass@{k}"] = pass_at_k(int(n[task]), int(c[task]), k) if k <= n[task] else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        mean = {"task": "mean", "n": int(out["n"].sum()), "c": int(out["c"].sum())}
        for k in ks:
            mean[f"pass@{k}"] = float(out[f"pass@{k}"].mean())
        out = pd.concat([out, pd.DataFrame([mean])], ignore_index=True)
    return out


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10_000, seed: int = 0, stat: Callable = np.mean, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap of `stat` over the (task-level) values."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A fail, B pass (discordant in favour of B)
    c: int  # A pass, B fail
    statistic: float
    p: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.b + self.c


def mcnemar(b: int, c: int, *, exact: bool = True) -> McNemarResult:
    """Paired binary comparison on the discordant counts. Exact = two-sided binomial test."""
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "exact" if exact else "chi2")
    if exact:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        return McNemarResult(b, c, float(min(b, c)), min(1.0, p), "exact")
    from scipy.stats import chi2

    stat = (abs(b - c) - 1) ** 2 / n
    return McNemarResult(b, c, float(stat), float(chi2.sf(stat, 1)), "chi2-cc")


def per_task_outcomes(df, *, reduce: str = "mean", task_col: str = "task", passed_col: str = "passed", status_col: str = "status"):
    """Series task -> outcome in [0,1] from a job_summary frame (clean trials only).
    reduce: mean (pass rate) | first (first trial) | majority | any."""
    d = df[df[status_col] == "clean"] if status_col in df else df
    g = d.groupby(task_col)[passed_col]
    if reduce == "mean":
        return g.mean().astype(float)
    if reduce == "first":
        return g.first().astype(float)
    if reduce == "majority":
        return (g.mean() > 0.5).astype(float)
    if reduce == "any":
        return g.max().astype(float)
    raise ValueError(reduce)


def paired_compare(a, b, *, reduce: str = "mean", n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired per-task comparison of two job_summary frames (A = baseline, B = candidate)."""
    pa = per_task_outcomes(a, reduce=reduce)
    pb = per_task_outcomes(b, reduce=reduce)
    common = sorted(set(pa.index) & set(pb.index))
    xa = np.array([pa[t] for t in common])
    xb = np.array([pb[t] for t in common])
    diff = xb - xa
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    ba = (xa >= 0.5).astype(int)
    bb = (xb >= 0.5).astype(int)
    b_disc = int(((ba == 0) & (bb == 1)).sum())
    c_disc = int(((ba == 1) & (bb == 0)).sum())
    mc = mcnemar(b_disc, c_disc)
    return {
        "n_tasks": len(common),
        "mean_a": float(xa.mean()) if len(common) else float("nan"),
        "mean_b": float(xb.mean()) if len(common) else float("nan"),
        "delta_pp": float(diff.mean() * 100) if len(common) else float("nan"),
        "delta_ci95_pp": (lo * 100, hi * 100),
        "gained": [t for t, d in zip(common, diff) if d > 0],
        "lost": [t for t, d in zip(common, diff) if d < 0],
        "mcnemar": {"b": mc.b, "c": mc.c, "p": mc.p, "method": mc.method},
        "reduce": reduce,
        "dropped_tasks": sorted((set(pa.index) | set(pb.index)) - set(common)),
    }

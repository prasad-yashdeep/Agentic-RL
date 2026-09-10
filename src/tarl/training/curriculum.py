"""Difficulty curriculum + dynamic sampling (docs/03-math.md §3, DAPO).

Per-task pass rate is tracked as a Beta posterior; sampling prefers tasks whose posterior mean
lies in the informative band [0.05, 0.95] — tasks the policy solves *sometimes*. Groups that come
back all-equal carry zero advantage and are dropped (`filter_informative_groups`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from tarl.training.advantages import group_advantages, group_is_informative


@dataclass
class TaskStats:
    n: int = 0
    successes: int = 0
    a: float = 1.0  # Beta prior
    b: float = 1.0

    @property
    def p_hat(self) -> float:
        return (self.successes + self.a) / (self.n + self.a + self.b)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        from scipy.stats import beta

        lo = float(beta.ppf(alpha / 2, self.successes + self.a, self.n - self.successes + self.b))
        hi = float(beta.ppf(1 - alpha / 2, self.successes + self.a, self.n - self.successes + self.b))
        return lo, hi


@dataclass
class Curriculum:
    tasks: Sequence[str]
    band: tuple[float, float] = (0.05, 0.95)
    prior: tuple[float, float] = (1.0, 1.0)
    seed: int = 0
    stats: dict[str, TaskStats] = field(default_factory=dict)
    last_sample_fallback: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        for t in self.tasks:
            self.stats.setdefault(t, TaskStats(a=self.prior[0], b=self.prior[1]))

    # updates ---------------------------------------------------------------
    def update(self, task: str, passed: bool) -> None:
        st = self.stats.setdefault(task, TaskStats(a=self.prior[0], b=self.prior[1]))
        st.n += 1
        st.successes += int(bool(passed))

    def update_group(self, task: str, rewards: Iterable[float]) -> None:
        for r in rewards:
            self.update(task, float(r) >= 1.0)

    # queries ---------------------------------------------------------------
    def p_hat(self, task: str) -> float:
        return self.stats[task].p_hat

    def in_band(self) -> list[str]:
        lo, hi = self.band
        return [t for t in self.tasks if lo <= self.stats[t].p_hat <= hi]

    def trainable_fraction(self) -> float:
        return len(self.in_band()) / max(1, len(self.tasks))

    def sample(self, n: int, strategy: str = "band_uniform") -> list[str]:
        """n distinct tasks. `band_uniform`: uniform within the band (falls back to the tasks
        closest to the band and sets `last_sample_fallback`); `uncertainty`: weight by p(1-p);
        `all`: uniform over every task."""
        pool = list(self.tasks)
        self.last_sample_fallback = False
        if strategy == "all":
            return self._rng.sample(pool, min(n, len(pool)))
        band = self.in_band()
        if strategy == "band_uniform":
            if len(band) >= n:
                return self._rng.sample(band, n)
            self.last_sample_fallback = True
            lo, hi = self.band
            dist = lambda t: 0.0 if lo <= self.stats[t].p_hat <= hi else min(abs(self.stats[t].p_hat - lo), abs(self.stats[t].p_hat - hi))
            rest = sorted((t for t in pool if t not in band), key=dist)
            return band + rest[: n - len(band)]
        if strategy == "uncertainty":
            weights = [max(1e-6, self.stats[t].p_hat * (1 - self.stats[t].p_hat)) for t in pool]
            chosen: list[str] = []
            cand, w = pool[:], weights[:]
            while cand and len(chosen) < n:
                pick = self._rng.choices(range(len(cand)), weights=w, k=1)[0]
                chosen.append(cand.pop(pick))
                w.pop(pick)
            return chosen
        raise ValueError(f"unknown strategy {strategy!r}")

    def table(self):
        import pandas as pd

        lo, hi = self.band
        return pd.DataFrame(
            [
                {
                    "task": t,
                    "n": st.n,
                    "successes": st.successes,
                    "p_hat": st.p_hat,
                    "in_band": lo <= st.p_hat <= hi,
                }
                for t, st in self.stats.items()
            ]
        )


def filter_informative_groups(groups: Mapping[str, Sequence[float]], tol: float = 1e-6) -> tuple[dict[str, list[float]], list[str]]:
    """DAPO dynamic sampling: keep only groups whose rewards are not all equal."""
    kept: dict[str, list[float]] = {}
    dropped: list[str] = []
    for k, rewards in groups.items():
        if group_is_informative(group_advantages(rewards), tol):
            kept[k] = [float(r) for r in rewards]
        else:
            dropped.append(k)
    return kept, dropped


def seed_from_summary(df, *, band=(0.05, 0.95), seed: int = 0) -> Curriculum:
    """Initialise per-task stats from a tarl.data.job_summary frame (clean trials only)."""
    clean = df[df["status"] == "clean"]
    tasks = sorted(df["task"].unique())
    cur = Curriculum(tasks, band=band, seed=seed)
    for task, grp in clean.groupby("task"):
        for passed in grp["passed"]:
            cur.update(task, bool(passed))
    return cur

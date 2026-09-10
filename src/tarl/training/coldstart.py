"""The zero-gradient cold start, quantified (docs/03-math.md §3) — plus a SYNTHETIC toy
simulation that illustrates why judge-shaped rewards keep learning alive in all-fail groups.

Nothing here is a prediction of real training; it exists to make the argument visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tarl.training.advantages import group_advantages, useful_group_fraction  # noqa: F401 (re-export)

DEFAULT_PS = (0.001, 0.01, 0.02, 0.033, 0.05, 0.08, 0.1, 0.2, 0.5)
DEFAULT_GS = (4, 8, 16)


def useful_fraction_table(ps=DEFAULT_PS, Gs=DEFAULT_GS):
    """DataFrame(p, G=4, G=8, G=16, …) of useful-group fractions."""
    import pandas as pd

    rows = []
    for p in ps:
        row = {"p": p}
        for G in Gs:
            row[f"G={G}"] = useful_group_fraction(p, G)
        rows.append(row)
    return pd.DataFrame(rows)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class SimConfig:
    p0: float = 0.02
    G: int = 8
    steps: int = 200
    n_tasks: int = 20
    lr_gain: float = 0.3
    judge_shaping: bool = False
    judge_corr: float = 0.6
    judge_noise: float = 0.2
    lam: float = 0.5
    seed: int = 0
    task_spread: float = 1.0

    def as_dict(self) -> dict:
        return {**self.__dict__, "synthetic": True}


def simulate_coldstart(cfg: SimConfig | None = None, **kw):
    """SYNTHETIC latent-skill bandit.

    Each task has a latent skill s (probit scale); a rollout draws quality q ~ N(s, 1) and passes
    iff q > 0, so P(pass) = Φ(s). Verifier reward R = 1[q>0]; with judge shaping a failed rollout
    gets R = lam · sigmoid(judge_corr·q + noise) — a noisy, monotone proxy of how close it came.
    Dr.GRPO advantages A_i = R_i − mean(R); skill moves along Σ_i A_i (q_i − q̄): rollouts that
    were better than their siblings pull the skill up. All-fail groups give A ≡ 0 without the
    judge, so the verifier-only arm cannot learn from tasks it never solves.
    Returns DataFrame(step, condition, mean_p, useful_group_frac, judged_calls).
    """
    import numpy as np

    cfg = cfg or SimConfig(**kw)
    rng = np.random.default_rng(cfg.seed)
    s0 = float(np.sqrt(2.0) * _inv_erf(2 * cfg.p0 - 1))  # Φ^{-1}(p0)
    s = s0 + rng.normal(0.0, cfg.task_spread, cfg.n_tasks)
    rows = []
    judged = 0
    for step in range(cfg.steps + 1):
        useful = 0
        for k in range(cfg.n_tasks):
            q = rng.normal(s[k], 1.0, cfg.G)
            passed = q > 0
            R = passed.astype(float)
            if cfg.judge_shaping:
                fails = ~passed
                nf = int(fails.sum())
                if nf:
                    judged += nf
                    proxy = 1.0 / (1.0 + np.exp(-(cfg.judge_corr * q[fails] + rng.normal(0, cfg.judge_noise, nf))))
                    R[fails] = cfg.lam * proxy
            A = np.asarray(group_advantages(R.tolist()))
            if np.abs(A).max() > 1e-9:
                useful += 1
                if step < cfg.steps:
                    s[k] += cfg.lr_gain * float(np.sum(A * (q - q.mean()))) / cfg.G
        rows.append(
            {
                "step": step,
                "condition": "judge-shaped" if cfg.judge_shaping else "verifier-only",
                "mean_p": float(np.mean([_phi(x) for x in s])),
                "useful_group_frac": useful / cfg.n_tasks,
                "judged_calls": judged,
                "seed": cfg.seed,
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)


def _inv_erf(y: float) -> float:
    """Inverse error function via scipy when available, else a Newton fallback."""
    try:
        from scipy.special import erfinv

        return float(erfinv(y))
    except Exception:  # pragma: no cover
        x = 0.0
        for _ in range(60):
            x -= (math.erf(x) - y) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
        return x


def simulate_both(seeds=(0, 1, 2, 3, 4), **kw):
    """Both arms over several seeds, concatenated."""
    import pandas as pd

    frames = []
    for sd in seeds:
        frames.append(simulate_coldstart(SimConfig(judge_shaping=False, seed=sd, **kw)))
        frames.append(simulate_coldstart(SimConfig(judge_shaping=True, seed=sd, **kw)))
    return pd.concat(frames, ignore_index=True)

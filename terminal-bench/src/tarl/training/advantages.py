"""Group-relative advantages (docs/03-math.md §2).

* ``dr_grpo`` (default, what 06 used): A_i = R_i − mean(R)   — no ÷std (Dr. GRPO, arXiv:2503.20783;
  the 1/std term up-weights near-solved / near-impossible tasks).
* ``grpo_std``: A_i = (R_i − mean) / (std + eps)             — the original DeepSeekMath form.

An all-equal group (all-fail or all-pass under a binary reward) yields zeros: no gradient. That
is the cold-start failure the judge-shaped reward exists to fix (§3).
"""

from __future__ import annotations

from typing import Mapping, Sequence


def group_advantages(rewards: Sequence[float], mode: str = "dr_grpo", eps: float = 1e-8) -> list[float]:
    r = [float(x) for x in rewards]
    n = len(r)
    if n == 0:
        return []
    mean = sum(r) / n
    adv = [x - mean for x in r]
    if mode == "grpo_std":
        var = sum(a * a for a in adv) / n
        std = var**0.5
        adv = [a / (std + eps) for a in adv]
    elif mode != "dr_grpo":
        raise ValueError(f"unknown advantage mode {mode!r} (dr_grpo | grpo_std)")
    return adv


def advantages_by_group(rewards_by_group: Mapping[str, Sequence[float]], mode: str = "dr_grpo") -> dict[str, list[float]]:
    return {k: group_advantages(v, mode) for k, v in rewards_by_group.items()}


def informative(adv: float, tol: float = 1e-6) -> bool:
    """06:317 rule — a rollout with |A| < tol carries no signal and is skipped."""
    return abs(float(adv)) >= tol


def group_is_informative(advs: Sequence[float], tol: float = 1e-6) -> bool:
    return any(informative(a, tol) for a in advs)


def useful_group_fraction(p: float, G: int) -> float:
    """P(a group of G Bernoulli(p) rollouts is neither all-fail nor all-pass) = 1 − p^G − (1−p)^G."""
    return 1.0 - p**G - (1.0 - p) ** G

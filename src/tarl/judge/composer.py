"""Gated composite reward. The verifier always dominates; the judge can only
rank failures against failures.

    R = 1.0                      if verifier passed
    R = lambda_judge * r_judge   otherwise (lambda_judge <= 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    lambda_judge: float = 0.5

    def __post_init__(self):
        if not (0.0 <= self.lambda_judge <= 0.5):
            raise ValueError(
                f"lambda_judge={self.lambda_judge}: gate requires <= 0.5 so a "
                "judged failure can never outrank a verified pass"
            )


def composite_reward(verifier_passed: bool, judge_score: float | None, cfg: RewardConfig) -> float:
    if verifier_passed:
        return 1.0
    if judge_score is None:  # judge unavailable -> verifier-only semantics
        return 0.0
    if not (0.0 <= judge_score <= 1.0):
        raise ValueError(f"judge_score out of range: {judge_score}")
    return cfg.lambda_judge * judge_score

"""Spend accounting for live judge calls (configs/judge/judge.yaml: cost_cap_usd_per_run)."""

from __future__ import annotations

from dataclasses import dataclass, field

from tarl.judge.client import JudgeVerdict
from tarl.judge.protocol import Judge, JudgeUnavailable


class JudgeBudgetExceeded(JudgeUnavailable):
    pass


@dataclass
class CostMeter:
    price_in_per_mtok: float
    price_out_per_mtok: float
    cap_usd: float | None = None
    spent_usd: float = 0.0
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = prompt_tokens / 1e6 * self.price_in_per_mtok + completion_tokens / 1e6 * self.price_out_per_mtok
        self.spent_usd += cost
        self.n_calls += 1
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        return cost

    @property
    def exhausted(self) -> bool:
        return self.cap_usd is not None and self.spent_usd >= self.cap_usd

    def snapshot(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "cap_usd": self.cap_usd,
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class CostCappedJudge:
    inner: Judge
    meter: CostMeter
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = getattr(self.inner, "name", "judge")

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        if self.meter.exhausted:
            raise JudgeBudgetExceeded(
                f"judge budget exhausted: spent ${self.meter.spent_usd:.2f} >= cap ${self.meter.cap_usd}"
            )
        v = self.inner.judge(system_prompt, user_prompt)
        self.meter.add(v.prompt_tokens, v.completion_tokens)
        return v

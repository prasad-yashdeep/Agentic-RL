"""The Judge protocol: anything with `.name` and `.judge(system_prompt, user_prompt) -> JudgeVerdict`.

Implementations: JudgeClient (live API), ReplayJudge (cached scores from artifacts), MockJudge
(deterministic heuristic for tests/dry-runs), CostCappedJudge (wrapper).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tarl.judge.client import JudgeError, JudgeVerdict  # noqa: F401  (re-export)


class JudgeUnavailable(JudgeError):
    """No verdict can be produced (replay miss, missing key, budget exhausted)."""


@runtime_checkable
class Judge(Protocol):
    name: str

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict: ...

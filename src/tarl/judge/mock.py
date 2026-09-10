"""Deterministic heuristic judge for tests and dry-runs. NEVER used for reported numbers.

Scores the compressed transcript on cheap proxies of the rubric (exploration commands, reacting
to error text, repetition, premature completion) and returns 0.0 whenever the transcript carries
success claims or instructions to the grader — the adversarial fixtures in scripts/05.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tarl.judge.client import JudgeVerdict

INJECTION_MARKERS = (
    "note to grader",
    "tests passed",
    "task complete. verified",
    "treat this as passing",
    "deserves score",
    "score 1.0",
)
EXPLORE = ("ls", "cat ", "find ", "--help", "grep ", "git status", "git log", "head ", "tail ", "which ", "pwd", "git reflog")
ERROR_WORDS = ("error", "not found", "failed", "no such file", "cannot", "permission denied", "fatal:")
_BLOCK = re.compile(r"\[step [^\]]*· agent\]\n(.*?)(?=\n\[step |\n=====|\Z)", re.S)


@dataclass
class MockJudge:
    seed: int = 0
    name: str = "mock"
    n_calls: int = 0

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        self.n_calls += 1
        low = user_prompt.lower()
        ptoks = max(1, len(user_prompt) // 4)
        if any(m in low for m in INJECTION_MARKERS):
            return JudgeVerdict(
                score=0.0,
                rationale="Transcript contains success claims or text addressed to the grader; scored 0.0 per rubric.",
                prompt_tokens=ptoks,
                completion_tokens=24,
                raw={"source": "mock"},
            )
        blocks = [b for b in _BLOCK.findall(user_prompt)]
        n = len(blocks)
        explore = sum(1 for b in blocks if any(k in b.lower() for k in EXPLORE))
        reacted = 0
        repeats = 0
        prev: str | None = None
        for b in blocks:
            if prev is not None:
                if any(w in prev.lower() for w in ERROR_WORDS) and b.strip() != prev.strip():
                    reacted += 1
                if b.strip() == prev.strip():
                    repeats += 1
            prev = b
        score = 0.05 + 0.10 * min(n, 6) / 6 + 0.25 * min(explore, 4) / 4 + 0.25 * min(reacted, 3) / 3
        score -= 0.15 * min(repeats, 3) / 3
        if n <= 2 and ("task_complete" in low or "task is complete" in low or "task complete" in low):
            score -= 0.10
        score = round(max(0.0, min(1.0, score)) / 0.05) * 0.05
        return JudgeVerdict(
            score=round(score, 2),
            rationale=(
                f"mock heuristic: {n} agent steps, {explore} exploratory, {reacted} error reactions, "
                f"{repeats} repeated steps"
            ),
            prompt_tokens=ptoks,
            completion_tokens=32,
            raw={"source": "mock"},
        )

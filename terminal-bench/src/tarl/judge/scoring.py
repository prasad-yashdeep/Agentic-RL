"""Gated composite reward for a batch of rollouts (extraction of scripts/06_grpo_micro.py:139-150)
plus the production wrappers that 06 lacked: alarms, cost cap, provenance of each score."""

from __future__ import annotations

from typing import Callable, Iterable

from tarl.judge.alarms import JudgeAlarms
from tarl.judge.client import JudgeError
from tarl.judge.composer import RewardConfig, composite_reward
from tarl.judge.compressor import compress, load_trajectory
from tarl.judge.cost import JudgeBudgetExceeded
from tarl.judge.protocol import Judge


def score_rollouts(
    rollouts: Iterable,
    judge: Judge | None,
    rubric: str,
    cfg: RewardConfig | None = None,
    *,
    alarms: JudgeAlarms | None = None,
    log: Callable[[str], None] | None = None,
    compress_kwargs: dict | None = None,
    step: int | None = None,
) -> list[str]:
    """Sets ``ro.judge``, ``ro.judge_source``, ``ro.judge_rationale``, ``ro.R`` on every rollout.

    Verifier passes are never judged (R=1). Judge failures/budget exhaustion degrade to
    verifier-only semantics for that rollout (R=0), exactly like 06:147-148. Returns alarms.
    """
    cfg = cfg or RewardConfig()
    log = log or (lambda s: None)
    fired: list[str] = []
    budget_hit = False
    for ro in rollouts:
        passed = float(ro.verifier or 0.0) >= 1.0
        jscore: float | None = None
        src: str | None = None
        rationale: str | None = None
        if not passed and judge is not None and cfg.lambda_judge > 0 and not budget_hit:
            try:
                prompt = compress(load_trajectory(ro.trajectory), **(compress_kwargs or {}))
                v = judge.judge(rubric, prompt)
                jscore, rationale, src = v.score, v.rationale, getattr(judge, "name", "judge")
                ro.judge_tokens = v.prompt_tokens
            except JudgeBudgetExceeded as e:
                budget_hit = True
                fired.append(f"BUDGET: {e}")
                log(f"  judge budget exhausted ({e}); verifier-only from here")
            except (JudgeError, OSError, ValueError) as e:
                log(f"  judge failed ({e}); verifier-only for this rollout")
        ro.judge = jscore
        ro.judge_source = src
        ro.judge_rationale = rationale
        ro.R = composite_reward(passed, jscore, cfg)
        if alarms is not None:
            fired.extend(
                alarms.record(
                    jscore,
                    passed,
                    meta={"task": ro.task, "trial": str(getattr(ro, "trial_dir", "")), "step": step, "source": src},
                )
            )
    return fired

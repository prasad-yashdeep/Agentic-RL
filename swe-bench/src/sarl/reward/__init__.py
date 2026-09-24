"""Reward: replay ONLY the candidate patch in a fresh container from the same image, run the task's eval script,
and grade the log with the official swebench (v5) parsers. Binary: 1.0 iff every FAIL_TO_PASS test now passes and
every PASS_TO_PASS test still passes."""

from sarl.reward.grade import GradeResult, grade

__all__ = ["GradeResult", "grade"]

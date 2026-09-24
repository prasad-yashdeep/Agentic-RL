"""ReplayJudge: serve judge verdicts recorded in artifacts instead of calling the API.

Sources
* experiments/judge-pilot.jsonl — 12 A0 failures scored during gate G2 (exact: keyed by path).
* g4.log + jobs-g4/ — the 40 micro-run rollouts (order-inferred: the log has no trial ids, the
  mapping relies on tarl.data.loader.legacy_rollout_order reproducing 06's glob order).

Key = sha256 of the exact prompt the judge received, i.e. ``compress(load_trajectory(path))`` with
default arguments — the same deterministic function 05/06 used, so hashing is consistent by
construction and any change to the legacy rendering would surface as replay misses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tarl.common.paths import G4_LOG, JOBS_G4, PILOT_JSONL, REPO
from tarl.judge.client import JudgeVerdict
from tarl.judge.compressor import compress, load_trajectory
from tarl.judge.protocol import JudgeUnavailable

OnMiss = Literal["error", "none", "mock"]


def prompt_key(user_prompt: str) -> str:
    return hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:24]


def trajectory_key(path: str | Path) -> str:
    return prompt_key(compress(load_trajectory(path)))


@dataclass
class ReplayEntry:
    score: float
    rationale: str
    source: str  # "pilot" | "g4log"
    task: str
    trial_dir: Path | None
    step: int | None
    order: int | None
    confidence: str  # "exact" | "order-inferred"
    prompt_tokens: int = 0


class ReplayJudge:
    name = "replay"

    def __init__(self, entries: dict[str, ReplayEntry], *, on_miss: OnMiss = "error"):
        self.entries = entries
        self.on_miss = on_miss
        self.n_calls = 0
        self.n_misses = 0
        self._mock = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_artifacts(
        cls,
        pilot_jsonl: Path | None = PILOT_JSONL,
        g4_log: Path | None = G4_LOG,
        jobs_g4: Path | None = JOBS_G4,
        *,
        on_miss: OnMiss = "error",
        strict: bool = True,
    ) -> "ReplayJudge":
        entries: dict[str, ReplayEntry] = {}
        if pilot_jsonl and Path(pilot_jsonl).exists():
            entries.update(cls._from_pilot(Path(pilot_jsonl)))
        if g4_log and jobs_g4 and Path(g4_log).exists() and Path(jobs_g4).exists():
            entries.update(cls._from_g4(Path(g4_log), Path(jobs_g4), strict=strict))
        return cls(entries, on_miss=on_miss)

    @staticmethod
    def _from_pilot(path: Path) -> dict[str, ReplayEntry]:
        out: dict[str, ReplayEntry] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = Path(rec["path"])
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                continue
            out[trajectory_key(p)] = ReplayEntry(
                score=float(rec["score"]),
                rationale=str(rec.get("rationale", "")),
                source="pilot",
                task=str(rec.get("task", p.parent.parent.name.rsplit("__", 1)[0])),
                trial_dir=p.parent.parent,
                step=None,
                order=None,
                confidence="exact",
                prompt_tokens=int(rec.get("prompt_tokens", 0)),
            )
        return out

    @staticmethod
    def _from_g4(g4_log: Path, jobs_g4: Path, *, strict: bool) -> dict[str, ReplayEntry]:
        from tarl.data.loader import legacy_rollout_order
        from tarl.judge.g4log import parse_g4_log

        run = parse_g4_log(g4_log)
        out: dict[str, ReplayEntry] = {}
        for st in run.steps:
            job = jobs_g4 / st.job
            if not job.exists():
                if strict:
                    raise FileNotFoundError(f"g4 step {st.step}: job dir {job} missing")
                continue
            trials = legacy_rollout_order(job)
            by_task: dict[str, list] = {}
            for t in trials:
                by_task.setdefault(t.task, []).append(t)
            for task, group in by_task.items():
                logged = st.group(task)
                if strict and len(logged) != len(group):
                    raise ValueError(
                        f"g4 step {st.step} task {task}: {len(logged)} logged rollouts vs {len(group)} trials on disk"
                    )
                for r in logged:
                    if r.order >= len(group) or r.judge is None:
                        continue
                    t = group[r.order]
                    out[trajectory_key(t.trajectory_path)] = ReplayEntry(
                        score=float(r.judge),
                        rationale="(rationale not persisted by 06_grpo_micro.py; score replayed from g4.log)",
                        source="g4log",
                        task=task,
                        trial_dir=t.trial_dir,
                        step=st.step,
                        order=r.order,
                        confidence="order-inferred",
                    )
        return out

    # ------------------------------------------------------------------ use
    def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        self.n_calls += 1
        e = self.entries.get(prompt_key(user_prompt))
        if e is None:
            self.n_misses += 1
            if self.on_miss == "none":
                raise JudgeUnavailable("no cached verdict for this transcript (replay miss)")
            if self.on_miss == "mock":
                if self._mock is None:
                    from tarl.judge.mock import MockJudge

                    self._mock = MockJudge()
                v = self._mock.judge(system_prompt, user_prompt)
                v.raw = {**(v.raw or {}), "replay_miss": True}
                return v
            raise JudgeUnavailable(
                "no cached verdict for this transcript — set judge mode 'mock', or 'live' with --yes-spend"
            )
        return JudgeVerdict(
            score=e.score,
            rationale=e.rationale,
            prompt_tokens=e.prompt_tokens or max(1, len(user_prompt) // 4),
            completion_tokens=0,
            raw={"source": e.source, "confidence": e.confidence, "step": e.step, "order": e.order},
        )

    def lookup_trial(self, trial_dir: str | Path) -> ReplayEntry | None:
        p = Path(trial_dir) / "agent" / "trajectory.json"
        if not p.exists():
            return None
        return self.entries.get(trajectory_key(p))

    def score_for_path(self, trajectory_path: str | Path) -> float | None:
        e = self.entries.get(trajectory_key(trajectory_path))
        return e.score if e else None

    def table(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "key": k,
                    "task": e.task,
                    "trial": e.trial_dir.name if e.trial_dir else None,
                    "trial_dir": str(e.trial_dir) if e.trial_dir else None,
                    "step": e.step,
                    "order": e.order,
                    "score": e.score,
                    "source": e.source,
                    "confidence": e.confidence,
                    "rationale": e.rationale,
                }
                for k, e in self.entries.items()
            ]
        )

    def write_jsonl(self, path: str | Path) -> int:
        """Materialise the cache (committed as experiments/derived/g4-replay.jsonl)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w") as f:
            for k, e in self.entries.items():
                rec = {
                    "key": k,
                    "task": e.task,
                    "trial_dir": str(e.trial_dir.relative_to(REPO)) if e.trial_dir and str(e.trial_dir).startswith(str(REPO)) else (str(e.trial_dir) if e.trial_dir else None),
                    "step": e.step,
                    "order": e.order,
                    "score": e.score,
                    "source": e.source,
                    "confidence": e.confidence,
                    "rationale": e.rationale,
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
        return n

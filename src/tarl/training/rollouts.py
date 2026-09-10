"""Where rollouts come from: a live Harbor run (node08) or recorded job directories (replay)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from tarl.common.env import refuse_on_headnode
from tarl.common.paths import REPO
from tarl.data.loader import Trial, legacy_rollout_order


@dataclass
class RolloutRecord:
    step: int
    task: str
    trial_dir: Path
    trajectory: Path
    verifier: float
    judge: float | None = None
    judge_source: str | None = None
    judge_rationale: str | None = None
    judge_tokens: int | None = None
    R: float | None = None
    adv: float | None = None

    @classmethod
    def from_trial(cls, step: int, t: Trial) -> "RolloutRecord":
        return cls(step=step, task=t.task, trial_dir=t.trial_dir, trajectory=t.trajectory_path, verifier=float(t.reward or 0.0))


class RolloutSource(Protocol):
    def collect(self, step: int) -> list[RolloutRecord]: ...


class ReplayRolloutSource:
    """Serve recorded Harbor jobs as if they were fresh rollouts (dry runs / parity tests)."""

    def __init__(self, jobs_dir: str | Path, step_to_job: dict[int, str] | None = None, tasks: Sequence[str] | None = None):
        self.jobs_dir = Path(jobs_dir)
        self.tasks = set(tasks) if tasks else None
        if step_to_job is None:
            jobs = sorted(p.name for p in self.jobs_dir.glob("2*") if p.is_dir())
            step_to_job = {i + 1: name for i, name in enumerate(jobs)}
        self.step_to_job = step_to_job
        self.last_job: Path | None = None

    def collect(self, step: int) -> list[RolloutRecord]:
        name = self.step_to_job.get(step)
        if name is None:
            raise KeyError(f"no recorded job for step {step} (have {sorted(self.step_to_job)})")
        job = self.jobs_dir / name
        self.last_job = job
        out = [RolloutRecord.from_trial(step, t) for t in legacy_rollout_order(job)]
        if self.tasks is not None:
            out = [r for r in out if r.task in self.tasks]
        return out


class HarborRolloutSource:
    """Launch `harbor run` (Terminus 2, singularity backend) against the local vLLM serve and read the
    newest job dir back — scripts/06_grpo_micro.py:105-135 behind one class. node08 only."""

    def __init__(self, cfg, *, env_sh: Path = REPO / "scripts" / "env.sh"):
        refuse_on_headnode("harbor rollouts")
        self.cfg = cfg
        self.env_sh = env_sh
        self.last_job: Path | None = None

    def collect(self, step: int) -> list[RolloutRecord]:
        from tarl.serving.harbor_cmd import build_harbor_cmd

        cfg = self.cfg
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_harbor_cmd(
            tb_path=cfg.tb_path,
            tasks=cfg.tasks,
            k=cfg.G,
            jobs_dir=cfg.jobs_dir,
            temperature=cfg.temperature,
            n_concurrent=cfg.n_concurrent,
            timeout_multiplier=cfg.timeout_multiplier,
            collect_rollout_details=cfg.collect_rollout_details,
        )
        shell = f"cd {REPO} && . {self.env_sh} >/dev/null 2>&1 && OPENAI_API_KEY=dummy " + " ".join(cmd)
        r = subprocess.run(["bash", "-lc", shell], capture_output=True, text=True, timeout=5400, env=dict(os.environ))
        jobs = sorted(cfg.jobs_dir.glob("2*"), key=lambda p: p.stat().st_mtime)
        if not jobs:
            raise RuntimeError(f"no rollout job dir; harbor stderr: {r.stderr[-500:]}")
        self.last_job = jobs[-1]
        out = [RolloutRecord.from_trial(step, t) for t in legacy_rollout_order(self.last_job)]
        return [ro for ro in out if ro.task in set(cfg.tasks)]

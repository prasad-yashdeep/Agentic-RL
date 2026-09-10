"""Parse g4.log (scripts/06_grpo_micro.py stdout) into structured records.

The log is the ONLY place the G4 judge scores and advantages survive (06 kept them in memory).
Rollout lines are emitted in ``job.glob("*/result.json")`` order, grouped by task; they carry no
trial id, so `order` (0-based within (step, task)) is the join key for
tarl.data.loader.legacy_rollout_order.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from tarl.common.paths import G4_LOG

_HEADER = re.compile(r"^\[g4\] rubric (\w+), judge (\S+), tasks (\[.*?\]), G=(\d+), steps=(\d+)")
_STEP = re.compile(r"^\[g4\] step (\d+): (\d+) clean rollouts from (\S+)")
_ROLLOUT = re.compile(r"^\[g4\]\s+(\S+) R=([\d.]+) \(verifier=([\d.]+), judge=(None|[\d.]+)\) adv=([+-]?[\d.]+)")
_UPDATE = re.compile(r"^\[g4\] step (\d+): UPDATE loss/tok=([-\d.]+) grad_norm=([\d.]+) tokens=(\d+) \((\d+)s\)")
_SYNC = re.compile(r"^\[g4\] step (\d+): SYNC adapter=(\w+) probe_changed=(True|False)")
_NOINFO = re.compile(r"^\[g4\] step (\d+): no informative rollouts")


@dataclass
class G4Rollout:
    step: int
    task: str
    order: int
    R: float
    verifier: float
    judge: float | None
    adv: float


@dataclass
class G4Update:
    step: int
    loss_tok: float
    grad_norm: float
    tokens: int
    seconds: int


@dataclass
class G4Sync:
    step: int
    adapter_hash: str
    probe_changed: bool


@dataclass
class G4Step:
    step: int
    n_clean: int
    job: str
    rollouts: list[G4Rollout] = field(default_factory=list)
    update: G4Update | None = None
    sync: G4Sync | None = None
    skipped: bool = False

    def group(self, task: str) -> list[G4Rollout]:
        return [r for r in self.rollouts if r.task == task]


@dataclass
class G4Run:
    rubric_hash: str
    judge_model: str
    tasks: list[str]
    G: int
    n_steps: int
    steps: list[G4Step] = field(default_factory=list)
    done: bool = False

    def step(self, n: int) -> G4Step:
        for s in self.steps:
            if s.step == n:
                return s
        raise KeyError(n)


def parse_g4_log(path: str | Path = G4_LOG) -> G4Run:
    run: G4Run | None = None
    cur: G4Step | None = None
    counters: dict[tuple[int, str], int] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _HEADER.match(line)
        if m:
            run = G4Run(
                rubric_hash=m.group(1),
                judge_model=m.group(2),
                tasks=list(ast.literal_eval(m.group(3))),
                G=int(m.group(4)),
                n_steps=int(m.group(5)),
            )
            continue
        if run is None:
            continue
        m = _STEP.match(line)
        if m:
            cur = G4Step(step=int(m.group(1)), n_clean=int(m.group(2)), job=m.group(3))
            run.steps.append(cur)
            continue
        m = _ROLLOUT.match(line)
        if m and cur is not None:
            task = m.group(1)
            key = (cur.step, task)
            order = counters.get(key, 0)
            counters[key] = order + 1
            cur.rollouts.append(
                G4Rollout(
                    step=cur.step,
                    task=task,
                    order=order,
                    R=float(m.group(2)),
                    verifier=float(m.group(3)),
                    judge=None if m.group(4) == "None" else float(m.group(4)),
                    adv=float(m.group(5)),
                )
            )
            continue
        m = _UPDATE.match(line)
        if m:
            st = run.step(int(m.group(1)))
            st.update = G4Update(int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5)))
            continue
        m = _SYNC.match(line)
        if m:
            st = run.step(int(m.group(1)))
            st.sync = G4Sync(int(m.group(1)), m.group(2), m.group(3) == "True")
            continue
        m = _NOINFO.match(line)
        if m:
            run.step(int(m.group(1))).skipped = True
            continue
        if "G4_MICRO_DONE" in line:
            run.done = True
    if run is None:
        raise ValueError(f"{path}: no '[g4] rubric …' header found")
    return run


def to_frame(run: G4Run):
    """One row per rollout with the step's update/sync joined in."""
    import pandas as pd

    rows = []
    for st in run.steps:
        for r in st.rollouts:
            rows.append(
                {
                    "step": st.step,
                    "job": st.job,
                    "task": r.task,
                    "order": r.order,
                    "R": r.R,
                    "verifier": r.verifier,
                    "judge": r.judge,
                    "adv": r.adv,
                    "loss_tok": st.update.loss_tok if st.update else None,
                    "grad_norm": st.update.grad_norm if st.update else None,
                    "tokens": st.update.tokens if st.update else None,
                    "seconds": st.update.seconds if st.update else None,
                    "adapter_hash": st.sync.adapter_hash if st.sync else None,
                    "probe_changed": st.sync.probe_changed if st.sync else None,
                }
            )
    return pd.DataFrame(rows)


def steps_frame(run: G4Run):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "step": st.step,
                "job": st.job,
                "n_clean": st.n_clean,
                "n_rollouts": len(st.rollouts),
                "loss_tok": st.update.loss_tok if st.update else None,
                "grad_norm": st.update.grad_norm if st.update else None,
                "tokens": st.update.tokens if st.update else None,
                "seconds": st.update.seconds if st.update else None,
                "adapter_hash": st.sync.adapter_hash if st.sync else None,
                "probe_changed": st.sync.probe_changed if st.sync else None,
                "skipped": st.skipped,
            }
            for st in run.steps
        ]
    )

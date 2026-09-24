"""Harbor job/trial artifacts -> Trial / Episode objects and one-row-per-trial DataFrames.

Layout (see CLAUDE.md, docs/02-architecture.md):
  <job_dir>/<task>__<id>/
      result.json                 phases, tokens, verifier reward, exception_info
      agent/trajectory.json       ATIF-v1.7: steps[{source user|agent, message, tool_calls,
                                  observation, reasoning_content, metrics}]
      agent/recording.cast        asciicast v2 (what was typed / shown)
      agent/terminus_2.pane       final tmux pane
      verifier/reward.txt         "0" | "1";   verifier/ctrf.json  pytest CTRF report

Every loader tolerates missing pieces (errored trials) and never raises on one bad trial.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tarl.judge.compressor import load_trajectory

PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
STATUSES = ("clean", "errored_with_trajectory", "errored_no_trajectory", "unverified", "incomplete")


# --------------------------------------------------------------------------- helpers
def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _seconds(a: str | None, b: str | None) -> float | None:
    ta, tb = _parse_ts(a), _parse_ts(b)
    return (tb - ta).total_seconds() if ta and tb else None


def _phase_seconds(result: dict) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for ph in PHASES:
        blk = result.get(ph) or {}
        out[ph] = _seconds(blk.get("started_at"), blk.get("finished_at"))
    return out


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        warnings.warn(f"unreadable JSON {path}: {e}")
        return None


def _task_from(result: dict, trial_dir: Path) -> str:
    tn = str(result.get("task_name") or "")
    if "/" in tn:
        return tn.rsplit("/", 1)[-1]
    if tn:
        return tn
    return trial_dir.name.rsplit("__", 1)[0]


def step_commands(step: dict) -> list[str]:
    """Keystroke strings of one agent step (task-complete markers excluded)."""
    out = []
    for tc in step.get("tool_calls") or []:
        args = tc.get("arguments") or {}
        ks = args.get("keystrokes")
        if ks:
            out.append(str(ks))
    return out


def step_observation(step: dict) -> str:
    res = (step.get("observation") or {}).get("results") or []
    return "\n".join(str(r.get("content") or "") for r in res)


def step_declares_complete(step: dict) -> bool:
    return any((tc.get("function_name") == "mark_task_complete") for tc in step.get("tool_calls") or [])


# --------------------------------------------------------------------------- Trial
@dataclass(frozen=True)
class Trial:
    """Cheap view of one trial (result.json only)."""

    trial_dir: Path
    job_dir: Path
    job_id: str | None
    task: str
    trial_name: str
    trial_id: str
    reward: float | None
    passed: bool
    exception_type: str | None
    status: str
    has_trajectory: bool
    has_ctrf: bool
    result: dict = field(repr=False, compare=False)

    @property
    def trajectory_path(self) -> Path:
        return self.trial_dir / "agent" / "trajectory.json"

    @property
    def cast_path(self) -> Path:
        return self.trial_dir / "agent" / "recording.cast"

    @property
    def pane_path(self) -> Path:
        return self.trial_dir / "agent" / "terminus_2.pane"

    @property
    def ctrf_path(self) -> Path:
        return self.trial_dir / "verifier" / "ctrf.json"

    @property
    def errored(self) -> bool:
        return self.exception_type is not None


def trial_from_dir(trial_dir: str | Path, *, allow_incomplete: bool = False) -> Trial | None:
    trial_dir = Path(trial_dir)
    rp = trial_dir / "result.json"
    result = _read_json(rp) if rp.exists() else None
    has_traj = (trial_dir / "agent" / "trajectory.json").exists()
    has_ctrf = (trial_dir / "verifier" / "ctrf.json").exists()
    if result is None:
        if not allow_incomplete:
            return None
        result, exc, reward, status = {}, None, None, "incomplete"
    else:
        exc = (result.get("exception_info") or {}).get("exception_type")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        reward = float(reward) if reward is not None else None
        if exc:
            status = "errored_with_trajectory" if has_traj else "errored_no_trajectory"
        elif reward is None:
            status = "unverified"
        else:
            status = "clean"
    name = trial_dir.name
    return Trial(
        trial_dir=trial_dir,
        job_dir=trial_dir.parent,
        job_id=(result.get("config") or {}).get("job_id"),
        task=_task_from(result, trial_dir),
        trial_name=name,
        trial_id=name.rsplit("__", 1)[-1] if "__" in name else name,
        reward=reward,
        passed=bool(reward is not None and reward >= 1.0),
        exception_type=exc,
        status=status,
        has_trajectory=has_traj,
        has_ctrf=has_ctrf,
        result=result,
    )


def iter_trials(
    job_dir: str | Path,
    *,
    include_errored: bool = True,
    require_trajectory: bool = False,
    include_incomplete: bool = False,
    sort: bool = True,
) -> Iterator[Trial]:
    """Deterministic (sorted) iteration over the trial directories of one Harbor job."""
    job_dir = Path(job_dir)
    dirs = [p for p in job_dir.iterdir() if p.is_dir() and "__" in p.name]
    if sort:
        dirs.sort()
    for d in dirs:
        t = trial_from_dir(d, allow_incomplete=include_incomplete)
        if t is None:
            continue
        if not include_errored and t.exception_type:
            continue
        if require_trajectory and not t.has_trajectory:
            continue
        yield t


def legacy_rollout_order(job_dir: str | Path) -> list[Trial]:
    """Exactly the trial order scripts/06_grpo_micro.py:121-133 used: ``job.glob("*/result.json")``
    (directory order, NOT sorted), skipping errored trials and trials without a trajectory.

    The G4 log's per-rollout lines are only reproducible through this order, which assumes the
    directory listing is unchanged since 2026-07-21 (plausible on an untouched NFS tree; not
    provable). ReplayJudge labels scores derived this way `order-inferred`.
    """
    out: list[Trial] = []
    for rj in Path(job_dir).glob("*/result.json"):
        t = trial_from_dir(rj.parent)
        if t is None or t.exception_type or not t.has_trajectory:
            continue
        out.append(t)
    return out


# --------------------------------------------------------------------------- CTRF
@dataclass
class CtrfSummary:
    tests: int
    passed: int
    failed: int
    failed_names: list[str]
    messages: dict[str, str]

    @property
    def label(self) -> str:
        return f"{self.passed}/{self.tests}"


def load_ctrf(path: str | Path) -> CtrfSummary | None:
    d = _read_json(Path(path))
    if not d:
        return None
    res = d.get("results") or {}
    summ = res.get("summary") or {}
    tests = res.get("tests") or []
    failed = [str(t.get("name", "?")) for t in tests if t.get("status") != "passed"]
    msgs = {
        str(t.get("name", "?")): str(t.get("message") or t.get("trace") or "")[:500]
        for t in tests
        if t.get("status") != "passed"
    }
    return CtrfSummary(
        tests=int(summ.get("tests", len(tests))),
        passed=int(summ.get("passed", sum(1 for t in tests if t.get("status") == "passed"))),
        failed=int(summ.get("failed", len(failed))),
        failed_names=failed,
        messages=msgs,
    )


# --------------------------------------------------------------------------- Episode
@dataclass
class Episode:
    """One trial with its trajectory unpacked (commands, observations, tokens, timings)."""

    trial: Trial
    trajectory: dict | None
    steps: list[dict]
    agent_steps: list[dict]
    commands: list[str]
    task_complete_declared: bool
    observations: list[str]
    reasoning: list[str | None]
    per_step_tokens: list[tuple[int | None, int | None]]
    tokens: dict
    n_episodes: int | None
    summarization_count: int | None
    api_request_times_ms: list[float]
    phases: dict[str, float | None]
    wall_s: float | None
    ctrf: CtrfSummary | None
    model_name: str | None
    temperature: float | None
    parser: str | None
    task_digest: str | None

    # passthroughs -----------------------------------------------------------
    @property
    def task(self) -> str:
        return self.trial.task

    @property
    def trial_dir(self) -> Path:
        return self.trial.trial_dir

    @property
    def reward(self) -> float | None:
        return self.trial.reward

    @property
    def passed(self) -> bool:
        return self.trial.passed

    @property
    def status(self) -> str:
        return self.trial.status

    @property
    def exception_type(self) -> str | None:
        return self.trial.exception_type

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_agent_steps(self) -> int:
        return len(self.agent_steps)

    @property
    def cast_path(self) -> Path:
        return self.trial.cast_path

    @property
    def pane_path(self) -> Path:
        return self.trial.pane_path

    def instruction(self) -> str | None:
        from tarl.data.instruction import task_instruction

        return task_instruction(self.trajectory)


def load_episode(trial_dir: str | Path) -> Episode:
    t = trial_from_dir(trial_dir, allow_incomplete=True)
    assert t is not None
    traj: dict | None = None
    if t.has_trajectory:
        try:
            traj = load_trajectory(t.trajectory_path)
        except (OSError, ValueError) as e:
            warnings.warn(f"unreadable trajectory {t.trajectory_path}: {e}")
            traj = None
    steps = list((traj or {}).get("steps") or [])
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    r = t.result
    ar = r.get("agent_result") or {}
    md = ar.get("metadata") or {}
    tokens = dict((traj or {}).get("final_metrics") or {})
    tokens.setdefault("n_input_tokens", ar.get("n_input_tokens"))
    tokens.setdefault("n_output_tokens", ar.get("n_output_tokens"))
    tokens.setdefault("n_cache_tokens", ar.get("n_cache_tokens"))
    agent_meta = (traj or {}).get("agent") or {}
    extra = agent_meta.get("extra") or {}
    return Episode(
        trial=t,
        trajectory=traj,
        steps=steps,
        agent_steps=agent_steps,
        commands=[c for s in agent_steps for c in step_commands(s)],
        task_complete_declared=any(step_declares_complete(s) for s in agent_steps),
        observations=[step_observation(s) for s in agent_steps],
        reasoning=[s.get("reasoning_content") for s in agent_steps],
        per_step_tokens=[
            ((s.get("metrics") or {}).get("prompt_tokens"), (s.get("metrics") or {}).get("completion_tokens"))
            for s in agent_steps
        ],
        tokens=tokens,
        n_episodes=md.get("n_episodes"),
        summarization_count=md.get("summarization_count"),
        api_request_times_ms=[float(x) for x in (md.get("api_request_times_msec") or [])],
        phases=_phase_seconds(r),
        wall_s=_seconds(r.get("started_at"), r.get("finished_at")),
        ctrf=load_ctrf(t.ctrf_path) if t.has_ctrf else None,
        model_name=agent_meta.get("model_name") or ((r.get("agent_info") or {}).get("model_info") or {}).get("name"),
        temperature=extra.get("temperature"),
        parser=extra.get("parser"),
        task_digest=r.get("task_checksum"),
    )


def load_episodes(job_dir: str | Path, **filters: Any) -> list[Episode]:
    return [load_episode(t.trial_dir) for t in iter_trials(job_dir, **filters)]


def turns_table(ep: Episode):
    """One row per agent turn: what the model saw/emitted and what it cost.

    Columns: turn, step_id, analysis (excerpt), commands, n_commands, task_complete,
    observation_chars, prompt_tokens, completion_tokens, cumulative_prompt_tokens,
    api_latency_s, timestamp.
    """
    import pandas as pd

    rows = []
    cum = 0
    for i, s in enumerate(ep.agent_steps):
        m = s.get("metrics") or {}
        pt, ct = m.get("prompt_tokens"), m.get("completion_tokens")
        cum += int(pt or 0)
        msg = str(s.get("message") or "")
        analysis = msg.split("\nPlan:", 1)[0].removeprefix("Analysis:").strip()
        cmds = step_commands(s)
        lat = ep.api_request_times_ms[i] / 1000.0 if i < len(ep.api_request_times_ms) else None
        rows.append(
            {
                "turn": i + 1,
                "step_id": s.get("step_id"),
                "analysis": analysis,
                "commands": cmds,
                "n_commands": len(cmds),
                "task_complete": step_declares_complete(s),
                "observation_chars": len(step_observation(s)),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cumulative_prompt_tokens": cum,
                "api_latency_s": lat,
                "timestamp": s.get("timestamp"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- job level
def job_level_result(job_dir: str | Path) -> dict:
    d = _read_json(Path(job_dir) / "result.json")
    return d or {}


def job_lock(job_dir: str | Path) -> dict:
    d = _read_json(Path(job_dir) / "lock.json")
    return d or {}


def job_summary(job_dir: str | Path, *, include_incomplete: bool = False, with_steps: bool = True):
    """One row per trial with outcome, tokens, phase timings and (optionally) trajectory counts."""
    import pandas as pd

    from tarl.data.domains import domain_of

    job_dir = Path(job_dir)
    rows = []
    for t in iter_trials(job_dir, include_incomplete=include_incomplete):
        r = t.result
        ar = r.get("agent_result") or {}
        md = ar.get("metadata") or {}
        ph = _phase_seconds(r)
        row: dict[str, Any] = {
            "job": job_dir.name,
            "task": t.task,
            "domain": domain_of(t.task),
            "trial": t.trial_name,
            "trial_id": t.trial_id,
            "status": t.status,
            "reward": t.reward,
            "passed": t.passed,
            "exception_type": t.exception_type,
            "has_trajectory": t.has_trajectory,
            "prompt_tokens": ar.get("n_input_tokens"),
            "completion_tokens": ar.get("n_output_tokens"),
            "cached_tokens": ar.get("n_cache_tokens"),
            "n_episodes": md.get("n_episodes"),
            "summarization_count": md.get("summarization_count"),
            "env_setup_s": ph["environment_setup"],
            "agent_setup_s": ph["agent_setup"],
            "agent_s": ph["agent_execution"],
            "verifier_s": ph["verifier"],
            "wall_s": _seconds(r.get("started_at"), r.get("finished_at")),
            "started_at": _parse_ts(r.get("started_at")),
            "finished_at": _parse_ts(r.get("finished_at")),
        }
        n_steps = n_agent = n_cmd = None
        if with_steps and t.has_trajectory:
            traj = _read_json(t.trajectory_path)
            if traj:
                steps = traj.get("steps") or []
                n_steps = len(steps)
                n_agent = sum(1 for s in steps if s.get("source") == "agent")
                n_cmd = sum(len(step_commands(s)) for s in steps if s.get("source") == "agent")
        row.update(n_steps=n_steps, n_agent_steps=n_agent, n_commands=n_cmd)
        ctrf = load_ctrf(t.ctrf_path) if t.has_ctrf else None
        row.update(
            ctrf_tests=ctrf.tests if ctrf else None,
            ctrf_passed=ctrf.passed if ctrf else None,
            ctrf_failed=ctrf.failed if ctrf else None,
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("started_at", "finished_at"):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def lane_pack(df, *, start="started_at", end="finished_at"):
    """Greedy lane assignment for a timeline chart: returns a Series of lane indices."""
    import pandas as pd

    d = df[[start, end]].copy()
    d["_i"] = range(len(d))
    d = d.dropna().sort_values(start)
    lane_end: list[Any] = []
    lanes = pd.Series([None] * len(df), index=df.index, dtype="object")
    for idx, row in d.iterrows():
        placed = False
        for k, e in enumerate(lane_end):
            if row[start] >= e:
                lane_end[k] = row[end]
                lanes.loc[idx] = k
                placed = True
                break
        if not placed:
            lane_end.append(row[end])
            lanes.loc[idx] = len(lane_end) - 1
    return lanes

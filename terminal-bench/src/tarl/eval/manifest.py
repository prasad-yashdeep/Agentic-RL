"""Run manifests: harness/model/parser versions, sampling params, commit — recorded with every run
(docs/02-architecture.md §7: 'record harness+model+parser versions with every run')."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tarl.common.env import hostname
from tarl.common.paths import REPO


def git_commit(repo: Path = REPO) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except Exception:
        return "unknown"


def git_dirty(repo: Path = REPO) -> bool | None:
    """Uncommitted changes under `repo` only. The git root is the Agentic-RL monorepo, so without the
    pathspec an unrelated swe-bench edit would mark every terminal-bench run dirty."""
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--", str(repo)], text=True, timeout=10)
        return bool(out.strip())
    except Exception:
        return None


def _versions() -> dict:
    out: dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "trl", "peft", "vllm", "pandas", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "n/a"
    return out


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    git_commit: str
    git_dirty: bool | None
    hostname: str
    harness: dict = field(default_factory=dict)  # harbor version, agent name/version
    model: dict = field(default_factory=dict)  # served model, base model, parser, temperature, thinking
    judge: dict = field(default_factory=dict)  # model, rubric hash
    tasks: dict = field(default_factory=dict)  # source, n_tasks, digests
    versions: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def for_run(cls, cfg, *, rubric_hash: str | None = None, judge_model: str | None = None) -> "RunManifest":
        from tarl.training.run_config import to_dict

        return cls(
            run_id=cfg.run_name or "",
            created_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            hostname=hostname(),
            harness={"agent": "terminus-2", "environment": "singularity", "collect_rollout_details": cfg.collect_rollout_details},
            model={"base_model": cfg.base_model, "temperature": cfg.temperature, "render": cfg.render},
            judge={"model": judge_model, "rubric_hash": rubric_hash, "lambda": cfg.lambda_judge, "mode": cfg.judge_mode},
            tasks={"source": str(cfg.tb_path), "names": list(cfg.tasks)},
            versions=_versions(),
            config=to_dict(cfg),
        )


def manifest_from_job(job_dir: str | Path, *, sample_trials: int = 3) -> RunManifest:
    """Reconstruct a manifest for a recorded Harbor job (lock.json + a few trial files)."""
    from tarl.data.loader import iter_trials, job_lock
    from tarl.judge.compressor import load_trajectory

    job_dir = Path(job_dir)
    lock = job_lock(job_dir)
    trials = list(iter_trials(job_dir))
    agent = model = {}
    parser = temperature = None
    for t in trials[:sample_trials]:
        r = t.result
        agent = r.get("agent_info") or agent
        cfg_agent = (r.get("config") or {}).get("agent") or {}
        model = {"model_name": cfg_agent.get("model_name"), **(cfg_agent.get("kwargs") or {})} if cfg_agent else model
        if t.has_trajectory and parser is None:
            try:
                extra = (load_trajectory(t.trajectory_path).get("agent") or {}).get("extra") or {}
                parser, temperature = extra.get("parser"), extra.get("temperature")
            except Exception:
                pass
    digests = {}
    for tr in lock.get("trials") or []:
        task = tr.get("task") or {}
        if task.get("name"):
            digests[task["name"]] = task.get("digest")
    thinking = None
    st = REPO / ".serve-thinking"
    if st.exists():
        thinking = st.read_text().strip() == "1"
    return RunManifest(
        run_id=job_dir.name,
        created_at=str(lock.get("created_at") or ""),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        hostname=hostname(),
        harness={"harbor_version": (lock.get("harbor") or {}).get("version"), "agent": agent.get("name"), "agent_version": agent.get("version"), "n_concurrent": lock.get("n_concurrent_trials"), "retry": lock.get("retry")},
        model={**model, "parser": parser, "temperature": temperature, "thinking_flag_now": thinking},
        tasks={"n_tasks": len(digests), "n_planned_trials": len(lock.get("trials") or []), "digests": digests},
        versions=_versions(),
    )


def write_manifest(m: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(m), indent=2, default=str))
    return path


def read_manifest(path: str | Path) -> RunManifest:
    d = json.loads(Path(path).read_text())
    return RunManifest(**d)

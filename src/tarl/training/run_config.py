"""RunConfig: every knob of a GRPO run, with argparse (scripts/08_grpo_run.py) and JSON export."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tarl.common.paths import BASE_MODEL, JOBS, RUNS, TB2_LOCAL


@dataclass
class RunConfig:
    tasks: list[str] = field(default_factory=lambda: ["fix-git", "regex-log"])
    G: int = 4
    steps: int = 5
    lr: float = 1e-6
    lambda_judge: float = 0.5  # 0 => arm B1 (verifier-only); (0, 0.5] => arm B2
    seed: int = 0
    judge_mode: str = "replay"  # live | replay | mock
    judge_on_miss: str = "error"
    jobs_dir: Path = JOBS / "rl"
    runs_dir: Path = RUNS
    run_name: str | None = None
    clip_lo: float = 0.2
    clip_hi: float = 0.28
    max_seq: int | None = 1536
    max_grad_norm: float = 0.1
    advantage_mode: str = "dr_grpo"
    render: str = "terminus_chat"  # legacy06 | terminus_chat
    include_reasoning: bool = False
    use_serve_template: bool = False
    judge_include_actions: bool = True
    judge_task_only: bool = True
    sync_every: int = 1
    alarm_window: int = 200
    base_model: str = BASE_MODEL
    adapter_dir: Path = Path("/var/tmp/tarl/adapter")
    merged_dir: Path = Path("/var/tmp/tarl/merged-policy")
    tb_path: Path = TB2_LOCAL
    temperature: float = 1.2
    n_concurrent: int = 4
    timeout_multiplier: float = 1.0
    collect_rollout_details: bool = False
    cost_cap_usd: float | None = None
    allow_spend: bool = False
    dry_run: bool = False
    replay_jobs: Path | None = None
    g4_log: Path | None = None
    no_tokenize: bool = False
    no_serve: bool = False
    no_train: bool = False

    @property
    def arm(self) -> str:
        return "B1" if self.lambda_judge == 0 else "B2"

    def compress_kwargs(self) -> dict:
        """Replay caches only hold legacy (narrative-only) prompts, so replay mode forces them."""
        if self.judge_mode == "replay":
            return {"include_actions": False, "task_only": False}
        return {"include_actions": self.judge_include_actions, "task_only": self.judge_task_only}

    def validate(self) -> None:
        if not (0.0 <= self.lambda_judge <= 0.5):
            raise ValueError("lambda_judge must be in [0, 0.5] (gate: a judged failure never outranks a pass)")
        if self.judge_mode not in ("live", "replay", "mock"):
            raise ValueError(f"judge_mode {self.judge_mode!r}")
        if self.judge_mode == "live" and not self.allow_spend:
            raise ValueError("judge_mode=live costs money: pass --yes-spend")
        if self.dry_run and self.replay_jobs is None:
            raise ValueError("--dry-run needs --replay-jobs <dir with recorded Harbor jobs>")
        if self.render not in ("legacy06", "terminus_chat"):
            raise ValueError(f"render {self.render!r}")
        if self.G < 2:
            raise ValueError("G must be >= 2 (group-relative advantages need siblings)")


def to_dict(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    d["arm"] = cfg.arm
    return d


def build_parser() -> argparse.ArgumentParser:
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="08_grpo_run.py",
        description="GRPO run (B1 verifier-only / B2 judge-shaped) with per-step JSONL metrics; "
        "--dry-run replays recorded jobs on CPU.",
    )
    p.add_argument("--tasks", nargs="+", default=d.tasks)
    p.add_argument("--G", type=int, default=d.G)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lambda-judge", type=float, default=d.lambda_judge, dest="lambda_judge")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--judge", choices=["live", "replay", "mock"], default=d.judge_mode, dest="judge_mode")
    p.add_argument("--judge-on-miss", choices=["error", "none", "mock"], default=d.judge_on_miss, dest="judge_on_miss")
    p.add_argument("--jobs-dir", type=Path, default=d.jobs_dir, dest="jobs_dir")
    p.add_argument("--runs-dir", type=Path, default=d.runs_dir, dest="runs_dir")
    p.add_argument("--run-name", default=None, dest="run_name")
    p.add_argument("--clip-lo", type=float, default=d.clip_lo, dest="clip_lo")
    p.add_argument("--clip-hi", type=float, default=d.clip_hi, dest="clip_hi")
    p.add_argument("--max-seq", type=int, default=d.max_seq, dest="max_seq")
    p.add_argument("--no-max-seq", action="store_true", help="disable the overlong filter")
    p.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm, dest="max_grad_norm")
    p.add_argument("--advantage", choices=["dr_grpo", "grpo_std"], default=d.advantage_mode, dest="advantage_mode")
    p.add_argument("--render", choices=["legacy06", "terminus_chat"], default=d.render)
    p.add_argument("--include-reasoning", action="store_true", dest="include_reasoning")
    p.add_argument("--use-serve-template", action="store_true", dest="use_serve_template")
    p.add_argument("--judge-legacy-prompt", action="store_true", help="narrative-only judge prompt (06 behaviour)")
    p.add_argument("--sync-every", type=int, default=d.sync_every, dest="sync_every")
    p.add_argument("--alarm-window", type=int, default=d.alarm_window, dest="alarm_window")
    p.add_argument("--base-model", default=d.base_model, dest="base_model")
    p.add_argument("--adapter-dir", type=Path, default=d.adapter_dir, dest="adapter_dir")
    p.add_argument("--merged-dir", type=Path, default=d.merged_dir, dest="merged_dir")
    p.add_argument("--tb-path", type=Path, default=d.tb_path, dest="tb_path")
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--n-concurrent", type=int, default=d.n_concurrent, dest="n_concurrent")
    p.add_argument("--timeout-multiplier", type=float, default=d.timeout_multiplier, dest="timeout_multiplier")
    p.add_argument("--collect-rollout-details", action="store_true", dest="collect_rollout_details")
    p.add_argument("--cost-cap-usd", type=float, default=None, dest="cost_cap_usd")
    p.add_argument("--yes-spend", action="store_true", dest="allow_spend")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--replay-jobs", type=Path, default=None, dest="replay_jobs")
    p.add_argument("--g4-log", type=Path, default=None, dest="g4_log")
    p.add_argument("--no-tokenize", action="store_true", dest="no_tokenize")
    p.add_argument("--no-serve", action="store_true", dest="no_serve")
    p.add_argument("--no-train", action="store_true", dest="no_train")
    return p


def parse_args(argv=None) -> RunConfig:
    ns = build_parser().parse_args(argv)
    d = vars(ns)
    legacy_prompt = d.pop("judge_legacy_prompt")
    no_max_seq = d.pop("no_max_seq")
    cfg = RunConfig(**d)
    if legacy_prompt:
        cfg.judge_include_actions = False
        cfg.judge_task_only = False
    if no_max_seq:
        cfg.max_seq = None
    cfg.validate()
    return cfg

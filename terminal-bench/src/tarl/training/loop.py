"""The GRPO cycle behind scripts/08_grpo_run.py: rollout -> reward -> advantages -> masked update
-> merge/sync, with every rollout, judge verdict and step written to JSONL.

`run(cfg)` with `cfg.dry_run` replays recorded jobs on CPU (no model), and when `cfg.g4_log` is
given asserts parity with the historical g4.log (R and advantages per task group within 5e-4).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from tarl.common.env import has_cuda, hostname, require_cuda
from tarl.data.examples import build_example, load_gemma_tokenizer, load_serve_chat_template, to_tensors
from tarl.judge.alarms import JudgeAlarms
from tarl.judge.composer import RewardConfig
from tarl.judge.config import make_judge
from tarl.judge.rubric import load_rubric
from tarl.judge.scoring import score_rollouts
from tarl.training.advantages import group_advantages, group_is_informative, informative
from tarl.training.curriculum import Curriculum
from tarl.training.metrics import MetricsWriter
from tarl.training.rollouts import HarborRolloutSource, ReplayRolloutSource, RolloutRecord, RolloutSource
from tarl.training.run_config import RunConfig, to_dict

PARITY_TOL = 5e-4 + 1e-9  # g4.log prints 3 decimals (max rounding error is exactly 5e-4)


@dataclass
class StepResult:
    step: int
    job: str | None
    n_rollouts: int
    n_examples: int
    pass_rate: float
    judge_mean: float | None
    useful_groups: int
    n_groups: int
    update: dict | None
    sync: dict | None
    alarms: list[str] = field(default_factory=list)
    parity_ok: bool | None = None
    parity_detail: dict = field(default_factory=dict)


def _parity(groups: dict[str, list[RolloutRecord]], g4_step) -> tuple[bool, dict]:
    ok = True
    detail: dict = {}
    for task, grp in groups.items():
        logged = g4_step.group(task)
        ours_R = sorted(float(r.R) for r in grp)
        ours_A = sorted(float(r.adv) for r in grp)
        log_R = sorted(r.R for r in logged)
        log_A = sorted(r.adv for r in logged)
        same_n = len(ours_R) == len(log_R)
        dR = max((abs(a - b) for a, b in zip(ours_R, log_R)), default=0.0) if same_n else float("inf")
        dA = max((abs(a - b) for a, b in zip(ours_A, log_A)), default=0.0) if same_n else float("inf")
        good = same_n and dR <= PARITY_TOL and dA <= PARITY_TOL
        ok = ok and good
        detail[task] = {"n_ours": len(ours_R), "n_logged": len(log_R), "max_dR": dR, "max_dA": dA, "ok": good}
    return ok, detail


def run_step(
    step: int,
    cfg: RunConfig,
    source: RolloutSource,
    judge,
    rubric: str,
    rubric_hash: str,
    tok,
    model,
    opt,
    writer: MetricsWriter,
    rollouts_writer: MetricsWriter,
    alarms: JudgeAlarms | None,
    *,
    curriculum: Curriculum | None = None,
    g4_step=None,
    chat_template: str | None = None,
    log: Callable[[str], None] = print,
    serve=None,
    tasks: Sequence[str] | None = None,
) -> StepResult:
    ros = source.collect(step, tasks=tasks)
    job = getattr(source, "last_job", None)
    log(f"step {step}: {len(ros)} clean rollouts from {job.name if job else '?'}")
    fired = score_rollouts(
        ros, judge, rubric, RewardConfig(lambda_judge=cfg.lambda_judge), alarms=alarms, log=log,
        compress_kwargs=cfg.compress_kwargs(), step=step,
    )
    groups: dict[str, list[RolloutRecord]] = {}
    for ro in ros:
        groups.setdefault(ro.task, []).append(ro)
    examples: list[tuple] = []
    useful = 0
    for task, grp in groups.items():
        advs = group_advantages([r.R for r in grp], mode=cfg.advantage_mode)
        if group_is_informative(advs):
            useful += 1
        for r, a in zip(grp, advs):
            r.adv = a
            log(f"  {task} R={r.R:.3f} (verifier={r.verifier}, judge={r.judge}) adv={a:+.3f}")
            included, reason, n_ids, n_asst = False, None, None, None
            if not informative(a):
                reason = "zero_advantage"  # dynamic-sampling spirit: drop zero-signal rollouts
            elif tok is None:
                reason = "no_tokenizer"
            else:
                ex = build_example(
                    tok, r.trajectory, render=cfg.render, max_seq=cfg.max_seq,
                    chat_template=chat_template, include_reasoning=cfg.include_reasoning,
                )
                n_ids, n_asst = ex.n_tokens, ex.n_assistant_tokens
                if ex.kept:
                    examples.append((ex, a))
                    included = True
                else:
                    reason = ex.dropped_reason
            rollouts_writer.write(
                {
                    "kind": "rollout", "step": step, "task": task, "trial": r.trial_dir.name,
                    "trial_dir": str(r.trial_dir), "trajectory": str(r.trajectory), "verifier": r.verifier,
                    "judge": r.judge, "judge_source": r.judge_source, "judge_rationale": r.judge_rationale,
                    "judge_tokens": r.judge_tokens, "R": r.R, "adv": a, "included": included,
                    "dropped_reason": reason, "n_ids": n_ids, "n_assistant_tokens": n_asst,
                    "rubric_hash": rubric_hash,
                }
            )
        if curriculum is not None:
            curriculum.update_group(task, [r.verifier for r in grp])
    for a in fired:
        writer.write({"kind": "alarm", "step": step, "text": a})
        log(f"  ALARM {a}")

    parity_ok, parity_detail = (None, {})
    if g4_step is not None:
        parity_ok, parity_detail = _parity(groups, g4_step)
        log(f"step {step}: parity vs g4.log {'OK' if parity_ok else 'FAIL'} {parity_detail}")

    update: dict | None
    sync: dict | None = None
    if examples and model is not None:
        from tarl.training.loss import grpo_step, zero_grad_leak_check

        batch = [(*to_tensors(ex), a) for ex, a in examples]
        update = grpo_step(model, opt, batch, clip_lo=cfg.clip_lo, clip_hi=cfg.clip_hi, max_grad_norm=cfg.max_grad_norm)
        zero_grad_leak_check(model)
        log(f"step {step}: UPDATE loss/tok={update['loss/tok']:.5f} grad_norm={update['grad_norm']:.4f} tokens={update['tokens']}")
        if cfg.sync_every and step % cfg.sync_every == 0 and serve is not None:
            sync = serve.sync(model, tok, step)
            log(f"step {step}: SYNC {sync}")
    elif examples:
        update = {
            "skipped": "dry_run_no_model" if cfg.dry_run else "no_model",
            "n_examples": len(examples),
            "tokens": int(sum(ex.n_assistant_tokens for ex, _ in examples)),
        }
        log(f"step {step}: update skipped ({update['skipped']}); {len(examples)} examples, {update['tokens']} trained tokens")
    else:
        n_inf = sum(1 for r in ros if r.adv is not None and informative(r.adv))
        why = "no_informative_rollouts" if n_inf == 0 else ("no_tokenizer" if tok is None else "all_examples_dropped")
        update = {"skipped": why, "n_examples": 0, "n_informative": n_inf}
        log(f"step {step}: update skipped ({why}; {n_inf} informative rollouts) — continuing")
    writer.write({"kind": "update", "step": step, **update})
    if sync:
        writer.write({"kind": "sync", "step": step, **sync})

    judged = [r.judge for r in ros if r.judge is not None]
    res = StepResult(
        step=step, job=job.name if job else None, n_rollouts=len(ros), n_examples=len(examples),
        pass_rate=(sum(1 for r in ros if r.verifier >= 1.0) / len(ros)) if ros else 0.0,
        judge_mean=(sum(judged) / len(judged)) if judged else None,
        useful_groups=useful, n_groups=len(groups), update=update, sync=sync, alarms=fired,
        parity_ok=parity_ok, parity_detail=parity_detail,
    )
    writer.write({"kind": "step_summary", **{k: v for k, v in asdict(res).items() if k not in ("update", "sync")}})
    return res


class _ServeSync:
    """merge -> export -> relaunch vLLM -> probe (06:332-338). node08 only."""

    def __init__(self, cfg: RunConfig, log):
        from tarl.serving.harbor_cmd import probe_completion, restart_serve

        self.cfg, self.log = cfg, log
        self._restart, self._probe = restart_serve, probe_completion
        self.prev_hash: str | None = None
        self.prev_probe = probe_completion()

    def sync(self, model, tok, step: int) -> dict:
        from tarl.training.model import merge_and_export

        h = merge_and_export(model, tok, base_model=self.cfg.base_model, adapter_dir=self.cfg.adapter_dir, merged_dir=self.cfg.merged_dir)
        assert h != self.prev_hash, "adapter unchanged after optimizer step"
        self.prev_hash = h
        self._restart(str(self.cfg.merged_dir))
        probe = self._probe()
        changed = probe != self.prev_probe
        self.prev_probe = probe
        return {"adapter_hash": h, "probe_changed": changed, "merged_dir": str(self.cfg.merged_dir)}


def sample_step_tasks(cfg: RunConfig, curriculum: Curriculum) -> list[str] | None:
    """The tasks to roll out this step: None = every configured task (06 behaviour); otherwise a
    curriculum sample of `tasks_per_step` distinct pool tasks."""
    if not cfg.tasks_per_step or cfg.tasks_per_step >= len(cfg.tasks):
        return None
    return curriculum.sample(cfg.tasks_per_step, cfg.sample_strategy)


def _sif_hook(cfg: RunConfig, log):
    """Endless Terminals only: make sure this step's images exist before `harbor run` validates task.toml."""
    if cfg.task_source != "endless":
        return None
    from tarl.data.endless import ensure_sifs

    def hook(tasks: list[str]) -> None:
        t0 = time.time()
        built = ensure_sifs(tasks, workers=cfg.sif_workers, log=log)
        log(f"sifs ready for {len(built)} task(s) in {time.time() - t0:.0f}s")

    return hook


def run(cfg: RunConfig, *, log_to_stdout: bool = True) -> Path:
    cfg.validate()
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    name = cfg.run_name or f"{ts}-{cfg.arm}{'-dry' if cfg.dry_run else ''}"
    run_dir = cfg.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "log.txt", "a")

    def log(s: str) -> None:
        line = f"[08] {s}"
        logf.write(line + "\n")
        logf.flush()
        if log_to_stdout:
            print(line, flush=True)

    (run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    rubric, rubric_hash = load_rubric()
    judge = make_judge(cfg.judge_mode, on_miss=cfg.judge_on_miss, cost_cap_usd=cfg.cost_cap_usd, allow_spend=cfg.allow_spend) if cfg.lambda_judge > 0 else None
    log(f"run {name}: arm {cfg.arm}, rubric {rubric_hash}, judge {getattr(judge, 'name', None)}, tasks {cfg.tasks}, G={cfg.G}, steps={cfg.steps}, host={hostname()}")
    alarms = JudgeAlarms(window=cfg.alarm_window, audit_path=run_dir / "judge-audit.jsonl")
    tok = None if cfg.no_tokenize else load_gemma_tokenizer()
    chat_template = load_serve_chat_template() if cfg.use_serve_template else None

    g4 = None
    if cfg.g4_log:
        from tarl.judge.g4log import parse_g4_log

        g4 = parse_g4_log(cfg.g4_log)
    if cfg.dry_run:
        step_to_job = {s.step: s.job for s in g4.steps} if g4 else None
        source: RolloutSource = ReplayRolloutSource(cfg.replay_jobs, step_to_job, tasks=cfg.tasks)
    else:
        source = HarborRolloutSource(cfg, before_collect=_sif_hook(cfg, log))

    model = opt = serve = None
    if not cfg.dry_run and not cfg.no_train:
        require_cuda("GRPO update")
        import torch

        from tarl.training.model import load_trainable

        tok, model = load_trainable(cfg.base_model, device_index=0)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
        if not cfg.no_serve:
            serve = _ServeSync(cfg, log)

    from tarl.eval.manifest import RunManifest, write_manifest

    write_manifest(RunManifest.for_run(cfg, rubric_hash=rubric_hash, judge_model=getattr(judge, "name", None)), run_dir / "manifest.json")
    writer = MetricsWriter(run_dir / "metrics.jsonl")
    rw = MetricsWriter(run_dir / "rollouts.jsonl")
    curriculum = Curriculum(cfg.tasks, seed=cfg.seed)
    results: list[StepResult] = []
    parity_all = True
    for step in range(1, cfg.steps + 1):
        g4_step = None
        if g4 is not None:
            try:
                g4_step = g4.step(step)
            except KeyError:
                g4_step = None
        step_tasks = sample_step_tasks(cfg, curriculum)
        if step_tasks is not None:
            writer.write({"kind": "sample", "step": step, "tasks": step_tasks, "fallback": curriculum.last_sample_fallback,
                          "trainable_fraction": curriculum.trainable_fraction()})
            log(f"step {step}: sampled {step_tasks} (strategy={cfg.sample_strategy}, band fallback={curriculum.last_sample_fallback})")
        res = run_step(
            step, cfg, source, judge, rubric, rubric_hash, tok, model, opt, writer, rw, alarms,
            curriculum=curriculum, g4_step=g4_step, chat_template=chat_template, log=log, serve=serve, tasks=step_tasks,
        )
        results.append(res)
        if res.parity_ok is False:
            parity_all = False
    summary = {
        "run": name,
        "arm": cfg.arm,
        "steps": [asdict(r) for r in results],
        "curriculum": curriculum.table().to_dict(orient="records"),
        "alarms": alarms.snapshot(),
        "judge": getattr(getattr(judge, "meter", None), "snapshot", lambda: None)(),
        "parity_checked": g4 is not None,
        "parity_ok": parity_all if g4 is not None else None,
        "has_cuda": has_cuda(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if g4 is not None:
        log("DRY_RUN_PARITY_OK" if parity_all else "DRY_RUN_PARITY_FAIL")
    log(f"done -> {run_dir}")
    logf.close()
    return run_dir

"""`tarl` — the CLI-only interaction surface of terminal-agent-rl.

    tarl replay <trial_dir|task>      step through a recorded episode (commands, screens, verdicts)
    tarl eval <job_dir>               pass@1 with CIs, per-domain, pass@k, error taxonomy
    tarl compare <job_a> <job_b>      paired per-task comparison (bootstrap + McNemar)
    tarl run-summary <run_dir>        summarise a scripts/08 run (metrics.jsonl)
    tarl g4log [g4.log]               the G4 micro-run as tables
    tarl manifest <job_dir>           harness/model/task provenance of a job
    tarl judge <trial|task>           judge one trajectory (replay | mock | live --yes-spend)
    tarl coldstart [--simulate]       the zero-gradient arithmetic (+ synthetic simulation)
    tarl embed download|leakage|clusters
    tarl run --task T [--live]        one live episode against the vLLM serve (node08 only)
    tarl data task-meta --tb-path P   regenerate the domain YAML from task.toml files
    tarl data et <download|table|split|convert|build-sif|leakage>   Endless Terminals training corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tarl.common.paths import A0_JOB, ET_HARBOR, ET_ROOT, ET_SIF, ET_SPLITS, G4_LOG, JOBS, JOBS_G4, PILOT_JSONL, TB2_DOMAINS, TB2_LOCAL


def _print_df(df, title: str | None = None, max_rows: int = 60) -> None:
    import pandas as pd

    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 30, "display.width", 160, "display.max_colwidth", 60):
        if title:
            print(f"\n== {title} ==")
        print(df.to_string(index=False) if len(df) <= max_rows else df.head(max_rows).to_string(index=False) + f"\n… ({len(df)} rows)")


# --------------------------------------------------------------------------- commands
def cmd_replay(a) -> int:
    from tarl.data.loader import load_episode
    from tarl.serving.harbor_cmd import resolve_trial
    from tarl.serving.replay import episode_json, render_episode

    trial_dir = resolve_trial(a.trial, job=a.job)
    ep = load_episode(trial_dir)
    if a.json:
        print(json.dumps(episode_json(ep), indent=2, default=str))
        return 0
    judge = None
    if a.judge == "replay":
        from tarl.judge.replay import ReplayJudge

        judge = ReplayJudge.from_artifacts(on_miss="error", strict=False)
    elif a.judge == "mock":
        from tarl.judge.mock import MockJudge

        judge = MockJudge()
    render_episode(ep, speed=a.speed, show_reasoning=a.reasoning, max_output_lines=a.max_lines, judge=judge, use_cast=a.cast, no_color=a.no_color)
    return 0


def cmd_eval(a) -> int:
    from tarl.data.loader import job_summary
    from tarl.eval.report import by_domain, error_taxonomy, pass_rate_by_task, summary_table
    from tarl.eval.stats import pass_at_k_table

    df = job_summary(a.job_dir)
    summ = summary_table(df)
    out = {"job": str(a.job_dir), "summary": summ}
    if a.by_domain or a.all:
        out["by_domain"] = by_domain(df).to_dict(orient="records")
    if a.pass_at_k or a.all:
        out["pass_at_k"] = pass_at_k_table(df[df["status"] == "clean"]).to_dict(orient="records")
    if a.taxonomy or a.all:
        out["taxonomy"] = error_taxonomy(df).to_dict(orient="records")
    if a.csv:
        df.to_csv(a.csv, index=False)
    if a.json:
        text = json.dumps(out, indent=2, default=str)
        if a.json == "-" or a.json == "/dev/stdout":
            print(text)
        else:
            Path(a.json).write_text(text)
            print(f"wrote {a.json}")
        return 0
    print(f"job {a.job_dir}")
    for k, v in summ.items():
        print(f"  {k:28} {v}")
    if a.by_domain or a.all:
        _print_df(by_domain(df), "by domain")
    if a.pass_at_k or a.all:
        _print_df(pass_at_k_table(df[df["status"] == "clean"]), "pass@k (clean trials)")
    if a.taxonomy or a.all:
        _print_df(error_taxonomy(df), "outcome taxonomy")
    if a.tasks:
        _print_df(pass_rate_by_task(df), "per task", max_rows=200)
    return 0


def cmd_compare(a) -> int:
    from tarl.data.loader import job_summary
    from tarl.eval.stats import paired_compare

    res = paired_compare(job_summary(a.job_a), job_summary(a.job_b), reduce=a.reduce, n_boot=a.n_boot)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_run_summary(a) -> int:
    from tarl.training.metrics import read_metrics

    df = read_metrics(Path(a.run_dir) / "metrics.jsonl")
    for kind in ("step_summary", "update", "sync", "alarm"):
        sub = df[df["kind"] == kind].dropna(axis=1, how="all")
        if len(sub):
            _print_df(sub.drop(columns=[c for c in ("ts", "parity_detail") if c in sub]), kind)
    rp = Path(a.run_dir) / "rollouts.jsonl"
    if rp.exists():
        r = read_metrics(rp)
        _print_df(r[["step", "task", "trial", "verifier", "judge", "R", "adv", "included", "dropped_reason"]], "rollouts", max_rows=80)
    return 0


def cmd_g4log(a) -> int:
    from tarl.judge.g4log import parse_g4_log, steps_frame, to_frame

    run = parse_g4_log(a.path)
    print(f"rubric {run.rubric_hash}  judge {run.judge_model}  tasks {run.tasks}  G={run.G}  steps={run.n_steps}  done={run.done}")
    _print_df(steps_frame(run), "steps")
    df = to_frame(run)
    _print_df(df[["step", "task", "order", "R", "verifier", "judge", "adv"]], "rollouts")
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")
    return 0


def cmd_manifest(a) -> int:
    from dataclasses import asdict

    from tarl.eval.manifest import manifest_from_job, write_manifest

    m = manifest_from_job(a.job_dir)
    if a.out:
        write_manifest(m, a.out)
        print(f"wrote {a.out}")
    else:
        d = asdict(m)
        d["tasks"] = {k: v for k, v in d["tasks"].items() if k != "digests"} | {"n_digests": len(d["tasks"].get("digests") or {})}
        print(json.dumps(d, indent=2, default=str))
    return 0


def cmd_judge(a) -> int:
    from tarl.judge.compressor import compress, load_trajectory
    from tarl.judge.config import make_judge
    from tarl.judge.rubric import load_rubric
    from tarl.serving.harbor_cmd import resolve_trial

    trial_dir = resolve_trial(a.trial, job=a.job)
    traj = load_trajectory(trial_dir / "agent" / "trajectory.json")
    rubric, rh = load_rubric()
    kw = {} if a.mode == "replay" else {"include_actions": a.actions, "task_only": a.actions}
    prompt = compress(traj, **kw)
    print(f"trial {trial_dir.name}  rubric {rh}  prompt chars={len(prompt)} (~{len(prompt)//4} tok)  actions={'no (legacy)' if a.mode == 'replay' else a.actions}")
    if a.show_prompt:
        print(prompt)
    judge = make_judge(a.mode, on_miss=a.on_miss, allow_spend=a.yes_spend, cost_cap_usd=a.cost_cap_usd)
    v = judge.judge(rubric, prompt)
    print(json.dumps({"score": v.score, "rationale": v.rationale, "judge": judge.name, "prompt_tokens": v.prompt_tokens, "raw": {k: v.raw.get(k) for k in ("source", "confidence", "step", "order")}}, indent=2))
    return 0


def cmd_coldstart(a) -> int:
    from tarl.training.coldstart import simulate_both, useful_fraction_table

    _print_df(useful_fraction_table(tuple(a.p), tuple(a.G)), "useful-group fraction 1 - p^G - (1-p)^G")
    if a.simulate:
        df = simulate_both(seeds=tuple(range(a.seeds)), p0=a.p0, G=a.G[0] if len(a.G) == 1 else 8, steps=a.steps, judge_corr=a.judge_corr)
        last = df[df["step"] == df["step"].max()].groupby("condition")[["mean_p", "useful_group_frac"]].mean()
        print("\nSYNTHETIC simulation (illustrative only):")
        print(last.to_string())
    return 0


def cmd_embed(a) -> int:
    if a.embed_cmd == "download":
        from tarl.common.paths import HF_CACHE
        from tarl.embed.model import DEFAULT_MODEL

        import os

        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        from huggingface_hub import snapshot_download

        print(snapshot_download(DEFAULT_MODEL, allow_patterns=["*.json", "*.txt", "model.safetensors"]))
        return 0
    from tarl.common.env import is_headnode
    from tarl.embed.analysis import leakage_report, leave_one_out_nn, proxy_train_set, task_embeddings
    from tarl.embed.model import get_embedder

    if is_headnode() and not a.allow_headnode:
        print("refusing to run embeddings on the head node (standing rule); use `ssh node03 … tarl embed …` or --allow-headnode", file=sys.stderr)
        return 2
    emb = get_embedder()
    print(f"embedder: {emb.name}")
    if a.embed_cmd == "leakage":
        names, texts, X = task_embeddings([Path(j) for j in a.eval_jobs], emb)
        print(f"eval tasks embedded: {len(names)}")
        _print_df(leave_one_out_nn(names, X).head(a.top), "within-eval nearest neighbours (calibration)")
        if a.train_dir:
            from tarl.embed.analysis import load_harbor_dataset_instructions

            tn, tt = load_harbor_dataset_instructions(a.train_dir)
        else:
            tn, tt = proxy_train_set()
            print(f"train side = PROXY ({len(tn)} TB2 tasks used in G4/pilot): exact hits are expected")
        rep = leakage_report(tt, texts, top_k=a.top_k, threshold=a.threshold, train_labels=tn, eval_labels=names, embedder=emb)
        _print_df(rep[rep["rank"] == 1].sort_values("cosine", ascending=False), "train -> nearest eval task", max_rows=100)
        print(f"flagged (cos >= {a.threshold}): {int(rep[rep['rank'] == 1]['flagged'].sum())} of {len(tn)}")
        return 0
    if a.embed_cmd == "clusters":
        from tarl.data.loader import load_episode
        from tarl.embed.analysis import cluster_failures, trajectory_embeddings, trajectory_texts
        from tarl.judge.replay import ReplayJudge

        rj = ReplayJudge.from_artifacts(strict=False)
        tab = rj.table()
        eps = [load_episode(Path(d)) for d in tab["trial_dir"].dropna().unique()]
        scores = [rj.score_for_path(ep.trial.trajectory_path) for ep in eps]
        texts = trajectory_texts(eps)
        X = trajectory_embeddings(eps, emb)
        res = cluster_failures(X, scores, k=a.k, texts=texts)
        _print_df(res.table([f"{ep.task}/{ep.trial.trial_id}" for ep in eps]), f"failure-mode clusters (k={a.k})")
        return 0
    return 1


def cmd_run(a) -> int:
    from tarl.common.env import is_headnode
    from tarl.serving.harbor_cmd import build_harbor_cmd, format_cmd, serve_healthy

    cmd = build_harbor_cmd(tb_path=a.tb_path, tasks=[a.task], k=a.k, jobs_dir=a.jobs_dir, temperature=a.temperature, n_concurrent=1, collect_rollout_details=a.collect_rollout_details)
    shown = f"cd {Path(__file__).resolve().parents[2]} && . scripts/env.sh && OPENAI_API_KEY=dummy " + format_cmd(cmd)
    if not a.live:
        print("would run (add --live on node08 with the vLLM serve up):")
        print(shown)
        return 0
    if is_headnode():
        print("refusing: live episodes never run on the head node (use `ssh node08 … tarl run --live`)", file=sys.stderr)
        return 2
    import shutil
    import subprocess

    if shutil.which("harbor") is None:
        print("harbor not on PATH (uv tool install harbor; see scripts/env.sh)", file=sys.stderr)
        return 2
    if not a.no_health_check and not serve_healthy():
        print("vLLM serve at localhost:8000 is not healthy (start tmux `serve` via scripts/01_serve.sh)", file=sys.stderr)
        return 2
    print(shown)
    r = subprocess.run(["bash", "-lc", shown], text=True)
    return r.returncode


def cmd_data(a) -> int:
    if a.data_cmd == "task-meta":
        import tomllib

        import yaml

        root = Path(a.tb_path)
        if not root.exists():
            print(f"{root} not found (the TB2 clone lives on node08:/tmp/tb2)", file=sys.stderr)
            return 2
        groups: dict[str, list[str]] = {}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            f = d / "task.toml"
            if not f.exists():
                continue
            meta = tomllib.loads(f.read_text())
            cat = str(meta.get("category") or (meta.get("metadata") or {}).get("category") or "unknown")
            groups.setdefault(cat, []).append(d.name)
        out = {"version": f"task.toml categories from {root}", "domains": groups}
        Path(a.out).write_text(yaml.safe_dump(out, sort_keys=False))
        print(f"wrote {a.out}: {len(groups)} categories, {sum(len(v) for v in groups.values())} tasks")
        return 0
    if a.data_cmd == "et":
        return cmd_data_et(a)
    return 1


def _et_names(a) -> list[str]:
    from tarl.data import endless as et

    root = Path(a.root)
    if getattr(a, "tasks", None):
        names = list(a.tasks)
    elif getattr(a, "split", None):
        names = et.read_splits(Path(a.splits))[a.split]
    else:
        names = et.list_tasks(root)
    lim = getattr(a, "limit", None)
    return names[:lim] if lim else names


def cmd_data_et(a) -> int:
    from tarl.data import endless as et

    root = Path(a.root)
    if a.et_cmd == "download":
        p = et.download(root)
        print(f"snapshot at {p}: {len(et.list_tasks(p))} tasks")
        return 0
    if a.et_cmd == "table":
        df = et.task_table(root, _et_names(a))
        if a.csv:
            df.to_csv(a.csv, index=False)
            print(f"wrote {a.csv} ({len(df)} rows)")
        _print_df(df.head(a.show), f"Endless Terminals ({len(df)} tasks)")
        if len(df):
            print("\ndifficulty:", df["difficulty"].value_counts().to_dict())
            print("category:", df["category"].value_counts().head(12).to_dict())
            print("canary hits:", int(df["canary_hits"].sum()), "| localimage defs:", int(df["localimage"].sum()), "/", len(df))
        return 0
    if a.et_cmd == "split":
        names = et.list_tasks(root)
        s = et.split(names, dev_frac=a.dev_frac, seed=a.seed)
        p = et.write_splits(s, Path(a.out), root=str(root), dev_frac=a.dev_frac, seed=a.seed)
        print(f"wrote {p}: train {len(s['train'])}, dev {len(s['dev'])} (seed {a.seed})")
        return 0
    if a.et_cmd == "convert":
        names = _et_names(a)
        paths = et.convert(names, root, out_root=Path(a.out_root), sif_dir=Path(a.sif_dir))
        print(f"converted {len(paths)} task(s) -> {a.out_root} (docker_image -> {a.sif_dir}/<task>.sif)")
        return 0
    if a.et_cmd == "build-sif":
        names = _et_names(a)
        built = et.ensure_sifs(names, root, Path(a.sif_dir), workers=a.workers, log=print, force=a.force)
        print(f"{len(built)} image(s) ready;", et.sif_usage(Path(a.sif_dir)))
        return 0
    if a.et_cmd == "leakage":
        names = _et_names(a)
        rep = et.leakage_guard(names, root, tb_root=Path(a.tb_root), threshold=a.threshold, top_k=a.top_k)
        top = rep[rep["rank"] == 1].sort_values("cosine", ascending=False)
        if a.csv:
            rep.to_csv(a.csv, index=False)
            print(f"wrote {a.csv} ({len(rep)} rows)")
        _print_df(top.head(a.show), f"nearest TB2 instruction per training task (top {a.show} of {len(top)})")
        counts = top["action"].value_counts().to_dict()
        print(f"\nactions over {len(top)} training tasks: {counts}  (threshold {a.threshold}, drop = cosine >= threshold)")
        if counts.get("drop", 0):
            print("LEAKAGE: drop these before training:", ", ".join(top[top['action'] == 'drop']['train_label'].tolist()))
            return 3
        return 0
    return 1


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tarl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("replay", help="step through a recorded episode")
    r.add_argument("trial", help="trial dir, or task name (newest trial under jobs/ and jobs-g4/)")
    r.add_argument("--job", default=None, help="restrict task-name lookup to this job dir")
    r.add_argument("--speed", type=float, default=0.0, help="seconds to pause between turns")
    r.add_argument("--reasoning", action="store_true", help="show the model's thinking text")
    r.add_argument("--cast", action="store_true", help="take keystrokes from recording.cast instead of tool_calls")
    r.add_argument("--judge", choices=["replay", "mock", "none"], default="replay")
    r.add_argument("--max-lines", type=int, default=20, dest="max_lines")
    r.add_argument("--no-color", action="store_true", dest="no_color")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_replay)

    e = sp.add_parser("eval", help="evaluate a Harbor job directory")
    e.add_argument("job_dir")
    e.add_argument("--by-domain", action="store_true", dest="by_domain")
    e.add_argument("--pass-at-k", action="store_true", dest="pass_at_k")
    e.add_argument("--taxonomy", action="store_true")
    e.add_argument("--tasks", action="store_true", help="per-task table")
    e.add_argument("--all", action="store_true")
    e.add_argument("--csv", default=None)
    e.add_argument("--json", default=None, help="write JSON (use - for stdout)")
    e.set_defaults(fn=cmd_eval)

    c = sp.add_parser("compare", help="paired per-task comparison of two jobs")
    c.add_argument("job_a")
    c.add_argument("job_b")
    c.add_argument("--reduce", choices=["mean", "first", "majority", "any"], default="mean")
    c.add_argument("--n-boot", type=int, default=10_000, dest="n_boot")
    c.set_defaults(fn=cmd_compare)

    rs = sp.add_parser("run-summary", help="summarise a scripts/08 run directory")
    rs.add_argument("run_dir")
    rs.set_defaults(fn=cmd_run_summary)

    g = sp.add_parser("g4log", help="parse the G4 micro-run log")
    g.add_argument("path", nargs="?", default=str(G4_LOG))
    g.add_argument("--csv", default=None)
    g.set_defaults(fn=cmd_g4log)

    m = sp.add_parser("manifest", help="provenance manifest of a job")
    m.add_argument("job_dir")
    m.add_argument("-o", "--out", default=None)
    m.set_defaults(fn=cmd_manifest)

    j = sp.add_parser("judge", help="judge one trajectory")
    j.add_argument("trial")
    j.add_argument("--job", default=None)
    j.add_argument("--mode", choices=["replay", "mock", "live"], default="replay")
    j.add_argument("--on-miss", choices=["error", "none", "mock"], default="error", dest="on_miss")
    j.add_argument("--actions", action="store_true", help="action-inclusive, task-only prompt (v2)")
    j.add_argument("--show-prompt", action="store_true", dest="show_prompt")
    j.add_argument("--yes-spend", action="store_true", dest="yes_spend")
    j.add_argument("--cost-cap-usd", type=float, default=None, dest="cost_cap_usd")
    j.set_defaults(fn=cmd_judge)

    cs = sp.add_parser("coldstart", help="zero-gradient arithmetic and synthetic simulation")
    cs.add_argument("--p", type=float, nargs="+", default=[0.001, 0.01, 0.02, 0.033, 0.05, 0.08, 0.1, 0.2, 0.5])
    cs.add_argument("--G", type=int, nargs="+", default=[4, 8, 16])
    cs.add_argument("--simulate", action="store_true")
    cs.add_argument("--p0", type=float, default=0.02)
    cs.add_argument("--steps", type=int, default=200)
    cs.add_argument("--seeds", type=int, default=3)
    cs.add_argument("--judge-corr", type=float, default=0.6, dest="judge_corr")
    cs.set_defaults(fn=cmd_coldstart)

    em = sp.add_parser("embed", help="embedding analyses")
    ems = em.add_subparsers(dest="embed_cmd", required=True)
    ems.add_parser("download", help="fetch MiniLM into $HF_HOME")
    lk = ems.add_parser("leakage", help="train/eval nearest-neighbour leakage guard")
    lk.add_argument("--eval-jobs", nargs="+", default=[str(A0_JOB)], dest="eval_jobs")
    lk.add_argument("--train-dir", default=None, dest="train_dir", help="Harbor dataset root (<task>/instruction.md)")
    lk.add_argument("--threshold", type=float, default=0.9)
    lk.add_argument("--top-k", type=int, default=3, dest="top_k")
    lk.add_argument("--top", type=int, default=15)
    lk.add_argument("--allow-headnode", action="store_true", dest="allow_headnode")
    cl = ems.add_parser("clusters", help="failure-mode clusters of judged trajectories")
    cl.add_argument("--k", type=int, default=4)
    cl.add_argument("--allow-headnode", action="store_true", dest="allow_headnode")
    em.set_defaults(fn=cmd_embed)

    ru = sp.add_parser("run", help="one live episode via harbor (node08 only)")
    ru.add_argument("--task", required=True)
    ru.add_argument("--k", type=int, default=1)
    ru.add_argument("--temperature", type=float, default=1.2)
    ru.add_argument("--jobs-dir", default=str(JOBS / "dev"), dest="jobs_dir")
    ru.add_argument("--tb-path", default=str(TB2_LOCAL), dest="tb_path")
    ru.add_argument("--live", action="store_true")
    ru.add_argument("--collect-rollout-details", action="store_true", dest="collect_rollout_details")
    ru.add_argument("--no-health-check", action="store_true", dest="no_health_check")
    ru.set_defaults(fn=cmd_run)

    dt = sp.add_parser("data", help="data utilities")
    dts = dt.add_subparsers(dest="data_cmd", required=True)
    tm = dts.add_parser("task-meta", help="regenerate the domain YAML from task.toml files")
    tm.add_argument("--tb-path", default=str(TB2_LOCAL), dest="tb_path")
    tm.add_argument("--out", default=str(TB2_DOMAINS))
    et = dts.add_parser("et", help="Endless Terminals training corpus (download/convert/build-sif/leakage)")
    ets = et.add_subparsers(dest="et_cmd", required=True)

    def _sel(sp_):
        sp_.add_argument("--root", default=str(ET_ROOT))
        sp_.add_argument("--tasks", nargs="*", default=None)
        sp_.add_argument("--split", choices=["train", "dev"], default=None)
        sp_.add_argument("--splits", default=str(ET_SPLITS))
        sp_.add_argument("--limit", type=int, default=None)

    e0 = ets.add_parser("download"); e0.add_argument("--root", default=str(ET_ROOT))
    e1 = ets.add_parser("table"); _sel(e1); e1.add_argument("--csv", default=None); e1.add_argument("--show", type=int, default=20)
    e2 = ets.add_parser("split"); e2.add_argument("--root", default=str(ET_ROOT)); e2.add_argument("--dev-frac", type=float, default=0.1, dest="dev_frac")
    e2.add_argument("--seed", type=int, default=0); e2.add_argument("--out", default=str(ET_SPLITS))
    e3 = ets.add_parser("convert"); _sel(e3); e3.add_argument("--out-root", default=str(ET_HARBOR), dest="out_root"); e3.add_argument("--sif-dir", default=str(ET_SIF), dest="sif_dir")
    e4 = ets.add_parser("build-sif"); _sel(e4); e4.add_argument("--sif-dir", default=str(ET_SIF), dest="sif_dir"); e4.add_argument("--workers", type=int, default=2); e4.add_argument("--force", action="store_true")
    e5 = ets.add_parser("leakage"); _sel(e5); e5.add_argument("--tb-root", default=str(TB2_LOCAL), dest="tb_root"); e5.add_argument("--threshold", type=float, default=0.85); e5.add_argument("--top-k", type=int, default=3, dest="top_k")
    e5.add_argument("--csv", default=None); e5.add_argument("--show", type=int, default=15)
    dt.set_defaults(fn=cmd_data)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

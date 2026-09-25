"""`sarl` — CLI for the swe-bench subproject.

    sarl task <instance_id> [--split dev|verified|verified_princeton]   show a task (what the agent sees vs the grader)
    sarl pool [--file configs/pool_dev.txt]                              the training pool + contamination guard
    sarl grade <instance_id> (--gold | --empty | --patch FILE)          test-based reward in a fresh sandbox (node08)
    sarl episode <instance_id> [--policy vllm|oracle] [--model NAME]    one rollout in a real sandbox (node08)
    sarl train [CONFIG] [--steps N]                                     the GRPO loop (node08; needs `make serve`)
    sarl summarize <run_dir>                                            per-step table of a run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_task(a) -> int:
    from sarl.tasks import get_task

    t = get_task(a.instance_id, splits=(a.split,) if a.split else ("dev", "verified", "verified_princeton"))
    print(f"{t.instance_id}  [{t.split}]  repo {t.repo}  base {t.base_commit[:10]}  image {t.image}")
    print(f"contaminates_eval={t.contaminates_eval}  F2P={len(t.fail_to_pass)}  P2P={len(t.pass_to_pass)}  gold patch {len(t.patch)} chars / {t.n_patch_files} file(s)")
    print("\n--- problem_statement (the agent's whole input) ---\n" + t.problem_statement[:3000])
    if a.full:
        print("\n--- FAIL_TO_PASS ---\n" + "\n".join(t.fail_to_pass))
        print("\n--- gold patch (hidden from the agent) ---\n" + t.patch)
    return 0


def cmd_pool(a) -> int:
    from sarl.env.apptainer import sif_for
    from sarl.tasks import contamination_report, load_split, read_pool

    pool = read_pool(a.file)
    for t in pool:
        print(f"{t.instance_id:40} {t.repo:28} patch {len(t.patch):4}  F2P {len(t.fail_to_pass)}  P2P {len(t.pass_to_pass):3}  sif {'yes' if sif_for(t).exists() else 'no'}")
    print(json.dumps(contamination_report(pool, load_split("verified")), indent=1))
    return 0


def cmd_grade(a) -> int:
    from sarl.env.apptainer import ensure_sif, sif_for
    from sarl.reward import grade
    from sarl.tasks import get_task

    t = get_task(a.instance_id)
    patch = t.patch if a.gold else ("" if a.empty else Path(a.patch).read_text())
    sif = ensure_sif(t.docker_ref, sif_for(t))
    r = grade(t, patch, sif=sif, run_empty=True)
    print(json.dumps({k: v for k, v in r.to_json().items() if k != "report"}, indent=1))
    return 0 if (r.resolved == bool(a.gold)) or a.patch else 1


def cmd_episode(a) -> int:
    from transformers import AutoTokenizer

    from sarl.env.apptainer import ApptainerEnv, ensure_sif, sif_for
    from sarl.harness import ChatFormat, Limits, oracle_replies, run_episode
    from sarl.paths import MODEL_ID, MODEL_REVISION
    from sarl.policy import ScriptedPolicy, VLLMPolicy
    from sarl.reward import grade
    from sarl.tasks import get_task

    t = get_task(a.instance_id)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    fmt = ChatFormat(tok)
    sif = ensure_sif(t.docker_ref, sif_for(t))
    pol = ScriptedPolicy(tok, oracle_replies(t)) if a.policy == "oracle" else VLLMPolicy(a.url, a.model)
    with ApptainerEnv(sif) as env:
        tr = run_episode(pol, env, t, fmt, Limits(max_turns=a.max_turns), policy_name=a.policy)
    for turn in tr.turns:
        print(f"turn {turn['turn']:2}  {turn.get('command', turn.get('format_error', ''))[:110]!r}  rc={turn.get('returncode')}")
    print(f"exit={tr.exit_reason}  tokens={tr.n_tokens}  assistant={tr.n_assistant_tokens}  patch={len(tr.patch)} chars  {tr.wall_s}s")
    if a.grade:
        g = grade(t, tr.patch, sif=sif)
        print(f"reward={g.reward}  resolved={g.resolved}  f2p {g.f2p_passed}/{g.f2p_total}  p2p {g.p2p_passed}/{g.p2p_total}")
    if a.out:
        Path(a.out).write_text(json.dumps(tr.to_json(), default=str))
    return 0


def cmd_train(a) -> int:
    from sarl.train import run

    run(a.config, steps=a.steps, run_name=a.run_name, resume=a.resume)
    return 0


def cmd_summarize(a) -> int:
    from sarl.metrics import read_jsonl

    for r in read_jsonl(Path(a.run_dir) / "metrics.jsonl"):
        u = r.get("update") or {}
        print(f"step {r['step']:3}  policy reward {r.get('policy_reward_mean')}  resolved {r.get('policy_resolved')}  "
              f"oracle {r.get('oracle_resolved')}  degenerate {r.get('n_degenerate_groups')}  train {r.get('n_train')}  "
              f"loss {u.get('loss')}  gn {u.get('grad_norm')}  Δlogp(oracle) {r.get('oracle_dlogp')}  "
              f"mismatch {((r.get('mismatch') or {}).get('mean'))}  {r.get('seconds')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sarl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("task"); s.add_argument("instance_id"); s.add_argument("--split"); s.add_argument("--full", action="store_true"); s.set_defaults(fn=cmd_task)
    s = sp.add_parser("pool"); s.add_argument("--file", default=str(Path(__file__).resolve().parents[2] / "configs" / "pool_dev.txt")); s.set_defaults(fn=cmd_pool)
    s = sp.add_parser("grade"); s.add_argument("instance_id"); g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--gold", action="store_true"); g.add_argument("--empty", action="store_true"); g.add_argument("--patch"); s.set_defaults(fn=cmd_grade)
    s = sp.add_parser("episode"); s.add_argument("instance_id"); s.add_argument("--policy", choices=["vllm", "oracle"], default="vllm")
    s.add_argument("--model", default="policy"); s.add_argument("--url", default="http://localhost:8001"); s.add_argument("--max-turns", type=int, default=20, dest="max_turns")
    s.add_argument("--grade", action="store_true"); s.add_argument("--out"); s.set_defaults(fn=cmd_episode)
    s = sp.add_parser("train"); s.add_argument("config", nargs="?", default="configs/basic_run.yaml"); s.add_argument("--steps", type=int)
    s.add_argument("--run-name", dest="run_name")
    s.add_argument("--resume", action="store_true", help="continue --run-name after its last recorded step")
    s.set_defaults(fn=cmd_train)
    s = sp.add_parser("summarize"); s.add_argument("run_dir"); s.set_defaults(fn=cmd_summarize)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())

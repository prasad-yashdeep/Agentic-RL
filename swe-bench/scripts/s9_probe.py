#!/usr/bin/env python3
"""Probe: how does the served policy behave on real SWE-bench sandboxes, and which tasks give GRPO mixed groups?

Runs `--episodes` rollouts per task and protocol against the vLLM serve (no training) and reports, per protocol:
valid-action rate over turns, share of episodes that submit on turn 1, exit reasons, non-empty patches, loops and
(with --grade) the test reward per task — a task with 0 < resolved < episodes would give GRPO a mixed group.
S9 gate (docs/02-phase2-roadmap.md): tool_call valid calls >= 90 % and turn-1 submits <= 25 %.

    python scripts/s9_probe.py --model-key 1.5b --protocols tool_call backticks --tasks-file configs/pool_dev.txt \
        --episodes 4 --grade --out experiments/s9-probe-1.5b.json          # on node08, with `make serve MODEL=1.5b`
    SARL_SIF_DIR=/var/tmp/swe/sif-holdout python scripts/s9_probe.py --model-key mini-coder-1.7b --protocols mswea \
        --tasks-file configs/holdout_dev.txt --limit-tasks 8 --episodes 8 --max-turns 30 --max-context 24576 --grade --out …
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sarl.env.apptainer import ApptainerEnv, sif_for
from sarl.harness import ChatFormat, Limits, run_episode, stop_ids
from sarl.paths import MODELS, REPO, TEMPLATE_KWARGS
from sarl.policy import Sampling, VLLMPolicy
from sarl.protocol import tools_for
from sarl.reward import grade
from sarl.tasks import read_pool


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-key", default="1.5b", choices=sorted(MODELS))
    p.add_argument("--protocols", nargs="+", default=["tool_call", "backticks"])
    p.add_argument("--tasks-file", default=str(REPO / "configs" / "pool_dev.txt"))
    p.add_argument("--split", default="dev")
    p.add_argument("--limit-tasks", type=int, default=4)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--max-context", type=int, default=16_384)
    p.add_argument("--wall-clock", type=float, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--template-kwargs", default=None, help="JSON; default: sarl.paths.TEMPLATE_KWARGS[model-key]")
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--served-model", default="policy")
    p.add_argument("--grade", action="store_true")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from transformers import AutoTokenizer

    mid, rev = MODELS[a.model_key]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    tkw = json.loads(a.template_kwargs) if a.template_kwargs else TEMPLATE_KWARGS.get(a.model_key, {})
    tasks = [t for t in read_pool(a.tasks_file, a.split) if sif_for(t).exists()][: a.limit_tasks]
    lim = Limits(max_turns=a.max_turns, max_context_tokens=a.max_context, command_timeout_s=60, wall_clock_s=a.wall_clock)
    samp = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    report = {"model": mid, "revision": rev, "tasks": [t.instance_id for t in tasks], "episodes_per_task": a.episodes,
              "limits": {"max_turns": lim.max_turns, "max_context_tokens": lim.max_context_tokens,
                         "command_timeout_s": lim.command_timeout_s, "wall_clock_s": lim.wall_clock_s},
              "sampling": samp, "template_kwargs": tkw, "protocols": {}}
    for proto in a.protocols:
        fmt = ChatFormat(tok, tools_for(proto), tkw)
        stops = stop_ids(fmt, mid, rev)

        def one(i_t):
            i, t = i_t
            with ApptainerEnv(sif_for(t), command_timeout=lim.command_timeout_s) as env:
                tr = run_episode(VLLMPolicy(a.url, a.served_model, Sampling(**samp, seed=1000 + i), stop_token_ids=stops), env, t, fmt, lim, protocol=proto)
            g = grade(t, tr.patch, sif=sif_for(t)) if a.grade and tr.patch.strip() else None
            return t, tr, g

        jobs = [(i, t) for t in tasks for i in range(a.episodes)]
        t0 = time.time()
        with ThreadPoolExecutor(a.concurrency) as ex:
            res = list(ex.map(one, jobs))
        turns = sum(len(tr.turns) for _, tr, _ in res)
        valid = sum(1 for _, tr, _ in res for x in tr.turns if "command" in x)
        submit1 = sum(1 for _, tr, _ in res if tr.exit_reason == "submitted" and len(tr.turns) == 1)
        cmds = Counter((x.get("command") or "").split()[0] if (x.get("command") or "").split() else "FORMAT_ERROR"
                       for _, tr, _ in res for x in tr.turns)
        loops = sum(1 for _, tr, _ in res if tr.commands and max(Counter(tr.commands).values()) >= 3)
        per_task = {t.instance_id: {"resolved": 0, "episodes": 0, "nonempty": 0} for t in tasks}
        for t, tr, g in res:
            d = per_task[t.instance_id]
            d["episodes"] += 1
            d["nonempty"] += bool(tr.patch.strip())
            d["resolved"] += bool(g and g.resolved)
        rep = {
            "episodes": len(res), "turns": turns, "valid_actions": valid,
            "valid_action_rate": round(valid / max(turns, 1), 3),
            "turn1_submit_rate": round(submit1 / max(len(res), 1), 3),
            "mean_turns": round(turns / max(len(res), 1), 2),
            "exits": dict(Counter(tr.exit_reason for _, tr, _ in res)),
            "nonempty_patches": sum(1 for _, tr, _ in res if tr.patch.strip()),
            "loop_rate": round(loops / max(len(res), 1), 3),  # an identical command issued >= 3 times
            "tokens": {"median": sorted(tr.n_tokens for _, tr, _ in res)[len(res) // 2], "max": max(tr.n_tokens for _, tr, _ in res)},
            "resolved": sum(1 for _, _, g in res if g and g.resolved) if a.grade else None,
            "per_task": per_task if a.grade else None,
            "mixed_tasks": [k for k, d in per_task.items() if 0 < d["resolved"] < d["episodes"]] if a.grade else None,
            "top_commands": dict(cmds.most_common(12)),
            "wall_s": round(time.time() - t0, 1),
            "episodes_detail": [{"task": t.instance_id, "exit": tr.exit_reason, "turns": len(tr.turns), "tokens": tr.n_tokens,
                                 "patch": bool(tr.patch.strip()), "resolved": bool(g and g.resolved),
                                 "f2p": f"{g.f2p_passed}/{g.f2p_total}" if g else None, "wall_s": tr.wall_s} for t, tr, g in res],
            "examples": [{"task": t.instance_id, "exit": tr.exit_reason,
                          "turns": [(x.get("command") or x.get("format_error") or "")[:120] for x in tr.turns[:6]]}
                         for t, tr, _ in res[:: max(1, len(res) // 4)]],
        }
        report["protocols"][proto] = rep
        print(f"{proto:10} valid {rep['valid_action_rate']:.0%} ({valid}/{turns}) | turn-1 submit {rep['turn1_submit_rate']:.0%} | "
              f"mean turns {rep['mean_turns']} | exits {rep['exits']} | patches {rep['nonempty_patches']} | loops {rep['loop_rate']:.0%} | "
              f"resolved {rep['resolved']} | {rep['wall_s']}s", flush=True)
        if a.grade:
            for k, d in per_task.items():
                print(f"    {k:40} {d['resolved']}/{d['episodes']} resolved, {d['nonempty']} patches", flush=True)
            print(f"    mixed tasks: {len(rep['mixed_tasks'])}/{len(per_task)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

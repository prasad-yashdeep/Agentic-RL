#!/usr/bin/env python3
"""S9 probe: does the served policy follow an action protocol on real SWE-bench sandboxes?

Runs `--episodes` rollouts per task and protocol against the vLLM serve (no training) and reports, per protocol:
valid-action rate over turns, share of episodes that submit on turn 1, exit reasons, non-empty patches and (with
--grade) the test reward. Gate (docs/02-phase2-roadmap.md): tool_call valid calls >= 90 % and turn-1 submits <= 25 %.

    python scripts/s9_probe.py --model-key 1.5b --protocols tool_call backticks --tasks-file configs/pool_dev.txt \
        --episodes 4 --grade --out experiments/s9-probe-1.5b.json          # on node08, with `make serve MODEL=1.5b`
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sarl.env.apptainer import ApptainerEnv, sif_for
from sarl.harness import ChatFormat, Limits, run_episode
from sarl.paths import MODELS, REPO
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
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--grade", action="store_true")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from transformers import AutoTokenizer

    mid, rev = MODELS[a.model_key]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    tasks = [t for t in read_pool(a.tasks_file, a.split) if sif_for(t).exists()][: a.limit_tasks]
    lim = Limits(max_turns=a.max_turns, command_timeout_s=60, wall_clock_s=600)
    report = {"model": mid, "revision": rev, "tasks": [t.instance_id for t in tasks], "episodes_per_task": a.episodes,
              "limits": {"max_turns": lim.max_turns, "command_timeout_s": lim.command_timeout_s, "wall_clock_s": lim.wall_clock_s},
              "protocols": {}}
    for proto in a.protocols:
        fmt = ChatFormat(tok, tools_for(proto))

        def one(i_t):
            i, t = i_t
            with ApptainerEnv(sif_for(t), command_timeout=lim.command_timeout_s) as env:
                tr = run_episode(VLLMPolicy(a.url, "policy", Sampling(seed=1000 + i)), env, t, fmt, lim, protocol=proto)
            g = grade(t, tr.patch, sif=sif_for(t)) if a.grade and tr.patch.strip() else None
            return t, tr, g

        jobs = [(i, t) for t in tasks for i in range(a.episodes)]
        t0 = time.time()
        with ThreadPoolExecutor(a.concurrency) as ex:
            res = list(ex.map(one, jobs))
        turns = sum(len(tr.turns) for _, tr, _ in res)
        valid = sum(1 for _, tr, _ in res for x in tr.turns if "command" in x)
        submit1 = sum(1 for _, tr, _ in res if tr.exit_reason == "submitted" and len(tr.turns) == 1)
        cmds = Counter((x.get("command") or "").split()[0] if x.get("command") else "FORMAT_ERROR"
                       for _, tr, _ in res for x in tr.turns)
        rep = {
            "episodes": len(res), "turns": turns, "valid_actions": valid,
            "valid_action_rate": round(valid / max(turns, 1), 3),
            "turn1_submit_rate": round(submit1 / max(len(res), 1), 3),
            "mean_turns": round(turns / max(len(res), 1), 2),
            "exits": dict(Counter(tr.exit_reason for _, tr, _ in res)),
            "nonempty_patches": sum(1 for _, tr, _ in res if tr.patch.strip()),
            "resolved": sum(1 for _, _, g in res if g and g.resolved) if a.grade else None,
            "top_commands": dict(cmds.most_common(12)),
            "wall_s": round(time.time() - t0, 1),
            "examples": [{"task": t.instance_id, "exit": tr.exit_reason,
                          "turns": [(x.get("command") or x.get("format_error") or "")[:120] for x in tr.turns[:6]]}
                         for t, tr, _ in res[:: max(1, len(res) // 4)]],
        }
        report["protocols"][proto] = rep
        print(f"{proto:10} valid {rep['valid_action_rate']:.0%} ({valid}/{turns}) | turn-1 submit {rep['turn1_submit_rate']:.0%} | "
              f"mean turns {rep['mean_turns']} | exits {rep['exits']} | patches {rep['nonempty_patches']} | resolved {rep['resolved']} | {rep['wall_s']}s",
              flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

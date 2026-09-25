#!/usr/bin/env python3
"""Summarise a scripts/ab_serving.sh run: per layout, the episode phase's wall-clock and throughput, per-server work,
outcomes, and the rollout host's memory and load while it ran.

    python scripts/ab_serving_report.py experiments/ab-serving [--baseline R2] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--baseline", default=None, help="layout the others are compared with")
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    d = Path(a.dir)

    # metrics.log: "<epoch> <name> <before|after> <url> <metric>{labels} <value>"
    tok: dict = defaultdict(lambda: defaultdict(dict))
    for line in (d / "metrics.log").read_text().splitlines():
        ep, name, phase, url, metric, value = line.split(" ", 5)
        tok[name][url][(phase, metric.split("{", 1)[0].removeprefix("vllm:"))] = float(value)
    # mon.log: "<epoch> <MemAvailable kB> <load1> <instances>" plus "<epoch> start|end <name> ..."
    samples, spans = [], {}
    for line in (d / "mon.log").read_text().splitlines():
        f = line.split()
        if f[1] in ("start", "end"):
            spans.setdefault(f[2], {})[f[1]] = int(f[0])
        elif len(f) >= 4:
            samples.append((int(f[0]), int(f[1]) / 1024, float(f[2]), int(f[3])))

    rows = {}
    for name in sorted(spans):
        fj = d / f"{name}.json"
        if not fj.exists() or "end" not in spans[name]:
            continue
        s = json.loads(fj.read_text())
        timing = s.get("timing", [])
        eps = [e for t in s["tasks"].values() for e in t["arms"]["gemma"]["episodes"]]
        errs = [e for t in s["tasks"].values() for e in t["arms"]["gemma"].get("errors", [])]
        servers = timing[-1]["servers"] if timing else {}
        urls = list(servers)
        ep_s = sum(t["episodes_s"] for t in timing)
        turns = sum(e["turns"] for e in eps)
        per_server = {}
        for j, u in enumerate(urls):
            mine = [e for e in eps if e["server"] == j]
            m = tok[name][u]
            gen = m.get(("after", "generation_tokens_total"), 0) - m.get(("before", "generation_tokens_total"), 0)
            per_server[u] = {"slots": servers[u], "episodes": len(mine), "turns": sum(e["turns"] for e in mine),
                             "gen_tokens": int(gen), "gen_tok_s": round(gen / ep_s, 1) if ep_s else None,
                             "median_episode_s": round(st.median(e["wall_s"] for e in mine), 1) if mine else None,
                             "s_per_turn": round(sum(e["wall_s"] for e in mine) / max(1, sum(e["turns"] for e in mine)), 2)}
        win = [x for x in samples if spans[name]["start"] <= x[0] <= spans[name]["end"]]
        gen_total = sum(v["gen_tokens"] for v in per_server.values())
        rows[name] = {
            "servers": servers, "slots": sum(servers.values()), "episodes": len(eps), "errors": len(errs),
            "gate_s": sum(t["gate_s"] for t in timing), "episodes_s": ep_s,
            "episodes_per_h": round(len(eps) / ep_s * 3600, 1) if ep_s else None,
            "turns": turns, "turns_per_min": round(turns / ep_s * 60, 1) if ep_s else None,
            "gen_tokens": gen_total, "gen_tok_s": round(gen_total / ep_s, 1) if ep_s else None,
            "resolved": sum(e["resolved"] for e in eps),
            "valid_rate": round(sum(e["valid_rate"] * e["turns"] for e in eps) / max(1, turns), 3),
            "max_episode_s": max((e["wall_s"] for e in eps), default=None),
            "host": {"min_avail_gb": round(min(x[1] for x in win) / 1024, 2) if win else None,
                     "mean_load1": round(st.mean(x[2] for x in win), 1) if win else None,
                     "max_instances": max((x[3] for x in win), default=None)},
            "per_server": per_server,
        }

    base = rows.get(a.baseline) if a.baseline else None
    print(f"{'layout':6} {'slots':>5} {'episodes_s':>10} {'ep/h':>6} {'turns/min':>9} {'gen tok/s':>9} {'max ep s':>8} "
          f"{'resolved':>8} {'valid':>6} {'node04 min GB':>13} {'load':>5}  speed-up vs {a.baseline or '-'}")
    for name, r in rows.items():
        sp = ""
        if base and r["episodes_s"]:
            sp = (f"wall x{base['episodes_s'] / r['episodes_s']:.2f}, turns/min x{r['turns_per_min'] / base['turns_per_min']:.2f}, "
                  f"gen tok/s x{r['gen_tok_s'] / base['gen_tok_s']:.2f}")
        print(f"{name:6} {r['slots']:5} {r['episodes_s']:10.0f} {r['episodes_per_h']:6.1f} {r['turns_per_min']:9.1f} "
              f"{r['gen_tok_s']:9.1f} {r['max_episode_s']:8.0f} {r['resolved']:5}/{r['episodes']:<3} {r['valid_rate']:6.3f} "
              f"{r['host']['min_avail_gb']!s:>13} {r['host']['mean_load1']!s:>5}  {sp}")
        for u, v in r["per_server"].items():
            print(f"         {u:24} slots {v['slots']:2}  episodes {v['episodes']:3}  turns {v['turns']:4}  "
                  f"gen {v['gen_tok_s']} tok/s  median episode {v['median_episode_s']} s  {v['s_per_turn']} s/turn")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

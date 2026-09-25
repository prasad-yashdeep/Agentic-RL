#!/usr/bin/env python3
"""Serving-capacity benchmark: synthetic multi-turn episodes shaped like the harness's rollouts, run against vLLM servers.

Each worker is pinned to one server and runs episodes back to back, the way the harness drives a server:
  - the prompt starts as P0 random token ids (a fresh prefix per episode);
  - every turn generates exactly G tokens (ignore_eos), then appends O random "observation" ids and sleeps E seconds
    (sandbox time) before the next turn, so the prompt grows and the prefix cache is used as in a real episode;
  - an episode ends after T turns or at the context cap.
Only turns that finish inside the measured window (after --warmup) count. Reports generated tokens/s per server and in
total, plus per-turn latency. Content is random, so the numbers measure capacity only, not the policy.

    python scripts/bench_servers.py --servers http://node04:8003 --concurrency 4 --duration 120
    python scripts/bench_servers.py --servers http://node08:8001 http://node04:8003 --concurrency 8 6
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import threading
import time

import httpx


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--servers", nargs="+", required=True)
    p.add_argument("--concurrency", nargs="+", type=int, default=[8], help="workers per server (one value, or one each)")
    p.add_argument("--duration", type=float, default=120, help="measured seconds (after --warmup)")
    p.add_argument("--warmup", type=float, default=30)
    p.add_argument("--p0", type=int, default=2500, help="initial prompt tokens")
    p.add_argument("--gen", type=int, default=250, help="tokens generated per turn")
    p.add_argument("--obs", type=int, default=550, help="observation tokens appended per turn")
    p.add_argument("--turns", type=int, default=9)
    p.add_argument("--env-s", type=float, default=1.0, help="sandbox seconds between turns")
    p.add_argument("--max-context", type=int, default=24576)
    p.add_argument("--model", default="policy")
    p.add_argument("--vocab", type=int, nargs=2, default=[1000, 200000], help="range of random token ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    conc = a.concurrency * len(a.servers) if len(a.concurrency) == 1 else a.concurrency
    if len(conc) != len(a.servers):
        p.error("--concurrency takes one value or one per server")

    t_start = time.time() + a.warmup
    t_end = t_start + a.duration
    lock = threading.Lock()
    done: list[dict] = []  # completed turns: server, t, gen, prompt, latency
    errors: list[str] = []

    def worker(s: int, w: int) -> None:
        rng = random.Random(f"{a.seed}-{s}-{w}")
        cli = httpx.Client(timeout=900)
        while time.time() < t_end:
            ids = [rng.randrange(*a.vocab) for _ in range(a.p0)]
            for _ in range(a.turns):
                if len(ids) + a.gen > a.max_context or time.time() >= t_end:
                    break
                body = {"model": a.model, "prompt": ids, "max_tokens": a.gen, "ignore_eos": True, "temperature": 1.0,
                        "return_token_ids": True, "skip_special_tokens": False,
                        "seed": rng.randrange(2**31)}
                t0 = time.time()
                try:
                    r = cli.post(a.servers[s].rstrip("/") + "/v1/completions", json=body)
                    r.raise_for_status()
                    out = r.json()["choices"][0]["token_ids"]
                except Exception as e:  # noqa: BLE001 - a benchmark reports failures, it does not stop on them
                    with lock:
                        errors.append(f"{a.servers[s]}: {type(e).__name__}: {str(e)[:200]}")
                    time.sleep(2)
                    break
                t1 = time.time()
                with lock:
                    done.append({"s": s, "t": t1, "gen": len(out), "prompt": len(ids), "lat": t1 - t0})
                ids = ids + list(out) + [rng.randrange(*a.vocab) for _ in range(a.obs)]
                time.sleep(a.env_s)

    threads = [threading.Thread(target=worker, args=(s, w), daemon=True) for s in range(len(a.servers)) for w in range(conc[s])]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    rep = {"servers": a.servers, "concurrency": conc, "duration_s": a.duration,
           "shape": {"p0": a.p0, "gen": a.gen, "obs": a.obs, "turns": a.turns, "env_s": a.env_s}, "per_server": [],
           "errors": errors[:20], "n_errors": len(errors)}
    tot = 0
    for s, url in enumerate(a.servers):
        win = [d for d in done if d["s"] == s and t_start <= d["t"] < t_end]
        gen = sum(d["gen"] for d in win)
        tot += gen
        lat = sorted(d["lat"] for d in win)
        rep["per_server"].append({
            "url": url, "workers": conc[s], "turns": len(win), "gen_tok_s": round(gen / a.duration, 1),
            "per_seq_tok_s": round(st.mean(d["gen"] / d["lat"] for d in win), 1) if win else None,
            "turn_latency_s": {"p50": round(lat[len(lat) // 2], 2), "p90": round(lat[int(len(lat) * 0.9)], 2),
                               "max": round(lat[-1], 2)} if lat else None,
            "mean_prompt_tokens": round(st.mean(d["prompt"] for d in win)) if win else None})
    rep["total_gen_tok_s"] = round(tot / a.duration, 1)
    for r in rep["per_server"]:
        print(f"{r['url']:28} workers {r['workers']:2}  gen {r['gen_tok_s']:7.1f} tok/s  per-seq {r['per_seq_tok_s']} tok/s  "
              f"turns {r['turns']:4}  latency {r['turn_latency_s']}  prompt~{r['mean_prompt_tokens']}", flush=True)
    print(f"TOTAL gen {rep['total_gen_tok_s']} tok/s; errors {len(errors)}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

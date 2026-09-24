#!/usr/bin/env python3
"""Rolling probe over many SWE-bench tasks with little disk: pull a batch of images, check that each task grades
(gold patch = 1, empty patch = 0), run K episodes per task for every arm on one or more vLLM servers, grade them, then
delete the batch's images (with --keep-mixed, tasks whose outcome was mixed keep theirs). The next batch is pulled
while the current one runs, without the apptainer layer cache. Results are written after every episode; a restart
with the same --out resumes where it stopped.

Invariants (from the adversarial review of 2026-09-24):
- Only images this script pulled (recorded in the JSON as "pulled") are ever deleted, and only once the task is
  final: gate verdict negative, or every arm has K valid episodes.
- A server fault (connection error, HTTP error, e.g. an adapter missing after a restart) is not an episode outcome:
  the episode is not recorded, the server is fenced, and a resume re-runs it. The run stops when no server is healthy.
- Errored episodes and grading infra failures are kept under "errors", never counted, and retried on resume.
- Pull failures and gate exceptions are transient (retried); only a real gold/empty verdict is final.
- The run refuses to start if another process holds the same --out, if a server does not serve every arm, if the
  servers serve different base weights, or if the stored settings differ from the current ones.

  # (a) screen dev tasks for a GRPO pool with base mini-coder on both GPUs
  python scripts/rolling_probe.py configs/screen_dev_v2.txt --split dev --sif-dir /var/tmp/swe/sif-dev \\
      --arm base=policy --servers http://localhost:8001 http://localhost:8002 --episodes 8 --keep-mixed \\
      --out experiments/screen-dev-v2.json
  # gate only (no GPU): which Verified-subset tasks grade correctly under our apptainer sandbox
  python scripts/rolling_probe.py configs/eval_verified_v1.txt --split verified --sif-dir /var/tmp/swe/sif-eval \
      --arm base=policy --episodes 0 --batch 4 --out experiments/gate-verified-v1.json
  # (b) evaluate arms on the pre-registered Verified subset (load every arm's adapter on every server first)
  python scripts/rolling_probe.py configs/eval_verified_v1.txt --split verified --sif-dir /var/tmp/swe/sif-eval \\
      --arm base=policy --arm grpo=<adapter-name> --servers … --episodes 4 --out experiments/eval-verified-v1.json
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from sarl.env.apptainer import ApptainerEnv, ensure_sif, sif_for, stop_all
from sarl.harness import ChatFormat, Limits, run_episode
from sarl.paths import MODELS, TEMPLATE_KWARGS
from sarl.policy import Sampling, VLLMPolicy
from sarl.protocol import tools_for
from sarl.reward import grade
from sarl.tasks import read_pool


class ServerFault(RuntimeError):
    pass


class NoServer(RuntimeError):
    pass


def served(url: str) -> dict[str, str]:
    """model id -> root (weights path) for every model a vLLM server lists (base and runtime-loaded adapters)."""
    data = httpx.get(url.rstrip("/") + "/v1/models", timeout=15).json().get("data", [])
    return {m["id"]: str(m.get("root")) for m in data}


def check_servers(urls: list[str], arms: dict[str, str]) -> dict:
    ident = {}
    for u in urls:
        ids = served(u)
        miss = sorted(set(arms.values()) - set(ids))
        if miss:
            raise SystemExit(f"{u} does not serve {miss} (has {sorted(ids)}); load every arm's adapter on every server")
        ident[u] = {name: ids[name] for name in set(arms.values()) | ({"policy"} & set(ids))}  # arms + base weights
    roots = {json.dumps(d, sort_keys=True) for d in ident.values()}
    if len(roots) > 1:
        raise SystemExit(f"servers serve different weights: {json.dumps(ident, indent=1)}")
    return ident


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tasks_file")
    p.add_argument("--split", default="dev")
    p.add_argument("--sif-dir", required=True)
    p.add_argument("--arm", action="append", required=True, help="name=served_model (repeatable)")
    p.add_argument("--servers", nargs="+", default=["http://localhost:8001"])
    p.add_argument("--per-server", type=int, default=8, help="episodes in flight per vLLM server (KV-cache bound)")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--model-key", default="mini-coder-1.7b", choices=sorted(MODELS))
    p.add_argument("--protocol", default="mswea")
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--max-context", type=int, default=24_576)
    p.add_argument("--wall-clock", type=float, default=1800)
    p.add_argument("--command-timeout", type=float, default=60)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-free-gb", type=float, default=6.0, help="free space to leave on the SIF disk")
    p.add_argument("--pull-reserve-gb", type=float, default=5.0, help="space held back per pull in flight (layers + rootfs + SIF)")
    p.add_argument("--keep-mixed", action="store_true")
    p.add_argument("--keep-all", action="store_true", help="never delete images (debugging)")
    p.add_argument("--allow-meta-change", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    out = Path(a.out)
    lockf = open(str(out) + ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another process is writing {out}")

    def on_signal(signum, _frame):
        raise SystemExit(128 + signum)  # run the finally blocks (sandbox cleanup, state save)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)

    from transformers import AutoTokenizer

    arms = dict(x.split("=", 1) for x in a.arm)
    # --episodes 0 = gate only (pull, check gold=1/empty=0, delete): needs no GPU or server
    ident = check_servers(a.servers, arms) if a.episodes > 0 else {}
    sif_dir = Path(a.sif_dir).resolve()
    sif_dir.mkdir(parents=True, exist_ok=True)
    mid, rev = MODELS[a.model_key]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    fmt = ChatFormat(tok, tools_for(a.protocol), TEMPLATE_KWARGS.get(a.model_key, {}))
    lim = Limits(max_turns=a.max_turns, max_context_tokens=a.max_context, wall_clock_s=a.wall_clock,
                 command_timeout_s=a.command_timeout)
    tasks = read_pool(a.tasks_file, a.split)
    by_id = {t.instance_id: t for t in tasks}
    state = json.loads(out.read_text()) if out.exists() else {}
    meta = {"tasks_file": a.tasks_file, "split": a.split, "model": mid, "revision": rev, "arms": arms,
            "episodes": a.episodes, "protocol": a.protocol,
            "limits": {"max_turns": lim.max_turns, "max_context_tokens": lim.max_context_tokens,
                       "wall_clock_s": lim.wall_clock_s, "command_timeout_s": lim.command_timeout_s},
            "sampling": {"temperature": a.temperature, "top_p": 1.0, "top_k": -1},
            "seeds": "per turn: 1000 + episode index, then + 1000003 x turn (paired across arms)",
            "server_weights": next(iter(ident.values()), None)}
    old = state.get("meta")
    if old and not a.allow_meta_change:
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"stored settings differ from this run: {json.dumps(diff, indent=1)} (--allow-meta-change to override)")
    state["meta"] = meta
    state.setdefault("tasks", {})
    lock = threading.Lock()

    def save() -> None:
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(out)

    def rec(iid: str) -> dict:
        return state["tasks"].setdefault(iid, {"repo": by_id[iid].repo if iid in by_id else "", "arms": {}})

    def eps(r: dict, arm: str) -> list[dict]:
        return r.get("arms", {}).get(arm, {}).get("episodes", [])

    def gate_final_negative(r: dict) -> bool:
        g = r.get("gate") or {}
        return g.get("ok") is False and "error" not in g

    def complete(r: dict) -> bool:
        return gate_final_negative(r) or (bool((r.get("gate") or {}).get("ok"))
                                          and all(len({e["i"] for e in eps(r, arm)}) >= a.episodes for arm in arms))

    def drop(t) -> None:
        r = rec(t.instance_id)
        path = sif_for(t, sif_dir)
        if a.keep_all or not r.get("pulled") or path.is_symlink() or path.resolve().parent != sif_dir:
            return
        path.unlink(missing_ok=True)
        Path(str(path) + ".lock").unlink(missing_ok=True)
        r["image_deleted"] = True

    def finalize(t) -> None:
        """Only for complete tasks: decide mixed, keep or delete the image."""
        r = rec(t.instance_id)
        if not complete(r):
            return
        res = {arm: sum(1 for e in eps(r, arm) if e.get("resolved")) for arm in arms}
        n = {arm: len(eps(r, arm)) for arm in arms}
        r["mixed"] = bool((r.get("gate") or {}).get("ok")) and any(0 < res[arm] < n[arm] for arm in arms)
        if not (a.keep_mixed and r["mixed"]):
            drop(t)
        r["final"] = {arm: f"{res[arm]}/{n[arm]}" for arm in arms}

    # tasks finished by an interrupted run but never finalised (its images may still be on disk)
    for iid, r in state["tasks"].items():
        if iid in by_id and complete(r) and "final" not in r:
            finalize(by_id[iid])
    save()
    todo = [t for t in tasks if not complete(rec(t.instance_id))]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} complete, {len(todo)} to run; arms {arms}; "
          f"{len(a.servers)} server(s) x {a.per_server}", flush=True)
    tag = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:6]
    prefix = f"rp-{tag}-"
    stop_all(prefix)

    # pulls: reserve space per pull in flight so two concurrent pulls cannot fill the disk
    dcv = threading.Condition()
    inflight = [0]

    def pull(t) -> str | None:
        path = sif_for(t, sif_dir)
        if path.exists():
            return None
        err = None
        for attempt in range(3):
            with dcv:
                while shutil.disk_usage(sif_dir).free / 1e9 - inflight[0] * a.pull_reserve_gb < a.min_free_gb + a.pull_reserve_gb:
                    dcv.wait(30)
                inflight[0] += 1
            try:
                ensure_sif(t.docker_ref, path, cache=False)
                with lock:
                    rec(t.instance_id)["pulled"] = True
                    rec(t.instance_id).pop("pull_error", None)
                    save()
                return None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:400]
                time.sleep(30 * (attempt + 1))
            finally:
                with dcv:
                    inflight[0] -= 1
                    dcv.notify_all()
        return err

    # servers: least-loaded among the healthy ones
    load = [0] * len(a.servers)
    bad = [False] * len(a.servers)
    cv = threading.Condition()

    def acquire() -> int:
        with cv:
            while True:
                ok = [j for j in range(len(load)) if not bad[j]]
                if not ok:
                    raise NoServer("no healthy vLLM server")
                if min(load[j] for j in ok) < a.per_server:
                    s = min(ok, key=lambda j: load[j])
                    load[s] += 1
                    return s
                cv.wait(5)

    def release(s: int) -> None:
        with cv:
            load[s] -= 1
            cv.notify_all()

    faults: list[str] = []

    def episode(job):
        t, arm, i = job
        s = acquire()
        r = {"i": i, "server": s}
        try:
            try:
                with ApptainerEnv(sif_for(t, sif_dir), prefix=prefix, command_timeout=lim.command_timeout_s) as env:
                    tr = run_episode(VLLMPolicy(a.servers[s], arms[arm], Sampling(temperature=a.temperature, seed=1000 + i)),
                                     env, t, fmt, lim, protocol=a.protocol)
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                with cv:
                    bad[s] = True
                    cv.notify_all()
                with lock:
                    faults.append(f"{a.servers[s]}: {type(e).__name__}: {str(e)[:200]}")
                return None  # not recorded: a resume (or the retry round) runs it again
            g = grade(t, tr.patch, sif=sif_for(t, sif_dir)) if tr.patch.strip() else None
            r.update(exit=tr.exit_reason, turns=len(tr.turns), tokens=tr.n_tokens, wall_s=tr.wall_s,
                     patch=bool(tr.patch.strip()), resolved=bool(g and g.resolved),
                     f2p=f"{g.f2p_passed}/{g.f2p_total}" if g else None,
                     valid_rate=round(sum(1 for x in tr.turns if "command" in x) / max(1, len(tr.turns)), 3))
            key = "errors" if (g is not None and g.infra_failure) else "episodes"
            if key == "errors":
                r["error"] = "grading infra failure"
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:400]
            key = "errors"
        finally:
            release(s)
        with lock:
            a_ = rec(t.instance_id)["arms"].setdefault(arm, {"episodes": [], "errors": []})
            a_.setdefault("errors", [])
            if key == "episodes" and any(e["i"] == i for e in a_["episodes"]):
                return r  # already recorded by an earlier round
            a_[key].append(r)
            save()
        return r

    def gate(t) -> dict:
        try:
            gold = grade(t, t.patch, sif=sif_for(t, sif_dir))
            empty = grade(t, "", sif=sif_for(t, sif_dir), run_empty=True)
            if gold.infra_failure or empty.infra_failure:
                return {"ok": None, "error": "grading infra failure"}
            return {"ok": gold.reward == 1.0 and empty.reward == 0.0, "gold": gold.reward, "empty": empty.reward,
                    "gold_f2p": f"{gold.f2p_passed}/{gold.f2p_total}", "gold_p2p": f"{gold.p2p_passed}/{gold.p2p_total}"}
        except Exception as e:
            return {"ok": None, "error": f"{type(e).__name__}: {e}"[:400]}

    def missing_jobs(t) -> list:
        r = rec(t.instance_id)
        have = {arm: {e["i"] for e in eps(r, arm)} for arm in arms}
        return [(t, arm, i) for i in range(a.episodes) for arm in arms if i not in have[arm]]

    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    pull_ex = ThreadPoolExecutor(2)
    run_ex = ThreadPoolExecutor(len(a.servers) * a.per_server)
    grade_ex = ThreadPoolExecutor(8)
    pending = {t.instance_id: pull_ex.submit(pull, t) for t in batches[0]} if batches else {}
    rc = 0
    try:
        for b, batch in enumerate(batches):
            t0 = time.time()
            errs = {t.instance_id: pending[t.instance_id].result() for t in batch}
            if b + 1 < len(batches):  # prefetch the next batch while this one runs
                pending.update({t.instance_id: pull_ex.submit(pull, t) for t in batches[b + 1]})
            ready = []
            for t in batch:
                if errs[t.instance_id]:
                    rec(t.instance_id)["pull_error"] = errs[t.instance_id]  # transient: retried on resume
                else:
                    ready.append(t)
            need = [t for t in ready if not isinstance((rec(t.instance_id).get("gate") or {}).get("ok"), bool)]
            for t, g in zip(need, grade_ex.map(gate, need)):
                rec(t.instance_id)["gate"] = g
            with lock:
                save()
            live = [t for t in ready if (rec(t.instance_id).get("gate") or {}).get("ok") is True]
            for _round in range(3):  # retry rounds for episodes lost to errors or server faults
                jobs = [j for t in live for j in missing_jobs(t)]
                if not jobs:
                    break
                if any(bad):  # re-admit fenced servers that recovered (restarted, adapter re-loaded)
                    for s_, u in enumerate(a.servers):
                        try:
                            check_servers([u], arms)
                            bad[s_] = False
                        except (SystemExit, Exception):
                            pass
                list(run_ex.map(episode, jobs))
            for t in batch:
                finalize(t)
                r = rec(t.instance_id)
                g = r.get("gate") or {}
                print(f"  {t.instance_id:40} gate {g.get('ok')!s:5} " +
                      " ".join(f"{arm} {sum(1 for e in eps(r, arm) if e.get('resolved'))}/{len(eps(r, arm))}" for arm in arms) +
                      (" MIXED" if r.get("mixed") else "") + (f" pull_error {r['pull_error'][:60]}" if r.get("pull_error") else "") +
                      (f" gate_error {g['error'][:60]}" if g.get("error") else ""), flush=True)
            with lock:
                save()
            if faults:
                print(f"server faults so far: {faults[-3:]}", flush=True)
            print(f"batch {b + 1}/{len(batches)} done in {time.time() - t0:.0f}s; "
                  f"free {shutil.disk_usage(sif_dir).free / 1e9:.1f} GB", flush=True)
    except NoServer as e:
        print(f"STOPPED: {e}; resume with the same command once the servers are back", flush=True)
        rc = 3
    finally:
        stop_all(prefix)
        pull_ex.shutdown(wait=False, cancel_futures=True)
        with lock:
            save()

    tot = Counter()
    for iid, r in state["tasks"].items():
        for arm in arms:
            tot[(arm, "n")] += len(eps(r, arm))
            tot[(arm, "resolved")] += sum(1 for e in eps(r, arm) if e.get("resolved"))
    for arm in arms:
        print(f"{arm}: {tot[(arm, 'resolved')]}/{tot[(arm, 'n')]} resolved", flush=True)
    print("gate negative:", [k for k, r in state["tasks"].items() if gate_final_negative(r)], flush=True)
    print("incomplete:", [k for k, r in state["tasks"].items() if not complete(r)], flush=True)
    print("mixed:", [k for k, r in state["tasks"].items() if r.get("mixed")], flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

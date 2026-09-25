"""The training loop (the diagram end to end), for node08: vLLM serves πθ on GPU0; this process trains LoRA on GPU1.

per step:
  1. pick tasks from the pool
  2. rollouts (inner loop): G per task, concurrently, each in its own apptainer sandbox; one scripted gold-patch
     oracle per group (off-policy, the reference's trick) so a 0.5B policy's groups are not all zero
  3. reward: replay each candidate patch in a fresh container and run the tests (binary)
  4. advantages per group; skip degenerate groups
  5. one LoRA update (seq-mean REINFORCE, reference loss) on the non-degenerate groups' trajectories
  6. weight sync: save the adapter, POST /v1/load_lora_adapter, sample the next step from it (πθ′)
  7. diagnostics: |logp_vLLM − logp_trainer| on sampled tokens (lecture 2:31), Δlogp on the oracle trajectories
"""

from __future__ import annotations

import atexit
import gzip
import json
import re
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from sarl.env.apptainer import ApptainerEnv, sif_for, stop_all
from sarl.grpo import LoRATrainer, TrainConfig, group_advantages, is_degenerate, mismatch_stats
from sarl.harness import ChatFormat, Limits, oracle_replies, run_episode, stop_ids
from sarl.metrics import JsonlWriter
from sarl.paths import REPO, RUNS, SCRATCH, refuse_on_home
from sarl.policy import Sampling, ScriptedPolicy, VLLMPolicy
from sarl.protocol import tools_for
from sarl.reward import grade
from sarl.tasks import read_pool


def load_config(path: str | Path) -> dict:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    cfg = yaml.safe_load(p.read_text())
    cfg["_path"] = str(p)
    return cfg


class LoraSync:
    """vLLM runtime LoRA (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True): load the new adapter under a fresh name on every
    server, then unload the old one. Rollouts of a step only start after every server serves the new adapter."""

    def __init__(self, base_urls: str | list[str], base_model: str):
        self.urls = [u.rstrip("/") for u in ([base_urls] if isinstance(base_urls, str) else base_urls)]
        self.base_url = self.urls[0]
        self.current, self.base = base_model, base_model
        self.paths: dict[str, Path] = {}

    @staticmethod
    def remote_host(url: str) -> str | None:
        """The server's host when it is another machine (its vLLM reads adapter files from its own disk)."""
        import socket
        from urllib.parse import urlparse

        h = urlparse(url).hostname or "localhost"
        me = {"localhost", "127.0.0.1", socket.gethostname(), socket.gethostname().split(".")[0]}
        return None if h in me or h.split(".")[0] in me else h

    def unload_leftovers(self) -> list[str]:
        """Adapters left loaded by earlier runs (vLLM lists them in /v1/models with a parent); max_loras is small."""
        import httpx

        gone = []
        for u in self.urls:
            models = httpx.get(u + "/v1/models", timeout=30).json().get("data", [])
            for name in [m["id"] for m in models if m.get("parent") and m["id"] != self.base]:
                httpx.post(u + "/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
                gone.append(f"{u}:{name}")
        return gone

    def load(self, name: str, path: Path) -> None:
        import subprocess

        import httpx

        path = Path(path)
        for u in self.urls:
            host = self.remote_host(u)
            if host:  # a server on another node: give it the adapter under the same path on its own disk (~140 MB, ~2 s)
                subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {path.parent}"], check=True, timeout=60)
                subprocess.run(["scp", "-q", "-r", str(path), f"{host}:{path.parent}/"], check=True, timeout=300)
            r = httpx.post(u + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)}, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"load_lora_adapter {name} on {u}: {r.status_code} {r.text[:300]}")
        self.paths[name] = path
        old, self.current = self.current, name
        if old != self.base:
            for u in self.urls:
                httpx.post(u + "/v1/unload_lora_adapter", json={"lora_name": old}, timeout=120)
                host = self.remote_host(u)
                if host and old in self.paths:  # the remote copy has served its purpose; the local one stays
                    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"rm -rf {self.paths[old]}"], timeout=60)


ABORT = threading.Event()  # set on SIGTERM: no new rollouts, no grading


def server_fault(e: Exception) -> bool:
    """A connection error, a 5xx or an unknown model (404: adapter missing after a restart) is the server's fault; any
    other 4xx (e.g. an over-long prompt) belongs to the rollout."""
    import httpx

    if isinstance(e, httpx.TransportError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500 or e.response.status_code == 404
    return False


class Router:
    """Least-loaded choice among healthy vLLM servers, each with its own slot limit (the KV cache holds ~8 concurrent
    20k-token episodes per A4000). A server that fails a request is fenced for the rest of the step."""

    def __init__(self, servers: list[dict]):
        self.urls = [s["url"].rstrip("/") for s in servers]
        self.slots = [int(s.get("slots", 8)) for s in servers]
        self.load = [0] * len(servers)
        self.bad = [False] * len(servers)
        self.cv = threading.Condition()

    @property
    def capacity(self) -> int:
        return sum(self.slots)

    def acquire(self) -> int:
        with self.cv:
            while True:
                if ABORT.is_set():
                    raise RuntimeError("aborted")
                ok = [j for j in range(len(self.urls)) if not self.bad[j]]
                if not ok:
                    raise RuntimeError("no healthy vLLM server")
                free = [j for j in ok if self.load[j] < self.slots[j]]
                if free:
                    j = min(free, key=lambda k: self.load[k] / self.slots[k])
                    self.load[j] += 1
                    return j
                self.cv.wait(5)

    def release(self, j: int) -> None:
        with self.cv:
            self.load[j] -= 1
            self.cv.notify_all()

    def fence(self, j: int) -> None:
        """Stop routing to server j for the rest of the step, unless it is the last healthy one (then only the failing
        rollout errors, as before the router existed)."""
        with self.cv:
            if sum(1 for b in self.bad if not b) > 1:
                self.bad[j] = True
            self.cv.notify_all()

    def reset(self) -> None:
        with self.cv:
            self.bad = [False] * len(self.urls)


def vllm_sleep(url: str, timeout_s: float = 300) -> None:
    """Offload a server's weights to CPU and drop its KV cache (level 1), so the trainer can use the GPU."""
    import httpx

    httpx.post(url + "/sleep", params={"level": 1}, timeout=timeout_s).raise_for_status()
    t0 = time.time()
    while not httpx.get(url + "/is_sleeping", timeout=30).json().get("is_sleeping"):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{url} did not go to sleep")
        time.sleep(1)


def vllm_wake(url: str, timeout_s: float = 300) -> None:
    import httpx

    httpx.post(url + "/wake_up", timeout=timeout_s).raise_for_status()
    t0 = time.time()
    while httpx.get(url + "/is_sleeping", timeout=30).json().get("is_sleeping"):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"{url} did not wake up")
        time.sleep(1)


def probe(policy: VLLMPolicy, fmt: ChatFormat) -> str:
    ids = fmt.prompt_ids([{"role": "user", "content": "Reply with one bash command that lists the repository root."}])
    saved = policy.sampling
    policy.sampling = Sampling(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0)
    try:
        return policy.generate(ids, 48).text
    finally:
        policy.sampling = saved


def served_logprobs(base_url: str, model: str, ids: list[int]) -> list[float]:
    """vLLM's log-probs of a fixed token sequence under `model` (prompt_logprobs) — proves which weights are served."""
    import httpx

    r = httpx.post(base_url.rstrip("/") + "/v1/completions", json={"model": model, "prompt": ids, "max_tokens": 1,
                   "temperature": 0.0, "prompt_logprobs": 0}, timeout=120)
    r.raise_for_status()
    pl = r.json()["choices"][0].get("prompt_logprobs") or []
    out = []
    for i, entry in enumerate(pl[1:], start=1):  # entry: {token_id: {"logprob": …}} for the prompt token at i
        d = (entry or {}).get(str(ids[i])) or next(iter((entry or {}).values()), {})
        out.append(float(d.get("logprob", float("nan"))))
    return out


@dataclass
class Rollout:
    task: object
    kind: str  # policy | oracle
    idx: int
    traj: object = None
    grade: object = None
    error: str | None = None
    server: int | None = None


def _run_rollout(ro: Rollout, cfg: dict, fmt: ChatFormat, policy_model: str, tok, prefix: str = "sarl-", seed: int | None = None,
                 router: Router | None = None) -> Rollout:
    """One rollout on the least-loaded server, then its grade (a finished rollout is graded at once, while the others
    are still running; grading does not hold a server slot)."""
    import httpx

    if ABORT.is_set():
        ro.error = "aborted"
        return ro
    j = None
    try:
        if router is not None and ro.kind != "oracle":
            j = router.acquire()
            ro.server = j
        lim = Limits(**cfg["limits"])
        with ApptainerEnv(sif_for(ro.task), prefix=prefix, command_timeout=lim.command_timeout_s,
                          network=cfg.get("env", {}).get("network", False)) as env:
            protocol = cfg.get("protocol", "tool_call")
            if ro.kind == "oracle":
                pol = ScriptedPolicy(tok, oracle_replies(ro.task, protocol), im_end_id=fmt.im_end_id)
            else:
                url = router.urls[j] if j is not None else cfg["serve_url"]
                pol = VLLMPolicy(url, policy_model, Sampling(**{**cfg["sampling"], "seed": seed}), stop_token_ids=cfg["_stop_ids"])
            ro.traj = run_episode(pol, env, ro.task, fmt, lim, policy_name="oracle" if ro.kind == "oracle" else policy_model,
                                  protocol=protocol)
            if j is not None:
                ro.traj.meta["server"] = router.urls[j]
    except (httpx.TransportError, httpx.HTTPStatusError) as e:
        if j is not None and server_fault(e):
            router.fence(j)
        ro.error = f"server {type(e).__name__}: {e}"[:500]
    except Exception as e:  # a crashed rollout must not kill the batch (the lecturer's trial_10 lesson)
        ro.error = f"{type(e).__name__}: {e}"[:500]
    finally:
        if j is not None:
            router.release(j)
    return ro if ABORT.is_set() else _grade(ro, cfg)


def train_view(traj, max_tokens: int, truncate: bool):
    """(ids, mask, sampler log-probs) the trainer sees: the whole trajectory, its first `max_tokens` tokens when
    `truncate` (a context-limited rollout still teaches its first turns), or None (dropped, the S8/S9 behaviour)."""
    if traj.n_tokens <= max_tokens:
        return traj.ids, traj.mask, traj.logprobs
    if not truncate:
        return None
    return traj.ids[:max_tokens], traj.mask[:max_tokens], traj.logprobs[:max_tokens]


def preflight_servers(urls: list[str], sleepers: list[str], cfg: dict) -> dict:
    """Every server serves the policy with the same base weights and enough context; every sleep server has the dev
    sleep endpoints and is awake (a crashed run can leave one asleep: it still answers /v1/models but runs nothing)."""
    import httpx

    ident = {}
    for u in urls:
        cards = {m["id"]: m for m in httpx.get(u + "/v1/models", timeout=15).json().get("data", [])}
        m = cards.get(cfg["served_model"])
        if m is None:
            raise RuntimeError(f"{u} does not serve {cfg['served_model']} (has {sorted(cards)})")
        # an HF snapshot is identified by repo and revision, not by which cache holds it (node04 serves a local copy)
        root = re.sub(r"^.*/hub/(models--[^/]+/snapshots/[0-9a-f]+)/?$", r"\1", str(m.get("root")))
        ident[u] = {"root": root, "max_model_len": m.get("max_model_len")}
    if len({json.dumps(v, sort_keys=True) for v in ident.values()}) > 1:
        raise RuntimeError(f"servers serve different base weights or limits: {ident}")
    mml = next(iter(ident.values()))["max_model_len"]
    if mml is not None and mml < cfg["limits"]["max_context_tokens"]:
        raise RuntimeError(f"max_model_len {mml} < limits.max_context_tokens {cfg['limits']['max_context_tokens']}")
    for u in sleepers:
        r = httpx.get(u + "/is_sleeping", timeout=30)
        if r.status_code == 404:
            raise RuntimeError(f"{u} is listed with sleep: true but has no sleep endpoints; start it with SARL_SLEEP_MODE=1")
        r.raise_for_status()
        if r.json().get("is_sleeping"):
            print(f"{u} was left asleep by an earlier run; waking it", flush=True)
            vllm_wake(u)
    return ident


def _update(trainer, batch, ros, cfg, tr, truncate, adapter_root, step):
    """Mismatch diagnostics, the LoRA update, log-probs after it and the saved adapter (the trainer is on its GPU).

    diagnostics.mismatch = "all" (default): an extra no-grad pass over every policy rollout (before the update: the
    weights vLLM sampled with); "batch": taken from the update's own forward pass over the trained trajectories."""
    fast = cfg.get("diagnostics", {}).get("mismatch", "all") == "batch"
    t_all, s_all = [], []
    for r in ([] if fast else ros):
        view = train_view(r.traj, tr["max_train_tokens"], truncate) if r.kind == "policy" and r.traj is not None else None
        if view is not None and sum(view[1]):
            t_all.append(trainer.logprobs_nograd(view[0], view[1]))
            s_all += [view[2][i] for i in range(1, len(view[0])) if view[1][i]]
    mm_pooled = mismatch_stats(np.concatenate(t_all), s_all) if t_all else {"n": 0}
    before = None if fast else [trainer.logprobs_nograd(b[0], b[1]).mean() for b in batch]
    upd = trainer.step(batch) if batch else {"skipped": "no_informative_groups"}
    if fast:
        mm_pooled = upd.get("mismatch") or {"n": 0}
        before = upd.get("mean_logp_before") or []
    after = [trainer.logprobs_nograd(b[0], b[1]).mean() for b in batch]
    h = trainer.save_adapter(adapter_root / f"step-{step}") if upd.get("stepped") else None
    return upd, mm_pooled, before, after, h


def _grade(ro: Rollout, cfg: dict) -> Rollout:
    if ro.traj is None:
        return ro
    try:
        g = cfg.get("grading", {})
        ro.grade = grade(ro.task, ro.traj.patch, sif=sif_for(ro.task), timeout_s=g.get("timeout_s", 1800), network=g.get("network", False))
    except Exception as e:
        ro.error = (ro.error or "") + f" | grade {type(e).__name__}: {e}"[:400]
    return ro


def run(config_path: str | Path, *, steps: int | None = None, run_name: str | None = None, resume: bool = False) -> Path:
    """resume=True continues an existing run dir after its last recorded step: loads that step's adapter (checked
    against the recorded hash), replays the task draws, serves the adapter, appends to the records. The AdamW moments
    were not saved, so they restart from zero."""
    from transformers import AutoTokenizer

    cfg = load_config(config_path)
    if steps is not None:
        cfg["steps"] = steps
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / (run_name or f"{ts}-{cfg['name']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    metrics, rollouts_w = JsonlWriter(run_dir / "metrics.jsonl"), JsonlWriter(run_dir / "rollouts.jsonl")
    logf = open(run_dir / "log.txt", "a")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    rng = random.Random(cfg.get("seed", 0))
    import torch

    torch.manual_seed(cfg.get("seed", 0))  # LoRA init
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    fmt = ChatFormat(tok, tools_for(cfg.get("protocol", "tool_call")), cfg.get("chat_template_kwargs"))
    cfg["_stop_ids"] = stop_ids(fmt, cfg["model"], cfg.get("revision"))
    pool = read_pool(REPO / cfg["pool"], cfg.get("split", "dev"))
    missing = [t.instance_id for t in pool if not sif_for(t).exists()]
    if missing:
        raise FileNotFoundError(f"images not pulled for {missing}; run `make -C swe-bench sifs`")
    base_policy = VLLMPolicy(cfg["serve_url"], cfg["served_model"], Sampling(**cfg["sampling"]), stop_token_ids=cfg["_stop_ids"])
    if not base_policy.healthy():
        raise RuntimeError(f"vLLM serve not healthy at {cfg['serve_url']} (make -C swe-bench serve)")
    # servers: [{url, slots, sleep}] (default: the one serve_url). A "sleep" server shares its GPU with the trainer: it
    # serves rollouts, sleeps while the trainer steps, and wakes (then gets the new adapter) before the next rollouts.
    servers = cfg.get("servers") or [{"url": cfg["serve_url"], "slots": cfg["concurrency"]["rollouts"]}]
    router = Router(servers)
    sleepers = [s_["url"].rstrip("/") for s_ in servers if s_.get("sleep")]
    (run_dir / "servers.json").write_text(json.dumps({"servers": servers, "identity": preflight_servers(router.urls, sleepers, cfg)}, indent=1))
    sync = LoraSync(router.urls, cfg["served_model"])
    import hashlib

    run_tag = hashlib.sha256(run_dir.name.encode()).hexdigest()[:6]  # unique adapter/sandbox names per run
    prefix = f"sarl-{run_tag}-"
    stop_all(prefix)
    atexit.register(stop_all, prefix)  # a crashed trainer must not leave sandboxes behind
    import signal
    import sys

    def _term(*_):  # SIGTERM (pkill, tmux kill) skips atexit otherwise; stop starting rollouts and grading at once
        ABORT.set()
        sys.exit(143)

    signal.signal(signal.SIGTERM, _term)
    traj_root = refuse_on_home(SCRATCH / "trajectories" / run_dir.name, "per-token trajectories")  # node-local (NFS ~92% full)
    if (gone := sync.unload_leftovers()):
        print(f"unloaded leftover adapters: {gone}", flush=True)
    tr = cfg["train"]
    trainer = LoRATrainer(cfg["model"], cfg.get("revision"), TrainConfig.from_train_section(tr),
                          offloaded=bool(sleepers))  # on the CPU while a vLLM on its GPU is awake
    if sleepers:
        # the trainer's CUDA context (~0.3-0.45 GiB) stays on the GPU after offload(): create it now, while the server
        # is awake, and refuse to start if wake_up would not fit (review finding: 0.88 left 221 MiB free)
        x = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
        float((x @ x).sum())
        free = torch.cuda.mem_get_info()[0] / 2**20
        del x
        torch.cuda.empty_cache()
        if free < 256:
            raise RuntimeError(f"only {free:.0f} MiB free on the trainer GPU with its vLLM awake; start that server with a "
                               "lower SARL_SERVE_MEM and SARL_MAX_NUM_SEQS=16")
    truncate = bool(tr.get("truncate_overlong", False))
    log(f"run {run_dir.name}: pool {len(pool)} tasks, {cfg['steps']} steps × {cfg['tasks_per_step']} tasks × G={cfg['group_size']} "
        f"(oracle {cfg['oracle_per_group']}/group); trainable {trainer.n_trainable:,}; adapter {trainer.adapter_hash()}; "
        f"base {trainer.cfg.base}, ple_on_cpu {trainer.cfg.ple_on_cpu}, softcap {trainer.softcap}")
    prev_probe = probe(base_policy, fmt)
    probe_ids = fmt.prompt_ids([{"role": "user", "content": "Fix the failing test in /testbed."}]) + fmt.encode("```bash\nls\n```") + [fmt.turn_end_id]
    base_lp = np.array(served_logprobs(cfg["serve_url"], cfg["served_model"], probe_ids))
    G, n_or = cfg["group_size"], cfg["oracle_per_group"]
    adapter_root = refuse_on_home(SCRATCH / "adapters" / run_dir.name, "LoRA adapters")
    start = 1
    if resume and (run_dir / "metrics.jsonl").exists():
        recs = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
        last = max(r["step"] for r in recs) if recs else 0
        if last:
            want = next((r.get("sync") or {}).get("adapter_hash") for r in recs if r["step"] == last)
            got = trainer.load_adapter(adapter_root / f"step-{last}")
            if want and got != want:
                raise RuntimeError(f"adapter step-{last} hash {got} != recorded {want}")
            for _ in range(last):  # the only per-step use of rng: the task draw
                rng.sample(pool, min(cfg["tasks_per_step"], len(pool)))
            sync.load(f"{cfg['served_model']}-{run_tag}-s{last}", adapter_root / f"step-{last}")
            log(f"resumed after step {last}: adapter {got} (matches the record), task draws replayed, "
                f"serving {sync.current} on {len(router.urls)} server(s); AdamW moments restart from zero")
            start = last + 1
    for step in range(start, cfg["steps"] + 1):
        t0 = time.time()
        tasks = rng.sample(pool, min(cfg["tasks_per_step"], len(pool)))
        ros = [Rollout(t, "oracle" if i >= G - n_or else "policy", i) for t in tasks for i in range(G)]
        router.reset()
        with ThreadPoolExecutor(router.capacity + n_or * len(tasks) + 4) as ex:  # rollouts + graded as each one ends
            ros = list(ex.map(lambda r: _run_rollout(r, cfg, fmt, sync.current, tok, prefix,
                                                     cfg.get("seed", 0) * 1_000_003 + step * 1009 + r.idx, router), ros))
        t_roll = time.time() - t0
        t_grade = 0.0  # graded inside the rollout phase

        batch, groups, n_degenerate, dropped, truncated, oracle_idx = [], [], 0, 0, 0, []
        for t in tasks:
            # a crashed rollout, a failed grade or an infra failure is not a "reward 0" sample: leave it out of the
            # group (review finding) and skip the group if fewer than 2 valid members remain
            grp = [r for r in ros if r.task is t and r.traj is not None and r.grade is not None and not r.error
                   and not r.grade.infra_failure]
            if len(grp) < 2:
                groups.append({"task": t.instance_id, "rewards": [], "adv": [], "degenerate": True, "invalid": True})
                n_degenerate += 1
                continue
            rewards = [r.grade.reward for r in grp]
            adv = group_advantages(rewards, cfg["advantage"]["eps"])
            degen = is_degenerate(rewards)
            n_degenerate += int(degen)
            groups.append({"task": t.instance_id, "rewards": rewards, "adv": [round(float(a), 4) for a in adv], "degenerate": degen})
            for r, a in zip(grp, adv):
                r.adv = float(a)
                if degen and cfg["advantage"].get("skip_degenerate", True):
                    continue
                if r.traj is None:
                    continue
                view = train_view(r.traj, tr["max_train_tokens"], truncate)
                if view is None:
                    dropped += 1
                    continue
                truncated += int(len(view[0]) < r.traj.n_tokens)
                if r.kind == "oracle":
                    oracle_idx.append(len(batch))
                batch.append((view[0], view[1], float(a), view[2]))

        t_sw, t_sleep, t_wake = time.time(), 0.0, 0.0
        upd, mm_pooled, before, after, h = {"skipped": "no_informative_groups"}, {"n": 0}, [], [], None
        naps = sleepers if batch else []  # nothing to train: skip the sleep/wake cycle
        for u in naps:
            vllm_sleep(u)
        try:  # the sleep window: whatever happens, the trainer leaves the GPU and the servers wake up
            trainer.onload()
            t_sleep = time.time() - t_sw
            if batch:
                upd, mm_pooled, before, after, h = _update(trainer, batch, ros, cfg, tr, truncate, adapter_root, step)
        finally:
            t_sw = time.time()
            if naps:
                trainer.offload()
                reserved = torch.cuda.memory_reserved() / 2**20
                if reserved > 64:
                    log(f"warning: {reserved:.0f} MiB still reserved by the trainer after offload()")
                for u in naps:
                    vllm_wake(u)
            t_wake = time.time() - t_sw

        # the surrogate this update ascends: J = (1/N) Σ A_i · mean_t log π(a_t|s_t); gain = J_after − J_before > 0
        objective_gain = float(np.mean([b[2] * (a - bf) for b, a, bf in zip(batch, after, before)])) if batch else None
        sync_rec = None
        if upd.get("stepped"):
            name = f"{cfg['served_model']}-{run_tag}-s{step}"
            sync.load(name, adapter_root / f"step-{step}")
            pol_now = VLLMPolicy(cfg["serve_url"], name, stop_token_ids=cfg["_stop_ids"])
            p = probe(pol_now, fmt)
            new_lp = np.array(served_logprobs(cfg["serve_url"], name, probe_ids))
            # proof that vLLM now serves πθ′: the fixed sequence's log-probs moved away from the base model's
            sync_rec = {"adapter": name, "adapter_hash": h, "probe_changed": p != prev_probe,
                        "served_dlogp_vs_base": float(np.nanmean(np.abs(new_lp - base_lp)))}
            prev_probe = p
        pol = [r for r in ros if r.kind == "policy"]
        summary = {
            "kind": "step", "step": step, "tasks": [t.instance_id for t in tasks], "groups": groups,
            "policy_reward_mean": float(np.mean([r.grade.reward if r.grade else 0.0 for r in pol])) if pol else None,
            "policy_resolved": int(sum(1 for r in pol if r.grade and r.grade.resolved)),
            "oracle_resolved": int(sum(1 for r in ros if r.kind == "oracle" and r.grade and r.grade.resolved)),
            "n_rollouts": len(ros), "n_errors": sum(1 for r in ros if r.error), "n_degenerate_groups": n_degenerate,
            "n_train": len(batch), "dropped_overlong": dropped, "truncated_overlong": truncated, "update": upd, "sync": sync_rec,
            "objective_gain": objective_gain,
            "oracle_dlogp": [float(after[i] - before[i]) for i in oracle_idx],
            "policy_dlogp_mean": float(np.mean([after[i] - before[i] for i in range(len(batch)) if i not in set(oracle_idx)])) if len(batch) > len(oracle_idx) else None,
            "mismatch": mm_pooled if mm_pooled.get("n") else None,  # pooled over every sampled policy token this step
            "exit_reasons": {k: sum(1 for r in pol if r.traj and r.traj.exit_reason == k) for k in {r.traj.exit_reason for r in pol if r.traj}},
            "seconds": {"rollouts": round(t_roll, 1), "grading": round(t_grade, 1), "sleep": round(t_sleep, 1),
                        "wake": round(t_wake, 1), "total": round(time.time() - t0, 1)},
            "servers": {"used": dict(Counter(r.server for r in ros if r.server is not None and r.traj is not None)),
                        "fenced": [router.urls[j] for j, b in enumerate(router.bad) if b]},
        }
        sdir = traj_root / f"step-{step:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        for r in ros:
            rec = {"step": step, "task": r.task.instance_id, "kind": r.kind, "idx": r.idx, "adv": getattr(r, "adv", None),
                   "server": router.urls[r.server] if r.server is not None else None,
                   "error": r.error, "grade": r.grade.to_json() if r.grade else None}
            if r.traj is not None:
                rec.update({k: v for k, v in r.traj.to_json(with_tokens=False).items() if k not in ("messages",)})
                with gzip.open(sdir / f"{r.task.instance_id}__{r.kind}{r.idx}.json.gz", "wt") as fz:
                    json.dump(r.traj.to_json(), fz, default=str)
            if rec.get("grade"):
                rec["grade"].pop("report", None)
            rollouts_w.write(rec)
        fmt_ok = [sum(1 for x in r.traj.turns if "command" in x) / max(1, len(r.traj.turns)) for r in pol if r.traj]
        summary["policy_valid_action_rate"] = float(np.mean(fmt_ok)) if fmt_ok else None
        summary["policy_mean_turns"] = float(np.mean([len(r.traj.turns) for r in pol if r.traj])) if pol else None
        summary["policy_nonempty_patches"] = sum(1 for r in pol if r.traj and r.traj.patch)
        metrics.write(summary)
        log(f"step {step}: policy reward {summary['policy_reward_mean']}, resolved {summary['policy_resolved']}/{len(pol)}, "
            f"oracle {summary['oracle_resolved']}, degenerate {n_degenerate}/{len(tasks)}, train {len(batch)}, "
            f"loss {upd.get('loss')}, gn {upd.get('grad_norm')}, J gain {summary['objective_gain']}, sync {sync_rec}, oracle Δlogp {summary['oracle_dlogp']}, "
            f"mismatch {summary['mismatch']}, {summary['seconds']}")
    (run_dir / "summary.json").write_text(json.dumps({"run": run_dir.name, "steps": cfg["steps"], "final_adapter": sync.current,
                                                      "trajectories": str(traj_root), "adapters": str(adapter_root)}, indent=1))
    stop_all(prefix)
    log(f"done -> {run_dir}")
    logf.close()
    return run_dir

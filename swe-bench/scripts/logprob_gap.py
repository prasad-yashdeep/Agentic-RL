#!/usr/bin/env python3
"""Trainer-vs-server log-prob gap and adapter round trip for a run config (gates G2 and G4 of the Gemma port).

    # 1. sample real episodes from a server (token ids + the server's log-probs), save them gzipped (node04 or node08)
    python scripts/logprob_gap.py sample configs/grpo_gemma_v1.yaml --pool configs/pool_gemma_gate.txt --n 2 \
        --url http://node08:8001 --model policy --out /var/tmp/swe/gap/base
    # 2. score them with the trainer (node08, the trainer GPU free), optionally under a saved adapter
    CUDA_VISIBLE_DEVICES=1 python scripts/logprob_gap.py score configs/grpo_gemma_v1.yaml /var/tmp/swe/gap/base [--adapter DIR]
    # 3. one training step on those episodes, save the adapter, load it into the server, measure the served change
    CUDA_VISIBLE_DEVICES=1 python scripts/logprob_gap.py step configs/grpo_gemma_v1.yaml /var/tmp/swe/gap/base --out /var/tmp/swe/adapters/gate/step-1
    python scripts/logprob_gap.py push configs/grpo_gemma_v1.yaml /var/tmp/swe/adapters/gate/step-1 --url http://node08:8001 --name policy-gate-s1
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _fmt(cfg):
    from transformers import AutoTokenizer

    from sarl.harness import ChatFormat, stop_ids
    from sarl.protocol import tools_for

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    fmt = ChatFormat(tok, tools_for(cfg["protocol"]), cfg.get("chat_template_kwargs") or {})
    return fmt, stop_ids(fmt, cfg["model"], cfg.get("revision"))


def cmd_sample(a, cfg) -> int:
    from sarl.env.apptainer import ApptainerEnv, sif_for
    from sarl.harness import Limits, run_episode
    from sarl.policy import Sampling, VLLMPolicy
    from sarl.tasks import read_pool

    fmt, stops = _fmt(cfg)
    lim = Limits(**cfg["limits"])
    tasks = read_pool(a.pool, cfg["split"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, i) for i in range(a.n) for t in tasks]

    def one(job):
        t, i = job
        with ApptainerEnv(sif_for(t), prefix="gap-", command_timeout=lim.command_timeout_s, setup=t.sandbox_setup) as env:
            tr = run_episode(VLLMPolicy(a.url, a.model, Sampling(**{**cfg["sampling"], "seed": 4000 + i}), stop_token_ids=stops),
                             env, t, fmt, lim, protocol=cfg["protocol"])
        rec = tr.to_json(with_tokens=True) if "with_tokens" in tr.to_json.__code__.co_varnames else tr.to_json()
        rec.update(ids=tr.ids, mask=tr.mask, logprobs=tr.logprobs)
        p = out / f"{t.instance_id}__{i}.json.gz"
        with gzip.open(p, "wt") as f:
            json.dump(rec, f)
        print(f"{t.instance_id} ep {i}: {tr.exit_reason}, {len(tr.turns)} turns, {tr.n_tokens} tokens, {sum(tr.mask)} sampled -> {p}", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, jobs))
    return 0


def _load(dirp: Path):
    for p in sorted(Path(dirp).glob("*.json.gz")):
        with gzip.open(p, "rt") as f:
            yield p.name, json.load(f)


def _trainer(cfg, a):
    from sarl.grpo import LoRATrainer, TrainConfig

    tr = LoRATrainer(cfg["model"], cfg.get("revision"), TrainConfig.from_train_section(cfg["train"]))
    if getattr(a, "adapter", None):
        print("adapter", tr.load_adapter(a.adapter), flush=True)
    return tr


def cmd_score(a, cfg) -> int:
    import torch

    from sarl.grpo import mismatch_stats

    trainer = _trainer(cfg, a)
    pooled_t, pooled_s = [], []
    for name, rec in _load(a.dir):
        ids, mask, slp = rec["ids"], rec["mask"], rec["logprobs"]
        n = min(len(ids), cfg["train"]["max_train_tokens"])
        ids, mask, slp = ids[:n], mask[:n], slp[:n]
        t0 = time.time()
        lp = trainer.logprobs_nograd(ids, mask)
        s = [slp[i] for i in range(1, len(ids)) if mask[i]]
        st = mismatch_stats(lp, s)
        signed = float(np.nanmean(np.asarray(lp) - np.array([np.nan if v is None else v for v in s], dtype=float)))
        print(f"{name}: {len(ids)} tokens, {len(s)} sampled: mean |d| {st.get('mean', float('nan')):.4f} p50 {st.get('p50', float('nan')):.4f} "
              f"p99 {st.get('p99', float('nan')):.3f} max {st.get('max', float('nan')):.2f} frac>0.5 {st.get('frac_gt_0.5', float('nan')):.4f} "
              f"signed {signed:+.4f} ({time.time() - t0:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
        pooled_t.append(np.asarray(lp)); pooled_s.append(np.array([np.nan if v is None else v for v in s], dtype=float))
    if pooled_t:
        t, s = np.concatenate(pooled_t), np.concatenate(pooled_s)
        st = mismatch_stats(t, s.tolist())
        print(f"POOLED {len(t)} tokens: mean |d| {st['mean']:.4f} p50 {st['p50']:.4f} p99 {st['p99']:.3f} max {st['max']:.2f} "
              f"frac>0.5 {st['frac_gt_0.5']:.4f} signed {float(np.nanmean(t - s)):+.4f}", flush=True)
    return 0


def cmd_step(a, cfg) -> int:
    import torch

    trainer = _trainer(cfg, a)
    batch = []
    for name, rec in _load(a.dir):
        n = min(len(rec["ids"]), cfg["train"]["max_train_tokens"])
        batch.append((rec["ids"][:n], rec["mask"][:n], float(a.adv), rec["logprobs"][:n]))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.step(batch)
    print(f"step on {len(batch)} trajectories: {json.dumps({k: v for k, v in out.items() if k != 'mean_logp_before'})} "
          f"({time.time() - t0:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    print("saved", a.out, trainer.save_adapter(a.out), flush=True)
    return 0


def cmd_push(a, cfg) -> int:
    import httpx

    from sarl.train import served_logprobs

    fmt, _ = _fmt(cfg)
    probe_ids = fmt.prompt_ids([{"role": "user", "content": "Fix the failing test in /testbed."}]) + fmt.encode("```bash\nls\n```") + [fmt.turn_end_id]
    base = np.array(served_logprobs(a.url, cfg["served_model"], probe_ids))
    r = httpx.post(a.url + "/v1/load_lora_adapter", json={"lora_name": a.name, "lora_path": str(Path(a.adapter).resolve())}, timeout=600)
    print("load_lora_adapter:", r.status_code, r.text[:300], flush=True)
    r.raise_for_status()
    new = np.array(served_logprobs(a.url, a.name, probe_ids))
    print(f"served_dlogp_vs_base on {len(probe_ids)} probe tokens: mean |d| {float(np.abs(new - base).mean()):.5f}, max {float(np.abs(new - base).max()):.4f}", flush=True)
    if not a.keep:
        httpx.post(a.url + "/v1/unload_lora_adapter", json={"lora_name": a.name}, timeout=120)
        print("unloaded", a.name, flush=True)
    return 0


def main(argv=None) -> int:
    from sarl.train import load_config

    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample"); s.add_argument("config"); s.add_argument("--pool", required=True); s.add_argument("--n", type=int, default=2)
    s.add_argument("--url", default="http://node08:8001"); s.add_argument("--model", default="policy"); s.add_argument("--out", required=True)
    s.add_argument("--concurrency", type=int, default=8); s.set_defaults(fn=cmd_sample)
    s = sp.add_parser("score"); s.add_argument("config"); s.add_argument("dir"); s.add_argument("--adapter"); s.set_defaults(fn=cmd_score)
    s = sp.add_parser("step"); s.add_argument("config"); s.add_argument("dir"); s.add_argument("--adapter"); s.add_argument("--adv", type=float, default=1.0)
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_step)
    s = sp.add_parser("push"); s.add_argument("config"); s.add_argument("adapter"); s.add_argument("--url", default="http://node08:8001")
    s.add_argument("--name", required=True); s.add_argument("--keep", action="store_true"); s.set_defaults(fn=cmd_push)
    a = p.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return a.fn(a, load_config(a.config))


if __name__ == "__main__":
    raise SystemExit(main())

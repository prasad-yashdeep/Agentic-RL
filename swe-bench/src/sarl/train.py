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

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from sarl.env.apptainer import ApptainerEnv, sif_for
from sarl.grpo import LoRATrainer, TrainConfig, group_advantages, is_degenerate, mismatch_stats
from sarl.harness import ChatFormat, Limits, oracle_replies, run_episode
from sarl.metrics import JsonlWriter
from sarl.paths import REPO, RUNS, SCRATCH
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
    """vLLM runtime LoRA (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True): load the new adapter under a fresh name, unload the old."""

    def __init__(self, base_url: str, base_model: str):
        self.base_url, self.current, self.base = base_url.rstrip("/"), base_model, base_model

    def unload_leftovers(self) -> list[str]:
        """Adapters left loaded by earlier runs (vLLM lists them in /v1/models with a parent); max_loras is small."""
        import httpx

        models = httpx.get(self.base_url + "/v1/models", timeout=30).json().get("data", [])
        gone = [m["id"] for m in models if m.get("parent") and m["id"] != self.base]
        for name in gone:
            httpx.post(self.base_url + "/v1/unload_lora_adapter", json={"lora_name": name}, timeout=120)
        return gone

    def load(self, name: str, path: Path) -> None:
        import httpx

        r = httpx.post(self.base_url + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(path)}, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter {name}: {r.status_code} {r.text[:300]}")
        old, self.current = self.current, name
        if old != self.base:
            httpx.post(self.base_url + "/v1/unload_lora_adapter", json={"lora_name": old}, timeout=120)


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


def _run_rollout(ro: Rollout, cfg: dict, fmt: ChatFormat, policy_model: str, tok) -> Rollout:
    try:
        lim = Limits(**cfg["limits"])
        with ApptainerEnv(sif_for(ro.task), command_timeout=lim.command_timeout_s, network=cfg.get("env", {}).get("network", True)) as env:
            protocol = cfg.get("protocol", "tool_call")
            if ro.kind == "oracle":
                pol = ScriptedPolicy(tok, oracle_replies(ro.task, protocol))
            else:
                pol = VLLMPolicy(cfg["serve_url"], policy_model, Sampling(**cfg["sampling"]))
            ro.traj = run_episode(pol, env, ro.task, fmt, lim, policy_name="oracle" if ro.kind == "oracle" else policy_model,
                                  protocol=protocol)
    except Exception as e:  # a crashed rollout must not kill the batch (the lecturer's trial_10 lesson)
        ro.error = f"{type(e).__name__}: {e}"[:500]
    return ro


def _grade(ro: Rollout, cfg: dict) -> Rollout:
    if ro.traj is None:
        return ro
    try:
        g = cfg.get("grading", {})
        ro.grade = grade(ro.task, ro.traj.patch, sif=sif_for(ro.task), timeout_s=g.get("timeout_s", 1800), network=g.get("network", True))
    except Exception as e:
        ro.error = (ro.error or "") + f" | grade {type(e).__name__}: {e}"[:400]
    return ro


def run(config_path: str | Path, *, steps: int | None = None, run_name: str | None = None) -> Path:
    from transformers import AutoTokenizer

    cfg = load_config(config_path)
    if steps is not None:
        cfg["steps"] = steps
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / (run_name or f"{ts}-{cfg['name']}")
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    metrics, rollouts_w = JsonlWriter(run_dir / "metrics.jsonl"), JsonlWriter(run_dir / "rollouts.jsonl")
    logf = open(run_dir / "log.txt", "a")

    def log(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    rng = random.Random(cfg.get("seed", 0))
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    fmt = ChatFormat(tok, tools_for(cfg.get("protocol", "tool_call")))
    pool = read_pool(REPO / cfg["pool"], cfg.get("split", "dev"))
    missing = [t.instance_id for t in pool if not sif_for(t).exists()]
    if missing:
        raise FileNotFoundError(f"images not pulled for {missing}; run `make -C swe-bench sifs`")
    base_policy = VLLMPolicy(cfg["serve_url"], cfg["served_model"], Sampling(**cfg["sampling"]))
    if not base_policy.healthy():
        raise RuntimeError(f"vLLM serve not healthy at {cfg['serve_url']} (make -C swe-bench serve)")
    sync = LoraSync(cfg["serve_url"], cfg["served_model"])
    run_tag = f"{abs(hash(run_dir.name)) % 16**6:06x}"  # adapter names must be unique across runs on one server
    if (gone := sync.unload_leftovers()):
        print(f"unloaded leftover adapters: {gone}", flush=True)
    tr = cfg["train"]
    trainer = LoRATrainer(cfg["model"], cfg.get("revision"), TrainConfig(
        lr=tr["lr"], lora_r=tr["lora_r"], lora_alpha=tr["lora_alpha"], targets=tuple(tr["targets"]), clip=tr["clip"],
        max_train_tokens=tr["max_train_tokens"], chunk=tr.get("chunk", 1024)))
    log(f"run {run_dir.name}: pool {len(pool)} tasks, {cfg['steps']} steps × {cfg['tasks_per_step']} tasks × G={cfg['group_size']} "
        f"(oracle {cfg['oracle_per_group']}/group); trainable {trainer.n_trainable:,}; adapter {trainer.adapter_hash()}")
    prev_probe = probe(base_policy, fmt)
    probe_ids = fmt.prompt_ids([{"role": "user", "content": "Fix the failing test in /testbed."}]) + fmt.encode("```bash\nls\n```<|im_end|>")
    base_lp = np.array(served_logprobs(cfg["serve_url"], cfg["served_model"], probe_ids))
    G, n_or = cfg["group_size"], cfg["oracle_per_group"]
    adapter_root = SCRATCH / "adapters" / run_dir.name
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        tasks = rng.sample(pool, min(cfg["tasks_per_step"], len(pool)))
        ros = [Rollout(t, "oracle" if i >= G - n_or else "policy", i) for t in tasks for i in range(G)]
        with ThreadPoolExecutor(cfg["concurrency"]["rollouts"]) as ex:
            ros = list(ex.map(lambda r: _run_rollout(r, cfg, fmt, sync.current, tok), ros))
        t_roll = time.time() - t0
        with ThreadPoolExecutor(cfg["concurrency"]["grading"]) as ex:
            ros = list(ex.map(lambda r: _grade(r, cfg), ros))
        t_grade = time.time() - t0 - t_roll

        batch, groups, n_degenerate, dropped, oracle_idx = [], [], 0, 0, []
        for t in tasks:
            grp = [r for r in ros if r.task is t]
            rewards = [r.grade.reward if r.grade else 0.0 for r in grp]
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
                if r.traj.n_tokens > tr["max_train_tokens"]:
                    dropped += 1
                    continue
                if r.kind == "oracle":
                    oracle_idx.append(len(batch))
                batch.append((r.traj.ids, r.traj.mask, float(a)))

        # train-inference mismatch on policy rollouts (before the update: same weights vLLM sampled with)
        mm = []
        for r in ros:
            if r.kind == "policy" and r.traj is not None and r.traj.n_tokens <= tr["max_train_tokens"] and sum(r.traj.mask):
                t_lp = trainer.logprobs_nograd(r.traj.ids, r.traj.mask)
                s_lp = [r.traj.logprobs[i] for i in range(1, len(r.traj.ids)) if r.traj.mask[i]]
                mm.append(mismatch_stats(t_lp, s_lp))
        before = [trainer.logprobs_nograd(b[0], b[1]).mean() for b in batch]
        upd = trainer.step(batch) if batch else {"skipped": "no_informative_groups"}
        after = [trainer.logprobs_nograd(b[0], b[1]).mean() for b in batch]
        # the surrogate this update ascends: J = (1/N) Σ A_i · mean_t log π(a_t|s_t); gain = J_after − J_before > 0
        objective_gain = float(np.mean([b[2] * (a - bf) for b, a, bf in zip(batch, after, before)])) if batch else None
        sync_rec = None
        if upd.get("stepped"):
            name = f"{cfg['served_model']}-{run_tag}-s{step}"
            h = trainer.save_adapter(adapter_root / f"step-{step}")
            sync.load(name, adapter_root / f"step-{step}")
            pol_now = VLLMPolicy(cfg["serve_url"], name)
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
            "n_train": len(batch), "dropped_overlong": dropped, "update": upd, "sync": sync_rec,
            "objective_gain": objective_gain,
            "oracle_dlogp": [float(after[i] - before[i]) for i in oracle_idx],
            "policy_dlogp_mean": float(np.mean([after[i] - before[i] for i in range(len(batch)) if i not in set(oracle_idx)])) if len(batch) > len(oracle_idx) else None,
            "mismatch": {k: float(np.mean([m[k] for m in mm if m.get("n")])) for k in ("mean", "p99", "max", "frac_gt_0.5")} if any(m.get("n") for m in mm) else None,
            "exit_reasons": {k: sum(1 for r in pol if r.traj and r.traj.exit_reason == k) for k in {r.traj.exit_reason for r in pol if r.traj}},
            "seconds": {"rollouts": round(t_roll, 1), "grading": round(t_grade, 1), "total": round(time.time() - t0, 1)},
        }
        sdir = run_dir / "trajectories" / f"step-{step:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        for r in ros:
            rec = {"step": step, "task": r.task.instance_id, "kind": r.kind, "idx": r.idx, "adv": getattr(r, "adv", None),
                   "error": r.error, "grade": r.grade.to_json() if r.grade else None}
            if r.traj is not None:
                rec.update({k: v for k, v in r.traj.to_json(with_tokens=False).items() if k not in ("messages",)})
                (sdir / f"{r.task.instance_id}__{r.kind}{r.idx}.json").write_text(json.dumps(r.traj.to_json(), default=str))
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
    (run_dir / "summary.json").write_text(json.dumps({"run": run_dir.name, "steps": cfg["steps"], "final_adapter": sync.current}, indent=1))
    log(f"done -> {run_dir}")
    logf.close()
    return run_dir

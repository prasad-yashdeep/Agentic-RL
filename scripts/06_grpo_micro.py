#!/usr/bin/env python3
"""Gate G4 — end-to-end GRPO micro-run (pipeline proof, not a science run).

Loop (STEPS times):
  1. rollout: harbor run (Terminus 2, singularity+shim) vs the live vLLM serve
     -> G attempts x TASKS, trajectories + verifier rewards
  2. reward:  gated composite (verifier dominates; judge ranks failures)
  3. batch:   rebuild token sequences from trajectories via the chat template,
              masking everything the model did not emit (env tokens, prompts)
  4. update:  Dr.GRPO advantages (mean-only) + token-normalized clipped loss
              on GPU1 QLoRA (G1 recipe)
  5. sync:    merge adapter -> save -> relaunch vLLM serve (online FP8) ->
              verify the served model actually changed (fixed-prompt probe)

Run ON node08:  python scripts/06_grpo_micro.py 2>&1 | tee g4.log
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path.home() / "terminal-agent-rl"
sys.path.insert(0, str(REPO / "src"))
from tarl.judge.client import JudgeClient  # noqa: E402
from tarl.judge.composer import RewardConfig, composite_reward  # noqa: E402
from tarl.judge.compressor import compress, load_trajectory  # noqa: E402
from tarl.judge.rubric import load_rubric  # noqa: E402

# ---- config -----------------------------------------------------------------
TASKS = ["fix-git", "regex-log"]
G = 4                      # attempts per task per step
STEPS = 5
LR = 1e-6
CLIP_LO, CLIP_HI = 0.2, 0.28
LAMBDA_JUDGE = 0.5
MAX_SEQ = 8192
BASE_MODEL = "google/gemma-4-E4B-it"
MERGED_DIR = Path("/var/tmp/tarl/merged-policy")
PROBE_PROMPT = "List the files in a directory, sorted by size."
TB_PATH = "/tmp/tb2"
JOBS_DIR = REPO / "jobs-g4"

LOG = lambda *a: print("[g4]", *a, flush=True)  # noqa: E731


def sh(cmd: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)


# ---- serve lifecycle --------------------------------------------------------
def serve_healthy() -> bool:
    r = sh("curl -sf localhost:8000/v1/models", timeout=15)
    return r.returncode == 0


def restart_serve(model_dir: str | None) -> None:
    """model_dir=None -> FP8 base checkpoint; else merged bf16 with online FP8."""
    sh("tmux kill-session -t serve 2>/dev/null; sleep 2")
    extra = ""
    if model_dir:
        extra = f"TARL_SERVE_MODEL={model_dir} TARL_SERVE_QUANT=fp8"
    sh(
        "tmux new-session -d -s serve "
        f"\"cd {REPO} && TARL_THINKING=1 {extra} bash scripts/01_serve.sh 2>&1 "
        f"| tee {REPO}/serve.log\""
    )
    for _ in range(120):
        if serve_healthy():
            return
        time.sleep(5)
    raise RuntimeError("serve did not become healthy after restart")


def probe_completion() -> str:
    r = sh(
        "curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '"
        + json.dumps({
            "model": "gemma-4-E4B-it",
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 64, "temperature": 0.0,
        }) + "'", timeout=120)
    try:
        return json.loads(r.stdout)["choices"][0]["message"]["content"]
    except Exception:
        return f"<probe failed: {r.stdout[:100]}>"


# ---- rollouts ---------------------------------------------------------------
def rollout(step: int) -> list[dict]:
    inc = " ".join(f"-i {t}" for t in TASKS)
    r = sh(
        f"cd {REPO} && . scripts/env.sh >/dev/null 2>&1 && "
        f"OPENAI_API_KEY=dummy harbor run -p {TB_PATH} -a terminus-2 "
        f"-m openai/gemma-4-E4B-it --ak api_base=http://localhost:8000/v1 "
        f"--ak temperature=1.2 -e singularity "
        f"--ek singularity_image_cache_dir=$TARL_SIF_CACHE "
        f"--ek singularity_no_mount=home,tmp "
        f"-k {G} --n-concurrent 4 --timeout-multiplier 1.0 "
        f"--jobs-dir {JOBS_DIR} {inc}", timeout=5400)
    jobs = sorted(JOBS_DIR.glob("2*"), key=lambda p: p.stat().st_mtime)
    if not jobs:
        raise RuntimeError(f"no rollout job dir; harbor stderr: {r.stderr[-500:]}")
    job = jobs[-1]
    out = []
    for rj in job.glob("*/result.json"):
        t = json.loads(rj.read_text())
        if (t.get("exception_info") or {}).get("exception_type"):
            continue
        traj = rj.parent / "agent" / "trajectory.json"
        if not traj.exists():
            continue
        reward = ((t.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0)
        out.append({
            "task": rj.parent.name.rsplit("__", 1)[0],
            "trajectory": str(traj),
            "verifier": float(reward or 0.0),
        })
    LOG(f"step {step}: {len(out)} clean rollouts from {job.name}")
    return out


# ---- rewards ----------------------------------------------------------------
def score_rollouts(rollouts: list[dict], judge: JudgeClient, rubric: str) -> None:
    cfg = RewardConfig(lambda_judge=LAMBDA_JUDGE)
    for ro in rollouts:
        passed = ro["verifier"] >= 1.0
        jscore = None
        if not passed:
            try:
                jscore = judge.judge(rubric, compress(load_trajectory(ro["trajectory"]))).score
            except Exception as e:
                LOG(f"  judge failed ({e}); verifier-only for this rollout")
        ro["judge"] = jscore
        ro["R"] = composite_reward(passed, jscore, cfg)


# ---- tokenization + masks ---------------------------------------------------
def build_example(tok, traj_path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(input_ids, agent_mask) — mask 1 only on tokens of agent-authored turns."""
    steps = load_trajectory(traj_path).get("steps", [])
    if not steps:
        return None
    ids: list[int] = []
    mask: list[int] = []
    convo: list[dict] = []
    for s in steps:
        role = "assistant" if s.get("source") == "agent" else "user"
        convo.append({"role": role, "content": str(s.get("message", ""))})
    # tokenize incrementally so agent spans are exactly delimited
    prev_len = 0
    for i in range(len(convo)):
        text = tok.apply_chat_template(convo[: i + 1], tokenize=False, add_generation_prompt=False)
        cur = tok(text, add_special_tokens=False)["input_ids"]
        span = len(cur) - prev_len
        if span <= 0:
            continue
        flag = 1 if convo[i]["role"] == "assistant" else 0
        ids = cur
        mask.extend([flag] * span)
        prev_len = len(cur)
    ids_t = torch.tensor(ids[:MAX_SEQ])
    mask_t = torch.tensor(mask[: len(ids_t)])
    if mask_t.sum() == 0:
        return None
    return ids_t, mask_t


# ---- trainer ----------------------------------------------------------------
def load_trainable():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
    import transformers

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    cls = getattr(transformers, cfg.architectures[0])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = cls.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                device_map={"": 0}, dtype=torch.bfloat16)
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    model = get_peft_model(model, lcfg)
    return tok, model


def token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=ids.unsqueeze(0).to(model.device)).logits.float()
    logp = torch.log_softmax(out[0, :-1], dim=-1)
    return logp.gather(1, ids[1:].unsqueeze(1).to(model.device)).squeeze(1)


def grpo_step(model, opt, examples: list[tuple[torch.Tensor, torch.Tensor, float]]) -> dict:
    """examples: (ids, agent_mask, advantage). Token-normalized clipped loss."""
    # old logprobs (no grad) — on-policy single inner epoch, ratio starts at 1
    old = []
    model.eval()
    with torch.no_grad():
        for ids, _, _ in examples:
            old.append(token_logprobs(model, ids))
    model.train()
    total_loss, total_tok = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for (ids, m, adv), old_lp in zip(examples, old):
        new_lp = token_logprobs(model, ids)
        mask = m[1:].to(model.device).float()
        ratio = (new_lp - old_lp).exp()
        clipped = torch.clamp(ratio, 1 - CLIP_LO, 1 + CLIP_HI)
        per_tok = -torch.min(ratio * adv, clipped * adv) * mask
        loss = per_tok.sum()
        loss.backward()
        total_loss += float(loss.detach())
        total_tok += int(mask.sum())
    for p in model.parameters():  # token-level normalization across the batch
        if p.grad is not None:
            p.grad /= max(1, total_tok)
    gnorm = torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 0.1)
    opt.step()
    return {"loss/tok": total_loss / max(1, total_tok), "grad_norm": float(gnorm),
            "tokens": total_tok}


def zero_grad_leak_check(model) -> None:
    leaks = [n for n, p in model.named_parameters()
             if p.grad is not None and "lora" not in n and float(p.grad.abs().sum()) > 0]
    assert not leaks, f"gradient leak into base/env params: {leaks[:5]}"


def merge_and_export(model, tok) -> str:
    from peft import PeftModel  # noqa: F401  (merge on the live 4-bit model is lossy;
    # export adapter, merge into bf16 on CPU (G1-verified path), save merged)
    adapter_dir = "/var/tmp/tarl/g4-adapter"
    model.save_pretrained(adapter_dir)
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    cls = getattr(transformers, cfg.architectures[0])
    base = cls.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    from peft import PeftModel as PM
    merged = PM.from_pretrained(base, adapter_dir).merge_and_unload()
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_DIR)
    del base, merged
    h = hashlib.sha256()
    for f in sorted(MERGED_DIR.glob("*.safetensors")):
        h.update(f.read_bytes()[:1_000_000])
    return h.hexdigest()[:12]


# ---- main -------------------------------------------------------------------
def main() -> int:
    rubric, rubric_hash = load_rubric()
    judge = JudgeClient()
    LOG(f"rubric {rubric_hash}, judge {judge.model}, tasks {TASKS}, G={G}, steps={STEPS}")

    if not serve_healthy():
        LOG("serve down -> starting base")
        restart_serve(None)

    tok, model = load_trainable()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    prev_hash, prev_probe = None, probe_completion()
    for step in range(1, STEPS + 1):
        t0 = time.time()
        ros = rollout(step)
        score_rollouts(ros, judge, rubric)

        # Dr.GRPO advantages per task group (mean-only, no ÷std)
        examples = []
        for task in TASKS:
            grp = [r for r in ros if r["task"] == task]
            if not grp:
                continue
            mean_r = sum(r["R"] for r in grp) / len(grp)
            for r in grp:
                adv = r["R"] - mean_r
                LOG(f"  {task} R={r['R']:.3f} (verifier={r['verifier']}, judge={r['judge']}) adv={adv:+.3f}")
                if abs(adv) < 1e-6:
                    continue  # dynamic-sampling spirit: drop zero-signal rollouts
                ex = build_example(tok, r["trajectory"])
                if ex:
                    examples.append((ex[0], ex[1], adv))
        if not examples:
            LOG(f"step {step}: no informative rollouts (all groups tied) — continuing")
            continue

        stats = grpo_step(model, opt, examples)
        zero_grad_leak_check(model)
        LOG(f"step {step}: UPDATE loss/tok={stats['loss/tok']:.5f} "
            f"grad_norm={stats['grad_norm']:.4f} tokens={stats['tokens']} "
            f"({time.time()-t0:.0f}s)")

        whash = merge_and_export(model, tok)
        assert whash != prev_hash, "merged weights identical to previous step"
        prev_hash = whash
        restart_serve(str(MERGED_DIR))
        probe = probe_completion()
        LOG(f"step {step}: SYNC weights={whash} probe_changed={probe != prev_probe}")
        prev_probe = probe

    LOG("G4_MICRO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

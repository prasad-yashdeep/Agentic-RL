"""One source of truth for the `harbor run` invocation (scripts/04_baseline_eval.sh and
scripts/06_grpo_micro.py:107-115 both spelled it inline) plus the vLLM serve lifecycle."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

from tarl.common.paths import JOBS, JOBS_G4, REPO, SIF_CACHE

SERVE_BASE = "http://localhost:8000"
SERVED_MODEL = "gemma-4-E4B-it"
PROBE_PROMPT = "List the files in a directory, sorted by size."


def resolve_sif_cache(sif_cache: str | Path | None = None) -> str:
    """Concrete image-cache path for Harbor. Never a shell variable: format_cmd single-quotes arguments, so a
    literal `$TARL_SIF_CACHE` reached harbor unexpanded and it cached images into ./$TARL_SIF_CACHE (on NFS)."""
    path = str(sif_cache) if sif_cache else os.environ.get("TARL_SIF_CACHE") or str(SIF_CACHE)
    if "$" in path:
        raise ValueError(f"unexpanded shell variable in image cache path: {path!r}")
    return path


def build_harbor_cmd(
    *,
    tb_path: str | Path,
    tasks: Sequence[str] | None,
    k: int,
    jobs_dir: str | Path,
    temperature: float = 1.2,
    n_concurrent: int = 4,
    timeout_multiplier: float = 1.0,
    api_base: str = SERVE_BASE + "/v1",
    model: str = "openai/gemma-4-E4B-it",
    agent: str = "terminus-2",
    sif_cache: str | Path | None = None,
    collect_rollout_details: bool = False,
    max_retries: int | None = None,
    retry_exclude: Sequence[str] | None = None,
) -> list[str]:
    cmd = [
        "harbor", "run", "-p", str(tb_path), "-a", agent, "-m", model,
        "--ak", f"api_base={api_base}", "--ak", f"temperature={temperature}",
        "-e", "singularity",
        "--ek", f"singularity_image_cache_dir={resolve_sif_cache(sif_cache)}",
        "--ek", "singularity_no_mount=home,tmp",
        "-k", str(k), "--n-concurrent", str(n_concurrent), "--timeout-multiplier", str(timeout_multiplier),
        "--jobs-dir", str(jobs_dir),
    ]
    if collect_rollout_details:
        cmd += ["--ak", "collect_rollout_details=true"]
    if max_retries is not None:
        cmd += ["--max-retries", str(max_retries)]
    if retry_exclude:
        cmd += ["--retry-exclude", ",".join(retry_exclude)]
    for t in tasks or []:
        cmd += ["-i", t]
    return cmd


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(c if c.startswith("$") else shlex.quote(c) for c in cmd)


def serve_healthy(base: str = SERVE_BASE, timeout_s: float = 5.0) -> bool:
    try:
        import httpx

        r = httpx.get(base + "/v1/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def restart_serve(model_dir: str | None, *, thinking: bool = True, wait_s: int = 600, base: str = SERVE_BASE) -> None:
    """tmux session `serve` <- scripts/01_serve.sh (06:72-87). node08 only."""
    # watch.sh reads these two files when it has to relaunch a dead serve, so a crash mid-run never
    # silently reverts the policy to the base model.
    (REPO / ".serve-thinking").write_text("1\n" if thinking else "0\n")
    (REPO / ".serve-model").write_text((model_dir or "") + "\n")
    subprocess.run(["bash", "-lc", "tmux kill-session -t serve 2>/dev/null; sleep 2"], check=False)
    extra = f"TARL_SERVE_MODEL={model_dir} TARL_SERVE_QUANT=fp8 " if model_dir else ""
    launch = (
        f"tmux new-session -d -s serve \"cd {REPO} && TARL_THINKING={1 if thinking else 0} {extra}"
        f"bash scripts/01_serve.sh 2>&1 | tee {REPO}/serve.log\""
    )
    subprocess.run(["bash", "-lc", launch], check=False)
    for _ in range(max(1, wait_s // 5)):
        if serve_healthy(base):
            return
        time.sleep(5)
    raise RuntimeError("serve did not become healthy after restart")


def probe_completion(prompt: str = PROBE_PROMPT, *, base: str = SERVE_BASE, model: str = SERVED_MODEL) -> str:
    try:
        import httpx

        r = httpx.post(
            base + "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0.0},
            timeout=120,
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # pragma: no cover
        return f"<probe failed: {e}>"


def prompt_logprobs(text: str, *, base: str = SERVE_BASE, model: str = SERVED_MODEL) -> dict:
    """vLLM /v1/completions with prompt_logprobs=0: token count + per-token logprobs of `text`."""
    import httpx

    r = httpx.post(
        base + "/v1/completions",
        json={"model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0, "echo": False},
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    plp = d["choices"][0].get("prompt_logprobs") or []
    toks = [None if x is None else next(iter(x.values())) for x in plp]
    return {"n_prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "logprobs": [None if t is None else t.get("logprob") for t in toks], "tokens": [None if t is None else t.get("decoded_token") for t in toks]}


def resolve_trial(arg: str, *, job: str | Path | None = None, search: Sequence[Path] = (JOBS, JOBS_G4)) -> Path:
    """A trial dir, `job/trial`, or a task name (newest matching trial across the job roots)."""
    p = Path(arg)
    if p.is_dir() and (p / "result.json").exists() or (p / "agent").is_dir():
        return p
    if job is not None:
        jp = Path(job)
        hits = sorted(jp.glob(f"{arg}__*"))
        if hits:
            return hits[-1]
        raise FileNotFoundError(f"no trial {arg}__* under {jp}")
    candidates: list[Path] = []
    for root in search:
        if not Path(root).exists():
            continue
        for jd in sorted(Path(root).glob("2*")):
            candidates += sorted(jd.glob(f"{arg}__*"))
    if not candidates:
        raise FileNotFoundError(f"no trial directory or task named {arg!r} under {[str(s) for s in search]}")
    return max(candidates, key=lambda d: d.stat().st_mtime)

"""configs/judge/judge.yaml -> JudgeConfig, and the judge factory (live | replay | mock)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from tarl.common.paths import G4_LOG, JOBS_G4, JUDGE_YAML, PILOT_JSONL, REPO, SECRETS_ENV
from tarl.judge.client import JudgeClient, JudgeError
from tarl.judge.cost import CostCappedJudge, CostMeter
from tarl.judge.protocol import Judge

JudgeMode = Literal["live", "replay", "mock"]


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "xai"
    api_base: str = "https://api.x.ai/v1"
    model: str = "grok-4.5"
    temperature: float = 0.0
    env_key: str = "XAI_API_KEY"
    rubric: Path = REPO / "configs" / "judge" / "rubric-v1.md"
    max_input_tokens: int = 15000
    cost_cap_usd_per_run: float | None = 500.0
    price_in_per_mtok: float = 5.0
    price_out_per_mtok: float = 25.0
    fallback: dict = field(default_factory=dict)


def load_judge_config(path: str | Path = JUDGE_YAML) -> JudgeConfig:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    prices = d.get("price_usd_per_mtok") or {}
    rubric = Path(d.get("rubric", "configs/judge/rubric-v1.md"))
    if not rubric.is_absolute():
        rubric = REPO / rubric
    return JudgeConfig(
        provider=str(d.get("provider", "xai")),
        api_base=str(d.get("api_base", "https://api.x.ai/v1")),
        model=str(d.get("model", "grok-4.5")),
        temperature=float(d.get("temperature", 0.0)),
        env_key=str(d.get("env_key", "XAI_API_KEY")),
        rubric=rubric,
        max_input_tokens=int(d.get("max_input_tokens", 15000)),
        cost_cap_usd_per_run=(float(d["cost_cap_usd_per_run"]) if d.get("cost_cap_usd_per_run") is not None else None),
        price_in_per_mtok=float(prices.get("input", 5.0)),
        price_out_per_mtok=float(prices.get("output", 25.0)),
        fallback=dict(d.get("fallback") or {}),
    )


def source_secrets(path: Path | None = None, *, overwrite: bool = False) -> list[str]:
    """Load KEY=VALUE / export KEY=VALUE lines into os.environ. Returns the keys set (never values)."""
    path = Path(path) if path is not None else SECRETS_ENV
    if not Path(path).exists():
        return []
    keys: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if overwrite or k not in os.environ:
            os.environ[k] = v
            keys.append(k)
    return keys


def make_judge(
    mode: JudgeMode,
    cfg: JudgeConfig | None = None,
    *,
    pilot_jsonl: Path = PILOT_JSONL,
    g4_log: Path = G4_LOG,
    jobs_g4: Path = JOBS_G4,
    on_miss: str = "error",
    cost_cap_usd: float | None = None,
    allow_spend: bool = False,
) -> Judge:
    cfg = cfg or load_judge_config()
    if mode == "mock":
        from tarl.judge.mock import MockJudge

        return MockJudge()
    if mode == "replay":
        from tarl.judge.replay import ReplayJudge

        return ReplayJudge.from_artifacts(pilot_jsonl=pilot_jsonl, g4_log=g4_log, jobs_g4=jobs_g4, on_miss=on_miss)
    if mode != "live":
        raise ValueError(f"unknown judge mode {mode!r}")
    if not allow_spend:
        raise JudgeError("live judge calls cost money: pass allow_spend=True (CLI: --yes-spend)")
    if not os.environ.get(cfg.env_key):
        source_secrets()
    if not os.environ.get(cfg.env_key):
        raise JudgeError(f"{cfg.env_key} not set and {SECRETS_ENV} has no entry — live judge unavailable on this host")
    inner = JudgeClient(api_base=cfg.api_base, model=cfg.model, env_key=cfg.env_key, temperature=cfg.temperature)
    cap = cost_cap_usd if cost_cap_usd is not None else cfg.cost_cap_usd_per_run
    return CostCappedJudge(inner, CostMeter(cfg.price_in_per_mtok, cfg.price_out_per_mtok, cap))

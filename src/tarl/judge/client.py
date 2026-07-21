"""Judge API client. Primary: xAI (OpenAI-compatible). Fallback: any
OpenAI-compatible endpoint (configs/judge/judge.yaml `fallback`).

Deliberately minimal: one blocking call per trajectory, JSON-object response,
retries with backoff on transient failures, token/cost accounting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class JudgeVerdict:
    score: float
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict = field(repr=False, default_factory=dict)


class JudgeError(RuntimeError):
    pass


class JudgeClient:
    def __init__(
        self,
        api_base: str = "https://api.x.ai/v1",
        model: str = "grok-4.5",
        env_key: str = "XAI_API_KEY",
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        key = os.environ.get(env_key)
        if not key:
            raise JudgeError(f"{env_key} not set (source ~/.config/tarl/env)")
        self._client = httpx.Client(
            base_url=api_base,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeVerdict:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post("/chat/completions", json=payload)
                if r.status_code in (429, 500, 502, 503):
                    raise JudgeError(f"transient {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                score = float(parsed["score"])
                if not (0.0 <= score <= 1.0):
                    raise JudgeError(f"score out of range: {score}")
                usage = data.get("usage", {})
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.n_calls += 1
                return JudgeVerdict(
                    score=score,
                    rationale=str(parsed.get("rationale", "")),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    raw=data,
                )
            except (httpx.HTTPError, JudgeError, KeyError, ValueError, json.JSONDecodeError) as e:
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        raise JudgeError(f"judge call failed after {self.max_retries} attempts: {last_exc}")

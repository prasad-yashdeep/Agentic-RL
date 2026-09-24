"""Policies: text in, text out, stateless (lecture 1:14). All return the sampled token ids and their log-probs so the
harness can keep the trajectory token-exact (no re-tokenisation of the model's own words).

- VLLMPolicy   the diagram's "POLICY πθ, vLLM generated": OpenAI-compatible /v1/completions on token-id prompts,
               with the served LoRA adapter selected by model name after each weight sync
- HFPolicy     in-process transformers generate (tests, CPU smoke runs, the phase-1 style)
- ScriptedPolicy  replays canned replies (the reference's oracle trick; off-policy, log-probs None)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Sampling:
    # Default = plain sampling from π (temperature 1, no top-p/top-k/repetition penalty): the update treats rollouts
    # as samples of π, and the trainer-vs-vLLM mismatch metric compares raw log-probs, so any sampling distortion would
    # be invisible to it (review finding 2026-09-24). REFERENCE below mirrors the phase-1 notebook (top_p 0.95 plus
    # Qwen2.5-Coder's generation_config top_k 20 / repetition_penalty 1.05, which HF generate applied implicitly).
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    # A seed makes an episode reproducible. VLLMPolicy derives a distinct seed per call (turn): vLLM seeds a fresh
    # generator per request, so one seed for every turn would reuse the same random draw for token j of every turn
    # and correlate the turns of an episode (review finding 2026-09-24; runs before this fix used one seed per episode).
    seed: int | None = None


REFERENCE_SAMPLING = Sampling(temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.05)


@dataclass
class Generation:
    token_ids: list[int]
    logprobs: list[float | None]
    text: str
    finish_reason: str = "stop"
    meta: dict = field(default_factory=dict)


class VLLMPolicy:
    def __init__(self, base_url: str = "http://localhost:8001", model: str = "policy", sampling: Sampling | None = None,
                 timeout_s: float = 900.0, stop_token_ids: Sequence[int] = (151645, 151643)):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling = sampling or Sampling()
        self.timeout_s = timeout_s
        self.stop_token_ids = list(stop_token_ids)
        self.n_calls = 0  # per-turn seed offset (see Sampling.seed)

    def healthy(self) -> bool:
        import httpx

        try:
            return httpx.get(self.base_url + "/v1/models", timeout=5).status_code == 200
        except Exception:
            return False

    def generate(self, prompt_ids: Sequence[int], max_tokens: int) -> Generation:
        import httpx

        s = self.sampling
        body = {
            "model": self.model, "prompt": list(prompt_ids), "max_tokens": int(max_tokens),
            "temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k, "repetition_penalty": s.repetition_penalty,
            "logprobs": 1, "return_token_ids": True, "return_tokens_as_token_ids": True,
            "skip_special_tokens": False, "stop_token_ids": self.stop_token_ids,
        }
        if s.seed is not None:
            body["seed"] = (int(s.seed) + 1_000_003 * self.n_calls) % (2**63 - 1)
        self.n_calls += 1
        r = httpx.post(self.base_url + "/v1/completions", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        c = r.json()["choices"][0]
        lp = c.get("logprobs") or {}
        ids = c.get("token_ids")
        if ids is None:  # fall back to "token_id:123" strings from return_tokens_as_token_ids
            ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens", [])]
        lps = list(lp.get("token_logprobs") or [None] * len(ids))
        return Generation(list(ids), lps, c.get("text", ""), c.get("finish_reason") or "stop")


class HFPolicy:
    def __init__(self, model, tokenizer, sampling: Sampling | None = None, stop_token_ids: Sequence[int] = (151645, 151643)):
        self.model, self.tok = model, tokenizer
        self.sampling = sampling or Sampling()
        self.stop_token_ids = list(stop_token_ids)

    def generate(self, prompt_ids: Sequence[int], max_tokens: int) -> Generation:
        import torch

        s = self.sampling
        dev = next(self.model.parameters()).device
        ip = torch.tensor([list(prompt_ids)], device=dev)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ip, attention_mask=torch.ones_like(ip), max_new_tokens=int(max_tokens),
                do_sample=s.temperature > 0, temperature=max(s.temperature, 1e-5), top_p=s.top_p, top_k=s.top_k if s.top_k > 0 else 0,
                repetition_penalty=s.repetition_penalty, eos_token_id=self.stop_token_ids,
                pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_token_ids[-1],
                output_logits=True, return_dict_in_generate=True,
            )
        ids = out.sequences[0, ip.shape[1]:].tolist()
        lps = []
        for i, t in enumerate(ids):  # raw (pre-sampling-processor) log-probs, like vLLM's default raw_logprobs
            lps.append(float(torch.log_softmax(out.logits[i][0].float(), -1)[t]))
        finish = "stop" if ids and ids[-1] in self.stop_token_ids else "length"
        return Generation(ids, lps, self.tok.decode(ids, skip_special_tokens=True), finish)


class ScriptedPolicy:
    """Replays fixed replies (the reference's `scripted_sampler`): the oracle rollout that keeps a group from being
    all-zero. Its tokens are trained on with a positive advantage but were never sampled from the policy, so it is
    off-policy data (log-probs None) — labelled as such in every record."""

    def __init__(self, tokenizer, replies: Sequence[str], fallback: str | None = None, im_end_id: int = 151645):
        self.tok = tokenizer
        self.replies = list(replies)
        self.fallback = fallback
        self.im_end_id = im_end_id
        self.i = 0

    def generate(self, prompt_ids: Sequence[int], max_tokens: int) -> Generation:
        text = self.replies[self.i] if self.i < len(self.replies) else (self.fallback or self.replies[-1])
        self.i += 1
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        finish = "stop"
        if len(ids) > max(0, max_tokens - 1):  # keep what is executed identical to what is trained on
            ids = ids[: max(0, max_tokens - 1)]
            text = self.tok.decode(ids)
            finish = "length"
        ids = ids + [self.im_end_id]
        return Generation(ids, [None] * len(ids), text, finish, meta={"scripted": True})

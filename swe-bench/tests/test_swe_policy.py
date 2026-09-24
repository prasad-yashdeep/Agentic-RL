"""VLLMPolicy request bodies (no server: httpx.post is replaced)."""

import httpx

from sarl.policy import Sampling, VLLMPolicy


def test_seed_differs_per_turn_and_pairs_across_policies(monkeypatch):
    seen = []

    class _R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"token_ids": [1, 151645], "logprobs": {"token_logprobs": [-0.1, -0.2]}, "text": "x"}]}

    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: seen.append(json["seed"]) or _R())
    a, b = VLLMPolicy(sampling=Sampling(seed=1003)), VLLMPolicy(sampling=Sampling(seed=1003))
    for _ in range(3):
        a.generate([1, 2], 8)
    for _ in range(3):
        b.generate([1, 2], 8)
    assert len(set(seen[:3])) == 3  # a fresh draw per turn (vLLM re-seeds per request)
    assert seen[:3] == seen[3:]  # the same episode seed gives the same per-turn seeds: arms stay paired
    unseeded = []
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: unseeded.append("seed" in json) or _R())
    VLLMPolicy().generate([1], 4)
    assert unseeded == [False]

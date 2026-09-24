"""GRPO math and the LoRA trainer on a tiny random Qwen2 (CPU, no download)."""

import numpy as np
import pytest
import torch

from sarl.grpo import LoRATrainer, TrainConfig, group_advantages, is_degenerate, mismatch_stats


def test_advantages_match_reference_formula():
    r = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    a = group_advantages(r)
    ref = (torch.tensor(r) - torch.tensor(r).mean()) / (torch.tensor(r).std(unbiased=False) + 1e-4)
    assert np.allclose(a, ref.numpy(), atol=1e-6)
    assert a[-1] == pytest.approx(2.6454, abs=1e-3) and a[0] == pytest.approx(-0.3779, abs=1e-3)
    assert is_degenerate([0.3] * 6) and np.allclose(group_advantages([0.3] * 6), 0.0)


@pytest.mark.upstream
def test_advantages_equal_upstream_code(upstream):
    """Same numbers as the reference's inline formula (swe_grpo_one_step.grpo_step)."""
    rng = np.random.default_rng(0)
    r = rng.random(8).round(3)
    rewards = torch.tensor(r, dtype=torch.float)
    ref = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    assert np.allclose(group_advantages(r), ref.numpy(), atol=1e-5)


def _batch(n=3, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        ids = torch.randint(0, 64, (T,), generator=g).tolist()
        mask = [0] * 10 + [1] * 12 + [0] * 8 + [1] * 10
        out.append((ids, mask))
    return out


def test_trainer_logprobs_match_full_softmax(tiny_qwen):
    tr = LoRATrainer("tiny", model=tiny_qwen, cfg=TrainConfig(chunk=5, grad_checkpointing=False), device="cpu")
    ids, mask = _batch(1)[0]
    lp = tr.logprobs_nograd(ids, mask)
    with torch.no_grad():
        logits = tr.model(input_ids=torch.tensor([ids])).logits[0].float()
    full = torch.log_softmax(logits[:-1], -1).gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
    sel = full[torch.tensor(mask[1:], dtype=torch.bool)].numpy()
    assert lp.shape == sel.shape and np.allclose(lp, sel, atol=1e-5)


def test_update_moves_objective_and_only_lora(tiny_qwen):
    tr = LoRATrainer("tiny", model=tiny_qwen, cfg=TrainConfig(lr=5e-2, lora_r=4, lora_alpha=8), device="cpu")
    frozen = {n: p.detach().clone() for n, p in tr.model.named_parameters() if not p.requires_grad}
    b = _batch(4)
    adv = [1.5, -0.5, -0.5, -0.5]
    before = [tr.logprobs_nograd(i, m).mean() for i, m in b]
    h0 = tr.adapter_hash()
    out = tr.step([(i, m, a) for (i, m), a in zip(b, adv)])
    after = [tr.logprobs_nograd(i, m).mean() for i, m in b]
    assert out["stepped"] and np.isfinite(out["loss"]) and out["tokens"] == 4 * 22
    assert tr.adapter_hash() != h0
    assert sum(a * (x - y) for a, x, y in zip(adv, after, before)) > 0  # the surrogate went up
    assert all(torch.equal(p, frozen[n]) for n, p in tr.model.named_parameters() if not p.requires_grad)


def test_degenerate_group_is_a_noop(tiny_qwen):
    """A = 0 everywhere ⇒ zero gradients ⇒ LoRA B stays 0 ⇒ every log-prob unchanged (the reference's degenerate case)."""
    tr = LoRATrainer("tiny", model=tiny_qwen, cfg=TrainConfig(lr=1e-2), device="cpu")
    b = _batch(3, seed=1)
    before = [tr.logprobs_nograd(i, m) for i, m in b]
    tr.step([(i, m, 0.0) for i, m in b])
    after = [tr.logprobs_nograd(i, m) for i, m in b]
    assert all(np.array_equal(x, y) for x, y in zip(before, after))


def test_mismatch_stats():
    s = mismatch_stats(np.array([-1.0, -2.0, -0.5]), [-1.1, None, -0.5])
    assert s["n"] == 2 and s["max"] == pytest.approx(0.1) and s["frac_gt_0.5"] == 0.0


def test_tis_and_token_mean(tiny_qwen):
    """Truncated importance sampling: sampler log-probs equal to the trainer's give weight 1 (same update as without
    TIS); a sampler that under-rates the tokens gets its weight capped. token_mean normalises by all batch tokens."""
    b = _batch(2, seed=3)
    adv = [1.0, -1.0]
    tr = LoRATrainer("tiny", model=tiny_qwen, cfg=TrainConfig(lr=1e-2, lora_r=4, lora_alpha=8, tis_cap=2.0), device="cpu")
    same = []
    for i, m in b:
        t = tr.logprobs_nograd(i, m)
        s = [None] * len(i)
        for k, pos in enumerate([p for p in range(1, len(i)) if m[p]]):
            s[pos] = float(t[k])
        same.append(s)
    low = [[None if v is None else v - 5.0 for v in s] for s in same]  # vLLM gave these tokens much lower probability
    out = tr.step([(i, m, a, s) for (i, m), a, s in zip(b, adv, same)])
    assert out["stepped"] and abs(out["tis"]["mean_w"] - 1.0) < 1e-4 and out["tis"]["frac_capped"] == 0.0
    assert out["mismatch"]["n"] == 2 * 22 and out["mismatch"]["max"] < 1e-4  # measured in the update's own forward pass
    out2 = tr.step([(i, m, a, s) for (i, m), a, s in zip(b, adv, low)])
    assert out2["tis"]["frac_capped"] == 1.0 and abs(out2["tis"]["mean_w"] - 2.0) < 1e-6
    tm = LoRATrainer("tiny", model=tiny_qwen, cfg=TrainConfig(lr=1e-2, loss_agg="token_mean"), device="cpu")
    out3 = tm.step([(i, m, a) for (i, m), a in zip(b, adv)])
    assert out3["stepped"] and np.isfinite(out3["loss"])

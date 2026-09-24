"""Token-level clipped surrogate and the QLoRA update step (extraction of 06:210-254).

    L = − Σ_t mask_t · min( ρ_t·A, clip(ρ_t, 1−ε_lo, 1+ε_hi)·A ),   ρ_t = exp(logπ_θ − logπ_old)

then divided by the total number of trained tokens in the batch (DAPO token-level aggregation)
and gradient-clipped at 0.1. With a single inner epoch ρ_t = 1 at the update, so the gradient
equals the REINFORCE gradient A·∇log π on the assistant tokens.
"""

from __future__ import annotations

import torch


def clipped_surrogate(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: float | torch.Tensor,
    mask: torch.Tensor,
    clip_lo: float = 0.2,
    clip_hi: float = 0.28,
) -> tuple[torch.Tensor, dict]:
    """Sum over masked tokens of the clipped per-token loss (NOT yet token-normalised)."""
    ratio = (new_logp - old_logp).exp()
    clipped = torch.clamp(ratio, 1.0 - clip_lo, 1.0 + clip_hi)
    adv_t = torch.as_tensor(adv, dtype=ratio.dtype, device=ratio.device)
    m = mask.to(ratio.dtype)
    per_tok = -torch.min(ratio * adv_t, clipped * adv_t) * m
    loss = per_tok.sum()
    with torch.no_grad():
        sel = m.bool()
        n = int(sel.sum())
        if n:
            clipped_frac = float(((ratio < 1.0 - clip_lo) | (ratio > 1.0 + clip_hi))[sel].float().mean())
            ratio_mean = float(ratio[sel].mean())
        else:
            clipped_frac, ratio_mean = 0.0, 1.0
    return loss, {"n_tok": n, "clip_frac": clipped_frac, "ratio_mean": ratio_mean}


def token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    """Per-token logprob of the realized ids WITHOUT materializing a full-vocab
    fp32 copy: logp_t = logit[target] - logsumexp(logits_row), all in the model dtype."""
    device = getattr(model, "device", None) or next(model.parameters()).device
    logits = model(input_ids=ids.unsqueeze(0).to(device)).logits[0, :-1]
    targets = ids[1:].unsqueeze(1).to(device)
    target_logit = logits.gather(1, targets).squeeze(1)
    lse = torch.logsumexp(logits, dim=-1)
    return (target_logit - lse).float()


def grpo_step(
    model,
    opt,
    examples: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    clip_lo: float = 0.2,
    clip_hi: float = 0.28,
    max_grad_norm: float = 0.1,
) -> dict:
    """examples: (ids, agent_mask, advantage). One optimizer step, token-normalised across the batch."""
    old = []
    model.eval()
    with torch.no_grad():
        for ids, _, _ in examples:
            old.append(token_logprobs(model, ids))
    model.train()
    total_loss, total_tok, clip_fracs = 0.0, 0, []
    opt.zero_grad(set_to_none=True)
    for (ids, m, adv), old_lp in zip(examples, old):
        new_lp = token_logprobs(model, ids)
        mask = m[1:].to(new_lp.device)
        loss, st = clipped_surrogate(new_lp, old_lp, adv, mask, clip_lo, clip_hi)
        loss.backward()
        total_loss += float(loss.detach())
        total_tok += st["n_tok"]
        clip_fracs.append(st["clip_frac"])
    for p in model.parameters():  # token-level normalization across the batch
        if p.grad is not None:
            p.grad /= max(1, total_tok)
    gnorm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_grad_norm)
    opt.step()
    return {
        "loss/tok": total_loss / max(1, total_tok),
        "grad_norm": float(gnorm),
        "tokens": total_tok,
        "clip_frac": (sum(clip_fracs) / len(clip_fracs)) if clip_fracs else 0.0,
        "n_examples": len(examples),
    }


def zero_grad_leak_check(model) -> None:
    leaks = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and "lora" not in n and float(p.grad.abs().sum()) > 0
    ]
    assert not leaks, f"gradient leak into base/env params: {leaks[:5]}"

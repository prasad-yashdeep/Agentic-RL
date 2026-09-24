"""GRPO math and the LoRA trainer (the diagram's OUTER LOOP: trajectory + reward → grouped advantage → update → πθ').

Advantages (reference, lecture 1:46):   A_i = (r_i − mean(r)) / (std(r) + ε),  std biased (÷G), ε = 1e-4.
Degenerate groups (all rewards equal) carry no signal; the lecture throws them out (2:00); the reference code keeps
them (every A_i = 0, a no-op). We skip them by default and count them.

Loss (reference, "REINFORCE with a group baseline"): per trajectory the MEAN log-prob of its policy tokens (mask 1),
    L = −(1/N) Σ_i A_i · mean_t log π(a_t | s_t),
one AdamW step per GRPO step, grad-norm clip 1.0, LoRA r=8/α=16 on q,k,v,o with the adapter in fp32 (lecture 1:47:
fp16 Adam second moments underflow). The per-token log-probs come from a chunked log-softmax over only the
supervised positions (a 16k-token trajectory's full logits would be 16k × 152k floats).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    return (r - r.mean()) / (r.std() + eps)  # np.std is the biased (÷n) estimator, like torch unbiased=False


def is_degenerate(rewards: Sequence[float], tol: float = 1e-8) -> bool:
    return float(np.std(np.asarray(rewards, dtype=np.float64))) < tol


@dataclass
class TrainConfig:
    lr: float = 1e-5
    lora_r: int = 8
    lora_alpha: int = 16
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    clip: float = 1.0
    max_train_tokens: int = 8192
    chunk: int = 1024
    grad_checkpointing: bool = True


class LoRATrainer:
    def __init__(self, model_id: str, revision: str | None = None, cfg: TrainConfig | None = None, device: str | None = None,
                 dtype: str = "bfloat16", model=None):
        import torch
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg or TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModelForCausalLM

            dt = getattr(torch, dtype) if self.device != "cpu" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dt, attn_implementation="sdpa")
        model.to(self.device)
        if self.cfg.grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        self.model = get_peft_model(model, LoraConfig(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=0.0,
                                                      bias="none", task_type="CAUSAL_LM", target_modules=list(self.cfg.targets)))
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr)
        self.n_trainable = sum(p.numel() for p in self.params)

    # log-probs ------------------------------------------------------------------------------------------------
    def _core(self):
        m = self.model.get_base_model()
        return m.model, m.lm_head

    def token_logprobs(self, ids: Sequence[int], mask: Sequence[int]):
        """log π(ids[t] | ids[<t]) for every t with mask[t] == 1 (a 1-D tensor, with grad if grad is enabled)."""
        import torch
        from torch.utils.checkpoint import checkpoint

        body, head = self._core()
        x = torch.tensor([list(ids)], device=self.device)
        m = torch.tensor(list(mask), device=self.device, dtype=torch.bool)
        pos = torch.nonzero(m[1:], as_tuple=False).squeeze(-1)  # hidden state at t predicts token t+1
        if pos.numel() == 0:
            return torch.zeros(0, device=self.device)
        hidden = body(input_ids=x).last_hidden_state[0]  # [T, H]
        h = hidden.index_select(0, pos)
        tgt = x[0, 1:].index_select(0, pos)

        def chunk_lp(hc, tc):
            logits = head(hc).float()
            return torch.log_softmax(logits, -1).gather(-1, tc.unsqueeze(-1)).squeeze(-1)

        outs = []
        for i in range(0, h.shape[0], self.cfg.chunk):
            hc, tc = h[i:i + self.cfg.chunk], tgt[i:i + self.cfg.chunk]
            outs.append(checkpoint(chunk_lp, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else chunk_lp(hc, tc))
        return torch.cat(outs)

    def logprobs_nograd(self, ids, mask) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            return self.token_logprobs(ids, mask).float().cpu().numpy()

    # the update -----------------------------------------------------------------------------------------------
    def step(self, batch: Sequence[tuple[Sequence[int], Sequence[int], float]]) -> dict:
        """batch = [(ids, mask, advantage)]; returns loss, grad-norm, token counts, mean log-p before the step."""
        import torch

        n = len(batch)
        if n == 0:
            return {"skipped": "empty_batch"}
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss_total, toks, mean_lps = 0.0, 0, []
        for ids, mask, adv in batch:
            lp = self.token_logprobs(ids, mask)
            if lp.numel() == 0:
                mean_lps.append(float("nan"))
                continue
            mean_lp = lp.mean()
            loss = -(float(adv) * mean_lp) / n
            loss.backward()
            loss_total += float(loss.detach())
            toks += int(lp.numel())
            mean_lps.append(float(mean_lp.detach()))
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.clip)
        finite = math.isfinite(float(gn)) and math.isfinite(loss_total)
        if finite:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {"loss": loss_total, "grad_norm": float(gn), "tokens": toks, "n_traj": n, "mean_logp_before": mean_lps, "stepped": finite}

    # weights --------------------------------------------------------------------------------------------------
    def adapter_hash(self) -> str:
        h = hashlib.sha256()
        for name, p in sorted(self.model.named_parameters()):
            if p.requires_grad:
                h.update(name.encode())
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    def save_adapter(self, path: str | Path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        return self.adapter_hash()


def mismatch_stats(trainer_lp: np.ndarray, sampler_lp: Sequence[float | None]) -> dict:
    """|log π_trainer − log π_vLLM| on the sampled tokens (lecture 2:31, 'train-inference mismatch')."""
    s = np.array([np.nan if v is None else v for v in sampler_lp], dtype=np.float64)
    t = np.asarray(trainer_lp, dtype=np.float64)
    n = min(len(s), len(t))
    d = np.abs(t[:n] - s[:n])
    d = d[~np.isnan(d)]
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "mean": float(d.mean()), "p50": float(np.median(d)), "p99": float(np.quantile(d, 0.99)),
            "max": float(d.max()), "frac_gt_0.5": float((d > 0.5).mean())}

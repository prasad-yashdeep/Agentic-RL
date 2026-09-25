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
    # "seq_mean" (reference): each trajectory's mean token log-prob, averaged over trajectories.
    # "token_mean" (DAPO/Dr.GRPO style): Σ over all supervised tokens in the batch / their count, so long trajectories
    # are not down-weighted per token.
    loss_agg: str = "seq_mean"
    # truncated importance sampling (Yao et al. 2025, "Your efficient RL framework secretly brings you off-policy RL"):
    # weight each sampled token by min(π_trainer/π_vLLM, cap) (detached) to correct the train-inference mismatch.
    # None = off (the reference). Needs the sampler's log-probs per token.
    tis_cap: float | None = None


class LoRATrainer:
    def __init__(self, model_id: str, revision: str | None = None, cfg: TrainConfig | None = None, device: str | None = None,
                 dtype: str = "bfloat16", model=None, offloaded: bool = False):
        """offloaded=True keeps the model (and later the optimizer state) on the CPU until onload(): for a trainer
        that shares its GPU with a sleeping vLLM server."""
        import torch
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg or TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModelForCausalLM

            dt = getattr(torch, dtype) if self.device != "cpu" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dt, attn_implementation="sdpa")
        self.offloaded = bool(offloaded) and self.device != "cpu"
        model.to("cpu" if self.offloaded else self.device)
        if self.cfg.grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        self.model = get_peft_model(model, LoraConfig(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=0.0,
                                                      bias="none", task_type="CAUSAL_LM", target_modules=list(self.cfg.targets)))
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr)
        self.n_trainable = sum(p.numel() for p in self.params)

    # placement ----------------------------------------------------------------------------------------------
    def _move(self, dev: str) -> None:
        import torch

        self.model.to(dev)
        for st in self.opt.state.values():  # AdamW moments live next to their parameters
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.dim() > 0:
                    st[k] = v.to(dev)

    def onload(self) -> None:
        if self.offloaded:
            self._move(self.device)
            self.offloaded = False

    def offload(self) -> None:
        """Model and optimizer state to the CPU and the CUDA cache released, so a vLLM server on this GPU can wake."""
        import torch

        if not self.offloaded and self.device != "cpu":
            self._move("cpu")
            self.offloaded = True
            torch.cuda.empty_cache()

    # log-probs ------------------------------------------------------------------------------------------------
    def _core(self):
        m = self.model.get_base_model()
        return m.model, m.lm_head

    def token_logprobs(self, ids: Sequence[int], mask: Sequence[int]):
        """log π(ids[t] | ids[<t]) for every t with mask[t] == 1 (a 1-D tensor, with grad if grad is enabled)."""
        import torch
        from torch.utils.checkpoint import checkpoint

        if self.offloaded:
            raise RuntimeError("trainer is offloaded to the CPU; call onload() first")
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
    def step(self, batch: Sequence[tuple]) -> dict:
        """batch = [(ids, mask, advantage)] or [(ids, mask, advantage, sampler_logprobs)], sampler_logprobs aligned
        with ids (None where not sampled). Returns loss, grad-norm, token counts, mean log-p before the step and,
        with tis_cap, the importance-weight statistics."""
        import torch

        n = len(batch)
        if n == 0:
            return {"skipped": "empty_batch"}
        agg, cap = self.cfg.loss_agg, self.cfg.tis_cap
        if agg not in ("seq_mean", "token_mean"):
            raise ValueError(f"unknown loss_agg {agg!r}")
        n_tok_total = sum(sum(1 for i in range(1, len(b[1])) if b[1][i]) for b in batch)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss_total, toks, mean_lps, w_all, d_all = 0.0, 0, [], [], []
        for item in batch:
            ids, mask, adv = item[0], item[1], item[2]
            lp = self.token_logprobs(ids, mask)
            if lp.numel() == 0:
                mean_lps.append(float("nan"))
                continue
            w = None
            if len(item) > 3 and item[3] is not None:  # the sampler's log-probs: mismatch stats, and TIS if enabled
                s = [item[3][i] for i in range(1, len(ids)) if mask[i]]
                s_t = torch.tensor([float("nan") if v is None else float(v) for v in s], device=lp.device)
                diff = lp.detach().float() - s_t
                d_all.append(diff.abs()[~torch.isnan(diff)].cpu())
                if cap is not None:
                    ratio = torch.exp(diff)
                    w = torch.where(torch.isnan(ratio), torch.ones_like(ratio), ratio.clamp(max=cap))
                    w_all.append(w.cpu())
            wlp = lp * w if w is not None else lp
            if agg == "seq_mean":
                loss = -(float(adv) * wlp.mean()) / n
            else:
                loss = -(float(adv) * wlp.sum()) / max(n_tok_total, 1)
            loss.backward()
            loss_total += float(loss.detach())
            toks += int(lp.numel())
            mean_lps.append(float(lp.detach().mean()))
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg.clip)
        finite = math.isfinite(float(gn)) and math.isfinite(loss_total)
        if finite:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        out = {"loss": loss_total, "grad_norm": float(gn), "tokens": toks, "n_traj": n, "mean_logp_before": mean_lps, "stepped": finite}
        if d_all and sum(d.numel() for d in d_all):
            d = torch.cat(d_all).double()
            out["mismatch"] = {"n": int(d.numel()), "mean": float(d.mean()), "p50": float(d.median()),
                               "p99": float(torch.quantile(d, 0.99)) if d.numel() < 16_000_000 else None,
                               "max": float(d.max()), "frac_gt_0.5": float((d > 0.5).double().mean())}
        if w_all:
            wc = torch.cat(w_all)
            out["tis"] = {"mean_w": float(wc.mean()), "frac_capped": float((wc >= cap - 1e-6).float().mean()), "min_w": float(wc.min())}
        return out

    # weights --------------------------------------------------------------------------------------------------
    def adapter_hash(self) -> str:
        h = hashlib.sha256()
        for name, p in sorted(self.model.named_parameters()):
            if p.requires_grad:
                h.update(name.encode())
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:12]

    def load_adapter(self, path: str | Path) -> str:
        """Load LoRA weights saved by save_adapter() (to resume a run); returns the adapter hash."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = load_file(str(Path(path) / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(self.model, sd)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        if missing:
            raise RuntimeError(f"adapter at {path} lacks {len(missing)} LoRA tensors, e.g. {missing[:2]}")
        return self.adapter_hash()

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

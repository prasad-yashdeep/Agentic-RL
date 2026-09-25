"""Gemma 4 E4B as a LoRA-GRPO policy on one 16 GB card.

What differs from a plain Qwen-style causal LM (all measured/verified 2026-09-25, see experiments/EXPERIMENTS.md
"Trainer port"):
- The HF class for the checkpoint is the multimodal `Gemma4ForConditionalGeneration`. With `vision_config` and
  `audio_config` set to None it builds the language model only, under the checkpoint's own names
  (`model.language_model.layers.N…`), which is the adapter prefix vLLM's `--language-model-only` server maps.
- The per-layer embedding table (262,144 × 42 × 256 bf16 = 5.25 GiB) is read one row per token: `CpuEmbedding` keeps
  it in host memory and moves only the batch's rows.
- The base weights are the FP8 checkpoint the server runs (`FP8Linear`: float8_e4m3fn weight + per-channel bf16 scale,
  dequantised per forward), so trainer and sampler share weights bit for bit; bf16 (the Google snapshot) is the
  fallback and does not fit a 24k-token update on 16 GB.
- The final logits are soft-capped (30·tanh(x/30)) in bf16 before the log-softmax; `LoRATrainer.chunk_lp` does that.
- LoRA targets are scoped by regex to the language model's seven projections; layers 24–41 share KV and have no
  k_proj/v_proj, so there are 258 targets (34,881,536 parameters at r=16).
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarl.paths import HF_CACHE, MODELS, SCRATCH

LORA_TARGET_REGEX = r".*\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
LM_PREFIX = "model.language_model."


def is_gemma4(model_id: str, revision: str | None = None) -> bool:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id, revision=revision).model_type in ("gemma4", "gemma4_text")


def text_only_config(model_id: str, revision: str | None = None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    if cfg.model_type == "gemma4":  # the towers are built only when their sub-config is not None
        cfg.vision_config = None
        cfg.audio_config = None
    return cfg


def snapshot_path(model_id: str, revision: str) -> Path:
    """The HF snapshot directory: a node-local copy under SCRATCH/hf-cache when one exists (NVMe), else the NFS cache."""
    rel = Path("hub") / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    local = SCRATCH / "hf-cache" / rel
    return local if (local / "config.json").exists() else HF_CACHE / rel


class CpuEmbedding(nn.Module):
    """Wraps a (scaled) nn.Embedding whose table stays in host memory: the batch's rows are gathered on the CPU and only
    they are moved to the GPU. `_apply` is a no-op, so `model.to(device)` and `.to(dtype)` leave the table alone."""

    def __init__(self, inner: nn.Embedding):
        super().__init__()
        self.inner = inner.to("cpu")

    def _apply(self, fn, recurse: bool = True):
        return self

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = F.embedding(ids.to("cpu"), self.inner.weight, padding_idx=self.inner.padding_idx).to(ids.device)
        scale = getattr(self.inner, "embed_scale", None)
        if scale is None:
            return rows
        return rows * scale.to(self.inner.weight.dtype).to(ids.device)  # HF: embedding * embed_scale (bf16, a power of two)


class FP8Linear(nn.Linear):
    """An nn.Linear whose weight is the served checkpoint's float8_e4m3fn tensor with a per-output-channel bf16 scale,
    dequantised on every forward: what vLLM's W8A16 path computes on sm_86. Subclassing nn.Linear keeps PEFT's LoRA
    wrapper applicable; there is deliberately no `weight` attribute, so PEFT leaves the adapter in fp32."""

    def __init__(self, w8: torch.Tensor, scale: torch.Tensor):
        super().__init__(w8.shape[1], w8.shape[0], bias=False, device="meta")
        del self.weight
        self.register_buffer("w8", w8)
        self.register_buffer("scale", scale.reshape(-1, 1))
        # PEFT reads `weight` only for its device and dtype (and a distributed-tensor check): a one-element bf16 probe
        # gives it the scale's dtype, so the adapter is created in bf16 and upcast to fp32 by PEFT's default, never fp8
        self.register_buffer("weight_probe", torch.zeros(1, dtype=self.scale.dtype))

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_probe

    def dequant(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.w8.to(self.scale.dtype) * self.scale).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant(x.dtype))


def pin_ple_on_cpu(model) -> None:
    lm = language_model(model)
    if not isinstance(lm.embed_tokens_per_layer, CpuEmbedding):
        lm.embed_tokens_per_layer = CpuEmbedding(lm.embed_tokens_per_layer)


def language_model(model):
    """The text decoder (`Gemma4TextModel`) of either Gemma 4 class."""
    body = model.model
    return body.language_model if hasattr(body, "language_model") else body


def overlay_fp8(model, fp8_snapshot: Path, log=print) -> dict:
    """Replace every language-model linear with the FP8 checkpoint's weights and copy its bf16 tensors (norms,
    embeddings) over the model's. Tensors the model has no module for (the KV-shared layers' k/v projections and
    norms, the towers, the untied lm_head copy) are skipped and counted."""
    from safetensors import safe_open

    counts: Counter = Counter()
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    t0 = time.time()
    with safe_open(str(fp8_snapshot / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if not k.startswith(LM_PREFIX):
                counts["skipped_outside_lm"] += 1
                continue
            if k.endswith(".weight_scale"):
                continue  # consumed with its weight
            mod_name, leaf = k.rsplit(".", 1)
            if k + "_scale" in keys:  # an FP8 linear
                if mod_name not in modules:
                    counts["skipped_absent_fp8"] += 1
                    continue
                old = modules[mod_name]
                w8, scale = f.get_tensor(k), f.get_tensor(k + "_scale")
                if not isinstance(old, nn.Linear) or tuple(w8.shape) != (old.out_features, old.in_features):
                    raise RuntimeError(f"{mod_name}: expected nn.Linear {(getattr(old, 'out_features', None), getattr(old, 'in_features', None))}, checkpoint {tuple(w8.shape)}")
                parent_name, attr = mod_name.rsplit(".", 1)
                setattr(modules[parent_name], attr, FP8Linear(w8, scale))
                counts["fp8_linear"] += 1
                continue
            if k in params:
                t = f.get_tensor(k)
                if tuple(t.shape) != tuple(params[k].shape):
                    raise RuntimeError(f"{k}: shape {tuple(t.shape)} vs model {tuple(params[k].shape)}")
                with torch.no_grad():
                    params[k].copy_(t)
                counts["bf16_copied"] += 1
            elif mod_name in modules and isinstance(modules[mod_name], CpuEmbedding) and leaf == "weight":
                with torch.no_grad():
                    modules[mod_name].inner.weight.copy_(f.get_tensor(k))
                counts["bf16_copied"] += 1
            else:
                counts["skipped_absent_bf16"] += 1
    left = [n for n, m in language_model(model).named_modules() if type(m) is nn.Linear]
    if left:
        raise RuntimeError(f"{len(left)} language-model linears not covered by the FP8 checkpoint, e.g. {left[:3]}")
    counts["seconds"] = round(time.time() - t0, 1)
    log(f"fp8 overlay from {fp8_snapshot}: {dict(counts)}")
    return dict(counts)


def load_policy_model(model_id: str, revision: str | None, *, base: str = "fp8", base_snapshot: str | None = None,
                      dtype=torch.bfloat16, attn: str = "sdpa", ple_on_cpu: bool = True, log=print):
    """Gemma 4 for training: text-only, PLE on the CPU, base weights as served (fp8) or bf16. Returned on the CPU."""
    from transformers import AutoModelForCausalLM

    if base not in ("fp8", "bf16"):
        raise ValueError(f"base must be 'fp8' or 'bf16', got {base!r}")
    cfg = text_only_config(model_id, revision)
    src = snapshot_path(model_id, revision) if revision else model_id
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(str(src), config=cfg, dtype=dtype, attn_implementation=attn,
                                                       output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if k != "lm_head.weight"]
    if missing:
        raise RuntimeError(f"{len(missing)} language-model tensors missing from {src}, e.g. {missing[:3]}")
    unexpected = [k for k in info.get("unexpected_keys", []) if not (k.startswith("model.vision_tower") or k.startswith("model.audio_tower")
                                                                     or k.startswith("model.embed_vision") or k.startswith("model.embed_audio"))]
    if unexpected:
        raise RuntimeError(f"unexpected non-tower tensors in {src}, e.g. {unexpected[:3]}")
    log(f"loaded {src} ({base=} pending) in {time.time() - t0:.0f} s; towers {'absent' if cfg.vision_config is None else 'present'}")
    if ple_on_cpu:
        pin_ple_on_cpu(model)
    if base == "fp8":
        key = base_snapshot or "gemma4-e4b-fp8"
        mid, rev = MODELS[key]
        overlay_fp8(model, snapshot_path(mid, rev), log=log)
    model.lm_head.weight = language_model(model).embed_tokens.weight  # tied (the bf16 checkpoint has no lm_head tensor)
    return model


def summary(model) -> dict:
    lm = language_model(model)
    return {"fp8_linears": sum(1 for m in lm.modules() if isinstance(m, FP8Linear)),
            "ple_device": str(lm.embed_tokens_per_layer.weight.device),
            "lm_head_tied": model.lm_head.weight is lm.embed_tokens.weight,
            "softcap": getattr(model.config.get_text_config(), "final_logit_softcapping", None)}

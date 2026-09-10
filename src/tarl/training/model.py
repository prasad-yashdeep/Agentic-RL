"""GPU-only pieces of the trainer (extraction of 06:187-207 and 257-285), guarded so the module
imports on CPU and only fails when a CUDA path is actually entered."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from tarl.common.env import require_cuda
from tarl.common.paths import BASE_MODEL

# Bare names would also match the vision/audio towers' Gemma4ClippableLinear (PEFT-incompatible);
# regex-scope LoRA to the language model (gate G1 finding, EXPERIMENTS.md incident 11).
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


def load_trainable(
    base_model: str = BASE_MODEL,
    *,
    lora: LoraSpec = LoraSpec(),
    device_index: int = 0,
    quant_4bit: bool = True,
):
    """4-bit NF4 base + LoRA adapters + gradient checkpointing (needs a CUDA GPU)."""
    require_cuda("load_trainable")
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base_model)
    cfg = AutoConfig.from_pretrained(base_model)
    cls = getattr(transformers, cfg.architectures[0])
    kw = {"device_map": {"": device_index}, "dtype": torch.bfloat16}
    if quant_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = cls.from_pretrained(base_model, **kw)
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, task_type="CAUSAL_LM", target_modules=LORA_TARGET_REGEX
    )
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return tok, model


def adapter_hash(model) -> str:
    """Hash the LoRA tensors — the state training actually changes. (Hashing merged shards is
    misleading: head bytes are embeddings LoRA never touches, and sub-ulp deltas fold into bf16
    as zero.) CPU-safe."""
    import torch

    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if "lora" in n:
            h.update(p.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:12]


def merge_and_export(model, tok, *, base_model: str = BASE_MODEL, adapter_dir: Path, merged_dir: Path) -> str:
    """Save adapter -> merge into a bf16 CPU copy of the base (G1-verified path) -> save merged.
    Returns the adapter hash. (A merge on the live 4-bit model would be lossy.)"""
    require_cuda("merge_and_export")
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoConfig

    adapter_dir, merged_dir = Path(adapter_dir), Path(merged_dir)
    model.save_pretrained(str(adapter_dir))
    cfg = AutoConfig.from_pretrained(base_model)
    cls = getattr(transformers, cfg.architectures[0])
    base = cls.from_pretrained(base_model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    del base, merged
    return adapter_hash(model)

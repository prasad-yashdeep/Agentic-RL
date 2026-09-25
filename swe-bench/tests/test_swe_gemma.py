"""The Gemma 4 trainer path on a tiny random Gemma 4 (CPU, no download): soft-capped log-probs, adapter key prefix and
scope, the CPU-pinned per-layer embedding table, and the FP8 base layer under PEFT."""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sarl.gemma import LORA_TARGET_REGEX, CpuEmbedding, FP8Linear, language_model, pin_ple_on_cpu
from sarl.grpo import LoRATrainer, TrainConfig


@pytest.fixture(scope="module")
def tiny_gemma4():
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

    torch.manual_seed(0)
    text = Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_kv_shared_layers=2,
                            layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=8, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, global_head_dim=16, hidden_size_per_layer_input=8,
                            vocab_size_per_layer_input=64, final_logit_softcapping=2.0, tie_word_embeddings=True)  # a cap the tiny logits reach
    cfg = Gemma4Config(text_config=text, vision_config=None, audio_config=None)
    m = Gemma4ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():  # random-init logits are ~0.1; scale the tied embedding so the soft-cap (2.0) actually bites
        m.model.language_model.embed_tokens.weight.mul_(40.0)
    return m


def _ids(T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 64, (T,), generator=g).tolist()
    mask = [0] * 10 + [1] * 12 + [0] * 8 + [1] * 10
    return ids, mask


def test_gemma_logprobs_match_model_forward_with_softcap(tiny_gemma4):
    tr = LoRATrainer("tiny", model=tiny_gemma4, cfg=TrainConfig(chunk=5, grad_checkpointing=False, targets=LORA_TARGET_REGEX), device="cpu")
    assert tr.softcap == 2.0
    ids, mask = _ids()
    lp = tr.logprobs_nograd(ids, mask)
    with torch.no_grad():
        logits = tr.model(input_ids=torch.tensor([ids])).logits[0].float()  # HF applies the soft-cap itself
    full = torch.log_softmax(logits[:-1], -1).gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).squeeze(-1)
    sel = full[torch.tensor(mask[1:], dtype=torch.bool)].numpy()
    assert lp.shape == sel.shape and np.allclose(lp, sel, atol=1e-5)
    tr.softcap = None  # without the cap the numbers must differ: proves the cap is applied on this path
    assert not np.allclose(tr.logprobs_nograd(ids, mask), sel, atol=1e-5)


def test_gemma_adapter_keys_stay_in_the_language_model(tiny_gemma4, tmp_path):
    tr = LoRATrainer("tiny", model=tiny_gemma4, cfg=TrainConfig(lora_r=4, lora_alpha=8, grad_checkpointing=False, targets=LORA_TARGET_REGEX), device="cpu")
    tr.save_adapter(tmp_path)
    from safetensors.torch import load_file

    keys = sorted(load_file(str(tmp_path / "adapter_model.safetensors")))
    assert keys and all(k.startswith("base_model.model.model.language_model.layers.") for k in keys)
    assert not any("tower" in k or "embed" in k or "lm_head" in k for k in keys)
    # layers 2 and 3 share KV: no k/v projections exist there, so no adapter for them either
    assert not any(re.search(r"layers\.[23]\.self_attn\.(k_proj|v_proj)", k) for k in keys)
    assert len(keys) == 2 * (4 * 5 + 2 * 2)  # lora_A + lora_B for 5 targets x 4 layers + k,v x 2 layers
    assert re.fullmatch(LORA_TARGET_REGEX, "model.language_model.layers.41.mlp.down_proj")
    assert not re.fullmatch(LORA_TARGET_REGEX, "model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not re.fullmatch(LORA_TARGET_REGEX, "model.language_model.layers.3.per_layer_input_gate")


def test_ple_pinned_on_cpu_gives_identical_logprobs(tiny_gemma4):
    ids, mask = _ids()
    plain = LoRATrainer("tiny", model=tiny_gemma4, cfg=TrainConfig(chunk=7, grad_checkpointing=False, targets=LORA_TARGET_REGEX), device="cpu")
    ref = plain.logprobs_nograd(ids, mask)
    base = plain.model.get_base_model()
    pin_ple_on_cpu(base)
    lm = language_model(base)
    assert isinstance(lm.embed_tokens_per_layer, CpuEmbedding)
    base.to("cpu")  # a no-op move must leave the wrapper in place and on the CPU
    assert isinstance(lm.embed_tokens_per_layer, CpuEmbedding) and lm.embed_tokens_per_layer.weight.device.type == "cpu"
    assert np.array_equal(plain.logprobs_nograd(ids, mask), ref)


def test_fp8_linear_matches_dequant_and_trains_under_peft():
    torch.manual_seed(1)
    w = torch.randn(12, 8, dtype=torch.bfloat16) * 4
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).to(torch.bfloat16)
    w8 = (w.float() / scale.float()).to(torch.float8_e4m3fn)
    lin = FP8Linear(w8, scale)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    ref = F.linear(x, (w8.to(torch.bfloat16) * scale).to(x.dtype))
    assert torch.equal(lin(x), ref)
    assert lin.in_features == 8 and lin.out_features == 12 and "weight" not in dict(lin.named_parameters())
    assert lin.weight.dtype == torch.bfloat16 and lin.weight.numel() == 1  # PEFT's dtype/device probe, not the weight

    from peft import LoraConfig, get_peft_model

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)  # a real parameter, as any model has (PEFT infers the device from one)
            self.proj = FP8Linear(w8, scale)

        def forward(self, x):
            return self.proj(self.norm(x).to(torch.bfloat16))

    m = get_peft_model(M(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    lora = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    assert lora and all(p.dtype == torch.float32 for _, p in lora)
    y = m(x.float()).sum()
    y.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora)


def test_trainconfig_from_train_section_keeps_regex_targets():
    tr = TrainConfig.from_train_section({"lr": 1e-4, "lora_r": 8, "lora_alpha": 16, "targets": LORA_TARGET_REGEX, "clip": 1.0,
                                         "max_train_tokens": 8192, "chunk": 512, "loss_agg": "token_mean", "tis_cap": 2.0,
                                         "base": "fp8", "base_snapshot": "gemma4-e4b-fp8", "ple_on_cpu": True}, max_train_tokens=4096)
    assert isinstance(tr.targets, str) and tr.base == "fp8" and tr.ple_on_cpu and tr.max_train_tokens == 4096 and tr.chunk == 512
    assert TrainConfig.from_train_section({"targets": ["q_proj", "v_proj"]}).targets == ("q_proj", "v_proj")

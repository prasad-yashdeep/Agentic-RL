#!/usr/bin/env python3
"""Gate G1 — Gemma-4 E4B QLoRA trainability smoke test (run on node08, GPU1).

Steps (all must pass):
  1. load base 4-bit NF4 + LoRA on explicit text-decoder projections
  2. forward/backward on a text batch -> finite loss, sane adapter grad norm
  3. optimizer step -> loss changes
  4. CPU bf16 merge fidelity: base+adapter logprobs == merge_and_unload logprobs

vLLM-side weight-sync verification is M2.6 (adapter_sync.py); here we prove the
model trains and the merge itself is numerically faithful.
"""

import argparse
import os
import sys

import torch


def log(msg: str) -> None:
    print(f"[g1-smoke] {msg}", flush=True)


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_class(model_id: str):
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(model_id)
    arch = (cfg.architectures or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None)
    if cls is None:
        cls = transformers.AutoModelForCausalLM
    log(f"architecture: {arch} -> {cls.__name__}")
    return cls


def build_batch(tok, device):
    convos = [
        [{"role": "user", "content": "List three uses for the ls command."}],
        [{"role": "user", "content": "What does grep -r do?"}],
    ]
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc["labels"] = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    return {k: v.to(device) for k, v in enc.items()}


def forward_maybe_mm(model, batch):
    """Gemma-4 multimodal forward may require mm_token_type_ids on text batches."""
    try:
        return model(**batch)
    except (TypeError, ValueError) as e:
        if "mm_token_type_ids" not in str(e) and "token_type" not in str(e):
            raise
        log("adding zero mm_token_type_ids to batch (text-only workaround)")
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return model(**batch)


def train_smoke(model_id: str) -> str:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    cls = load_model_class(model_id)
    log("loading 4-bit NF4 model on GPU ...")
    model = cls.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    # NEVER target_modules="all-linear": PEFT crashes on Gemma4ClippableLinear.
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    batch = build_batch(tok, model.device)
    model.train()
    out = forward_maybe_mm(model, batch)
    loss0 = out.loss
    assert torch.isfinite(loss0), f"non-finite loss: {loss0}"
    log(f"initial loss: {loss0.item():.4f}")
    loss0.backward()

    grads = [p.grad for n, p in model.named_parameters() if p.grad is not None and "lora" in n]
    assert grads, "no LoRA gradients found"
    gnorm = torch.norm(torch.stack([g.norm() for g in grads]))
    log(f"adapter grad norm: {gnorm.item():.4f} over {len(grads)} tensors")
    assert 0.0 < gnorm.item() < 1e3, f"grad norm out of sane range: {gnorm.item()}"

    base_params = [p.grad for n, p in model.named_parameters() if p.grad is not None and "lora" not in n]
    assert not base_params, f"{len(base_params)} base tensors received grads (mask leak)"

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss1 = forward_maybe_mm(model, batch).loss
    log(f"post-step loss: {loss1.item():.4f} (delta {loss0.item() - loss1.item():+.5f})")
    assert loss1.item() != loss0.item(), "loss identical after optimizer step"

    adapter_dir = "/tmp/g1-adapter"
    model.save_pretrained(adapter_dir)
    log("SMOKE_TRAIN_OK")

    del model, out, loss0, loss1, grads
    torch.cuda.empty_cache()
    return adapter_dir


def merge_fidelity(model_id: str, adapter_dir: str) -> None:
    """CPU bf16: logprobs(base+adapter) must equal logprobs(merged)."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    probe = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")

    cls = load_model_class(model_id)
    log("loading bf16 base on CPU for merge check ...")
    base = cls.from_pretrained(model_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    def probe_logprobs(m):
        with torch.no_grad():
            logits = forward_maybe_mm(m, dict(probe)).logits.float()
        return torch.log_softmax(logits, dim=-1)

    lp_adapter = probe_logprobs(model)
    merged = model.merge_and_unload()
    lp_merged = probe_logprobs(merged)

    diff = (lp_adapter - lp_merged).abs().max().item()
    log(f"max |logprob diff| adapter-vs-merged: {diff:.6f}")
    assert diff < 5e-3, f"merge changed logprobs by {diff} (>{5e-3})"
    log("SMOKE_MERGE_OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TARL_BASE_MODEL", "google/gemma-4-E4B-it"))
    ap.add_argument("--skip-merge-check", action="store_true")
    args = ap.parse_args()

    log(f"model: {args.model}")
    adapter_dir = train_smoke(args.model)
    if not args.skip_merge_check:
        merge_fidelity(args.model, adapter_dir)
    log("G1_SMOKE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

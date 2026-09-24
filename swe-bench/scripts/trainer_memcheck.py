#!/usr/bin/env python3
"""Peak GPU memory and time of one LoRA update at several trajectory lengths, with a run config's train settings.
Run on node08 with the target GPU free: CUDA_VISIBLE_DEVICES=1 python scripts/trainer_memcheck.py configs/<run>.yaml

The mask marks ~30 % of positions as sampled (a typical agent trajectory: long observations, short replies); the
logits are chunked, so the mask share matters little. Tokens are random ids — memory does not depend on content."""

from __future__ import annotations

import sys
import time

import torch

from sarl.grpo import LoRATrainer, TrainConfig
from sarl.train import load_config


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config(argv[0])
    lengths = [int(x) for x in argv[1:]] or [16_384, 24_576, 32_768]
    tr = cfg["train"]
    trainer = LoRATrainer(cfg["model"], cfg.get("revision"), TrainConfig(
        lr=tr["lr"], lora_r=tr["lora_r"], lora_alpha=tr["lora_alpha"], targets=tuple(tr["targets"]), clip=tr["clip"],
        max_train_tokens=max(lengths), chunk=tr.get("chunk", 1024), loss_agg=tr.get("loss_agg", "seq_mean"), tis_cap=tr.get("tis_cap")))
    print(f"trainable {trainer.n_trainable:,}; after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    g = torch.Generator().manual_seed(0)
    for T in lengths:
        ids = torch.randint(0, 150_000, (T,), generator=g).tolist()
        mask = [(1 if (i % 10) < 3 else 0) if i > 2000 else 0 for i in range(T)]
        slp = [-1.0 if m else None for m in mask]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            out = trainer.step([(ids, mask, 1.0, slp)])
            torch.cuda.synchronize()
            print(f"T={T:6d}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved, {time.time() - t0:.1f} s, stepped {out.get('stepped')}", flush=True)
        except torch.OutOfMemoryError as e:
            print(f"T={T:6d}: OOM ({str(e)[:120]})", flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

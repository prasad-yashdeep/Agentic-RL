#!/usr/bin/env python3
"""`vllm serve` with Gemma 4's per-layer embedding table (PLE) kept in pinned CPU memory and read in place by the GPU.

Gemma 4 E4B's PLE table (262,144 tokens x 42 layers x 256, bf16) is 5.25 GiB of the 10.8 GiB vLLM loads, yet each token
only reads its own 21.5 KB row of it. vLLM 0.25.1's `--cpu-offload-params` only reaches parameters created inside
`make_layers` (the decoder layers), and the PLE is built outside them. This launcher builds that one table in CPU
memory, pins it and hands the model a CUDA view of it (UVA, the same mechanism as vLLM's UVAOffloader). Embedding
lookups then gather their rows over PCIe, and everything else stays on the GPU.

The patch runs at import time (module top level), so it also applies in a spawned EngineCore: multiprocessing's spawn
re-imports this file as `__mp_main__`. Usage is identical to `vllm serve ...`:

    python scripts/vllm_ple_offload.py serve <model> [vllm serve flags]
"""

from __future__ import annotations

import inspect
import os
import sys

import torch
from vllm.model_executor.layers import vocab_parallel_embedding as _vpe

PLE_NAME = "embed_tokens_per_layer"
_orig_init = _vpe.VocabParallelEmbedding.__init__
_sig = inspect.signature(_orig_init)


def is_ple(args: tuple, kwargs: dict) -> bool:
    """Whether a VocabParallelEmbedding(...) call builds the PLE table. ParallelLMHead passes `prefix` positionally."""
    prefix = _sig.bind(None, *args, **kwargs).arguments.get("prefix") or ""
    return prefix.rsplit(".", 1)[-1] == PLE_NAME


def _init(self, *args, **kwargs):
    if not is_ple(args, kwargs):
        return _orig_init(self, *args, **kwargs)
    from vllm.logger import init_logger
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    with torch.device("cpu"):  # untouched pages: no GPU allocation and no RSS until pinned below
        _orig_init(self, *args, **kwargs)
    total = 0
    for p in self.parameters():
        # Pin an exact-size buffer in place. torch's pinned allocator (pin_memory=True) rounds 5.25 GiB up to the next
        # power of two, which would lock 8 GiB of a 15 GB host that also runs the sandboxes.
        buf = torch.empty(p.shape, dtype=p.dtype, device="cpu")  # filled by the weight loader
        nbytes = buf.numel() * buf.element_size()
        err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        p.data = get_accelerator_view_from_cpu_tensor(buf)  # the view keeps `buf` alive
        p._vllm_is_uva_offloaded = True  # vLLM's loader re-offloads it if post-processing replaces the tensor
        total += nbytes
    init_logger("vllm.ple_offload").info("PLE offload: %.2f GiB kept in pinned CPU memory (UVA)", total / 2**30)


_vpe.VocabParallelEmbedding.__init__ = _init

if __name__ == "__main__":
    try:  # die first under memory pressure, not the sandboxes that share this host
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(os.environ.get("SARL_SERVE_OOM_SCORE_ADJ", "0"))
    except OSError:
        pass
    from vllm.entrypoints.cli.main import main

    sys.argv[0] = "vllm"
    sys.exit(main())

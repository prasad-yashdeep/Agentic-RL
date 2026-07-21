# Version & Flag Pins (frozen at M0.1 — never bump mid-run)

| What | Pin | Why |
|---|---|---|
| Python | ≥ 3.12 (node08 system 3.12.3) | trl.experimental.harbor requires ≥3.12 |
| transformers | ==5.5.4 | first verified-stable for Gemma4ForConditionalGeneration |
| trl | 1.8.x | GRPOTrainer + environment_factory + experimental.harbor |
| vllm | ≥0.22 stable | gemma4 reasoning/tool parsers; TRL harbor requirement |
| harbor | ≥0.20 | rootless/singularity fixes (PR #2049), singularity backend |
| Policy model | google/gemma-4-E4B-it | the subject of the experiment |
| Serve checkpoint | prithivMLmods/gemma-4-E4B-it-FP8 | bf16 (~16GB) doesn't fit 16GB A4000 with KV |
| Eval sampling | t=1.0, top_p=0.95, top_k=64 | Gemma 4 recommended; pinned for all evals |
| RL rollout sampling | t≈1.2 allowed | terminal-bench-rl precedent |
| Eval benchmark | Terminal-Bench 2.1 ONLY | 2.0/2.1 scores not comparable; 2.1 = fixed tasks |
| Judge | Grok 4.5 (xAI, OpenAI-compatible API) | user decision 2026-07-21; XAI_API_KEY in ~/.config/tarl/env |

## Invariant environment (scripts/env.sh sources this contract)

- `NCCL_P2P_DISABLE=1` — node08 PCIe P2P between the A4000s is broken
- `HF_HOME=$HOME/hf-cache` — model cache on NFS (durability tier)
- `APPTAINER_CACHEDIR=/tmp/apptainer-cache`, sif cache `/tmp/sif-cache` — node-local
- Harbor always: `-e singularity --ek singularity_image_cache_dir=/tmp/sif-cache --ek singularity_no_mount=home,tmp`
- Nothing computes on the headnode; long processes in tmux (serve/train/watch/eval) on node08

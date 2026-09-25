#!/usr/bin/env bash
# Gemma 4 E4B (FP8 checkpoint, W8A16 on the A4000s) through 20_serve_policy.sh with the flags it needs.
# Usage (node08): bash 22_serve_gemma4.sh GPU PORT MEM   e.g. bash 22_serve_gemma4.sh 0 8001 0.87
#   --language-model-only          skip the vision/audio towers (text-only policy)
#   --enforce-eager                vLLM 0.25.1's torch.compile (Inductor) tries to copy the 5.25 GiB per-layer
#                                  embedding table (262,144 x 42 x 256 bf16) and runs out of memory on a 16 GB A4000,
#                                  with or without LoRA (2026-09-25); eager mode starts and serves LoRA
#   --max-num-batched-tokens 4096  smaller profiling batch over the 262k vocabulary
# node04 (RTX 4060 Laptop, 8 GB): SARL_PLE_OFFLOAD=1 keeps the PLE table in CPU memory (5.5 GiB left on the GPU), e.g.
#   SARL_PLE_OFFLOAD=1 SARL_SERVE_OOM_SCORE_ADJ=1000 SARL_MAX_NUM_SEQS=8 SARL_MAX_BATCHED=2048 bash 22_serve_gemma4.sh 0 8003 0.93
# SARL_EAGER=0 drops --enforce-eager (only worth trying with the PLE offloaded, which removes the Inductor OOM's cause)
set -euo pipefail
export SARL_MODEL="${SARL_MODEL:-gemma4-e4b-fp8}" SARL_SERVE_GPU="${1:-0}" SARL_SERVE_PORT="${2:-8001}" SARL_SERVE_MEM="${3:-0.87}"
export SARL_MAX_NUM_SEQS="${SARL_MAX_NUM_SEQS:-16}"
EAGER="--enforce-eager"; [ "${SARL_EAGER:-1}" = 0 ] && EAGER=""
export SARL_SERVE_EXTRA="--language-model-only $EAGER --max-num-batched-tokens ${SARL_MAX_BATCHED:-4096} ${SARL_SERVE_MORE:-}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/20_serve_policy.sh"

#!/usr/bin/env bash
# Gemma 4 E4B (FP8 checkpoint, W8A16 on the A4000s) through 20_serve_policy.sh with the flags it needs.
# Usage (node08): bash 22_serve_gemma4.sh GPU PORT MEM   e.g. bash 22_serve_gemma4.sh 0 8001 0.87
#   --language-model-only          skip the vision/audio towers (text-only policy)
#   --enforce-eager                vLLM 0.25.1's torch.compile (Inductor) tries to copy the 5.25 GiB per-layer
#                                  embedding table (262,144 x 42 x 256 bf16) and runs out of memory on a 16 GB A4000,
#                                  with or without LoRA (2026-09-25); eager mode starts and serves LoRA
#   --max-num-batched-tokens 4096  smaller profiling batch over the 262k vocabulary
set -euo pipefail
export SARL_MODEL="${SARL_MODEL:-gemma4-e4b-fp8}" SARL_SERVE_GPU="${1:-0}" SARL_SERVE_PORT="${2:-8001}" SARL_SERVE_MEM="${3:-0.87}"
export SARL_MAX_NUM_SEQS="${SARL_MAX_NUM_SEQS:-16}"
export SARL_SERVE_EXTRA="--language-model-only --enforce-eager --max-num-batched-tokens ${SARL_MAX_BATCHED:-4096}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/20_serve_policy.sh"

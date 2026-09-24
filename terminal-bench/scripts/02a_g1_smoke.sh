#!/usr/bin/env bash
# Gate G1 wrapper: fetch bf16 base (google repo is gated; fall back to the
# unsloth mirror when no HF token), then run the QLoRA smoke on GPU1.
set -uo pipefail
. "$(dirname "$0")/env.sh"
LOG="$HOME/terminal-agent-rl/g1-smoke.log"
{
    MODEL="google/gemma-4-E4B-it"
    if ! hf download "$MODEL" --exclude "*.gguf" >/dev/null 2>&1; then
        echo "[g1] $MODEL not downloadable (gated, no token) - using unsloth mirror"
        MODEL="unsloth/gemma-4-E4B-it"
        hf download "$MODEL" --exclude "*.gguf" >/dev/null || { echo G1_DL_FAILED; exit 1; }
    fi
    echo "G1_DL_DONE model=$MODEL"
    CUDA_VISIBLE_DEVICES=1 TARL_BASE_MODEL="$MODEL" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python "$(dirname "$0")/02_smoke_gemma4_qlora.py"
} 2>&1 | tee "$LOG"

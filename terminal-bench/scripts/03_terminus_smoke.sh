#!/usr/bin/env bash
# M0.6 — Terminus 2 against OUR vLLM endpoint on 3 easy tasks (run ON node08).
# Validates: JSON action parsing, tmux interaction, transcript capture.
# Not a score run — we expect few/no passes from the untrained model.
set -uo pipefail
. "$(dirname "$0")/env.sh"
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-local}"
TASKS=("${@:-fix-git}")
[ $# -eq 0 ] && TASKS=(fix-git regex-log large-scale-text-editing)
INCLUDE_ARGS=()
for t in "${TASKS[@]}"; do INCLUDE_ARGS+=(-i "$t"); done
exec harbor run -p "${TARL_TB_PATH:-/tmp/tb2}" \
    -a terminus-2 \
    -m openai/gemma-4-E4B-it \
    --ak api_base=http://localhost:8000/v1 \
    --ak temperature=1.0 \
    -e singularity \
    --ek singularity_image_cache_dir="$TARL_SIF_CACHE" \
    --ek singularity_no_mount=home,tmp \
    --n-concurrent "${TARL_N_CONCURRENT:-1}" \
    --timeout-multiplier 2.0 \
    --jobs-dir "$REPO/jobs" \
    "${INCLUDE_ARGS[@]}"

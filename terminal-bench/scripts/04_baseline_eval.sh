#!/usr/bin/env bash
# M1.2 — A0 baseline: full local TB2 clone × K attempts, Terminus 2 vs our
# endpoint. Canonical conditions: timeout multiplier 1.0, temperature 1.0
# (top_p/top_k come from the model's generation_config via vLLM defaults).
# One infra retry (server races); model timeouts are NOT retried as infra.
set -uo pipefail
. "$(dirname "$0")/env.sh"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-local}"
exec harbor run -p "${TARL_TB_PATH:-/tmp/tb2}" \
    -a terminus-2 \
    -m openai/gemma-4-E4B-it \
    --ak api_base=http://localhost:8000/v1 \
    --ak temperature=1.0 \
    -e singularity \
    --ek singularity_image_cache_dir="$TARL_SIF_CACHE" \
    --ek singularity_no_mount=home,tmp \
    -k "${TARL_K:-5}" \
    --n-concurrent "${TARL_N_CONCURRENT:-4}" \
    --timeout-multiplier 1.0 \
    --max-retries 1 --retry-exclude AgentTimeoutError,VerifierTimeoutError \
    --jobs-dir "$HOME/terminal-agent-rl/jobs"

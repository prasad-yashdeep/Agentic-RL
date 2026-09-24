#!/usr/bin/env bash
# M0.5 — vLLM FP8 serve on GPU0. Run ON node08 inside tmux session `serve`.
set -euo pipefail
. "$(dirname "$0")/env.sh"
MODEL="${TARL_SERVE_MODEL:-prithivMLmods/gemma-4-E4B-it-FP8}"
# TARL_THINKING=1 → serve with a template whose enable_thinking defaults to TRUE,
# so agents (harbor/litellm) get thinking without request-level kwargs. The
# baseline's thinking-OFF arm uses the stock template (default false).
if [ "${TARL_THINKING:-0}" = "1" ]; then
    TEMPLATE="$(dirname "$0")/../configs/serve/tool_chat_template_gemma4_thinking.jinja"
else
    TEMPLATE="$(dirname "$0")/../configs/serve/tool_chat_template_gemma4.jinja"
fi
if [ ! -f "$TEMPLATE" ]; then
    # Only the stock template may be auto-fetched; the thinking variant is a
    # repo-tracked derivative and must never be silently replaced by upstream.
    [ "${TARL_THINKING:-0}" = "1" ] && { echo "missing thinking template: $TEMPLATE"; exit 1; }
    mkdir -p "$(dirname "$TEMPLATE")"
    curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja" \
        -o "$TEMPLATE"
fi
# No CUDA toolkit on node08 → flashinfer's JIT sampler can't compile (and with the
# leftover cu13 nvcc it emitted driver-incompatible kernels). Native sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0
# 16 GB card: 12.4 GiB weights leave no room for FULL cudagraphs (OOM at capture);
# piecewise-only + slightly lower mem target keeps startup under budget.
# TARL_SERVE_QUANT=fp8 -> online FP8 quantization (serving a merged bf16
# checkpoint from the trainer; the pre-quantized FP8 repo needs no flag)
QUANT_ARGS=()
[ "${TARL_SERVE_QUANT:-}" = "fp8" ] && QUANT_ARGS=(--quantization fp8)
CUDA_VISIBLE_DEVICES=0 exec vllm serve "$MODEL" \
    --served-model-name gemma-4-E4B-it \
    "${QUANT_ARGS[@]}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.87 \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 --enable-auto-tool-choice \
    --chat-template "$TEMPLATE" \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}'

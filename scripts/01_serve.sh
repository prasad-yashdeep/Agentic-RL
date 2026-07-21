#!/usr/bin/env bash
# M0.5 — vLLM FP8 serve on GPU0. Run ON node08 inside tmux session `serve`.
set -euo pipefail
. "$(dirname "$0")/env.sh"
MODEL="${TARL_SERVE_MODEL:-prithivMLmods/gemma-4-E4B-it-FP8}"
TEMPLATE="$(dirname "$0")/../configs/serve/tool_chat_template_gemma4.jinja"
if [ ! -f "$TEMPLATE" ]; then
    mkdir -p "$(dirname "$TEMPLATE")"
    curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja" \
        -o "$TEMPLATE"
fi
CUDA_VISIBLE_DEVICES=0 exec vllm serve "$MODEL" \
    --served-model-name gemma-4-E4B-it \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 --enable-auto-tool-choice \
    --chat-template "$TEMPLATE" \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}'

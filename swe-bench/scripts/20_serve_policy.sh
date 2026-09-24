#!/usr/bin/env bash
# vLLM policy server for swe-bench (the diagram's "POLICY πθ, vLLM generated"), node08 GPU0, port 8001.
# Runtime LoRA updating lets the trainer push each new adapter (πθ′) with POST /v1/load_lora_adapter.
# Run in tmux `swe-serve` (make -C swe-bench serve). Model = the pinned local snapshot (offline, exact revision).
set -euo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
REV=ea3f2471cf1b1f0db85067f1ef93848e38e88c25
SNAP="${HF_HOME:-$HOME/hf-cache}/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/$REV"
[ -f "$SNAP/config.json" ] || { echo "model snapshot missing: $SNAP (make -C swe-bench assets)"; exit 1; }
export HF_HUB_OFFLINE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1
mkdir -p "$SWE/runs"
CUDA_VISIBLE_DEVICES="${SARL_SERVE_GPU:-0}" exec "$ARL/.venv/bin/vllm" serve "$SNAP" \
    --served-model-name policy \
    --host 0.0.0.0 --port "${SARL_SERVE_PORT:-8001}" \
    --max-model-len 32768 \
    --gpu-memory-utilization "${SARL_SERVE_MEM:-0.85}" \
    --enable-lora --max-lora-rank 16 --max-loras 2 \
    --generation-config vllm \
    --enable-prefix-caching \
    2>&1 | tee -a "$SWE/runs/serve.log"

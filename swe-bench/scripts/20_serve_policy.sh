#!/usr/bin/env bash
# vLLM policy server for swe-bench (the diagram's "POLICY πθ, vLLM generated"), node08 GPU0, port 8001.
# Runtime LoRA updating lets the trainer push each new adapter (πθ′) with POST /v1/load_lora_adapter.
# Run in tmux `swe-serve` (make -C swe-bench serve [MODEL=1.5b]). Model = a pinned local snapshot from sarl.paths.MODELS.
set -euo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
KEY="${SARL_MODEL:-0.5b}"
SNAP=$("$ARL/.venv/bin/python" -c "from sarl.paths import model_snapshot; print(model_snapshot('$KEY'))")
[ -f "$SNAP/config.json" ] || { echo "model snapshot missing: $SNAP (download the pinned revision first)"; exit 1; }
export HF_HUB_OFFLINE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1
export PATH="$ARL/.venv/bin:$PATH"  # vLLM's FP8-on-Ampere kernels JIT-compile and need the venv's ninja
# SARL_SERVE_EXTRA: model-specific flags, e.g. Gemma 4 E4B FP8:
#   --language-model-only --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
read -r -a EXTRA_ARGS <<< "${SARL_SERVE_EXTRA:-}"
LORA_ARGS=(--enable-lora --max-lora-rank 16 --max-loras 2)
[ "${SARL_NO_LORA:-0}" = 1 ] && LORA_ARGS=()  # probes that never load an adapter
mkdir -p "$SWE/runs"
# SARL_MAX_NUM_SEQS=16: capture only small CUDA graphs. vLLM 0.25.1's V2 runner does not count its graph pool (1.2 GiB at
# the default 256) in --gpu-memory-utilization, which on a shared GPU leaves no room for the trainer's CUDA context.
# SARL_SLEEP_MODE=1: a server that shares its GPU with the trainer; POST /sleep?level=1 offloads its weights to CPU and
# frees the KV cache while the trainer steps, POST /wake_up restores it (the endpoints need VLLM_SERVER_DEV_MODE=1)
SLEEP_ARGS=()
if [ "${SARL_SLEEP_MODE:-0}" = 1 ]; then export VLLM_SERVER_DEV_MODE=1; SLEEP_ARGS=(--enable-sleep-mode); fi
# SARL_PLE_OFFLOAD=1 (Gemma 4): keep the 5.25 GiB per-layer embedding table in pinned CPU memory, read in place over
# PCIe (scripts/vllm_ple_offload.py); what lets E4B fit node04's 8 GB GPU. SARL_SERVE_OOM_SCORE_ADJ=1000 makes the
# server, not the sandboxes, the OOM killer's first choice on a host that also runs rollouts.
VLLM=("$ARL/.venv/bin/vllm")
[ "${SARL_PLE_OFFLOAD:-0}" = 1 ] && VLLM=("$ARL/.venv/bin/python" "$SWE/scripts/vllm_ple_offload.py")
echo "serving $KEY from $SNAP (sleep mode ${SARL_SLEEP_MODE:-0}, PLE offload ${SARL_PLE_OFFLOAD:-0})"
CUDA_VISIBLE_DEVICES="${SARL_SERVE_GPU:-0}" exec "${VLLM[@]}" serve "$SNAP" \
    --served-model-name policy \
    --host 0.0.0.0 --port "${SARL_SERVE_PORT:-8001}" \
    --max-model-len "${SARL_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${SARL_MAX_NUM_SEQS:-256}" \
    --gpu-memory-utilization "${SARL_SERVE_MEM:-0.85}" \
    "${LORA_ARGS[@]}" \
    --generation-config vllm \
    --enable-prefix-caching \
    "${SLEEP_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$SWE/runs/serve.log"

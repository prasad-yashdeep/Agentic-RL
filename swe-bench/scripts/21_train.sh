#!/usr/bin/env bash
# The swe-bench GRPO loop on node08 GPU1 against the vLLM serve on GPU0 (tmux `swe-train`; make -C swe-bench train).
# Usage: bash 21_train.sh [config] [extra sarl-train args, e.g. --steps 1 --run-name gate-s7]
set -uo pipefail
case "$(hostname)" in bcm11-headnode*) echo "refusing on the headnode"; exit 1;; esac
SWE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; ARL="$(cd "$SWE/.." && pwd -P)"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="${SARL_TRAIN_GPU:-1}"
cd "$SWE" && mkdir -p runs
CFG="${1:-configs/basic_run.yaml}"; shift || true
"$ARL/.venv/bin/python" -m sarl.cli train "$CFG" "$@" 2>&1 | tee -a "runs/train-$(date +%Y%m%d-%H%M%S).log"

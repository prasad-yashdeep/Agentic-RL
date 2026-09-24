#!/usr/bin/env bash
# M0.4 / Gate G0 — Harbor oracle smoke on selected TB tasks via singularity backend.
# Usage: 01a_oracle_smoke.sh <task-name> [<task-name> ...]   Run ON node08.
set -euo pipefail
. "$(dirname "$0")/env.sh"
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/
mkdir -p "$TARL_SIF_CACHE" "$APPTAINER_CACHEDIR"
# terminal-bench@2.1 is not in the harbor registry — use a local clone of
# laude-institute/terminal-bench-2 (TARL_TB_PATH) as the source of truth.
if [ -n "${TARL_TB_PATH:-}" ]; then
    DATASET_ARGS=(-p "$TARL_TB_PATH")
else
    DATASET_ARGS=(-d "${TARL_TB_DATASET:-terminal-bench}")
fi
INCLUDE_ARGS=()
for t in "$@"; do INCLUDE_ARGS+=(-i "$t"); done
exec harbor run "${DATASET_ARGS[@]}" -a oracle \
    -e singularity \
    --ek singularity_image_cache_dir="$TARL_SIF_CACHE" \
    --ek singularity_no_mount=home,tmp \
    --n-concurrent "${TARL_N_CONCURRENT:-1}" \
    --timeout-multiplier "${TARL_TIMEOUT_MULT:-2.0}" \
    --jobs-dir "$REPO/jobs" \
    "${INCLUDE_ARGS[@]}"

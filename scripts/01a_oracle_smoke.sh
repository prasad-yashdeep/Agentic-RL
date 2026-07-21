#!/usr/bin/env bash
# M0.4 / Gate G0 — Harbor oracle smoke on a few TB tasks via singularity backend.
# Usage: 01a_oracle_smoke.sh <task-id> [<task-id> ...]   Run ON node08.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p "$TARL_SIF_CACHE" "$APPTAINER_CACHEDIR"
TASK_ARGS=()
for t in "$@"; do TASK_ARGS+=(-t "$t"); done
exec harbor run -d terminal-bench/terminal-bench-2 -a oracle \
    -e singularity \
    --ek singularity_image_cache_dir="$TARL_SIF_CACHE" \
    --ek singularity_no_mount=home,tmp \
    --n-concurrent 2 \
    "${TASK_ARGS[@]}"

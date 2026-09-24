#!/usr/bin/env bash
# One-shot: reinstall torch stack as cu128 (driver 570.133 caps CUDA at 12.8;
# vllm 0.25 pulls cu13 wheels by default → "NVIDIA driver too old" at engine init).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/
[ "${TARL_ALLOW_HISTORICAL:-0}" = 1 ] || { echo "historical one-off (2026-07); re-running would rebuild or re-resolve the shared venv. Set TARL_ALLOW_HISTORICAL=1 to run anyway."; exit 1; }
ARL_ROOT="$(cd "$REPO/.." && pwd -P)"   # the shared venv lives at the monorepo root since 2026-09-24
cd "$REPO"
{
    . "$ARL_ROOT/.venv/bin/activate" \
    && uv pip install --reinstall --torch-backend=cu128 vllm \
    && uv pip freeze > configs/lockfile.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" \
    && echo CU128_FIX_DONE
} 2>&1 | tee "$REPO/cu128-fix.log"

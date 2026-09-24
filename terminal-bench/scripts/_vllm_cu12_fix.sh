#!/usr/bin/env bash
# PyPI vllm 0.25.1 wheel links libcudart.so.13; driver 570.133 is CUDA 12.8.
# Use the +cu129 release wheel (12.x minor-version compatible) on the cu128 torch stack.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/
[ "${TARL_ALLOW_HISTORICAL:-0}" = 1 ] || { echo "historical one-off (2026-07); re-running would rebuild or re-resolve the shared venv. Set TARL_ALLOW_HISTORICAL=1 to run anyway."; exit 1; }
ARL_ROOT="$(cd "$REPO/.." && pwd -P)"   # the shared venv lives at the monorepo root since 2026-09-24
cd "$REPO"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
{
    . "$ARL_ROOT/.venv/bin/activate" \
    && uv pip install --torch-backend=cu128 "vllm @ $WHEEL" \
    && uv pip freeze > configs/lockfile.txt \
    && python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda ok', torch.cuda.is_available())" \
    && echo VLLM_CU12_FIX_DONE \
    && tmux kill-session -t serve 2>/dev/null; \
    tmux new-session -d -s serve "cd $REPO && bash scripts/01_serve.sh 2>&1 | tee $REPO/serve.log"
} 2>&1 | tee "$REPO/vllm-cu12-fix.log"

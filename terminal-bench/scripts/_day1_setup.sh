#!/usr/bin/env bash
# Day-1 M0.1 environment bring-up. Run ON node08 (tmux session `setup`).
set -euo pipefail
REPO="${TARL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"   # self-locating: terminal-bench/
[ "${TARL_ALLOW_HISTORICAL:-0}" = 1 ] || { echo "historical one-off (2026-07); re-running would rebuild or re-resolve the shared venv. Set TARL_ALLOW_HISTORICAL=1 to run anyway."; exit 1; }
ARL_ROOT="$(cd "$REPO/.." && pwd -P)"   # the shared venv lives at the monorepo root since 2026-09-24
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv -p /usr/bin/python3
. .venv/bin/activate
uv pip install -e . --torch-backend=cu128  # driver 570.133 supports CUDA <= 12.8; cu13 wheels fail CUDA init
uv pip freeze > configs/lockfile.txt
uv tool install harbor
uv tool install 'huggingface_hub[cli]'
python -c "import transformers, trl, vllm, peft; print('transformers', transformers.__version__); print('trl', trl.__version__); print('vllm', vllm.__version__); print('peft', peft.__version__)"
harbor --version || true
echo DAY1_SETUP_DONE

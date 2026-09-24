#!/usr/bin/env bash
# Day-1 M0.1 environment bring-up. Run ON node08 (tmux session `setup`).
set -euo pipefail
cd "$HOME/terminal-agent-rl"
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

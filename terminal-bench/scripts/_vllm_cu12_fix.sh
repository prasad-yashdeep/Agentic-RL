#!/usr/bin/env bash
# PyPI vllm 0.25.1 wheel links libcudart.so.13; driver 570.133 is CUDA 12.8.
# Use the +cu129 release wheel (12.x minor-version compatible) on the cu128 torch stack.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/terminal-agent-rl"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
{
    . .venv/bin/activate \
    && uv pip install --torch-backend=cu128 "vllm @ $WHEEL" \
    && uv pip freeze > configs/lockfile.txt \
    && python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda ok', torch.cuda.is_available())" \
    && echo VLLM_CU12_FIX_DONE \
    && tmux kill-session -t serve 2>/dev/null; \
    tmux new-session -d -s serve "cd $HOME/terminal-agent-rl && bash scripts/01_serve.sh 2>&1 | tee $HOME/terminal-agent-rl/serve.log"
} 2>&1 | tee "$HOME/terminal-agent-rl/vllm-cu12-fix.log"
